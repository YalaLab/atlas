"""
Engine base class for distributed training, evaluation.
"""
import os
from collections import OrderedDict
from typing import Literal

import time
import torch
import wandb
from tqdm import tqdm

from sandstone import losses, metrics
from sandstone.utils.engine import gather_step_outputs
from sandstone.utils.misc import rank_zero_only, logger

Split = Literal["train", "val", "test"]


def parse_amp_precision(precision):
    parse = {
        "bf16-mixed": torch.bfloat16,
        "fp16-mixed": torch.float16,
        "32": None,  # disable AMP
    }

    return parse[str(precision)]


class Engine(object):
    def __init__(
        self,
        args,
        *,
        accumulate_grad_batches=1,
        resume,
        max_epochs,
        precision,
        clip_grad=None,
        log_grad_norm=False,
        dataset_info=None,
        limit_num_batches=None,
        **kwargs,
    ):
        self.args = args
        self.accum_iter = accumulate_grad_batches
        self.training_step_outputs = []
        self.dataset_info = dataset_info
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.global_step = 0
        self.amp_precision = parse_amp_precision(precision)

        self.limit_num_batches = limit_num_batches

        # resume and max_epochs are handled outside of the engine

        if kwargs:
            logger.warning(f"Ignoring unrecognized kwargs to engine: {kwargs}")

    @rank_zero_only
    def save_on_master(self, ckpt_dir=None, epoch=0, state=None):
        # Outside of the engine (in main) we call this function only with master. Here we still use `rank_zero_only` to be safe.
        os.makedirs(ckpt_dir, exist_ok=True)
        if epoch == -1:
            path = f"{ckpt_dir}/latest.ckpt"
        else:
            path = f"{ckpt_dir}/epoch={epoch}.ckpt"
        torch.save(state, path)

    def load(self, path, map_location="cpu"):
        return torch.load(path, map_location=map_location)

    def get_epoch_metrics(self, split: Split):
        if (
            "epoch_metrics" not in self.args.metrics
            or split not in self.args.metrics.epoch_metrics
        ):
            return []
        epoch_metrics = []
        for metric_info in self.args.metrics.epoch_metrics[split]:
            name = metric_info["type"]
            kwargs = metric_info["kwargs"] if "kwargs" in metric_info else {}
            # Survivor metric requires training dataset to compute
            metric = metrics.__dict__[name](
                self.args, dataset_info=self.dataset_info, **kwargs
            )
            epoch_metrics.append(metric)
        return epoch_metrics

    def compute_metrics(self, metric_input):
        if "step_metrics" not in self.args.metrics:
            return []
        metric_dict = OrderedDict()

        for metric_info in self.args.metrics.step_metrics:
            name = metric_info["type"]
            kwargs = metric_info["kwargs"] if "kwargs" in metric_info else {}
            metric_fn = metrics.__dict__[name](self.args, **kwargs)
            local_metric_dict = metric_fn(**metric_input)
            metric_dict.update(local_metric_dict)
        return metric_dict

    def compute_step_metrics(self, loss_input: dict, metric_input: dict, train: bool):
        logging_dict = OrderedDict()

        loss, loss_dict = self.compute_loss(loss_input, train)
        logging_dict.update(loss_dict)

        if metric_input is not None:
            metric_dict = self.compute_metrics(metric_input)
            logging_dict.update(metric_dict)

        return loss, logging_dict

    def compute_epoch_metrics(
        self, result_dict, args, device, *, key_prefix, epoch, split, epoch_metrics
    ):
        stats_dict = OrderedDict()
        assert split is not None
        # Now call additional metric functions that are specialized
        """
            Remove prefix from keys. For instance, convert:
            val_probs -> probs for standard handling in the metric fucntions
        """
        result_dict_wo_key_prefix = {}

        for k, v in result_dict.items():
            if k == "meta":
                continue
            if key_prefix != "" and k.startswith(key_prefix):
                k_wo_prefix = k[len(key_prefix) :]
                result_dict_wo_key_prefix[k_wo_prefix] = v
            else:
                result_dict_wo_key_prefix[k] = v

        result_dict_wo_key_prefix["split"] = split

        epoch_metrics = self.get_epoch_metrics(split)
        for metric_fn in epoch_metrics:
            stats_wo_prefix = metric_fn(**result_dict_wo_key_prefix)
            for k, v in stats_wo_prefix.items():
                if isinstance(v, torch.Tensor):
                    stats_dict[key_prefix + k] = v.to(device=device)
                else:
                    stats_dict[key_prefix + k] = torch.tensor(v, device=device)

        return stats_dict

    def get_losses(self):
        assert (
            len(self.args.metrics.losses) > 0
        ), "Must specify at least one loss function in args.metrics.losses"
        loss_functions = []
        for loss_info in self.args.metrics.losses:
            name = loss_info["type"]
            kwargs = loss_info["kwargs"] if "kwargs" in loss_info else {}
            weight = loss_info["weight"]
            enable_at_train = loss_info.get("enable_at_train", True)
            enable_at_eval = loss_info.get("enable_at_eval", True)
            fn = losses.__dict__[name](self.args, **kwargs)
            loss_functions.append((fn, name, weight, enable_at_train, enable_at_eval))
        return loss_functions

    def compute_loss(self, loss_input, train):
        total_loss = 0
        losses = self.get_losses()
        if not losses:
            logger.warning(f"losses are not configured ({losses}), returning 0")
            total_loss = torch.tensor([0.])
        loss_dict = OrderedDict()
        for loss_fn, _, weight, enable_at_train, enable_at_eval in losses:
            if train:
                if not enable_at_train:
                    continue
            else:
                if not enable_at_eval:
                    continue
            loss, local_loss_dict = loss_fn(**loss_input)
            total_loss += weight * loss
            loss_dict.update(local_loss_dict)
        return total_loss, loss_dict

    def step(self, batch, batch_idx, device=None):
        raise NotImplementedError

    def on_epoch_end(self, split="train", device="cuda", epoch=0):
        epoch_metrics = self.get_epoch_metrics(split)
        # We don't need to gather step outputs when there are no metrics defined.
        if len(epoch_metrics) == 0:
            if split == "train":
                self.training_step_outputs.clear()
            elif split == "val":
                self.validation_step_outputs.clear()
            elif split == "test":
                self.test_step_outputs.clear()
            return
        if split == "train":
            outputs = gather_step_outputs(self.training_step_outputs)
            self.training_step_outputs.clear()
        elif split == "val":
            outputs = gather_step_outputs(self.validation_step_outputs)
            self.validation_step_outputs.clear()
        elif split == "test":
            outputs = gather_step_outputs(self.test_step_outputs)
            self.test_step_outputs.clear()

        if len(outputs) == 0:
            return
        epoch_metrics = self.compute_epoch_metrics(
            outputs,
            self.args,
            device,
            key_prefix=f"{split}_",
            epoch=epoch,
            split=split,
            epoch_metrics=epoch_metrics,
        )
        for k, v in outputs["logs"].items():
            epoch_metrics[k] = v.float().mean()
        for k, v in epoch_metrics.items():
            logger.info(f"{k}: {v:.4f}")
        wandb.log(epoch_metrics, step=self.global_step)

        self.args.main.status = f"done {split} epoch"
        return epoch_metrics

    def train_one_epoch(
        self,
        model,
        dataloader,
        optimizer,
        device,
        epoch,
        loss_scaler,
        lr_scheduler,
        args,
    ):
        raise NotImplementedError

    def evaluate(
        self,
        model,
        dataloader,
        device,
        epoch=None, 
        test=False, 
        gather_predictions=False
    ):
        raise NotImplementedError


    def profile_engine(self, model, dataloader, optimizer, device, loss_scaler, lr_scheduler, args, profile_steps=10, log_dir=None):
        model.train()

        with torch.profiler.profile(record_shapes=True, profile_memory=True, with_stack=True) as prof:
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                if batch_idx >= profile_steps:
                    break
                if batch is None:
                    continue
                with torch.cuda.amp.autocast(dtype=self.amp_precision, enabled=self.amp_precision is not None):
                    loss, _, _ = self.step(model, batch, batch_idx, device=device)
                # update model params
                loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(batch_idx % self.accum_iter == 0))

                # gradient accumulation
                if (batch_idx + 1) % self.accum_iter == 0:
                    optimizer.zero_grad()
                    self.global_step += 1

        prefix = f"nocompile__profiler_{int(time.time())}"        
        logger.info("Trace saved to: " + f"{log_dir}/{prefix}_trace.json.gz")
        prof.export_chrome_trace(f"{log_dir}/{prefix}_trace.json.gz")
        prof.export_memory_timeline(f"{log_dir}/{prefix}_memory.html")
        
        
