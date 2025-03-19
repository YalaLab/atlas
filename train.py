import datetime
import numpy as np
import os
from os.path import dirname, realpath
import random
import socket
import sys
import time

sys.path.append(dirname(dirname(realpath(__file__))))

from sandstone import datasets
from sandstone import engines
from sandstone import models
from sandstone import optimizers
from sandstone import schedulers
from sandstone.utils.loading import get_eval_dataset_loader, get_train_dataset_loader
from sandstone.models.attention import init_attn_impl
from sandstone.utils import misc
from sandstone.utils.misc import logger, set_loglevel, get_augmentations, set_all_seeds
from sandstone.utils.optim import NativeScalerWithGradNormCount as NativeScaler
import sandstone.utils.parsing as parsing

import torch
import torch.distributed as dist
from timm.optim import optim_factory
import wandb




# Global variables
exp_name = None

ATTN_IMPL = os.environ.get("ATTN_IMPL", "flash_attention2")
init_attn_impl(ATTN_IMPL)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.global_rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.local_rank = args.gpu
    elif "SLURM_PROCID" in os.environ:
        args.global_rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.global_rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.global_rank = -1
        args.local_rank = -1
        misc.is_master = True
        return
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    args.dist_url = "env://"
    print(
        "| distributed init (rank {}): {}, gpu {} / {}".format(
            args.global_rank, args.dist_url, args.gpu, args.world_size
        ),
        flush=True,
    )
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.local_rank,
    )
    dist.barrier()
    misc.setup_for_distributed(args.global_rank == 0)
    misc.setup_dirs(args, args.global_rank == 0)
    print(
        "| initialized host {} as rank {}".format(
            socket.gethostname(), args.global_rank
        ),
        flush=True,
    )

    return args


def build_experiment(args):
    global exp_name

    main_args = args.main
    exp_name = main_args.exp_name
    main_args.experiment_checkpoints_dir = os.path.join(
        main_args.checkpoints_dir, exp_name
    )
    os.makedirs(main_args.experiment_checkpoints_dir, exist_ok=True)
    init_distributed_mode(main_args)

    set_all_seeds(main_args.seed)

    global_rank = main_args.get("global_rank", -1)
    main_args.multi_gpu = global_rank > -1
    logger.info("Multi-GPU: {}".format(main_args.multi_gpu))

    if global_rank <= 0:
        allow_overwriting_experiment_checkpoints = (
            main_args.allow_overwriting_experiment_checkpoints
            or not os.path.exists(main_args.experiment_checkpoints_dir)
        )
        assert allow_overwriting_experiment_checkpoints, f"Checkpoint path ({main_args.experiment_checkpoints_dir}) already exists, set main.allow_overwriting_experiment_checkpoints to True to skip the check"
    else:
        # Disable wandb on other training processes
        main_args.disable_wandb = True

    main_args.status = ""

    if "PMIX_RANK" in os.environ:
        os.environ["NODE_RANK"] = os.environ["PMIX_RANK"]

    if main_args.disable_wandb:
        wandb_mode = "disabled"
    elif os.environ.get("SGE_IN_USE", False):
        wandb_mode = "offline"
    else:
        wandb_mode = None

    os.makedirs(main_args.experiment_checkpoints_dir, exist_ok=True)
    # Initialize wandb
    wandb.init(
        project=main_args.wandb_project,
        entity=main_args.wandb_entity,
        name=exp_name,
        dir=main_args.experiment_checkpoints_dir,
        mode=wandb_mode,
        tags=args.main.tags,
        settings=wandb.Settings(start_method="thread"),
    )

    logger.info(f"Checkpoint directory: {main_args.experiment_checkpoints_dir}")

    if wandb.run is not None:
        wandb.run.log_code(main_args.experiment_checkpoints_dir)

    if main_args.debug:
        logger.info(f"DEBUG MODE: Truncating to {NUM_DEBUG_BATCHES} batches")
        # raise NotImplementedError("debug mode has not been implemented")

    logger.info(str(args))

    main_args.callbacks = None  # Remove callbacks for pickling

    logger.info("Loading data-augmentation scheme...")
    # TODO: make augmentations part of the kwargs for the dataset

    augmentations = get_augmentations(args.dataset.image_augmentations, args)
    test_augmentations = get_augmentations(args.dataset.test_image_augmentations, args)

    # Load dataset and add dataset specific information to args
    logger.info("Loading data...")
    dataset_cls = datasets.__dict__[args.dataset.type]
    dataset_info = {}

    if main_args.phases.train or main_args.force_loading_train_dataloader:
        train_dataset = dataset_cls(
            args,
            augmentations=augmentations,
            split_group="train",
            **{
                **args.dataset.shared_dataset_kwargs,
                **args.dataset.dataset_train_kwargs,
            },
        )
        train_dataloader = get_train_dataset_loader(args, train_dataset)
        dataset_info["train"] = getattr(train_dataset, "info", None)

    if main_args.phases.train or main_args.phases.dev or (args.main.use_val_as_test and main_args.phases.test):
        dev_dataset = dataset_cls(
            args,
            augmentations=test_augmentations,
            split_group="dev",
            **{**args.dataset.shared_dataset_kwargs, **args.dataset.dataset_dev_kwargs},
        )
        multi_gpu_eval = args.main.multi_gpu and args.dataloader.multi_gpu_eval
        eval_dataloader = get_eval_dataset_loader(args, dev_dataset, shuffle=False, multi_gpu_eval=multi_gpu_eval)
        dataset_info["dev"] = getattr(dev_dataset, "info", None)

    if main_args.phases.test:
        if args.main.use_val_as_test:
            test_dataset = dev_dataset
        else:
            test_dataset = dataset_cls(
                args,
                augmentations=test_augmentations,
                split_group="test",
                **{
                    **args.dataset.shared_dataset_kwargs,
                    **args.dataset.dataset_test_kwargs,
                },
            )
        # We should not use multi-gpu in testing: it is often inaccurate due to drop or pad with the last batch.
        test_dataloader = get_eval_dataset_loader(args, test_dataset, shuffle=False, multi_gpu_eval=False)
        dataset_info["test"] = getattr(test_dataset, "info", None)

    engine_args = args.engine.kwargs
    engine_full_kwargs = dict(
        args=args, dataset_info=dataset_info, **args.engine.kwargs
    )
    engine: engines.Engine = engines.__dict__[args.engine.type](**engine_full_kwargs)
    logger.info("Engine built!")

    # build the model
    model = models.__dict__[args.model.type](args=args, **args.model.kwargs)

    model_summary = misc.get_model_summary(model, keys=["params"])

    misc.log_dict(model_summary)


    device = torch.device(main_args.get("gpu", "cuda"))
    model.to(device, non_blocking=True)

    model_without_ddp = model
    
    # Uncomment to print out the model:
    logger.info(f"{model_without_ddp}")

    if main_args.multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[main_args.gpu],
            output_device=main_args.gpu,
            broadcast_buffers=True,
            find_unused_parameters=main_args.get("find_unused_parameters", False)
        )
    
    if main_args.get("sync_bn", False):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if main_args.get("compile", False):
        model = torch.compile(model)

    # build optimizer, lr_scheduler, loss_scaler
    # following timm (referenced MAE): set wd as 0 for bias and norm layers if `timm_weight_decay` is set
    if args.optimizer.get('layer_decay', None) is not None:
        param_groups = model_without_ddp.param_groups_lrd(args.optimizer.timm_weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.optimizer.layer_decay
        )
        logger.info(f"Layer decay: {args.optimizer.get('layer_decay', None)}, weight decay: {args.optimizer.get('timm_weight_decay', None)}")
    elif args.optimizer.get('timm_weight_decay', None) is not None:
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.optimizer.timm_weight_decay)
    else:
        param_groups = model.parameters()

    optimizer: optimizers.Optimizer = optimizers.__dict__[args.optimizer.type](param_groups, **args.optimizer.kwargs)
    scheduler_interval = args.optimizer.scheduler.interval
    if scheduler_interval == "epoch":
        lr_scheduler = schedulers.__dict__[args.optimizer.scheduler.type](
            optimizer, **args.optimizer.scheduler.kwargs
        )
    else:
        lr_scheduler = schedulers.__dict__[args.optimizer.scheduler.type](
            optimizer, lr=args.optimizer.kwargs.lr, **args.optimizer.scheduler.kwargs
        )
        assert scheduler_interval == "step", f"scheduler type is not epoch or step: {scheduler_interval}"

    lr_scheduler.interval = scheduler_interval

    loss_scaler = NativeScaler()

    # fit, pass the dataloader
    if main_args.multi_gpu:
        torch.distributed.barrier()

    if engine_args.resume:
        checkpoint = torch.load(engine_args.resume, map_location="cpu")

        if list(checkpoint["model"].keys())[0].startswith("module."):
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }

        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            engine_args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            if "global_step" in checkpoint:
                engine.global_step = checkpoint["global_step"]
    else:
        engine_args.start_epoch = 0

    # run training
    logger.info(f"Start training for {engine_args.max_epochs} epochs")

    start_time = time.time()

    if main_args.phases.train:
        for epoch in range(engine_args.start_epoch, engine_args.max_epochs):
            # set current epoch for DDP
            if main_args.multi_gpu:
                train_dataloader.sampler.set_epoch(epoch)

            # train for one epoch
            train_stats = engine.train_one_epoch(
                model,
                train_dataloader,
                optimizer,
                device,
                epoch,
                loss_scaler,
                lr_scheduler=None if scheduler_interval == "epoch" else lr_scheduler,
                args=engine_args,
                clip_grad=engine_args.get("clip_grad", None),
                log_grad_norm=engine_args.get("log_grad_norm", None),
                log_interval=engine_args.get("log_interval", 200),
            )
            
            if misc.get_is_master():
                state_dict = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
                    "scaler": loss_scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": engine.global_step,
                    "args": args,
                }
                # always save the latest ckpt
                engine.save_on_master(
                    ckpt_dir=main_args.experiment_checkpoints_dir,
                    epoch=-1,
                    state=state_dict
                )
                
                if epoch % main_args.ckpt_freq == 0 or epoch == engine_args.max_epochs - 1:
                    engine.save_on_master(
                        ckpt_dir=main_args.experiment_checkpoints_dir,
                        epoch=epoch,
                        state=state_dict,
                    )

            # run evaluation
            if main_args.phases.dev:
                eval_stats = engine.evaluate(model, eval_dataloader, device, epoch, test=False, gather_predictions=multi_gpu_eval)

            if scheduler_interval == "epoch":
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # step scheduler is handled inside `train_one_epoch`
                    lr_scheduler.step(eval_stats['val_loss'])
                else:
                    lr_scheduler.step()


    # run testing
    if main_args.phases.test:
        # We do not do multi_gpu_eval in test so we do not gather predictions.
        test_stats = engine.evaluate(model, test_dataloader, device, epoch=None, test=True, gather_predictions=False)

    # print stats
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Experiment time {}".format(total_time_str))


if __name__ == "__main__":
    __spec__ = None

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    logger.info(f"Local rank: {local_rank}")

    if local_rank <= 0:
        set_loglevel(debug=True)
    else:
        set_loglevel(debug=False)

    config_args = parsing.parse_args()

    args = config_args

    build_experiment(args)
