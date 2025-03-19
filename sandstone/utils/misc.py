import hashlib
import os
from enum import Enum
import logging
import sys
import random
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from sandstone import augmentations

# Global variable
is_master = None


def set_loglevel(debug):
    loglevel = logging.INFO if debug else logging.WARN
    logger.setLevel(loglevel)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(loglevel)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False


# Set it to debug by default. Will override in main.
logger = logging.getLogger("main")
set_loglevel(debug=True)


def get_model_summary(model, keys=["params"]):
    summary_dict = {}
    for key in keys:
        if key == "params":
            trainable_params = 0
            non_trainable_params = 0
            for name, params in model.named_parameters():
                if params.requires_grad:
                    trainable_params += torch.numel(params)
                else:
                    non_trainable_params += torch.numel(params)
            summary_dict["trainable_params"] = trainable_params
            summary_dict["non_trainable_params"] = non_trainable_params
    return summary_dict


def log_dict(summary_dict):
    print("=========================================")
    print("Number of parameters")
    for k, v in summary_dict.items():
        print(f"{k}: {v}")
    print("=========================================")


def md5(key):
    """
    returns a hashed with md5 string of the key
    """
    return hashlib.md5(key.encode()).hexdigest()


# This should be used as a decorator
def rank_zero_only(fn):
    def new_fn(*args, **kwargs):
        if is_master:
            return fn(*args, **kwargs)
        assert (
            is_master is not None
        ), "is_master has not been initialized by `setup_for_distributed`"

    return new_fn


def get_is_master():
    return is_master


@rank_zero_only
def setup_dirs(main_args, is_master=False):
    if not is_master:
        return

    os.makedirs(main_args.experiment_checkpoints_dir, exist_ok=True)

    print("experiment_checkpoints_dir: {}".format(main_args.experiment_checkpoints_dir))


def setup_for_distributed(is_master_arg):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master_arg or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    global is_master
    is_master = is_master_arg

@rank_zero_only
def rank_zero_tqdm_write(*args, **kwargs):
    return tqdm.write(*args, **kwargs)

# Reference: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L414
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, tqdm_write=False):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if tqdm_write:
            rank_zero_tqdm_write("\t".join(entries))
        else:
            print("\t".join(entries))

    def display_summary(self, tqdm_write=False):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        if tqdm_write:
            rank_zero_tqdm_write(" ".join(entries))
        else:
            print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_augmentations(image_augmentations, args):
    augmentations_list = []
    for augmentation in image_augmentations:
        name = augmentation["type"]
        kwargs = augmentation["kwargs"] if "kwargs" in augmentation else {}
        augmentation = augmentations.__dict__[name](args, **kwargs)
        augmentations_list.append(augmentation)

    return augmentations_list


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
