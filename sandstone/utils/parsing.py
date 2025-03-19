import argparse
import os
import json

import numpy as np
import yaml
from easydict import EasyDict
from sandstone.utils.misc import logger

# Constants
NUM_DEBUG_BATCHES = 2



def dump_args(args, filename="config.yaml", allow_overwrite=False):
    if not allow_overwrite:
        assert not os.path.exists(filename), f"File exists: {filename}"

    # The interval easydicts are not converted to dict.
    with open(filename, "w") as f:
        yaml.dump(dict(args), f, sort_keys=False)


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate {key!r} key found in YAML.")
            mapping.add(key)
        return super().construct_mapping(node, deep)


# Does not support base_config in this function
def load_config_simple(filename):
    with open(filename, "r") as f:
        config = yaml.safe_load(f)

    assert "base_config" not in config
    args = EasyDict(config)

    return args


def merge_cli_opt(config, key, value):
    key_hierarchy = key.split(".")
    item_container = config
    for hierarchy in key_hierarchy[:-1]:
        if isinstance(item_container, list):
            hierarchy = int(hierarchy)
        item_container = item_container[hierarchy]

    try:
        original_value = item_container[key_hierarchy[-1]]
    except KeyError as e:
        raise KeyError(f"KeyError: {e}, the current parent structure: {item_container}")

    if isinstance(original_value, bool):
        if value == "True" or value == "true":
            value = True
        elif value == "False" or value == "false":
            value = False
        else:
            raise ValueError(f"Value {value} is not a boolean value")
    elif isinstance(original_value, int):
        value = int(value)
    elif isinstance(original_value, float):
        value = float(value)
    elif isinstance(original_value, list):
        value = json.loads(value)
        if len(original_value) > 0:
            assert type(original_value[0]) == type(value[0])
            assert all([type(v) == type(value[0]) for v in value])

    assert original_value == None or type(original_value) == type(
        value
    ), f"{type(original_value)} != {type(value)}"

    logger.info(f"Overriding {key} with {value} (original value: {original_value})")
    item_container[key_hierarchy[-1]] = value


def merge_cli_opts(config, cli_opts):
    assert len(cli_opts) % 2 == 0, f"{len(cli_opts)} should be even"
    for key, value in zip(cli_opts[::2], cli_opts[1::2]):
        merge_cli_opt(config, key, value)


# https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries


def merge_dict(a, b, path=None, allow_replace=False):
    """Merges b into a"""

    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(
                    a[key], b[key], path + [str(key)], allow_replace=allow_replace
                )
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                if allow_replace:
                    logger.info(
                        f"Replacing key at {'.'.join(path + [str(key)])} with {b[key]}"
                    )
                    a[key] = b[key]
                else:
                    raise ValueError("Conflict at {'.'.join(path + [str(key)])}")
        else:
            a[key] = b[key]
    return a


def load_config(
    config_path,
    easydict=True,
    cli_opts=None,
    override_base_config=None,
    return_base_config=True,
):
    """
    Load yaml into dictionary. Only merges dictionary. Lists will be replaced.

    Args:
        config_path (str): yaml path
        easydict (bool): return EasyDict
        cli_opts (list): the options from CLI to override the options in the config
        override_base_config (str, optional): override the base config in the yaml
        return_base_config (bool): return the config with base_config key
    Returns:
        config (dict): config in dictionary
    """

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=UniqueKeyLoader)

    if override_base_config is not None:
        config["base_config"] = override_base_config

    if "base_config" not in config:
        if cli_opts is not None:
            merge_cli_opts(config, cli_opts)
        if easydict:
            config = EasyDict(config)
        return config

    base_path = os.path.join(os.path.dirname(config_path), config["base_config"])
    overwrite_path = config_path

    logger.info(
        f"Loading base config {base_path} and overwrite config {overwrite_path}"
    )

    base_config = load_config(base_path, easydict=False)

    merged_config = merge_dict(base_config, config, allow_replace=True)

    if cli_opts is not None:
        merge_cli_opts(merged_config, cli_opts)
    if easydict:
        merged_config = EasyDict(merged_config)

    if not return_base_config:
        del merged_config["base_config"]

    return merged_config

    # TODO (Krishna): adapt this
    # if (isinstance(args.devices, str) and len(args.devices.split(",")) > 1) or (
    #     isinstance(args.devices, int) and args.devices > 1
    # ):
    #     args.distributed_backend = "ddp"
    #     args.replace_sampler_ddp = False


def parse_args(args_strings=None):
    parser = argparse.ArgumentParser(description="Sandstone research repo.")
    parser.add_argument(
        "config",
        metavar="C",
        type=str,
        nargs="?",
        help="path for config",
        default="configs/classif/mnist_demo.yaml",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="shortcut to disabling wandb (useful for debugging)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="shortcut to turn on debugging mode",
    )
    parser.add_argument(
        "--resume",
        type=str,
        required=False,
        help="shortcut to resuming",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=False,
        help="shortcut to main.exp_name",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="shortcut to evaluation: turn on test, turn off train/val (dev), and turn off wandb",
    )
    parser.add_argument(
        "--opts",
        help="Overriding config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    if args_strings is None:
        cli_args = parser.parse_args()
    else:
        cli_args = parser.parse_args(args_strings)

    config_path = cli_args.config

    logger.info(f"Loading config from {config_path}")

    args = load_config(config_path, cli_opts=cli_args.opts)
    args.config_path = config_path

    if cli_args.no_wandb:
        args.main.disable_wandb = True

    if cli_args.resume:
        args.engine.kwargs.resume = cli_args.resume
        
    if cli_args.evaluate:
        args.main.disable_wandb = True
        args.main.phases.train = False
        args.main.phases.dev = False
        args.main.phases.test = True
        args.main.force_loading_train_dataloader = True 

    if cli_args.exp_name:
        args.main.exp_name = cli_args.exp_name

    if cli_args.debug:
        args.main.disable_wandb = True
        args.main.seed = 42
        args.dataloader.num_workers = 0
        args.engine.kwargs.limit_num_batches = NUM_DEBUG_BATCHES

    return args

