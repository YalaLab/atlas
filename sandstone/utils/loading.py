import torch
from torch.utils import data
from torch.utils.data import default_collate
from torch.utils.data import sampler as torch_sampler

from sandstone.utils.misc import logger

string_classes = (str, bytes)
int_classes = int


def ignore_None_collate(batch):
    """
    default_collate wrapper that creates batches only of not None values.
    Useful for cases when the dataset.__getitem__ can return None because of some
    exception and then we will want to exclude that sample from the batch.
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def null_collate(batch):
    """
    This is for generating cache. We return None to make cache generation faster by reducing inter-process communication.
    """
    return None


def get_train_dataset_loader(args, train_data):
    """
    Given arg configuration, return appropriate torch.DataLoader
    for train_data and dev_data

    returns:
    train_data_loader: iterator that returns batches
    dev_data_loader: iterator that returns batches
    """

    if not args.dataloader.no_shuffle_training_set:
        if args.main.multi_gpu:
            logger.info("Using distributed sampler")
            sampler = data.distributed.DistributedSampler(
                train_data,
                shuffle=True,
                rank=args.main.global_rank,
                num_replicas=args.main.world_size,
                drop_last=args.dataloader.train_drop_last,
            )
        else:
            sampler = torch_sampler.RandomSampler(train_data)
    else:
        # We need to read training set without shuffling for cache generation.
        assert (
            not args.main.multi_gpu
        ), "MultiGPU without shuffling is not supported"
        sampler = torch_sampler.SequentialSampler(train_data)

    use_null_collate = args.dataloader.use_null_collate

    if use_null_collate:
        logger.info(
            "Using null collate function for dataloader: this is only supposed to be the case in cache generation"
        )

    train_data_loader = data.DataLoader(
        train_data,
        num_workers=args.dataloader.num_workers,
        sampler=sampler,
        pin_memory=True,
        batch_size=args.dataloader.batch_size,
        prefetch_factor=args.dataloader.prefetch_factor,
        persistent_workers=args.dataloader.persistent_workers,
        collate_fn=null_collate if use_null_collate else ignore_None_collate,
        drop_last=args.dataloader.train_drop_last,
    )

    return train_data_loader


def get_eval_dataset_loader(args, eval_data, shuffle, multi_gpu_eval=False):
    if multi_gpu_eval:
        logger.info("Multi-GPU distributed evaluation is enabled: the results may be a little inaccurate due to the padding or drop_last in DistributedSampler")
        sampler = torch.utils.data.distributed.DistributedSampler(
            eval_data,
            shuffle=shuffle,
            rank=args.main.global_rank,
            num_replicas=args.main.world_size,
            drop_last=args.dataloader.val_drop_last,
        )
    else:
        sampler = (
            torch_sampler.RandomSampler(eval_data)
            if shuffle
            else torch_sampler.SequentialSampler(eval_data)
        )

    use_null_collate = args.dataloader.use_null_collate
    if use_null_collate:
        logger.info(
            "Using null collate function for dataloader: this is only supposed to be the case in cache generation"
        )

    data_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=args.dataloader.eval_batch_size,
        num_workers=args.dataloader.num_workers,
        collate_fn=null_collate if use_null_collate else ignore_None_collate,
        pin_memory=True,
        drop_last=args.dataloader.val_drop_last,
        sampler=sampler,
    )

    return data_loader


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
