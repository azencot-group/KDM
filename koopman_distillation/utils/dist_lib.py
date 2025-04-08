import torch
import contextlib
from torch.distributed import all_gather_object, all_gather


@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield

def is_main_process(args):
    return args.rank == 0


def is_distributed(args):
    return args.gpu_num > 1


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

#----------------------------------------------------------------------------

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def gather_logs(args, to_log):

    to_log_list = [None for _ in range(get_world_size())]
    all_gather_object(to_log_list, to_log)
    if is_main_process(args):
        to_log = {k: sum([v[k].detach().cpu() for v in to_log_list]) for k in to_log.keys()}
    return to_log