from enum import Enum
from typing import List

import torch
from torch import Tensor

# CUDA Version
from torch.distributed import all_gather as all_gather_no_backprop
from torch.distributed.nn.functional import all_gather as all_gather_with_backprop

class BackpropType(Enum):
    """
    How to backpropagate gradients during all-gather op. GLOBAL will backpropagate
    to all workers, LOCAL to only the current worker, and NONE will not backpropagate
    at all.
    """

    GLOBAL = 0
    LOCAL = 1
    NONE = 2


def gather_tensor(
    tensor: Tensor,
    backprop_type: BackpropType = BackpropType.GLOBAL,
) -> List[Tensor]:
    """Gathers a tensor across all GPUs.

    Args:
        tensor (Tensor): Tensors that need to be gathered.
        backprop_type (BackpropType): whether to backpropagate gradients to all
            workers (GLOBAL), just the local worker (LOCAL), or not at all (NONE).
            Default: BackpropType.GLOBAL

    Returns:
        List[Tensor]: List of gathered tensors across all GPUs.
    """
    world_size = torch.distributed.get_world_size()

    # This uses the all_gather from torch.distributed.nn.functional,
    # which backpropagates gradients to all workers
    if world_size == 1:
        return tensor

    if backprop_type == BackpropType.GLOBAL:
        return all_gather_with_backprop(tensor)

    else:
        tensor_all_gpus = [torch.zeros_like(tensor) for _ in range(world_size)]
        all_gather_no_backprop(tensor_all_gpus, tensor)
        # just backprop to the current worker
        # This means that the image gradients on a given worker will only
        # consider the text samples from the same worker
        if backprop_type == BackpropType.LOCAL:
            tensor_all_gpus[get_rank()] = tensor
        return tensor_all_gpus


def get_rank() -> int:
    """get rank util for distributed training"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0
