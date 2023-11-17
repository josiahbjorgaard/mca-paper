# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import List

import torch
from torch import Tensor
#from torch.distributed import all_gather as all_gather_no_backprop
#from torch.distributed.nn.functional import all_gather as all_gather_with_backprop
from torch_xla.core.xla_model import all_gather as all_gather_no_backprop
from torch_xla.core.functions import all_gather as all_gather_with_backprop
import torch_xla.core.xla_model as xm

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
    world_size = xm.xrt_world_size()
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
    #if torch.distributed.is_available() and torch.distributed.is_initialized():
    return xm.get_ordinal()
    #return 0
