# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""TritonRSAG communication backend for token-aware all_gather / reduce_scatter.

Handles uneven token distribution across ranks using Triton RS/AG state.
Lazily creates and caches Triton RS/AG state keyed by (group_tuple, hidden_size).
"""

import torch
from tokenspeed_kernel.ops.communication.triton import (
    all_gather,
    create_state,
    reduce_scatter,
)

from tokenspeed.runtime.distributed.comm_backend.base import Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.utils import ceil_div
from tokenspeed.runtime.utils.env import global_server_args_dict


class TritonRSAGBackend:
    """Backend using TritonRSAG for token-aware reduce_scatter / all_gather.

    Unlike NCCL backends, TritonRSAG handles uneven token distribution
    across ranks (scattered tokens). Each instance is specific to a
    (group, hidden_size) pair because RSAG pre-allocates buffers.
    """

    def __init__(self):
        # (group_tuple, hidden_size) -> Triton RS/AG state
        self._instances = {}

    def _get_or_create(self, rank: int, group: Group, hidden_size: int):
        key = (group, hidden_size)
        if key in self._instances:
            return self._instances[key]

        max_num_tokens = self._get_max_num_gathered_tokens()
        rsag = create_state(
            group=pg_manager.get_process_group("nccl", group),
            rank_in_group=rank,
            max_tokens=max_num_tokens,
            hidden_size=hidden_size,
        )
        self._instances[key] = rsag
        return rsag

    def token_all_gather(
        self,
        tensor: torch.Tensor,
        rank: int,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        rsag = self._get_or_create(rank, group, tensor.size(-1))
        return all_gather(rsag, tensor, token_list_in_group=scattered_num_tokens)

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        rank: int,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        rsag = self._get_or_create(rank, group, tensor.size(-1))
        return reduce_scatter(rsag, tensor, token_list_in_group=scattered_num_tokens)

    def _get_max_num_gathered_tokens(self):
        """Compute max buffer size for TritonRSAG.

        global_server_args_dict read is intentional — this is one-time RSAG buffer
        init infrastructure. Passing mapping through all signatures would be too invasive.
        """
        mapping = global_server_args_dict["mapping"]
        chunked_prefill_size = global_server_args_dict["chunked_prefill_size"]
        max_prefill_tokens = global_server_args_dict["max_prefill_tokens"]
        max_model_len = global_server_args_dict["max_model_len"]
        if chunked_prefill_size > 0:
            max_attn_tp_num_tokens = chunked_prefill_size
        else:
            max_attn_tp_num_tokens = max_prefill_tokens + max_model_len
        max_scattered_num_tokens = ceil_div(
            max_attn_tp_num_tokens, mapping.attn.tp_size
        )
        return max_scattered_num_tokens * max(
            mapping.attn.tp_size, mapping.dense.tp_size, mapping.moe.tp_ep_size
        )
