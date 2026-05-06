# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

import torch

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    DEEPSEEK_V4_SWA_SCALE_DIM,
    DEEPSEEK_V4_SWA_TOKEN_STRIDE,
    DeepseekV4AttentionOpUnavailable,
    dequantize_deepseek_v4_fp8_ds_mla_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    DeepseekV4ForwardMetadata,
)
from tokenspeed.runtime.layers.attention.registry import register_backend


def _cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(
        torch.cumsum(lengths.to(torch.int32), dim=0, dtype=torch.int32),
        (1, 0),
    )


class DeepseekV4AttentionBackend(AttentionBackend):
    """Metadata owner for the model-local DeepSeek V4 attention path."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.page_size = config.page_size
        self.forward_metadata: DeepseekV4ForwardMetadata | None = None
        self._decode_tile_metadata = {}

    def _query_lens(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode | None,
        extend_seq_lens_cpu: torch.Tensor | None,
        extend_prefix_lens_cpu: torch.Tensor | None,
        extend_prefix_lens: torch.Tensor | None,
    ) -> torch.Tensor:
        if forward_mode is not None and forward_mode.is_decode_or_idle():
            return torch.ones(bs, dtype=torch.int32, device=seq_lens.device)
        if extend_seq_lens_cpu is not None:
            return extend_seq_lens_cpu[:bs].to(seq_lens.device, dtype=torch.int32)
        if extend_prefix_lens_cpu is not None:
            prefix = extend_prefix_lens_cpu[:bs].to(seq_lens.device, dtype=torch.int32)
            return (seq_lens[:bs].to(torch.int32) - prefix).clamp_min(0)
        if extend_prefix_lens is not None:
            prefix = extend_prefix_lens[:bs].to(torch.int32)
            return (seq_lens[:bs].to(torch.int32) - prefix).clamp_min(0)
        return seq_lens[:bs].to(torch.int32)

    def init_forward_metadata(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        del num_tokens, kwargs
        device = seq_lens.device
        req_pool_indices = req_pool_indices[:bs]
        seq_lens = seq_lens[:bs].to(torch.int32)
        query_lens = self._query_lens(
            bs,
            seq_lens,
            forward_mode,
            extend_seq_lens_cpu,
            extend_prefix_lens_cpu,
            extend_prefix_lens,
        )
        max_seq_len = int(seq_lens.max().item()) if bs else 0
        max_pages = (max_seq_len + self.page_size - 1) // self.page_size
        if req_to_page is None:
            block_table = torch.zeros(
                (bs, max(max_pages, 1)),
                dtype=torch.int32,
                device=device,
            )
        else:
            block_table = req_to_page[req_pool_indices, : max(max_pages, 1)]
        req_ids = torch.arange(bs, device=device, dtype=torch.int32)
        token_to_req = torch.repeat_interleave(req_ids, query_lens.clamp_min(0))
        self.forward_metadata = DeepseekV4ForwardMetadata(
            page_size=self.page_size,
            req_pool_indices=req_pool_indices,
            block_table=block_table,
            seq_lens=seq_lens,
            query_lens=query_lens,
            query_start_loc=_cu_seqlens(query_lens),
            token_to_req_indices=token_to_req,
            forward_mode=forward_mode,
        )
        self._decode_tile_metadata = {}

    def _slots_from_local_indices(
        self,
        metadata: DeepseekV4ForwardMetadata,
        req_idx: int,
        local_indices: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        if local_indices.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=local_indices.device)
        pages = torch.div(local_indices, block_size, rounding_mode="floor")
        offsets = local_indices % block_size
        page_ids = metadata.block_table[req_idx, pages.long()].to(torch.int64)
        return page_ids * block_size + offsets

    def _decode_swa_indices_and_lens(
        self,
        positions: torch.Tensor,
        *,
        window_size: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 decode requires forward metadata")
        num_tokens = positions.numel()
        indices = torch.full(
            (num_tokens, max(window_size, 1)),
            -1,
            dtype=torch.int32,
            device=positions.device,
        )
        lens = torch.zeros(num_tokens, dtype=torch.int32, device=positions.device)
        for token_idx in range(num_tokens):
            position = int(positions[token_idx].item())
            start = max(0, position - window_size + 1)
            local = torch.arange(
                start,
                position + 1,
                dtype=torch.int64,
                device=positions.device,
            )
            req_idx = int(metadata.token_to_req_indices[token_idx].item())
            slots = self._slots_from_local_indices(
                metadata,
                req_idx,
                local,
                block_size,
            )
            count = slots.numel()
            if count:
                indices[token_idx, :count] = slots.to(torch.int32)
                lens[token_idx] = count
        return indices, lens

    def _decode_compressed_indices_and_lens(
        self,
        positions: torch.Tensor,
        *,
        compress_ratio: int,
        block_size: int,
        topk_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if compress_ratio <= 1:
            return None, None
        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 decode requires forward metadata")
        per_token_slots = []
        max_len = 0
        min_width = topk_indices.shape[1] if topk_indices is not None else 1
        for token_idx in range(positions.numel()):
            position = int(positions[token_idx].item())
            if compress_ratio == 4:
                if topk_indices is None:
                    raise RuntimeError("DeepSeek V4 CSA decode requires top-k indices")
                local = topk_indices[token_idx].to(torch.int64)
                local = local[local >= 0]
            else:
                num_compressed = (position + 1) // compress_ratio
                local = torch.arange(
                    num_compressed,
                    dtype=torch.int64,
                    device=positions.device,
                )
            req_idx = int(metadata.token_to_req_indices[token_idx].item())
            slots = self._slots_from_local_indices(
                metadata,
                req_idx,
                local,
                block_size,
            )
            per_token_slots.append(slots)
            max_len = max(max_len, slots.numel())

        width = max(64, ((max(max_len, min_width) + 63) // 64) * 64)
        indices = torch.full(
            (positions.numel(), 1, width),
            -1,
            dtype=torch.int32,
            device=positions.device,
        )
        lens = torch.zeros(
            positions.numel(), dtype=torch.int32, device=positions.device
        )
        for token_idx, slots in enumerate(per_token_slots):
            count = slots.numel()
            if count:
                indices[token_idx, 0, :count] = slots.to(torch.int32)
                lens[token_idx] = count
        return indices, lens

    def _get_decode_tile_metadata(self, kind: str):
        tile_metadata = self._decode_tile_metadata.get(kind)
        if tile_metadata is not None:
            return tile_metadata
        try:
            from flash_mla import get_mla_metadata
        except Exception as exc:
            raise DeepseekV4AttentionOpUnavailable(
                "DeepSeek V4 decode requires FlashMLA latent attention. "
                "Build/install `tokenspeed-kernel/python` with FlashMLA."
            ) from exc
        tile_metadata = get_mla_metadata()[0]
        self._decode_tile_metadata[kind] = tile_metadata
        return tile_metadata

    def _pad_query(self, q: torch.Tensor, padded_heads: int) -> torch.Tensor:
        if q.shape[1] == padded_heads:
            return q
        q_padded = torch.zeros(
            (q.shape[0], padded_heads, q.shape[2]),
            dtype=q.dtype,
            device=q.device,
        )
        q_padded[:, : q.shape[1]].copy_(q)
        return q_padded

    def _fp8_ds_mla_cache_view(
        self,
        cache_2d: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        row_bytes = DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM
        return torch.as_strided(
            cache_2d,
            (cache_2d.shape[0], block_size, 1, row_bytes),
            (
                cache_2d.stride(0),
                row_bytes,
                row_bytes,
                1,
            ),
        )

    def forward_deepseek_v4_decode(
        self,
        *,
        q: torch.Tensor,
        positions: torch.Tensor,
        token_to_kv_pool,
        layer_id: int,
        kind: str,
        compress_ratio: int,
        num_local_heads: int,
        padded_heads: int,
        head_dim: int,
        window_size: int,
        softmax_scale: float,
        attn_sink: torch.Tensor,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 decode requires forward metadata")
        if metadata.forward_mode is None or not metadata.forward_mode.is_decode():
            raise RuntimeError(
                "forward_deepseek_v4_decode only supports ForwardMode.DECODE"
            )
        try:
            from flash_mla import flash_mla_with_kvcache
        except Exception as exc:
            raise DeepseekV4AttentionOpUnavailable(
                "DeepSeek V4 decode requires FlashMLA latent attention. "
                "Build/install `tokenspeed-kernel/python` with FlashMLA."
            ) from exc

        q_padded = self._pad_query(q, padded_heads).contiguous()
        swa_indices, swa_lens = self._decode_swa_indices_and_lens(
            positions,
            window_size=window_size,
            block_size=token_to_kv_pool.swa_block_size,
        )
        extra_indices, extra_lens = self._decode_compressed_indices_and_lens(
            positions,
            compress_ratio=compress_ratio,
            block_size=token_to_kv_pool.compressed_block_size,
            topk_indices=topk_indices,
        )

        swa_cache = self._fp8_ds_mla_cache_view(
            token_to_kv_pool.get_swa_kv_buffer(layer_id),
            token_to_kv_pool.swa_block_size,
        )
        compressed_cache = None
        if compress_ratio > 1:
            compressed_cache = self._fp8_ds_mla_cache_view(
                token_to_kv_pool.get_compressed_kv_buffer_2d(layer_id),
                token_to_kv_pool.compressed_block_size,
            )

        out, _ = flash_mla_with_kvcache(
            q=q_padded.unsqueeze(1),
            k_cache=swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=head_dim,
            tile_scheduler_metadata=self._get_decode_tile_metadata(kind),
            softmax_scale=softmax_scale,
            is_fp8_kvcache=True,
            indices=swa_indices.unsqueeze(1),
            attn_sink=attn_sink,
            extra_k_cache=compressed_cache,
            extra_indices_in_kvcache=extra_indices,
            topk_length=swa_lens,
            extra_topk_length=extra_lens,
        )
        if out.dim() == 4:
            out = out.squeeze(1)
        return out[:, :num_local_heads]

    def _prefill_gather_lens(
        self,
        *,
        window_size: int,
    ) -> torch.Tensor:
        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 prefill requires forward metadata")
        prefix_lens = metadata.seq_lens - metadata.query_lens
        return metadata.query_lens + torch.minimum(
            prefix_lens,
            torch.full_like(prefix_lens, max(window_size - 1, 0)),
        )

    def _copy_prefill_rows(
        self,
        *,
        kv_workspace: torch.Tensor,
        req_idx: int,
        offset: int,
        cache_2d: torch.Tensor,
        block_size: int,
        local_start: int,
        local_end: int,
    ) -> None:
        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 prefill requires forward metadata")
        count = max(local_end - local_start, 0)
        if count == 0:
            return
        local = torch.arange(
            local_start,
            local_end,
            dtype=torch.int64,
            device=kv_workspace.device,
        )
        slots = self._slots_from_local_indices(metadata, req_idx, local, block_size)
        rows = dequantize_deepseek_v4_fp8_ds_mla_cache(
            cache_2d,
            slots,
            block_size=block_size,
        )
        kv_workspace[req_idx, offset : offset + count].copy_(rows)

    def _prefill_workspace(
        self,
        *,
        positions: torch.Tensor,
        token_to_kv_pool,
        layer_id: int,
        compress_ratio: int,
        window_size: int,
        head_dim: int,
        topk_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 prefill requires forward metadata")
        num_reqs = metadata.seq_lens.numel()
        gather_lens = self._prefill_gather_lens(window_size=window_size)
        max_gather_len = int(gather_lens.max().item()) if num_reqs else 1
        compressed_lens = (
            torch.div(metadata.seq_lens, compress_ratio, rounding_mode="floor")
            if compress_ratio > 1
            else torch.zeros_like(metadata.seq_lens)
        )
        compressed_base = (
            int(compressed_lens.max().item()) if compress_ratio > 1 and num_reqs else 0
        )
        workspace_width = max(1, compressed_base + max_gather_len)
        kv_workspace = torch.zeros(
            (num_reqs, workspace_width, head_dim),
            dtype=torch.bfloat16,
            device=positions.device,
        )

        swa_cache = token_to_kv_pool.get_swa_kv_buffer(layer_id)
        compressed_cache = (
            token_to_kv_pool.get_compressed_kv_buffer_2d(layer_id)
            if compress_ratio > 1
            else None
        )
        for req_idx in range(num_reqs):
            if compress_ratio > 1:
                assert compressed_cache is not None
                self._copy_prefill_rows(
                    kv_workspace=kv_workspace,
                    req_idx=req_idx,
                    offset=0,
                    cache_2d=compressed_cache,
                    block_size=token_to_kv_pool.compressed_block_size,
                    local_start=0,
                    local_end=int(compressed_lens[req_idx].item()),
                )
            seq_len = int(metadata.seq_lens[req_idx].item())
            gather_len = int(gather_lens[req_idx].item())
            self._copy_prefill_rows(
                kv_workspace=kv_workspace,
                req_idx=req_idx,
                offset=compressed_base,
                cache_2d=swa_cache,
                block_size=token_to_kv_pool.swa_block_size,
                local_start=seq_len - gather_len,
                local_end=seq_len,
            )

        per_token_indices: list[torch.Tensor] = []
        max_candidates = 0
        for req_idx in range(num_reqs):
            query_start = int(metadata.query_start_loc[req_idx].item())
            query_end = int(metadata.query_start_loc[req_idx + 1].item())
            seq_len = int(metadata.seq_lens[req_idx].item())
            gather_start = seq_len - int(gather_lens[req_idx].item())
            compressed_len = int(compressed_lens[req_idx].item())
            request_base = req_idx * workspace_width
            for token_idx in range(query_start, query_end):
                position = int(positions[token_idx].item())
                chunks = []
                if compress_ratio > 1:
                    if compress_ratio == 4:
                        if topk_indices is None:
                            raise RuntimeError(
                                "DeepSeek V4 CSA prefill requires top-k indices"
                            )
                        local = topk_indices[token_idx].to(torch.int64)
                        local = local[(local >= 0) & (local < compressed_len)]
                    else:
                        num_compressed = min(
                            (position + 1) // compress_ratio,
                            compressed_len,
                        )
                        local = torch.arange(
                            num_compressed,
                            dtype=torch.int64,
                            device=positions.device,
                        )
                    if local.numel() > 0:
                        chunks.append(request_base + local)
                swa_start = max(0, position - window_size + 1)
                swa_end = position + 1
                swa_local = torch.arange(
                    swa_start,
                    swa_end,
                    dtype=torch.int64,
                    device=positions.device,
                )
                swa_offsets = compressed_base + (swa_local - gather_start)
                chunks.append(request_base + swa_offsets)
                token_indices = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
                per_token_indices.append(token_indices.to(torch.int32))
                max_candidates = max(max_candidates, token_indices.numel())

        padded_topk = max(128, ((max_candidates + 127) // 128) * 128)
        indices = torch.full(
            (positions.numel(), padded_topk),
            -1,
            dtype=torch.int32,
            device=positions.device,
        )
        lens = torch.zeros(
            positions.numel(), dtype=torch.int32, device=positions.device
        )
        for token_idx, token_indices in enumerate(per_token_indices):
            count = token_indices.numel()
            indices[token_idx, :count] = token_indices
            lens[token_idx] = count
        return kv_workspace, indices, lens

    def forward_deepseek_v4_prefill(
        self,
        *,
        q: torch.Tensor,
        positions: torch.Tensor,
        token_to_kv_pool,
        layer_id: int,
        compress_ratio: int,
        num_local_heads: int,
        padded_heads: int,
        head_dim: int,
        window_size: int,
        softmax_scale: float,
        attn_sink: torch.Tensor,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 prefill requires forward metadata")
        if metadata.forward_mode is None or not metadata.forward_mode.is_extend():
            raise RuntimeError(
                "forward_deepseek_v4_prefill only supports extend/prefill modes"
            )
        try:
            from flash_mla import (
                flash_mla_sparse_fwd,
            )
        except Exception as exc:
            raise DeepseekV4AttentionOpUnavailable(
                "DeepSeek V4 prefill requires FlashMLA sparse attention. "
                "Build/install `tokenspeed-kernel/python` with FlashMLA."
            ) from exc

        q_padded = self._pad_query(q, padded_heads).contiguous()
        kv_workspace, indices, lens = self._prefill_workspace(
            positions=positions,
            token_to_kv_pool=token_to_kv_pool,
            layer_id=layer_id,
            compress_ratio=compress_ratio,
            window_size=window_size,
            head_dim=head_dim,
            topk_indices=topk_indices,
        )
        out, _, _ = flash_mla_sparse_fwd(
            q=q_padded,
            kv=kv_workspace.view(-1, 1, head_dim),
            indices=indices.unsqueeze(1),
            sm_scale=softmax_scale,
            attn_sink=attn_sink,
            topk_length=lens,
        )
        return out[:, :num_local_heads]

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf=None):
        del max_bs, seq_lens_buf

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "DeepSeek V4 baseline does not support CUDA graph capture; "
            "pass --enforce-eager."
        )

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "DeepSeek V4 baseline does not support CUDA graph replay; "
            "pass --enforce-eager."
        )

    def advance_draft_forward_metadata(self):
        raise NotImplementedError(
            "DeepSeek V4 attention does not support draft graphs yet"
        )

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek V4 uses the model-local attention forward")

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek V4 uses the model-local attention forward")


register_backend("deepseek_v4", {AttentionArch.MLA}, DeepseekV4AttentionBackend)
