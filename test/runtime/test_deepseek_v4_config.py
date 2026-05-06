import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tokenspeed.runtime.configs.deepseek_v4_config import DeepseekV4Config
from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    ModelConfig,
    configure_deepseek_v4_attention,
    is_deepseek_v4,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    DeepseekV4AttentionOpUnavailable,
    fused_qnorm_rope_kv_insert,
    has_fused_qnorm_rope_kv_insert,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    DeepseekV4ForwardMetadata,
    DeepseekV4TokenToKVPool,
    deepseek_v4_cache_layout_from_config,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.flashinfer import (
    _get_flashinfer_mxfp4_device_permute_indices,
    _reorder_w1w3_to_w3w1,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.weights import MXFP4_SCALE_DTYPE
from tokenspeed.runtime.layers.quantization import QUANTIZATION_METHODS
from tokenspeed.runtime.models.deepseek_v4 import (
    _deepseek_v4_reorder_c4_ape_2604,
    deepseek_v4_attention_layout,
    deepseek_v4_rope_config,
    deepseek_v4_select_experts,
    hc_head,
    mhc_post,
    mhc_pre,
    pack_topk_as_router_logits,
)
from tokenspeed.runtime.utils.hf_transformers_utils import (
    _CONFIG_REGISTRY,
    _wrap_deepseek_v4_tokenizer,
    get_tokenizer,
    prefers_deepseek_v4_tokenizer,
)


class TestDeepseekV4Config(unittest.TestCase):
    quant_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "scale_fmt": "ue8m0",
    }

    def test_config_registry(self):
        self.assertEqual(DeepseekV4Config.model_type, "deepseek_v4")
        self.assertIs(_CONFIG_REGISTRY["deepseek_v4"], DeepseekV4Config)

    def test_deepseek_v4_tokenizer_wrapper_uses_model_encoder(self):
        calls = []

        class DummyTokenizer:
            vocab_size = 5

            def encode(self, text, add_special_tokens=False, **kwargs):
                self.last_encode = (text, add_special_tokens, kwargs)
                return [len(text)]

            def get_added_vocab(self):
                return {"<extra>": 5}

        def encode_messages(messages, **kwargs):
            calls.append((messages, kwargs))
            return "<encoded>"

        tokenizer = _wrap_deepseek_v4_tokenizer(DummyTokenizer(), encode_messages)

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            enable_thinking=True,
            reasoning_effort="medium",
        )
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            truncation=True,
            max_length=16,
        )

        self.assertEqual(prompt, "<encoded>")
        self.assertEqual(token_ids, [9])
        self.assertEqual(len(tokenizer), 6)
        self.assertEqual(calls[0][1]["thinking_mode"], "thinking")
        self.assertIsNone(calls[0][1]["reasoning_effort"])
        self.assertEqual(calls[1][1]["thinking_mode"], "chat")
        self.assertEqual(
            tokenizer.last_encode,
            ("<encoded>", False, {"truncation": True, "max_length": 16}),
        )

    def test_deepseek_v4_tokenizer_is_auto_selected_by_architecture(self):
        self.assertTrue(prefers_deepseek_v4_tokenizer(["DeepseekV4ForCausalLM"]))
        self.assertFalse(prefers_deepseek_v4_tokenizer(["KimiK2ForCausalLM"]))
        self.assertFalse(prefers_deepseek_v4_tokenizer(None))

    def test_auto_tokenizer_mode_wraps_deepseek_v4_architecture(self):
        class DummyTokenizer:
            vocab_size = 5

            def encode(self, text, add_special_tokens=False, **kwargs):
                return [len(text)]

            def get_added_vocab(self):
                return {}

        def encode_messages(messages, **kwargs):
            return "<encoded>"

        with patch(
            "tokenspeed.runtime.utils.hf_transformers_utils.AutoTokenizer.from_pretrained",
            return_value=DummyTokenizer(),
        ), patch(
            "tokenspeed.runtime.utils.hf_transformers_utils._load_deepseek_v4_encode_messages",
            return_value=encode_messages,
        ):
            tokenizer = get_tokenizer(
                "deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_mode="auto",
                architectures=["DeepseekV4ForCausalLM"],
            )

        self.assertEqual(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}],
            ),
            [9],
        )

    def test_fp8_quantization_config(self):
        quantization = QUANTIZATION_METHODS["fp8"]

        config = quantization.from_config(self.quant_config)

        self.assertEqual(quantization.get_name(), "fp8")
        self.assertIsNone(
            quantization.override_quantization_method(self.quant_config, None)
        )
        self.assertEqual(config.activation_scheme, "dynamic")
        self.assertTrue(config.is_checkpoint_fp8_serialized)

    def test_model_config_maps_deepseek_v4_to_standard_fp8(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            model_type="deepseek_v4", quantization_config=self.quant_config
        )
        model_config.quantization = None

        model_config._verify_quantization()

        self.assertEqual(model_config.quantization, "fp8")

    def test_model_config_keeps_incompatible_user_quantization_error(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            model_type="deepseek_v4", quantization_config=self.quant_config
        )
        model_config.quantization = "mxfp4"

        with self.assertRaisesRegex(ValueError, "does not match"):
            model_config._verify_quantization()

    def test_deepseek_v4_attention_op_boundary_fails_loudly_when_missing(self):
        if has_fused_qnorm_rope_kv_insert():
            self.skipTest("DeepSeek V4 fused attention op is available in this build")

        q = torch.empty(1, 1, 512)
        kv = torch.empty(1, 512)
        cache = torch.empty(1, 584, dtype=torch.uint8)
        slots = torch.zeros(1, dtype=torch.int32)
        positions = torch.zeros(1, dtype=torch.int32)
        cos_sin = torch.empty(1, 128)

        with self.assertRaisesRegex(
            DeepseekV4AttentionOpUnavailable,
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        ):
            fused_qnorm_rope_kv_insert(
                q, kv, cache, slots, positions, cos_sin, 1e-6, 256
            )

    def test_deepseek_v4_model_config_uses_mla_runtime_metadata(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            rope_scaling=None,
        )

        self.assertTrue(is_deepseek_v4(model_config.hf_config))

        configure_deepseek_v4_attention(model_config)

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.head_dim, 512)
        self.assertEqual(model_config.kv_lora_rank, 512)
        self.assertEqual(model_config.qk_rope_head_dim, 64)
        self.assertEqual(model_config.qk_nope_head_dim, 448)
        self.assertEqual(model_config.v_head_dim, 512)
        self.assertEqual(model_config.index_head_dim, 128)
        self.assertAlmostEqual(model_config.scaling, 512**-0.5)

    def test_deepseek_v4_attention_layout_matches_vllm_cache_contract(self):
        config = SimpleNamespace(
            compress_ratios=[0, 4, 128],
            num_attention_heads=64,
            head_dim=512,
            qk_rope_head_dim=64,
            sliding_window=128,
            index_head_dim=128,
        )

        swa = deepseek_v4_attention_layout(config, 0, attn_tp_size=4)
        csa = deepseek_v4_attention_layout(config, 1, attn_tp_size=4)
        csa_fp4 = deepseek_v4_attention_layout(
            config, 1, attn_tp_size=4, use_fp4_indexer_cache=True
        )
        hca = deepseek_v4_attention_layout(config, 2, attn_tp_size=4)

        self.assertEqual(swa.kind, "swa")
        self.assertEqual(swa.compress_ratio, 1)
        self.assertEqual(swa.num_local_heads, 16)
        self.assertEqual(swa.padded_heads, 64)
        self.assertEqual(swa.nope_head_dim, 448)
        self.assertEqual(swa.swa_head_bytes, 584)
        self.assertFalse(swa.needs_compressed_cache)
        self.assertFalse(swa.needs_indexer)

        self.assertEqual(csa.kind, "csa")
        self.assertEqual(csa.compress_ratio, 4)
        self.assertTrue(csa.needs_compressed_cache)
        self.assertTrue(csa.needs_indexer)
        self.assertEqual(csa.compressed_cache_alignment, 576)
        self.assertEqual(csa.indexer_cache_head_bytes, 132)
        self.assertEqual(csa_fp4.indexer_cache_head_bytes, 68)

        self.assertEqual(hca.kind, "hca")
        self.assertEqual(hca.compress_ratio, 128)
        self.assertTrue(hca.needs_compressed_cache)
        self.assertFalse(hca.needs_indexer)

    def test_deepseek_v4_attention_layout_rejects_unknown_ratio(self):
        config = SimpleNamespace(
            compress_ratios=[8],
            num_attention_heads=64,
            head_dim=512,
            qk_rope_head_dim=64,
            sliding_window=128,
            index_head_dim=128,
        )

        with self.assertRaisesRegex(ValueError, "compress_ratio=8"):
            deepseek_v4_attention_layout(config, 0)

    def test_deepseek_v4_rope_config_matches_layer_type(self):
        config = SimpleNamespace(
            rope_theta=10000,
            compress_rope_theta=160000,
            rope_scaling={
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
            },
        )

        swa_base, swa_scaling = deepseek_v4_rope_config(config, compress_ratio=1)
        csa_base, csa_scaling = deepseek_v4_rope_config(config, compress_ratio=4)

        self.assertEqual(swa_base, 10000.0)
        self.assertIsNone(swa_scaling)
        self.assertEqual(csa_base, 160000.0)
        self.assertIsNot(csa_scaling, config.rope_scaling)
        self.assertEqual(csa_scaling["rope_type"], "deepseek_yarn")
        self.assertEqual(csa_scaling["factor"], 16)
        self.assertEqual(csa_scaling["mscale"], 0)
        self.assertEqual(csa_scaling["mscale_all_dim"], 0)

    def test_deepseek_v4_kv_pool_allocates_v4_cache_families(self):
        config = SimpleNamespace(
            compress_ratios=[1, 4, 128],
            head_dim=512,
            index_head_dim=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=64,
            use_fp4_indexer_cache=True,
        )

        self.assertEqual(layout.cache_cell_size(3), 17329)

        pool = DeepseekV4TokenToKVPool(
            size=128,
            model_dtype=torch.bfloat16,
            layout=layout,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=128,
            page_size=64,
            rank=0,
        )

        self.assertEqual(tuple(pool.get_swa_kv_buffer(0).shape), (3, 37440))
        self.assertIsNone(pool.compressed_kv_buffer[0])
        self.assertEqual(tuple(pool.get_compressed_kv_buffer_2d(1).shape), (3, 37440))
        self.assertEqual(
            tuple(pool.get_compressor_state_buffer(1).shape), (3, 64, 2048)
        )
        self.assertEqual(
            tuple(pool.get_compressor_state_buffer(2).shape), (3, 64, 1024)
        )
        self.assertEqual(pool.get_compressor_state_buffer(1).dtype, torch.float32)
        self.assertEqual(pool.get_compressor_state_buffer(2).dtype, torch.float32)
        self.assertEqual(tuple(pool.get_indexer_kv_buffer_2d(1).shape), (3, 64 * 68))
        self.assertEqual(tuple(pool.get_indexer_state_buffer(1).shape), (3, 64, 512))
        self.assertEqual(pool.get_indexer_state_buffer(1).dtype, torch.float32)

    def test_deepseek_v4_kv_pool_rejects_nonpositive_size(self):
        config = SimpleNamespace(
            compress_ratios=[1],
            head_dim=512,
            index_head_dim=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=64,
            use_fp4_indexer_cache=True,
        )

        with self.assertRaisesRegex(ValueError, "must be positive"):
            DeepseekV4TokenToKVPool(
                size=0,
                model_dtype=torch.bfloat16,
                layout=layout,
                layer_num=1,
                device="cpu",
                enable_memory_saver=False,
                max_batch_size=2,
                max_context_len=128,
                page_size=64,
                rank=0,
            )

    def test_deepseek_v4_metadata_maps_compressed_slots(self):
        metadata = DeepseekV4ForwardMetadata(
            page_size=64,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            block_table=torch.tensor([[0, 1], [3, 4]], dtype=torch.int32),
            seq_lens=torch.tensor([70, 5], dtype=torch.int32),
            query_lens=torch.tensor([3, 5], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3, 8], dtype=torch.int32),
            token_to_req_indices=torch.tensor(
                [0, 0, 0, 1, 1, 1, 1, 1],
                dtype=torch.int32,
            ),
        )

        self.assertTrue(
            torch.equal(
                metadata.token_to_req_indices,
                torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int32),
            )
        )
        slots = metadata.compressed_slot_mapping(
            torch.tensor([3, 7, 127], dtype=torch.int64),
            compress_ratio=4,
        )
        self.assertTrue(torch.equal(slots, torch.tensor([0, 1, 31])))

    def test_hidden_compression_helpers_preserve_expected_shapes(self):
        import torch

        torch.manual_seed(0)
        tokens, hc_mult, hidden = 3, 4, 5
        mix_hc = (2 + hc_mult) * hc_mult
        residual = torch.randn(tokens, hc_mult, hidden, dtype=torch.float32)
        fn = torch.randn(mix_hc, hc_mult * hidden, dtype=torch.float32)
        scale = torch.ones(3, dtype=torch.float32)
        base = torch.zeros(mix_hc, dtype=torch.float32)

        layer_input, post, comb = mhc_pre(
            residual,
            fn,
            scale,
            base,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=2,
        )
        updated = mhc_post(layer_input, residual, post, comb)

        self.assertEqual(tuple(layer_input.shape), (tokens, hidden))
        self.assertEqual(tuple(post.shape), (tokens, hc_mult, 1))
        self.assertEqual(tuple(comb.shape), (tokens, hc_mult, hc_mult))
        self.assertEqual(tuple(updated.shape), tuple(residual.shape))

    def test_hidden_compression_pre_matches_vllm_reference_math(self):
        import torch
        import torch.nn.functional as F

        torch.manual_seed(1)
        tokens, hc_mult, hidden = 2, 3, 4
        mix_hc = (2 + hc_mult) * hc_mult
        residual = torch.randn(tokens, hc_mult, hidden, dtype=torch.bfloat16)
        fn = torch.randn(mix_hc, hc_mult * hidden, dtype=torch.float32)
        scale = torch.tensor([0.7, 1.1, 0.5], dtype=torch.float32)
        base = torch.randn(mix_hc, dtype=torch.float32)
        eps = 1e-5

        layer_input, post, comb = mhc_pre(
            residual, fn, scale, base, rms_eps=1e-6, hc_eps=eps, sinkhorn_iters=3
        )

        x = residual.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        mixes = F.linear(x, fn) * rsqrt
        pre_raw, post_raw, comb_raw = torch.split(
            mixes, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
        )
        pre_base, post_base, comb_base = torch.split(
            base, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
        )
        expected_pre = torch.sigmoid(pre_raw * scale[0] + pre_base) + eps
        expected_post = (
            torch.sigmoid(post_raw * scale[1] + post_base) * 2.0
        ).unsqueeze(-1)
        expected_comb = (
            F.softmax(
                comb_raw.reshape(tokens, hc_mult, hc_mult) * scale[2]
                + comb_base.reshape(1, hc_mult, hc_mult),
                dim=-1,
            )
            + eps
        )
        expected_comb = expected_comb / (expected_comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(2):
            expected_comb = expected_comb / (
                expected_comb.sum(dim=-1, keepdim=True) + eps
            )
            expected_comb = expected_comb / (
                expected_comb.sum(dim=-2, keepdim=True) + eps
            )
        expected_layer_input = torch.sum(
            expected_pre.unsqueeze(-1) * residual.float(), dim=1
        ).to(residual.dtype)

        self.assertTrue(torch.allclose(layer_input, expected_layer_input))
        self.assertTrue(torch.allclose(post, expected_post))
        self.assertTrue(torch.allclose(comb, expected_comb))

    def test_hidden_compression_post_matches_vllm_lane_orientation(self):
        import torch

        hidden_states = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
        residual = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
        post = torch.tensor([[[0.5], [0.25]]], dtype=torch.float32)
        comb = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)

        updated = mhc_post(hidden_states, residual, post, comb)

        expected = torch.empty_like(residual)
        expected[:, 0] = (
            comb[:, 0, 0:1] * residual[:, 0]
            + comb[:, 1, 0:1] * residual[:, 1]
            + post[:, 0] * hidden_states
        )
        expected[:, 1] = (
            comb[:, 0, 1:2] * residual[:, 0]
            + comb[:, 1, 1:2] * residual[:, 1]
            + post[:, 1] * hidden_states
        )
        self.assertTrue(torch.allclose(updated, expected))

    def test_hc_head_matches_vllm_shape_contract(self):
        import torch

        tokens, hc_mult, hidden = 2, 4, 6
        x = torch.randn(tokens, hc_mult, hidden)
        fn = torch.randn(hc_mult, hc_mult * hidden)
        scale = torch.ones(1)
        base = torch.zeros(hc_mult)

        y = hc_head(x, fn, scale, base, rms_norm_eps=1e-6, hc_eps=1e-6)

        self.assertEqual(tuple(y.shape), (tokens, hidden))

    def test_deepseek_v4_router_matches_vllm_bias_semantics(self):
        import torch
        import torch.nn.functional as F

        logits = torch.tensor(
            [
                [0.2, 1.0, -0.5, 0.7],
                [1.5, -0.3, 0.8, 0.0],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor([0.0, -0.4, 0.6, 0.0], dtype=torch.float32)

        topk_weights, topk_ids, scores = deepseek_v4_select_experts(
            logits,
            top_k=2,
            renormalize=True,
            correction_bias=bias,
        )

        expected_scores = F.softplus(logits).sqrt()
        expected_ids = torch.topk(expected_scores + bias, k=2, dim=-1, sorted=False)[1]
        expected_weights = expected_scores.gather(1, expected_ids)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.allclose(scores, expected_scores))
        self.assertTrue(torch.equal(topk_ids, expected_ids.to(torch.int32)))
        self.assertTrue(torch.allclose(topk_weights, expected_weights))

    def test_deepseek_v4_hash_router_uses_table_ids_and_gate_scores(self):
        import torch
        import torch.nn.functional as F

        logits = torch.tensor(
            [
                [0.5, 1.0, -0.5, 0.1],
                [-0.2, 0.3, 1.4, 0.0],
            ],
            dtype=torch.float32,
        )
        input_ids = torch.tensor([3, 1], dtype=torch.long)
        table = torch.tensor(
            [
                [0, 1],
                [2, 3],
                [1, 0],
                [3, 1],
            ],
            dtype=torch.int32,
        )

        topk_weights, topk_ids, _ = deepseek_v4_select_experts(
            logits,
            top_k=2,
            renormalize=True,
            hash_indices_table=table,
            input_ids=input_ids,
        )

        expected_ids = torch.tensor([[3, 1], [2, 3]], dtype=torch.int32)
        expected_scores = F.softplus(logits).sqrt()
        expected_weights = expected_scores.gather(1, expected_ids.long())
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.equal(topk_ids, expected_ids))
        self.assertTrue(torch.allclose(topk_weights, expected_weights))

    def test_packed_topk_router_logits_recover_weights_after_softmax(self):
        import torch

        topk_ids = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)
        topk_weights = torch.tensor([[0.7, 0.3], [0.55, 0.45]], dtype=torch.float32)

        packed = pack_topk_as_router_logits(topk_weights, topk_ids, num_experts=4)
        recovered = packed.softmax(dim=-1).gather(1, topk_ids.long())

        self.assertTrue(torch.allclose(recovered, topk_weights))

    def test_mxfp4_scale_dtype_preserves_e8m0_checkpoint_bits(self):
        import torch

        if not hasattr(torch, "float8_e8m0fnu"):
            self.skipTest("float8_e8m0fnu is unavailable")

        loaded = torch.tensor(
            [[0.0078125, 0.015625], [0.03125, 0.0625]], dtype=torch.float32
        ).to(torch.float8_e8m0fnu)
        param = torch.empty_like(loaded, dtype=MXFP4_SCALE_DTYPE)
        param.copy_(loaded)

        self.assertEqual(MXFP4_SCALE_DTYPE, torch.float8_e8m0fnu)
        self.assertTrue(torch.equal(param.view(torch.uint8), loaded.view(torch.uint8)))

    def test_mxfp4_flashinfer_reorders_w1w3_halves_for_trtllm(self):
        import torch

        weight = torch.arange(4, dtype=torch.uint8).reshape(1, 4, 1)
        scale = torch.arange(8, dtype=torch.uint8).reshape(1, 4, 2)
        bias = torch.arange(4, dtype=torch.float32).reshape(1, 4)

        self.assertTrue(
            torch.equal(
                _reorder_w1w3_to_w3w1(weight, -2).flatten(),
                torch.tensor([2, 3, 0, 1], dtype=torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                _reorder_w1w3_to_w3w1(scale, -2).flatten(),
                torch.tensor([4, 5, 6, 7, 0, 1, 2, 3], dtype=torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                _reorder_w1w3_to_w3w1(bias, -1).flatten(),
                torch.tensor([2, 3, 0, 1], dtype=torch.float32),
            )
        )
        if hasattr(torch, "float8_e8m0fnu"):
            scale_f8 = torch.tensor(
                [[0.0078125, 0.015625, 0.03125, 0.0625]], dtype=torch.float32
            ).to(torch.float8_e8m0fnu)
            reordered = _reorder_w1w3_to_w3w1(scale_f8, -1)
            self.assertEqual(reordered.dtype, torch.float8_e8m0fnu)
            self.assertTrue(
                torch.equal(
                    reordered.view(torch.uint8),
                    torch.tensor([[122, 123, 120, 121]], dtype=torch.uint8),
                )
            )

    def test_mxfp4_flashinfer_uses_gated_permute_for_w13(self):
        import torch
        from tokenspeed_kernel.thirdparty.flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        x = torch.empty((4096, 2048), dtype=torch.uint8)
        expected_w13 = _maybe_get_cached_w3_w1_permute_indices({}, x, 128)
        expected_w2 = get_w2_permute_indices_with_cache({}, x, 128)

        actual_w13 = _get_flashinfer_mxfp4_device_permute_indices(x, 128, kind="w13")
        actual_w2 = _get_flashinfer_mxfp4_device_permute_indices(x, 128, kind="w2")

        self.assertTrue(torch.equal(actual_w13.cpu(), expected_w13.cpu()))
        self.assertTrue(torch.equal(actual_w2.cpu(), expected_w2.cpu()))
        self.assertFalse(torch.equal(actual_w13.cpu(), actual_w2.cpu()))

    def test_c4_ape_reorder_matches_overlap_window_layout(self):
        import torch

        ape = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)

        reordered = _deepseek_v4_reorder_c4_ape_2604(ape)
        expected = torch.tensor(
            [
                [0, 1, 2, 3, 8, 9, 10, 11],
                [16, 17, 18, 19, 24, 25, 26, 27],
                [4, 5, 6, 7, 12, 13, 14, 15],
                [20, 21, 22, 23, 28, 29, 30, 31],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.equal(reordered, expected))


if __name__ == "__main__":
    unittest.main()
