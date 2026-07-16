# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS GraniteMoeForCausalLM (MoE) parity with HuggingFace."""

import torch
from transformers.models.granitemoe.modeling_granitemoe import (
    GraniteMoeConfig,
    GraniteMoeForCausalLM as HFGraniteMoeForCausalLM,
)

from coreai_models.models.macos.granitemoe import GraniteMoeForCausalLM
from coreai_models.primitives.macos.cache import KVCache


def _make_granitemoe_config(
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    num_hidden_layers: int = 2,
    intermediate_size: int = 128,
    vocab_size: int = 100,
    max_position_embeddings: int = 32,
    rope_theta: float = 1e7,
    logits_scaling: float = 6.0,
    embedding_multiplier: float = 12.0,
    residual_multiplier: float = 0.22,
    attention_multiplier: float = 0.015625,
    rms_norm_eps: float = 1e-6,
    tie_word_embeddings: bool = True,
    num_local_experts: int = 4,
    num_experts_per_tok: int = 2,
    attention_bias: bool = False,
) -> GraniteMoeConfig:
    config = GraniteMoeConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        logits_scaling=logits_scaling,
        embedding_multiplier=embedding_multiplier,
        residual_multiplier=residual_multiplier,
        attention_multiplier=attention_multiplier,
        rms_norm_eps=rms_norm_eps,
        tie_word_embeddings=tie_word_embeddings,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
        attention_bias=attention_bias,
    )
    return config


class TestmacOSGraniteMoeForCausalLM:
    """Test macOS GraniteMoeForCausalLM against HuggingFace reference."""

    def test_forward_parity_single_token(self):
        """Single-token decode: our macOS model matches HF logits."""
        hf_config = _make_granitemoe_config()
        our_config = _make_granitemoe_config()

        torch.manual_seed(42)
        hf_model = HFGraniteMoeForCausalLM(hf_config).to(torch.float32).eval()

        our_model = GraniteMoeForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, 1))
        position_ids = torch.tensor([[0]], dtype=torch.int32)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        max_diff = (our_out - hf_out.logits).abs().max().item()
        mean_diff = (our_out - hf_out.logits).abs().mean().item()
        print(f"Single-token: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_multi_token(self):
        """Multi-token prefill: our macOS model matches HF logits."""
        seq_len = 8
        hf_config = _make_granitemoe_config()
        our_config = _make_granitemoe_config()

        torch.manual_seed(42)
        hf_model = HFGraniteMoeForCausalLM(hf_config).to(torch.float32).eval()

        our_model = GraniteMoeForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        max_diff = (our_out - hf_out.logits).abs().max().item()
        mean_diff = (our_out - hf_out.logits).abs().mean().item()
        print(f"Multi-token (seq={seq_len}): max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_long_prefill(self):
        """Long prefill at float32 (seq_len=32)."""
        seq_len = 32
        hf_config = _make_granitemoe_config()
        our_config = _make_granitemoe_config()

        torch.manual_seed(42)
        hf_model = HFGraniteMoeForCausalLM(hf_config).to(torch.float32).eval()

        our_model = GraniteMoeForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        max_diff = (our_out - hf_out.logits).abs().max().item()
        mean_diff = (our_out - hf_out.logits).abs().mean().item()
        print(f"Long prefill (seq={seq_len}): max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_multi_layer(self):
        """Multi-layer model at float32."""
        hf_config = _make_granitemoe_config(num_hidden_layers=3)
        our_config = _make_granitemoe_config(num_hidden_layers=3)

        torch.manual_seed(42)
        hf_model = HFGraniteMoeForCausalLM(hf_config).to(torch.float32).eval()

        our_model = GraniteMoeForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, 6))
        position_ids = torch.arange(6, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        max_diff = (our_out - hf_out.logits).abs().max().item()
        mean_diff = (our_out - hf_out.logits).abs().mean().item()
        print(f"Multi-layer (3): max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_float16(self):
        """Verify parity in float16 precision."""
        hf_config = _make_granitemoe_config()
        our_config = _make_granitemoe_config()

        torch.manual_seed(42)
        hf_model = HFGraniteMoeForCausalLM(hf_config).to(torch.float16).eval()

        our_model = GraniteMoeForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float16).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, 6))
        position_ids = torch.arange(6, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float16)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        max_diff = (our_out - hf_out.logits).abs().max().item()
        mean_diff = (our_out - hf_out.logits).abs().mean().item()
        print(f"Float16: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
        torch.testing.assert_close(our_out, hf_out.logits, atol=5e-3, rtol=5e-3)

    def test_output_shape(self):
        """Output shape is (batch, seq_len, vocab_size)."""
        our_config = _make_granitemoe_config()
        our_model = GraniteMoeForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float32).eval()

        batch, seq_len, vocab = 1, 5, 100
        input_ids = torch.randint(0, vocab, (batch, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float32)

        with torch.no_grad():
            out = our_model(input_ids, position_ids, k_cache, v_cache)

        assert out.shape == (batch, seq_len, vocab), f"Expected {(batch, seq_len, vocab)}, got {out.shape}"
        print(f"Output shape: {out.shape} - OK")

    def test_mutate_state_dict_splits_experts(self):
        """_mutate_state_dict splits fused gate+up into separate gate_proj and up_proj for SwitchGLU."""
        our_config = _make_granitemoe_config(num_hidden_layers=1)
        our_model = GraniteMoeForCausalLM(our_config, model_device="cpu")

        hidden = 64
        n_heads = 4
        n_kv_heads = 2
        head_dim = hidden // n_heads
        num_experts = 4
        moe_intermediate = 128

        sd = {}
        sd["model.embed_tokens.weight"] = torch.randn(100, hidden)
        sd["model.norm.weight"] = torch.randn(hidden)
        sd["lm_head.weight"] = torch.randn(100, hidden)
        sd["model.layers.0.self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.k_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.v_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        sd["model.layers.0.input_layernorm.weight"] = torch.randn(hidden)
        sd["model.layers.0.post_attention_layernorm.weight"] = torch.randn(hidden)

        # GraniteMoe stores experts as fused input_linear [E, 2*I, D]
        sd["model.layers.0.block_sparse_moe.input_linear.weight"] = torch.randn(
            num_experts, 2 * moe_intermediate, hidden
        )
        sd["model.layers.0.block_sparse_moe.output_linear.weight"] = torch.randn(
            num_experts, hidden, moe_intermediate
        )
        sd["model.layers.0.block_sparse_moe.router.layer.weight"] = torch.randn(
            num_experts, hidden
        )

        our_model._mutate_state_dict(sd)

        # input_linear should be split into gate_proj and up_proj
        assert "model.layers.0.moe.switch_mlp.gate_proj.weight" in sd
        assert "model.layers.0.moe.switch_mlp.up_proj.weight" in sd
        assert sd["model.layers.0.moe.switch_mlp.gate_proj.weight"].shape == (1, num_experts, moe_intermediate, hidden)
        assert sd["model.layers.0.moe.switch_mlp.up_proj.weight"].shape == (1, num_experts, moe_intermediate, hidden)
        assert "model.layers.0.block_sparse_moe.input_linear.weight" not in sd

        # output_linear should become down_proj
        assert "model.layers.0.moe.switch_mlp.down_proj.weight" in sd
        assert sd["model.layers.0.moe.switch_mlp.down_proj.weight"].shape == (1, num_experts, hidden, moe_intermediate)
        assert "model.layers.0.block_sparse_moe.output_linear.weight" not in sd

        # router.layer should become gate
        assert "model.layers.0.moe.gate.weight" in sd
        assert sd["model.layers.0.moe.gate.weight"].shape == (num_experts, hidden)
        assert "model.layers.0.block_sparse_moe.router.layer.weight" not in sd

        print("State dict mutation: experts split for SwitchGLU - OK")

    def test_forward_parity_real_config(self):
        """Verify with real model config parameters (rope_theta=1e7, rms_norm_eps=1e-6, 40 experts)."""
        hf_config = _make_granitemoe_config(
            rope_theta=1e7,
            rms_norm_eps=1e-6,
            num_hidden_layers=2,
            num_local_experts=40,
            num_experts_per_tok=8,
        )
        our_config = _make_granitemoe_config(
            rope_theta=1e7,
            rms_norm_eps=1e-6,
            num_hidden_layers=2,
            num_local_experts=40,
            num_experts_per_tok=8,
        )

        torch.manual_seed(42)
        hf_model = HFGraniteMoeForCausalLM(hf_config).to(torch.float32).eval()

        our_model = GraniteMoeForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, 6))
        position_ids = torch.arange(6, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        max_diff = (our_out - hf_out.logits).abs().max().item()
        mean_diff = (our_out - hf_out.logits).abs().mean().item()
        print(f"Real config (f32, seq=6, 40 experts): max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_real_config_f16(self):
        """Verify with real model config parameters in float16."""
        hf_config = _make_granitemoe_config(
            rope_theta=1e7,
            rms_norm_eps=1e-6,
            num_hidden_layers=2,
            num_local_experts=40,
            num_experts_per_tok=8,
        )
        our_config = _make_granitemoe_config(
            rope_theta=1e7,
            rms_norm_eps=1e-6,
            num_hidden_layers=2,
            num_local_experts=40,
            num_experts_per_tok=8,
        )

        torch.manual_seed(42)
        hf_model = HFGraniteMoeForCausalLM(hf_config).to(torch.float16).eval()

        our_model = GraniteMoeForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float16).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, 6))
        position_ids = torch.arange(6, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float16)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        max_diff = (our_out - hf_out.logits).abs().max().item()
        mean_diff = (our_out - hf_out.logits).abs().mean().item()
        print(f"Real config (f16, seq=6, 40 experts): max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
        torch.testing.assert_close(our_out, hf_out.logits, atol=5e-3, rtol=5e-3)
