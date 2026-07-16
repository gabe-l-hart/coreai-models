# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""macOS implementation of GraniteMoeForCausalLM (Mixture-of-Experts variant).

Based on transformers.models.granitemoe.modeling_granitemoe.GraniteMoeForCausalLM.

Key architectural differences from regular GraniteForCausalLM:
  - MoE feed-forward: router + multiple experts per layer (block_sparse_moe)
  - Separate q/k/v attention projections (like Mixtral), not fused qkv
  - RoPE via rope_parameters dict (rope_type + rope_theta) instead of rope_theta scalar
  - attention_bias defaults to False
"""

import torch
import torch.nn as nn
from transformers.models.granitemoe.configuration_granitemoe import (
    GraniteMoeConfig,
)
from transformers.models.granitemoe.modeling_granitemoe import (
    GraniteMoeForCausalLM as HFGraniteMoeForCausalLM,
)
from typing_extensions import Self, override

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives.macos.switch import SwitchGLU


class SparseMoeBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate(x).to(torch.float32)

        top_logits, active_experts_indices = torch.topk(
            router_logits, self.top_k, dim=-1, largest=True
        )
        active_experts_scores = torch.softmax(top_logits, dim=-1).to(x.dtype)

        y_active_experts = self.switch_mlp(x, active_experts_indices)
        active_experts_scores = active_experts_scores.unsqueeze(-1).to(y_active_experts.device)
        y_active_experts_weighted_by_scores = y_active_experts * active_experts_scores
        y_active_experts_summary = torch.sum(y_active_experts_weighted_by_scores, dim=-2)
        return y_active_experts_summary.to(device=x.device, dtype=x.dtype)


class GraniteMoeAttention(nn.Module):
    """Multi-headed attention with separate q/k/v projections.

    GraniteMoe uses separate projections (like Mixtral/Llama) rather than
    fused qkv (like Qwen2/GPT-OSS). This avoids needing qkv fusion in
    _mutate_state_dict.
    """

    def __init__(self, config: GraniteMoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", None) or dim // n_heads
        self.num_key_value_groups = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=config.attention_bias)

        self.sdpa = SDPA(is_causal=True, scale=config.attention_multiplier)
        assert resolve_rope_theta(config) is not None, "rope_theta is required"
        self.rope = initialize_rope(
            base=resolve_rope_theta(config),
            dims=self.head_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        query = self.q_proj(x).reshape(batch_size, query_len, n_heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.k_proj(x).reshape(batch_size, query_len, n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.v_proj(x).reshape(batch_size, query_len, n_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        query = self.rope(query, position_ids=rope_positions)
        key = self.rope(key, position_ids=rope_positions)

        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx, offset, key, value, seq_len=seq_len, query_len=query_len
            )

        output = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, n_heads * self.head_dim)
        )
        return self.o_proj(output)


class GraniteMoeDecoderLayer(nn.Module):
    def __init__(self, config: GraniteMoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = GraniteMoeAttention(config, layer_idx=layer_idx)
        self.moe = SparseMoeBlock(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.residual_multiplier = config.residual_multiplier

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids, cache)
        h = x + r * self.residual_multiplier
        r = self.moe(self.post_attention_layernorm(h))
        return h + r * self.residual_multiplier


class GraniteMoeModel(nn.Module):
    def __init__(self, config: GraniteMoeConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.embedding_multiplier = config.embedding_multiplier
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [GraniteMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids) * self.embedding_multiplier
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.norm(h)


class GraniteMoeForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFGraniteMoeForCausalLM

    @override
    def _init_model(self, config: GraniteMoeConfig) -> None:
        self.model = GraniteMoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(input_ids, position_ids, cache)
        logits = self.lm_head(out)
        # Granite-specific: logits scaling
        return logits / self.config.logits_scaling

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        """Transform HF state dict to match our SwitchGLU layout.

        HF experts store:
          input_linear.weight: [num_experts, 2*intermediate_size, hidden_size]  (fused gate+up)
          output_linear.weight: [num_experts, hidden_size, intermediate_size]
          router.layer.weight: [num_experts, hidden_size]

        SwitchGLU expects:
          gate_proj.weight: [1, num_experts, intermediate_size, hidden_size]
          up_proj.weight: [1, num_experts, intermediate_size, hidden_size]
          down_proj.weight: [1, num_experts, hidden_size, intermediate_size]
          router.weight: [hidden_size, num_experts]
        """
        max_layer = -1
        for k in state_dict:
            name_split = k.split(".")
            if len(name_split) != 6:
                continue
            if not k.startswith("model.layers."):
                continue
            max_layer = max(max_layer, int(name_split[2]))

        if max_layer < 0:
            raise ValueError("invalid state_dict")

        for i in range(max_layer + 1):
            # Skip layers whose keys are not present in this (single-layer) state dict
            moe_prefix = f"model.layers.{i}.block_sparse_moe"
            if f"{moe_prefix}.input_linear.weight" not in state_dict:
                continue

            # ---- Split fused gate+up into separate gate_proj and up_proj ----
            hf_input_linear = state_dict[f"{moe_prefix}.input_linear.weight"]  # [E, 2*I, D]
            # Split along the intermediate dimension: first half = gate, second half = up
            hf_gate_up = hf_input_linear.chunk(2, dim=1)  # [E, I, D], [E, I, D]
            gate_weight = hf_gate_up[0].unsqueeze(0)  # [1, E, I, D]
            up_weight = hf_gate_up[1].unsqueeze(0)  # [1, E, I, D]

            state_dict[f"model.layers.{i}.moe.switch_mlp.gate_proj.weight"] = gate_weight
            state_dict[f"model.layers.{i}.moe.switch_mlp.up_proj.weight"] = up_weight

            del state_dict[f"{moe_prefix}.input_linear.weight"]

            # ---- Output linear (down_proj) ----
            hf_output_linear = state_dict[f"{moe_prefix}.output_linear.weight"]  # [E, D, I]
            down_weight = hf_output_linear.unsqueeze(0)  # [1, E, D, I]
            state_dict[f"model.layers.{i}.moe.switch_mlp.down_proj.weight"] = down_weight
            del state_dict[f"{moe_prefix}.output_linear.weight"]

            # ---- Router: HF weight is [E, D], nn.Linear(num_experts, dim) expects [E, D] ----
            hf_router_w = state_dict[f"{moe_prefix}.router.layer.weight"]
            state_dict[f"model.layers.{i}.moe.gate.weight"] = hf_router_w
            del state_dict[f"{moe_prefix}.router.layer.weight"]

            # ---- Attention: HF uses separate q/k/v (already matches our layout) ----
            # No mutation needed - GraniteMoeAttention uses q_proj, k_proj, v_proj separately.

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
