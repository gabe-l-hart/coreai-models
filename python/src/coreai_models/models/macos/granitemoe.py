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


class GraniteMoeMoEExperts(nn.Module):
    """Mixture-of-experts block with top-k routing.

    The HF state dict stores:
      gate_up_proj: [num_experts, 2*intermediate_size, hidden_size]
      down_proj:    [num_experts, hidden_size, intermediate_size]
    and a separate router weight [num_experts, hidden_size].

    Forward: for each token, route to top-k experts, compute, and combine.
    """

    def __init__(self, config: GraniteMoeConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.act = nn.SiLU()

        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        # Fused gate + up: [num_experts, 2*intermediate_size, hidden_size]
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.num_local_experts, 2 * config.intermediate_size, config.hidden_size)
        )
        # [num_experts, hidden_size, intermediate_size]
        self.down_proj = nn.Parameter(
            torch.empty(config.num_local_experts, config.hidden_size, config.intermediate_size)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        bsz, length, _ = hidden_states.shape
        # Flatten tokens
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_tokens = hidden_states.size(0)

        # Router: [num_tokens, num_experts] -> top-k indices and weights
        router_logits = self.router(hidden_states)
        top_k_logits, top_k_index = router_logits.topk(self.num_experts_per_tok, dim=-1)
        top_k_weights = torch.softmax(top_k_logits, dim=-1).type_as(hidden_states)

        # Compute output using index_add pattern
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.view(bsz, length, self.hidden_size)


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
        self.moe = GraniteMoeMoEExperts(config)

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
        """Transform HF state dict to match our parameter layout.

        HF experts store:
          gate_up_proj: [num_experts, 2*intermediate_size, hidden_size]  (already fused)
          down_proj:    [num_experts, hidden_size, intermediate_size]

        HF router stores weight as [num_experts, hidden_size] which
        matches our nn.Linear(num_experts, hidden_size, bias=False)
        reversed → Linear expects [hidden_size, num_experts].
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
            # ---- MoE Experts: gate_up_proj is already [E, 2*I, D] in HF ----
            hf_gate_up = state_dict[f"model.layers.{i}.block_sparse_moe.experts.gate_up_proj"]
            hf_down = state_dict[f"model.layers.{i}.block_sparse_moe.experts.down_proj"]

            state_dict[f"model.layers.{i}.moe.gate_up_proj"] = hf_gate_up
            state_dict[f"model.layers.{i}.moe.down_proj"] = hf_down

            del state_dict[f"model.layers.{i}.block_sparse_moe.experts.gate_up_proj"]
            del state_dict[f"model.layers.{i}.block_sparse_moe.experts.down_proj"]

            # ---- Router: HF weight is [E, D], Linear expects [D, E] ----
            hf_router_w = state_dict[f"model.layers.{i}.block_sparse_moe.router.weight"]
            state_dict[f"model.layers.{i}.moe.router.weight"] = hf_router_w.t()
            del state_dict[f"model.layers.{i}.block_sparse_moe.router.weight"]

            # ---- Attention: HF uses separate q/k/v (already matches our layout) ----
            # No mutation needed — GraniteMoeAttention uses q_proj, k_proj, v_proj separately.