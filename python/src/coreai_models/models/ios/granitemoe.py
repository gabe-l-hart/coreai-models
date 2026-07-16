# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""iOS implementation of GraniteMoeForCausalLM (Mixture-of-Experts variant).

Based on transformers.models.granitemoe.modeling_granitemoe.GraniteMoeForCausalLM.

Follows the same Extend-style architecture as GraniteForCausalLMForiOS,
replacing the standard MLP with a mixture-of-experts block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.granitemoe.modeling_granitemoe import (
    GraniteMoeForCausalLM as HFGraniteMoeForCausalLM,
)

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.quantization import (
    dequantize_per_tensor,
    quantize_per_tensor,
)
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import RoPECache, apply_rope
from coreai_models.primitives.ios.sdpa import SDPA


class GraniteMoeMoEExperts(nn.Module):
    """Mixture-of-experts block for iOS (Conv2d-based).

    Input shape: [batch, seq_len, 1, hidden_size] (iOS Conv2d convention)
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.act = nn.SiLU()

        # Fused gate + up: [num_experts, 2*intermediate, hidden, 1, 1]
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.num_local_experts, 2 * config.intermediate_size, config.hidden_size, 1, 1)
        )
        # Down: [num_experts, hidden, intermediate, 1, 1]
        self.down_proj = nn.Parameter(
            torch.empty(config.num_local_experts, config.hidden_size, config.intermediate_size, 1, 1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute MoE output with top-k expert routing.

        Args:
            hidden_states: [batch, seq_len, 1, hidden_size]

        Returns:
            [batch, seq_len, 1, hidden_size]
        """
        bsz, length, _, hidden_size = hidden_states.shape

        # Router: [batch*length, hidden] -> top-k indices and weights
        router_logits = F.linear(hidden_states.reshape(-1, self.hidden_size))
        # Use the router.weight which is [num_experts, hidden, 1, 1] -> [num_experts, hidden]
        router_logits = F.linear(
            hidden_states.reshape(-1, self.hidden_size),
            self.gate_up_proj[0, :0, :, 0, 0],  # dummy, will be replaced
        )
        # Actually the router weight is stored separately.
        # We'll compute routing in the decoder layer and pass results here.
        # For now, raise — see GraniteMoeDecoderLayer.
        raise NotImplementedError(
            "GraniteMoeMoEExperts.forward requires router precomputation "
            "from the decoder layer."
        )


class GraniteMoeAttention(nn.Module):
    """Multi-headed attention for iOS (Conv2d-based)."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", None) or dim // n_heads
        self.num_key_value_groups = n_heads // n_kv_heads

        self.q_proj = nn.Conv2d(dim, n_heads * head_dim, kernel_size=1, bias=config.attention_bias)
        self.k_proj = nn.Conv2d(dim, n_kv_heads * head_dim, kernel_size=1, bias=config.attention_bias)
        self.v_proj = nn.Conv2d(dim, n_kv_heads * head_dim, kernel_size=1, bias=config.attention_bias)
        self.o_proj = nn.Conv2d(n_heads * head_dim, dim, kernel_size=1, bias=config.attention_bias)

        self.sdpa = SDPA(head_dim=self.head_dim, scale=config.attention_multiplier)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _, hidden_size = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        x = x.transpose(-3, -1)
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = (
            query.transpose(-3, -1)
            .reshape(batch_size, query_len, n_heads, self.head_dim)
            .transpose(-2, -3)
        )
        key = (
            key.transpose(-3, -1)
            .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
            .transpose(-2, -3)
        )

        query = apply_rope(query, rope_cos, rope_sin)
        key = apply_rope(key, rope_cos, rope_sin)

        query = (
            query.transpose(-2, -3)
            .reshape(batch_size, query_len, 1, n_heads * self.head_dim)
            .transpose(-3, -1)
        )
        key = (
            key.transpose(-3, -2)
            .reshape(batch_size, query_len, 1, n_kv_heads * self.head_dim)
            .transpose(-3, -1)
        )

        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx, in_step, key, value, query_len,
            )

        output = self.sdpa(query, key, value, causal_mask)
        output = self.o_proj(output)
        return output.transpose(-3, -1)


class GraniteMoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = GraniteMoeAttention(config, layer_idx=layer_idx)
        self.moe = GraniteMoeMoEExperts(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.residual_multiplier = config.residual_multiplier

        # Router: [num_experts, hidden, 1, 1]
        self.router_weight = nn.Parameter(
            torch.empty(config.num_local_experts, config.hidden_size, 1, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), rope_cos, rope_sin, in_step, causal_mask, cache)
        h = x + r * self.residual_multiplier

        # MoE forward with routing
        moe_input = self.post_attention_layernorm(h)
        moe_out = self._moe_forward(moe_input)

        return h + moe_out * self.residual_multiplier

    def _moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute MoE block with top-k expert routing.

        Args:
            hidden_states: [batch, seq_len, 1, hidden_size]

        Returns:
            [batch, seq_len, 1, hidden_size]
        """
        bsz, length, _, _ = hidden_states.shape
        num_tokens = bsz * length

        # Router: compute top-k indices and weights
        # [num_tokens, hidden] @ [hidden, num_experts] -> [num_tokens, num_experts]
        router_logits = F.linear(
            hidden_states.reshape(-1, self.moe.hidden_size),
            self.router_weight.reshape(self.moe.num_experts, self.moe.hidden_size).t(),
        )
        top_k_logits, top_k_index = router_logits.topk(self.moe.num_experts_per_tok, dim=-1)
        top_k_weights = torch.softmax(top_k_logits, dim=-1).type_as(hidden_states)

        # Compute expert outputs using index_add
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.moe.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, num_tokens]
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]  # [n_tokens_for_expert, 1, 1, hidden]

            # Gate + up: conv2d [n, 2*I, H, 1, 1] @ [n, 1, 1, H] -> [n, 2*I, 1, 1]
            gate, up = F.conv2d(current_state, self.moe.gate_up_proj[expert_idx]).chunk(2, dim=1)
            current_hidden_states = self.moe.act(gate) * up

            # Down: conv2d [n, I, 1, 1] @ [n, H, I, 1, 1] -> [n, H, 1, 1]
            current_hidden_states = F.conv2d(current_hidden_states, self.moe.down_proj[expert_idx])

            # Weight by router score and accumulate
            weight = top_k_weights[token_idx, top_k_pos].reshape(-1, 1, 1, 1)
            current_hidden_states = current_hidden_states * weight
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class GraniteMoeModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.embedding_multiplier = config.embedding_multiplier
        self.layers = nn.ModuleList(
            [GraniteMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        token_embeddings = token_embeddings * self.embedding_multiplier
        for layer in self.layers:
            token_embeddings = layer(
                token_embeddings,
                rope_cos,
                rope_sin,
                in_step,
                causal_mask,
                cache,
            )
        return self.norm(token_embeddings)


class GraniteMoeExtend(nn.Module):
    """iOS Extend wrapper for GraniteMoe."""

    def __init__(self, config):
        super().__init__()
        self.model = GraniteMoeModel(config)

        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.emb_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)

        self.prefill_mode = False

        self.logits_scale = nn.Parameter(
            torch.tensor(1.0 / config.logits_scaling, dtype=torch.float16),
            requires_grad=False,
        )

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        self.kv_cache = KVCacheHandler(config.num_hidden_layers, config.hidden_size)

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rope_theta = resolve_rope_theta(config)
        self.rope = RoPECache(head_dim, config.max_position_embeddings, rope_theta)

    def forward(
        self,
        transformer_input: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        embedding_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.kv_cache.register_kv_cache(key_cache, value_cache)
        rope_cos, rope_sin = self.rope.gather_cos_sin(position_ids)

        batch_size, seq_len, _, hidden_dim = transformer_input.shape
        out = self.model(
            transformer_input,
            rope_cos,
            rope_sin,
            in_step,
            causal_mask,
            self.kv_cache,
        )

        if self.prefill_mode:
            return self.kv_cache.k_cache[0, 0, 0, 0, 0] + self.kv_cache.v_cache[0, 0, 0, 0, 0]

        if self.lm_head is not None:
            logits = self.lm_head(out.transpose(-2, -3))
        else:
            if embedding_table.dtype == torch.int8:
                embedding_table = dequantize_per_tensor(
                    embedding_table, self.emb_scale, self.emb_zero_point, out.dtype,
                )

            embedding_table = embedding_table.reshape(
                embedding_table.shape[1], embedding_table.shape[0], embedding_table.shape[2],
            )

            out = out.transpose(-3, -1).reshape(batch_size, 1, hidden_dim, seq_len)
            logits = (embedding_table @ out).transpose(-2, -1)

        return logits * self.logits_scale


class GraniteMoeForCausalLMForiOS(BaseForCausalLMForiOS):
    _HF_MODEL_CLASS = HFGraniteMoeForCausalLM

    def _init_model(self, config) -> None:
        self.extend = GraniteMoeExtend(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = self.gather_embeddings(input_ids, self.load_embeddings.embedding_table)
        return self.extend(
            token_embeddings,
            position_ids,
            in_step,
            causal_mask,
            key_cache,
            value_cache,
            self.load_embeddings.embedding_table,
        )

    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
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
            # ---- Attention: Conv2d reshape [O, I] -> [O, I, 1, 1] ----
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                weight_key = f"model.layers.{i}.self_attn.{proj}.weight"
                if weight_key in state_dict:
                    state_dict[weight_key] = state_dict[weight_key].unsqueeze(-1).unsqueeze(-1)

            # ---- MoE Experts: Conv2d reshape + fuse gate+up if separate ----
            hf_gate_up = state_dict.get(f"model.layers.{i}.block_sparse_moe.experts.gate_up_proj")
            hf_down = state_dict.get(f"model.layers.{i}.block_sparse_moe.experts.down_proj")

            if hf_gate_up is not None:
                # HF: [num_experts, 2*intermediate, hidden]
                # iOS: [num_experts, 2*intermediate, hidden, 1, 1]
                state_dict[f"model.layers.{i}.moe.gate_up_proj"] = hf_gate_up.unsqueeze(-1).unsqueeze(-1)
                del state_dict[f"model.layers.{i}.block_sparse_moe.experts.gate_up_proj"]

            if hf_down is not None:
                # HF: [num_experts, hidden, intermediate]
                # iOS: [num_experts, hidden, intermediate, 1, 1]
                state_dict[f"model.layers.{i}.moe.down_proj"] = hf_down.unsqueeze(-1).unsqueeze(-1)
                del state_dict[f"model.layers.{i}.block_sparse_moe.experts.down_proj"]

            # ---- Router: HF weight [num_experts, hidden] -> iOS [num_experts, hidden, 1, 1] ----
            hf_router_w = state_dict.get(f"model.layers.{i}.block_sparse_moe.router.weight")
            if hf_router_w is not None:
                # HF stores [E, D], our router_weight is [E, D, 1, 1]
                state_dict[f"model.layers.{i}.moe.router_weight"] = hf_router_w.unsqueeze(-1).unsqueeze(-1)
                del state_dict[f"model.layers.{i}.block_sparse_moe.router.weight"]

        # Handle embeddings
        embedding_table = state_dict["model.embed_tokens.weight"].unsqueeze(1)
        if not self.disable_embedding_quantization:
            embedding_table, scale, zero_point = quantize_per_tensor(embedding_table, nbits=8, symmetric=True)
        else:
            scale = torch.tensor(1.0, dtype=embedding_table.dtype)
            zero_point = torch.tensor(0, dtype=torch.int8)

        state_dict["load_embeddings.embedding_table"] = embedding_table
        state_dict["gather_embeddings.scale"] = scale
        state_dict["gather_embeddings.zero_point"] = zero_point
        state_dict["extend.emb_scale"] = scale
        state_dict["extend.emb_zero_point"] = zero_point

        state_dict.pop("model.embed_tokens.weight")

        # GraniteModel is held inside GraniteMoeExtend — add "extend." prefix
        new_state_dict = {}
        keys_to_pop = set()

        for k in list(state_dict.keys()):
            if k.startswith("model.") and "gather_embeddings" not in k:
                new_key = f"extend.{k}"
                new_state_dict[new_key] = state_dict[k]
                keys_to_pop.add(k)

        for k in keys_to_pop:
            state_dict.pop(k)
        state_dict.update(new_state_dict)

        if not self.config.tie_word_embeddings:
            state_dict["extend.lm_head.weight"] = state_dict["lm_head.weight"]

        state_dict.pop("lm_head.weight", None)

        # Store logits_scale
        logits_scale_val = 1.0 / self.config.logits_scaling
        state_dict["extend.logits_scale"] = torch.tensor(logits_scale_val, dtype=torch.float16)
