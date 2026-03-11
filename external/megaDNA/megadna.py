"""MEGADNA model — hierarchical MEGABYTE-style DNA transformer.

This module provides the MEGADNA class whose instances are stored in the
megaDNA .pt checkpoint files.  It is needed so that torch.load (with
weights_only=False) can reconstruct the pickled model object.

Architecture:
    3-stage hierarchy with token_embs / transformers /
    to_next_transformer_projections matching the MEGABYTE_pytorch package,
    but with multi-query attention (single KV head) instead of full MHA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, unpack


# ---------------------------------------------------------------------------
# Compatibility shims — the .pt was saved with an older megabyte-pytorch that
# used EfficientAttentionConfig.  We register it so pickle can reconstruct
# the Attend objects, then patch forward() to work with current PyTorch.
# ---------------------------------------------------------------------------

def _patch_megabyte_pytorch() -> None:
    """Ensure MEGABYTE_pytorch is importable and compatible with the checkpoint."""
    import MEGABYTE_pytorch.attend as attend_mod
    import MEGABYTE_pytorch.megabyte as megabyte_mod

    # 1. Register the old EfficientAttentionConfig so pickle doesn't fail.
    if not hasattr(attend_mod, "EfficientAttentionConfig"):
        @dataclass
        class EfficientAttentionConfig:
            enable_flash: bool
            enable_math: bool
            enable_mem_efficient: bool

        attend_mod.EfficientAttentionConfig = EfficientAttentionConfig

    # 2. Replace Attend.forward with a version that works regardless of which
    #    attributes are present (cpu_config vs attn_cfg).
    def _attend_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        mask=None) -> torch.Tensor:
        # k/v may be single-head (b, 1, n, d) — scaled_dot_product_attention
        # broadcasts across the query heads automatically.
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(self.causal if mask is None else False),
        )

    attend_mod.Attend.forward = _attend_forward

    # 3. Replace Attention.forward to handle multi-query attention:
    #    to_q → (b, h, n, d)   /   to_kv → (b, 1, n, d) for k and v.
    from einops import rearrange

    def _exists(val):
        return val is not None

    def _attention_forward(self, x: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        h = self.heads
        x = self.norm(x)
        q = self.to_q(x)
        kv = self.to_kv(x)

        # Multi-query: project q to (b, h, n, d), k/v to (b, 1, n, d)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k, v = kv.chunk(2, dim=-1)
        k = k.unsqueeze(1)   # (b, 1, n, dim_head)
        v = v.unsqueeze(1)   # (b, 1, n, dim_head)

        if _exists(rotary_emb):
            from MEGABYTE_pytorch.megabyte import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(rotary_emb, q), apply_rotary_pos_emb(rotary_emb, k)

        out = self.attend(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    megabyte_mod.Attention.forward = _attention_forward


_patch_megabyte_pytorch()


# ---------------------------------------------------------------------------
# Helpers (mirrors MEGABYTE_pytorch internals)
# ---------------------------------------------------------------------------

def _pack_one(t: torch.Tensor, pattern: str):
    return pack([t], pattern)


def _unpack_one(t: torch.Tensor, ps, pattern: str) -> torch.Tensor:
    return unpack(t, ps, pattern)[0]


# ---------------------------------------------------------------------------
# MEGADNA
# ---------------------------------------------------------------------------

class MEGADNA(nn.Module):
    """Hierarchical 3-stage DNA language model.

    When loaded from a .pt checkpoint all sub-modules (start_tokens,
    token_embs, transformers, to_next_transformer_projections, to_logits)
    are restored by pickle; __init__ is NOT called.

    forward(ids, return_value='loss')
        ids           : LongTensor (batch, seq_len) — flat token ids.
        return_value  : 'loss'      → scalar cross-entropy LM loss.
                        'embedding' → list of 3 tensors, one per stage
                                      (coarse→fine), each (batch, seq, dim).
    """

    # ------------------------------------------------------------------ #
    # forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        ids: torch.Tensor,
        return_value: Literal["loss", "embedding"] = "loss",
    ) -> torch.Tensor | list[torch.Tensor]:
        batch = ids.shape[0]
        seq_len = ids.shape[-1]

        # Pad flat (batch, seq) input to multiple of inner_chunk * outer_chunk
        inner = self.max_seq_len[2]   # 16
        mid   = self.max_seq_len[1]   # 64
        multiple = mid * inner        # 1024
        pad = (-seq_len) % multiple
        ids_padded = F.pad(ids, (0, pad), value=self.pad_id)
        # Reshape: (batch, N, mid, inner)
        ids_nd = ids_padded.reshape(batch, -1, mid, inner)

        # -------------------------------------------------------------- #
        # Build per-stage token embeddings (fine → coarse)               #
        # tokens_at_stages[0] = coarsest, [2] = finest                   #
        # -------------------------------------------------------------- #
        ids_view = ids_nd
        tokens_at_stages = []
        for ind in range(len(self.token_embs)):
            toks = self.token_embs[ind](ids_view)
            tokens_at_stages.insert(0, toks)
            if ind == 0:
                continue
            ids_view = ids_view.flatten(-2)   # merge last two dims

        # -------------------------------------------------------------- #
        # Forward through transformers: coarse → fine                    #
        # -------------------------------------------------------------- #
        prev_repr: torch.Tensor | None = None
        hidden_states: list[torch.Tensor] = []

        for stage_start, stage_tokens, transformer, proj in zip(
            self.start_tokens,
            tokens_at_stages,
            self.transformers,
            self.to_next_transformer_projections,
        ):
            stage_packed, ps = _pack_one(stage_tokens, "* n d")

            start_tok = stage_start.unsqueeze(0).unsqueeze(0).expand(
                stage_packed.shape[0], 1, -1
            )
            stage_packed = torch.cat([start_tok, stage_packed], dim=1)

            if prev_repr is not None:
                prev_packed, _ = _pack_one(prev_repr, "* n d")
                prev_padded = F.pad(prev_packed, (0, 0, 1, 0))
                stage_packed = stage_packed + prev_padded

            attended = transformer(stage_packed)
            attended_nd = _unpack_one(attended, ps, "* n d")
            hidden_states.append(attended_nd)

            prev_repr = proj(attended_nd[..., :-1, :])

        # -------------------------------------------------------------- #
        # Return embeddings or loss                                       #
        # -------------------------------------------------------------- #
        if return_value == "embedding":
            # Each hidden state: remove start token, flatten seq dims →
            # (batch, total_tokens, embed_dim) for mean-pooling.
            return [
                h[..., 1:, :].flatten(1, -2)
                for h in hidden_states
            ]

        # return_value == 'loss'  — language-model cross-entropy
        # Use the finest-stage attended output for next-token prediction.
        finest_attended = hidden_states[-1]  # (batch, N, mid, inner+1, dim)
        logits = self.to_logits(finest_attended)

        # Build labels from padded ids
        labels = ids_nd.flatten(1)
        logits_flat = logits[..., 1:, :].flatten(1, -2)  # drop start token

        logits_flat = logits_flat[:, : labels.shape[1]]  # trim to label length

        loss = F.cross_entropy(
            logits_flat.reshape(-1, logits_flat.shape[-1]),
            labels.reshape(-1),
            ignore_index=self.pad_id,
        )
        return loss
