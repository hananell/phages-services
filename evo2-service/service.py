"""Evo2 Embedding Service.

A self-hosted FastAPI service that generates per-layer mean-pooled embeddings
for phage DNA sequences using the pre-trained Evo2 7B model.
"""

from __future__ import annotations

import base64
import gc
import types
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

# =============================================================================
# Configuration
# =============================================================================

CONFIG_PATH: Path = Path(__file__).parent / "config.yaml"

DeviceType = Literal["cpu", "cuda"]


class ModelConfig(BaseModel):
    name: str = "evo2_7b"
    device: DeviceType | None = None
    max_sequence_length: int = 1_000_000


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8001


class EmbeddingConfig(BaseModel):
    layer_names: list[str] = Field(default_factory=lambda: ["blocks.28.mlp.l3", "blocks.31"])
    # Chunk size used when a full-sequence forward pass triggers CUDA OOM.
    # Sequences that fit in VRAM are always processed whole (best quality).
    # Only sequences that OOM fall back to this chunk size.
    fallback_chunk_size: int = 65_536


class Config(BaseModel):
    model: ModelConfig
    server: ServerConfig
    embedding: EmbeddingConfig


def load_config(config_path: Path = CONFIG_PATH) -> Config:
    with open(config_path) as f:
        raw: dict = yaml.safe_load(f)
    return Config(**raw)


# =============================================================================
# Model wrapper
# =============================================================================

VALID_NUCLEOTIDES: frozenset[str] = frozenset("ATCGN")
PAD_TOKEN_ID: int = 0


def validate_sequence(sequence: str) -> str | None:
    if not sequence:
        return "Sequence cannot be empty"
    invalid = set(sequence.upper()) - VALID_NUCLEOTIDES
    if invalid:
        return f"Invalid characters: {invalid}. Only A, T, C, G, N allowed."
    return None


# =============================================================================
# Memory-efficient compute_filter patch
# =============================================================================

def _memory_efficient_compute_filter(self: object, L: int, device: torch.device) -> tuple:
    """Drop-in replacement for HyenaCascade.compute_filter.

    The original computes `(log_poles * self.t).exp()` as a single broadcast,
    producing an intermediate of shape (num_systems, state_size, L) in float32.
    For L=50_000, D=4096, state_size=16 this is 12.21 GiB — the root cause of OOM.

    This version loops over state_size (16 iterations), keeping the peak
    allocation at O(num_systems × L) ≈ 0.8 GiB for L=50_000.
    """
    self.update_time(L, device)  # sets self.t → (1, 1, L) on device
    filter_dtype = torch.float32
    residues = self.residues.to(filter_dtype)   # (num_systems, state_size)
    log_poles = self.log_poles.to(filter_dtype)  # (num_systems, state_size, 1)

    t = self.t.reshape(-1)  # (L,)
    num_systems, state_size = residues.shape
    h = torch.zeros(num_systems, L, dtype=filter_dtype, device=device)
    for s in range(state_size):
        # (num_systems, 1) * (L,) → (num_systems, L) — one state at a time
        h.add_(residues[:, s : s + 1] * (log_poles[:, s, :1] * t).exp())
    return h[None], filter_dtype, log_poles, residues


def _patch_compute_filter(model: object) -> None:
    """Monkey-patch all HyenaCascade IIR filter blocks in a loaded StripedHyena model."""
    from vortex.model.model import HyenaCascade

    patched = 0
    for module in model.modules():
        if isinstance(module, HyenaCascade) and module.h is None:
            module.compute_filter = types.MethodType(_memory_efficient_compute_filter, module)
            patched += 1
    logger.info(f"Patched compute_filter on {patched} HyenaCascade IIR blocks "
                f"(peak memory: O(D×L) instead of O(D×S×L))")


# =============================================================================
# Memory-efficient fftconv patch
# =============================================================================

# Channel group size for chunked FFT convolution.
# Peak per chunk = _FFTCONV_CHANNEL_CHUNK * L * 8 * 3 bytes (k_f + u_f + product).
# 512 channels × 100K tokens × 24 bytes ≈ 1.2 GiB — safe on 32 GiB VRAM.
_FFTCONV_CHANNEL_CHUNK: int = 512


def _chunked_fftconv_func(
    u: torch.Tensor,
    k: torch.Tensor,
    D: torch.Tensor,
    dropout_mask,
    gelu: bool = True,
    k_rev=None,
    bidirectional: bool = False,
    print_activations: bool = False,
    layer_idx=None,
    **kwargs,
) -> torch.Tensor:
    """Channel-chunked replacement for vortex.model.engine.fftconv_func.

    The original allocates (batch, D, 2L) complex64 tensors all at once.
    For D=4096, L=75K this is ~9 GiB of intermediates.

    This version processes _FFTCONV_CHANNEL_CHUNK channels at a time,
    reducing the peak to O(chunk × L) ≈ 1.2 GiB.
    """
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    C = _FFTCONV_CHANNEL_CHUNK

    # k may be (D, L), (1, D, L), or (D, 1, L) — normalise to (D, L)
    k_sq = k.squeeze()
    if k_sq.dim() == 1:
        # scalar filter broadcast to all channels
        k_sq = k_sq.unsqueeze(0).expand(u.shape[-2], -1)

    # u: (B, D, L)
    D_size = u.shape[-2]
    out = torch.empty_like(u)

    for c0 in range(0, D_size, C):
        c1 = min(c0 + C, D_size)
        u_c = u[:, c0:c1, :]          # view (B, C, L)
        k_c = k_sq[c0:c1, :]          # view (C, L) or (1, L) depending on shape

        k_f = torch.fft.rfft(k_c.to(torch.float32), n=fft_size) / fft_size  # (C, L+1)
        # Unsqueeze to broadcast over batch dim
        k_f = k_f.unsqueeze(0)         # (1, C, L+1)

        if bidirectional:
            u_f = torch.fft.rfft(u_c.to(torch.float32), n=fft_size)
            k_c2, k_c3 = k_c.split(k_c.shape[0] // 2, dim=0) if k_c.shape[0] > 1 else (k_c, k_c)
            k2_f = torch.fft.rfft(k_c3.to(torch.float32), n=fft_size) / fft_size
            k2_f = k2_f.unsqueeze(0)
            y1 = u_f * k_f
            y2 = u_f.conj() * k2_f.conj()
            y = torch.fft.irfft(y1 + y2, n=fft_size, norm="forward")[..., :seqlen]
        else:
            if k_rev is not None:
                k_rev_c = k_rev.squeeze()[c0:c1, :]
                k_rev_f = torch.fft.rfft(k_rev_c.to(torch.float32), n=fft_size) / fft_size
                k_f = k_f + k_rev_f.unsqueeze(0).conj()
            u_f = torch.fft.rfft(u_c.to(torch.float32), n=fft_size)
            y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

        D_c = D[c0:c1] if D.shape[0] > 1 else D
        out[:, c0:c1, :] = (y + u_c.to(y.dtype) * D_c.unsqueeze(-1)).to(u.dtype)

    return out


def _patch_fftconv(model: object) -> None:
    """Replace fftconv_func in the vortex.model.engine module with a channel-chunked version.

    All HyenaInferenceEngine instances look up fftconv_func by name from their
    module's global scope, so patching the module-level name is sufficient.
    """
    import vortex.model.engine as _engine
    _engine.fftconv_func = _chunked_fftconv_func
    logger.info(
        f"Patched fftconv_func with channel-chunked version "
        f"(chunk={_FFTCONV_CHANNEL_CHUNK} channels, "
        f"peak O(chunk×L) instead of O(D×L))"
    )


# =============================================================================
# Memory-efficient parallel_iir patch
# =============================================================================

def _chunked_parallel_iir(
    self: object,
    z_pre: torch.Tensor,
    h: torch.Tensor,
    D: torch.Tensor,
    L: int,
    poles,
    residues,
    t,
    dims,
    layer_idx: int,
    inference_params=None,
    prefill_style: str = "fft",
    fftconv_fn=None,
    padding_mask=None,
    use_flashfft: bool = False,
    column_split_hyena: bool = False,
    long_fir_threshold=None,
) -> torch.Tensor:
    """Channel-chunked replacement for HyenaInferenceEngine.parallel_iir.

    The original allocates:
      H   = rfft(h, n=2L)            → (1, D, L+1)  complex64 = D·L·8 bytes
      X_s = fft(x1v, n=2L)           → (1, D, 2L)   complex64 = D·L·16 bytes ← main culprit
      y   = irfft(X_s[:L+1] * H, 2L) → (1, D, 2L)   float32   = D·L·8 bytes

    For D=4096, L=75K that peak is ~11 GiB.  This version processes
    _FFTCONV_CHANNEL_CHUNK channels at a time, keeping peak at ~1 GiB.

    For the stateless-inference path (inference_params=None) the X_s tensor
    is never reused after computing y, so it is safe to discard per chunk.
    If inference_params is not None we fall back to the original implementation.
    """
    from vortex.model.engine import fftconv_func  # picks up our patched version
    import torch.nn.functional as F

    fft_size = 2 * L
    hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims

    # ── Split projections ────────────────────────────────────────────────────
    if column_split_hyena:
        z = z_pre.reshape(
            z_pre.shape[0],
            num_attention_heads,
            3 * hidden_size_per_attention_head,
            z_pre.shape[2],
        )
        x2 = z[:, :, :hidden_size_per_attention_head].reshape(z_pre.shape[0], -1, z_pre.shape[2])
        x1 = z[:, :, hidden_size_per_attention_head:2*hidden_size_per_attention_head].reshape(z_pre.shape[0], -1, z_pre.shape[2])
        v  = z[:, :, 2*hidden_size_per_attention_head:].reshape(z_pre.shape[0], -1, z_pre.shape[2])
    else:
        x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)

    if self.hyena_flip_x1x2:
        x1, x2 = x2, x1

    x1v = x1 * v

    # ── FFT convolution ──────────────────────────────────────────────────────
    if inference_params is not None and prefill_style == "recurrence":
        y = self.prefill_via_direct_recurrence(
            inference_params=inference_params,
            x1v=x1v,
            L=L,
            poles=poles,
            residues=residues,
        )
    elif use_flashfft and (L % 2) == 0:
        y = fftconv_fn(
            x1v.to(dtype=torch.bfloat16).contiguous(),
            h.to(dtype=torch.float32),
        )
    elif inference_params is not None:
        # Stateful path needs X_s for prefill — use the original (no chunking).
        # This branch is never hit during embedding extraction.
        H = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
        X_s = torch.fft.fft(x1v.to(dtype=torch.float32), n=fft_size)
        X = X_s[..., : H.shape[-1]]
        if len(z_pre.shape) > 3:
            H = H.unsqueeze(1)
        y = torch.fft.irfft(X * H, n=fft_size, norm="forward")[..., :L]
        self.prefill_via_modal_fft(
            inference_params=inference_params,
            x1v=x1v,
            X_s=X_s,
            L=L,
            t=t,
            poles=poles,
            dims=dims,
            layer_idx=layer_idx,
            use_flashfft=use_flashfft,
            fftconv_fn=fftconv_fn,
        )
    elif long_fir_threshold is not None:
        assert h.shape[0] == 1, "batch size must be 1 for long_fir_threshold"
        h_fir = h[0][:, None][..., :long_fir_threshold]
        y = F.conv1d(
            x1v, h_fir.to(dtype=x1v.dtype), stride=1,
            groups=x1v.shape[1], padding=h_fir.shape[-1] - 1,
        )[..., :L]
    else:
        # ── Chunked FFT (stateless, no flashfft, no long_fir) ────────────────
        # Peak per chunk: H_c + X_c + product + irfft ≈ 1 GiB for C=256, L=75K
        C = _FFTCONV_CHANNEL_CHUNK
        D_size = h.shape[1]
        y = torch.empty(
            x1v.shape[0], D_size, L, dtype=torch.float32, device=x1v.device
        )
        for c0 in range(0, D_size, C):
            c1 = min(c0 + C, D_size)
            h_c = h[:, c0:c1, :]        # view (1, C, L)
            x1v_c = x1v[:, c0:c1, :]   # view (1, C, L)
            H_c = torch.fft.rfft(h_c.to(torch.float32), n=fft_size) / fft_size
            X_c = torch.fft.fft(x1v_c.to(torch.float32), n=fft_size)
            y[:, c0:c1, :] = torch.fft.irfft(
                X_c[..., : H_c.shape[-1]] * H_c, n=fft_size, norm="forward"
            )[..., :L]
            del H_c, X_c

    # ── Post-gate ─────────────────────────────────────────────────────────────
    y = y.to(dtype=x1v.dtype)
    y = (y + x1v * D.unsqueeze(-1)) * x2

    if type(padding_mask) == torch.Tensor:
        y = y * padding_mask[:, None]

    return y.permute(0, 2, 1)


def _patch_parallel_iir(model: object) -> None:
    """Monkey-patch parallel_iir on all HyenaInferenceEngine instances inside IIR blocks."""
    from vortex.model.model import HyenaCascade

    patched = 0
    for module in model.modules():
        if isinstance(module, HyenaCascade) and module.h is None:
            module.engine.parallel_iir = types.MethodType(_chunked_parallel_iir, module.engine)
            patched += 1
    logger.info(
        f"Patched parallel_iir on {patched} HyenaInferenceEngine IIR instances "
        f"(peak O(chunk×L) instead of O(D×L))"
    )


# =============================================================================
# Memory-efficient parallel_fir patch (bfloat16 short conv)
# =============================================================================

def _memory_efficient_parallel_fir(
    self: object,
    fir_fn,
    u,
    weight,
    bias,
    L,
    dims,
    groups=None,
    gated_bias=False,
    column_split_hyena=False,
    dim_last=True,
    fir_length=3,
    gate=False,
    inference_params=None,
    prefill_mode=None,
    padding_mask=None,
):
    """Drop-in replacement for HyenaInferenceEngine.parallel_fir.

    The original short-conv path (fir_length < 128) converts u to float32
    before calling depthwise conv1d, creating an intermediate of shape
    (B, 3*D, L) in float32.  For B=1, D=4096, L=100K this is:
      u.float():   1 × 12288 × 100000 × 4 B = 4.58 GiB
      conv output: same = 4.58 GiB
      z.to(bf16):  1 × 12288 × 100000 × 2 B = 2.29 GiB
      peak total ≈ 11.45 GiB — causes OOM at L≥100K.

    This version runs the short-conv entirely in the input dtype (bfloat16):
      u bf16:      2.29 GiB
      conv output: 2.29 GiB
      peak total ≈ 4.58 GiB — safe at L=100-200K.

    PyTorch ≥ 2.x supports bfloat16 depthwise conv1d on CUDA.
    """
    import vortex.model.engine as _engine
    from vortex.model.utils import column_split

    L = u.shape[1] if dim_last else u.shape[2]
    if gate:
        hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
        if column_split_hyena:
            x2, x1, v = column_split(u, num_attention_heads, hidden_size_per_attention_head)
        else:
            x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1
        u = x1 * v

    if fir_fn != torch.nn.functional.conv1d:
        # Deprecated non-conv1d kernel path — unchanged
        if dim_last:
            u = u.permute(0, 2, 1)
        z = fir_fn(u)[:, :L]

    elif fir_length >= 128:
        # Long FIR — uses fftconv which is already channel-chunked via our patch
        with torch.autocast("cuda"):
            z = _engine.fftconv_func(
                u.to(torch.float32),
                weight[:, :, :L].to(torch.float32),
                bias,
                None,
                gelu=False,
                bidirectional=False,
                print_activations=self.print_activations,
                groups=groups,
                layer_idx=self.layer_idx,
            )
            z = z.to(u.dtype)

    else:
        # Short depthwise conv (fir_length=3).
        # Run entirely in u.dtype (bfloat16) — saves ~7 GiB at L=100K.
        if dim_last:
            u = u.permute(0, 2, 1)

        z = fir_fn(
            u,
            weight.to(u.dtype),
            bias=None,
            stride=1,
            padding=fir_length - 1,
            groups=u.shape[1],
        )[..., :L]
        # z is already in u.dtype, no conversion needed

        if bias is not None:
            if gated_bias:
                z = z + bias[None, :, None] * u
            else:
                z = z + bias[None, :, None]

    if type(padding_mask) == torch.Tensor:
        z = z * padding_mask[:, None]

    if gate:
        z = x2 * z

    # Return (output, fir_state). For stateless (embedding) inference the state
    # is always None; the two-tuple matches the API expected by newer vortex builds.
    return z, None


def _patch_parallel_fir(model: object) -> None:
    """Patch HyenaInferenceEngine.parallel_fir at class level to skip float32 conversion.

    All engine instances share the same class method, so one class-level patch
    covers all 23 Hyena blocks (9 IIR + 9 HCM + 5 HCS).
    """
    from vortex.model.engine import HyenaInferenceEngine
    HyenaInferenceEngine.parallel_fir = _memory_efficient_parallel_fir
    logger.info(
        "Patched HyenaInferenceEngine.parallel_fir: short-conv runs in bfloat16 "
        "(saves ~7 GiB per block for L=100K; covers all 23 Hyena blocks)"
    )


class Evo2EmbeddingModel:
    """Wraps the Evo2 model for single-sequence embedding extraction."""

    def __init__(self, model_name: str, device: DeviceType | None = None) -> None:
        self.model_name = model_name
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def load(self) -> None:
        from evo2 import Evo2
        logger.info(f"Loading Evo2 model '{self.model_name}' on {self.device}")
        self._model = Evo2(self.model_name)
        _patch_compute_filter(self._model.model)
        _patch_fftconv(self._model.model)
        _patch_parallel_iir(self._model.model)
        _patch_parallel_fir(self._model.model)
        logger.info("Evo2 model loaded successfully")

    def get_embedding_single(
        self,
        sequence: str,
        layer_names: list[str],
        fallback_chunk_size: int = 65_536,
    ) -> dict[str, np.ndarray]:
        """Return mean-pooled embeddings for a single sequence.

        Attempts a full-sequence forward pass first (best quality: the model
        sees the whole phage at once). If that triggers a CUDA OOM, clears the
        cache and retries with non-overlapping chunks of `fallback_chunk_size`
        tokens, averaging their embeddings.

        Args:
            sequence: DNA sequence (uppercase, validated).
            layer_names: Intermediate layers to extract.
            fallback_chunk_size: Chunk size for the OOM fallback path.

        Returns:
            Dict mapping layer_name -> float32 ndarray of shape (embed_dim,).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        tokens: list[int] = self._model.tokenizer.tokenize(sequence)

        # ── Always use chunked processing ───────────────────────────────────
        # Full-sequence forward pass for long sequences triggers:
        #   • INT32 overflow in FFT intermediates (D=4096, 2*L > INT_MAX at L > 262K)
        #   • CUFFT_INTERNAL_ERROR for non-power-of-2 FFT sizes (e.g. 2×225K=450K)
        #   • CUDA illegal memory access that permanently corrupts the CUDA context
        # Using chunk_size=65536 (= 2^16) forces fft_size = 2^17, which is
        # cuFFT-optimal and well within VRAM for all sequence lengths.
        return self._embed_tokens_chunked(tokens, layer_names, fallback_chunk_size)

        # ── Try the full sequence first ─────────────────────────────────────
        _fallback = False
        _fallback_reason: str = ""
        try:
            return self._embed_tokens_full(tokens, layer_names)
        except torch.cuda.OutOfMemoryError:
            _fallback = True
            _fallback_reason = f"CUDA OOM for {len(tokens)} tokens"
        except RuntimeError as e:
            # TransformerEngine GEMM hits a 32-bit integer index limit for
            # very long sequences (≥~512K tokens × 4096 dims > INT_MAX).
            # The error surfaces as a RuntimeError with "CUDA" in the message.
            if "CUDA" not in str(e) and "out of memory" not in str(e).lower():
                raise  # non-CUDA RuntimeError — propagate normally
            _fallback = True
            _fallback_reason = f"CUDA error for {len(tokens)} tokens: {e}"

        # ── Fallback recovery — MUST be outside the except block ───────────
        # Python keeps exception frame-locals (= GPU tensors) alive through
        # __traceback__ for as long as the except-clause is executing.  Only
        # after the except block exits does sys.exc_info() clear those refs.
        # So we set a flag above and do all cleanup here, outside the block.
        if _fallback:
            gc.collect()             # finalise tensors whose ref-count just dropped
            torch.cuda.empty_cache() # return freed CUDA memory to the OS pool
            gc.collect()             # second sweep (cyclic garbage)
            torch.cuda.empty_cache()
            logger.warning(
                f"{_fallback_reason}. "
                f"Falling back to chunked processing (chunk_size={fallback_chunk_size}). "
                "Embedding quality will be reduced for this sequence."
            )
            return self._embed_tokens_chunked(tokens, layer_names, fallback_chunk_size)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _embed_tokens_full(
        self, tokens: list[int], layer_names: list[str]
    ) -> dict[str, np.ndarray]:
        """Single forward pass over the entire token sequence."""
        input_ids = torch.tensor([tokens], dtype=torch.int).to(self.device)

        try:
            with torch.inference_mode():
                _, raw_embeddings = self._model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=layer_names,
                )
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            del input_ids  # free before re-raising so caller's GC sweep works
            raise

        del input_ids

        results: dict[str, np.ndarray] = {}
        for layer in layer_names:
            layer_emb: torch.Tensor = raw_embeddings[layer].float()  # (1, L, d)
            results[layer] = layer_emb.mean(dim=1).squeeze(0).cpu().numpy()

        del raw_embeddings
        torch.cuda.empty_cache()

        return results

    def _embed_tokens_chunked(
        self, tokens: list[int], layer_names: list[str], chunk_size: int
    ) -> dict[str, np.ndarray]:
        """Process tokens in non-overlapping chunks; return length-weighted mean."""
        total_len = len(tokens)
        chunk_embeddings: dict[str, list[torch.Tensor]] = {l: [] for l in layer_names}
        chunk_lengths: list[int] = []

        for start in range(0, total_len, chunk_size):
            chunk = tokens[start : start + chunk_size]
            chunk_len = len(chunk)
            chunk_lengths.append(chunk_len)

            input_ids = torch.tensor([chunk], dtype=torch.int).to(self.device)

            with torch.inference_mode():
                _, raw_embeddings = self._model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=layer_names,
                )

            del input_ids

            for layer in layer_names:
                layer_emb: torch.Tensor = raw_embeddings[layer].float()  # (1, L, d)
                chunk_embeddings[layer].append(
                    layer_emb.mean(dim=1).squeeze(0).cpu()  # (d,) on CPU
                )

            del raw_embeddings
            torch.cuda.empty_cache()

        # Length-weighted average across chunks
        total_tokens = sum(chunk_lengths)
        results: dict[str, np.ndarray] = {}
        for layer in layer_names:
            weighted_sum: torch.Tensor = sum(
                emb * w
                for emb, w in zip(chunk_embeddings[layer], chunk_lengths)
            )
            results[layer] = (weighted_sum / total_tokens).numpy()

        return results


# =============================================================================
# API models
# =============================================================================

class BatchEmbedRequest(BaseModel):
    sequences: list[str] = Field(..., min_length=1, description="List of DNA sequences")
    layer_names: list[str] | None = Field(
        default=None,
        description="Layers to extract. Uses config default if omitted.",
    )


class SequenceResult(BaseModel):
    embeddings: dict[str, str]  # base64-encoded little-endian float32 arrays
    sequence_length: int


class BatchEmbedResponse(BaseModel):
    results: list[SequenceResult]
    embedding_dimensions: dict[str, int]


# =============================================================================
# Application
# =============================================================================

config: Config = load_config()
embedding_model = Evo2EmbeddingModel(
    model_name=config.model.name,
    device=config.model.device,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if embedding_model._model is None:
        embedding_model.load()
    yield
    logger.info("Shutting down evo2-service")


app = FastAPI(
    title="Evo2 Embedding Service",
    description="Generate per-layer embeddings for phage DNA sequences using Evo2 7B",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/embed/batch", response_model=BatchEmbedResponse)
async def embed_batch(request: BatchEmbedRequest) -> BatchEmbedResponse:
    """Generate embeddings for a batch of DNA sequences.

    Sequences are processed one at a time to minimise peak GPU memory.
    Each sequence is forwarded through the model as a whole (best quality);
    if a sequence is too long and triggers a CUDA OOM it is automatically
    retried with chunked (tiled) inference.
    """
    layer_names = request.layer_names or config.embedding.layer_names

    sequences: list[str] = []
    for raw_seq in request.sequences:
        seq = raw_seq.upper().strip()
        if err := validate_sequence(seq):
            raise HTTPException(status_code=400, detail=err)
        if len(seq) > config.model.max_sequence_length:
            raise HTTPException(
                status_code=400,
                detail=f"Sequence too long: {len(seq)} > {config.model.max_sequence_length}",
            )
        sequences.append(seq)

    logger.info(
        f"Processing batch of {len(sequences)} sequences, "
        f"lengths={[len(s) for s in sequences]}, layers={layer_names}"
    )

    all_embeddings: list[dict[str, np.ndarray]] = []
    for seq in sequences:
        emb = embedding_model.get_embedding_single(
            seq,
            layer_names,
            fallback_chunk_size=config.embedding.fallback_chunk_size,
        )
        all_embeddings.append(emb)

    results = [
        SequenceResult(
            embeddings={
                layer: base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")
                for layer, arr in emb.items()
            },
            sequence_length=len(seq),
        )
        for seq, emb in zip(sequences, all_embeddings)
    ]

    emb_dims = (
        {layer: all_embeddings[0][layer].size for layer in layer_names}
        if all_embeddings else {}
    )

    return BatchEmbedResponse(results=results, embedding_dimensions=emb_dims)


@app.get("/health")
async def health_check() -> dict[str, object]:
    return {"status": "healthy", "model_loaded": embedding_model._model is not None}


def main() -> None:
    import uvicorn
    logger.info(f"Starting evo2-service on {config.server.host}:{config.server.port}")
    uvicorn.run("service:app", host=config.server.host, port=config.server.port, reload=False)


if __name__ == "__main__":
    main()
