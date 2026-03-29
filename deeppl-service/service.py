"""DeepPL phage lifestyle prediction service.

Loads a fine-tuned DNABERT model (BertForSequenceClassification) and serves
batch predictions via FastAPI.

Algorithm (replicates DeepPL's two-threshold voting):
  1. Slide a window of `window_bp` (default 105 bp) across the genome with
     `stride` (default 10 bp), generating overlapping k-mer regions.
  2. Tokenize each window batch via a fully vectorized numpy pipeline — no
     Python-level string operations at all (see "Vectorized tokenizer" below).
  3. Run tokenized batches through the BERT classifier on GPU with fp16 autocast.
  4. Aggregate with DeepPL's two-threshold voting:
       - A window is "high-confidence Lysogenic" if its Lysogenic softmax
         probability exceeds `confidence_threshold` (default 0.9).
       - The genome is called Temperate if the fraction of high-confidence
         Lysogenic windows >= `lysogenic_window_fraction` (default 0.016).
  5. Return per-sequence lifestyle label, mean probabilities, and window count.

Sequences shorter than `window_bp` produce 0 windows and are predicted
Virulent by default (same behaviour as the original DeepPL for short seqs).

Model weights:
  Must be a directory in HuggingFace format containing config.json,
  pytorch_model.bin (or model.safetensors), and vocab.txt.
  The id2label mapping in config.json is used to identify which output neuron
  corresponds to Virulent/Lytic vs Temperate/Lysogenic.

─────────────────────────────────────────────────────────────────────────────
Vectorized tokenizer
─────────────────────────────────────────────────────────────────────────────
The original DeepPL implementation calls the HuggingFace BertTokenizer for
every GPU batch.  For large genomes (>50 kbp, stride=1) that tokenizer is the
dominant bottleneck: it builds k-mer strings in Python and runs per-string
parsing — about 1-2 s per batch of 4096 windows, while the GPU forward pass
takes ~50-100 ms (a 10-20× gap).

We replace the HF tokenizer with a two-stage numpy lookup built once at
startup:

  nucl_arr      uint8 array[256]: maps every ASCII byte to a base-4 nucleotide
                index (A/a=0, C/c=1, G/g=2, T/t=3, anything else=4).

  token_id_arr  int32 array[4**k]: maps base-4 encoded 6-mer index to
                DNABERT token ID.  Built from vocab.txt; defaults to unk_id
                for any k-mer not in the vocabulary (e.g. those containing N).

At inference time, for a chunk of B window start positions:

  1. Compute global genome positions for every k-mer character in the chunk:
       global_pos[b, j, m] = starts[b] + j + m  (shape B×100×6, no Python loop)
  2. Index genome_arr with global_pos to get per-character nucleotide indices.
  3. Compute base-4 k-mer index by multiplying by powers-of-4 and summing.
  4. Look up token IDs in token_id_arr (one numpy fancy-index call).
  5. Prepend [CLS] and append [SEP], wrap in torch tensors.

There are zero Python loops over windows or k-mer positions.  The resulting
token IDs are identical to what BertTokenizer would produce for the same
windows.

─────────────────────────────────────────────────────────────────────────────
Stride note
─────────────────────────────────────────────────────────────────────────────
DeepPL's original paper uses stride=1 (fully overlapping windows).  We use
stride=10 for practical throughput: it reduces the window count per genome by
10×, making the calculation ~10× faster.  The two-threshold voting logic is
identical; only the sampling density changes.  Because adjacent windows are
highly correlated (105-bp windows shifted by 10 bp share 95 bp of sequence),
the fraction of high-confidence Lysogenic windows is statistically similar
between stride=1 and stride=10.  The `confidence_threshold` and
`lysogenic_window_fraction` parameters remain at their published values (0.9
and 0.016) since they operate on the fraction of sampled windows, not the
absolute count.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer


# ── Configuration ─────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Vectorized numpy tokenizer ────────────────────────────────────────────────
#
# Two arrays are built once at startup:
#
#   nucl_arr[256]       — maps ASCII byte → base-4 nucleotide index (0–3),
#                         or 4 for any non-ACGT character (including N).
#
#   token_id_arr[4**k]  — maps base-4 6-mer index → DNABERT token ID.
#                         Defaults to unk_id for k-mers absent from the vocab.
#
# At inference time, for a batch of B window start positions we:
#   1. Compute the (B, max_seq_length, k) array of global genome positions.
#   2. Index genome_arr to get per-character nucl indices.
#   3. Multiply by powers of 4 and sum along axis 2 → (B, max_seq_length) k-mer indices.
#   4. Fancy-index token_id_arr → (B, max_seq_length) token IDs.
#   5. Overwrite positions with invalid chars (nucl_idx == 4) with unk_id.
#   6. Prepend [CLS] and append [SEP] → (B, max_seq_length+2) input_ids.
#
# No Python loops over individual windows or k-mer positions.


def _build_fast_tokenizer(
    tokenizer: BertTokenizer,
    k: int,
    unk_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build numpy lookup arrays for fully vectorized k-mer tokenization.

    Args:
        tokenizer: The loaded DNABERT BertTokenizer (provides vocab.txt mapping).
        k:         K-mer size (6 for DNABERT).
        unk_id:    Token ID to use for k-mers absent from the vocab (e.g. those
                   containing N).  Must match tokenizer.unk_token_id.

    Returns:
        nucl_arr:      uint8 ndarray of shape (256,).  Maps each ASCII byte to
                       a base-4 nucleotide index: A/a→0, C/c→1, G/g→2, T/t→3,
                       everything else→4.
        token_id_arr:  int32 ndarray of shape (4**k,).  Maps each base-4
                       encoded k-mer index to its DNABERT token ID.  Indices
                       not covered by the vocab are pre-filled with unk_id.

    The base-4 k-mer index for a string s[0..k-1] is:
        index = sum(base4(s[i]) * 4**i  for i in 0..k-1)
    i.e. s[0] is the least-significant "digit".  This ordering is used
    consistently in both this builder and _vectorized_tokenize.
    """
    # Nucleotide → base-4 index.  Handle both upper and lower case so the
    # caller does not need to upper-case the genome string first.
    nucl_arr = np.full(256, 4, dtype=np.uint8)  # default = 4 (invalid / N)
    for char, idx in [('A', 0), ('a', 0), ('C', 1), ('c', 1),
                      ('G', 2), ('g', 2), ('T', 3), ('t', 3)]:
        nucl_arr[ord(char)] = idx

    # Base-4 k-mer index → token ID.  Default to unk_id for k-mers absent
    # from the vocab (e.g. those containing N or ambiguity codes).
    n_kmers = 4 ** k
    token_id_arr = np.full(n_kmers, unk_id, dtype=np.int32)

    bases = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for token, token_id in tokenizer.vocab.items():
        if len(token) == k and all(c in bases for c in token):
            # Encode token as a base-4 integer; s[0] is least-significant.
            idx = sum(bases[c] * (4 ** i) for i, c in enumerate(token))
            token_id_arr[idx] = token_id

    return nucl_arr, token_id_arr


def _vectorized_tokenize(
    genome_arr: np.ndarray,
    starts: np.ndarray,
    k: int,
    max_seq_length: int,
    nucl_arr: np.ndarray,
    token_id_arr: np.ndarray,
    cls_id: int,
    sep_id: int,
    unk_id: int,
) -> dict[str, torch.Tensor]:
    """Tokenize a batch of windows from a genome using only numpy operations.

    Args:
        genome_arr:     uint8 ndarray of shape (genome_len,) — ASCII bytes of
                        the genome (no prior upper-casing needed; nucl_arr
                        handles both cases).
        starts:         int64 ndarray of shape (B,) — start positions of the B
                        windows to tokenize in this batch.
        k:              K-mer size (6).
        max_seq_length: Number of k-mer tokens per window (100).
        nucl_arr:       From _build_fast_tokenizer — ASCII byte → nucl index.
        token_id_arr:   From _build_fast_tokenizer — k-mer index → token ID.
        cls_id:         DNABERT [CLS] token ID (2, not 101 as in standard BERT).
        sep_id:         DNABERT [SEP] token ID (3, not 102).
        unk_id:         DNABERT [UNK] token ID (1).

    Returns:
        Dict with keys "input_ids", "attention_mask", "token_type_ids", each a
        torch.LongTensor of shape (B, max_seq_length + 2).  Identical in
        content to what BertTokenizer would produce for the same windows.

    Memory footprint at B=4096, max_seq_length=100, k=6:
        global_pos  — (B, 100, 6) int64  ≈ 20 MB
        char_idx    — (B, 100, 6) uint8  ≈  2 MB
        kmer_idx    — (B, 100)    int64  ≈  3 MB
        ids         — (B, 102)    int64  ≈  3 MB
        Total CPU   ≈ 28 MB per batch.
    """
    B = len(starts)
    L = max_seq_length + 2  # [CLS] + 100 tokens + [SEP]

    # ── Step 1: compute global genome positions for every k-mer character ──
    # global_pos[b, j, m] = starts[b] + j + m
    # - b ∈ [0, B): window index
    # - j ∈ [0, max_seq_length): k-mer position within the window
    # - m ∈ [0, k): character position within the k-mer
    j_idx = np.arange(max_seq_length, dtype=np.int64)[None, :, None]  # (1, 100, 1)
    m_idx = np.arange(k, dtype=np.int64)[None, None, :]               # (1,   1, 6)
    global_pos = starts[:, None, None] + j_idx + m_idx                # (B, 100, 6)

    # ── Step 2: look up nucleotide indices via genome_arr → nucl_arr ──────
    # genome_arr[global_pos] : (B, 100, 6) uint8 — ASCII bytes of each char
    # nucl_arr[...]           : (B, 100, 6) uint8 — base-4 index (0-3) or 4
    char_idx = nucl_arr[genome_arr[global_pos]]                        # (B, 100, 6)

    # ── Step 3: compute base-4 k-mer index ────────────────────────────────
    # Powers: [4^0, 4^1, ..., 4^(k-1)] = [1, 4, 16, 64, 256, 1024]
    # Clip invalid chars (value 4) to 0 for safe multiplication; we will
    # override those positions with unk_id after the lookup.
    powers = np.array([4 ** i for i in range(k)], dtype=np.int64)     # (6,)
    valid_char = char_idx.clip(0, 3).astype(np.int64)                 # (B, 100, 6)
    kmer_idx = (valid_char * powers[None, None, :]).sum(axis=2)       # (B, 100)

    # ── Step 4: look up token IDs and fix invalid k-mers ──────────────────
    has_invalid = (char_idx > 3).any(axis=2)                          # (B, 100) bool
    token_ids = token_id_arr[kmer_idx].astype(np.int64)               # (B, 100)
    token_ids[has_invalid] = unk_id

    # ── Step 5: assemble [CLS] + tokens + [SEP] ───────────────────────────
    ids = np.empty((B, L), dtype=np.int64)
    ids[:, 0] = cls_id
    ids[:, 1:L - 1] = token_ids
    ids[:, L - 1] = sep_id

    # All windows are the same length → no padding → attention_mask all ones.
    # Single-sequence input → token_type_ids all zeros.
    return {
        "input_ids": torch.from_numpy(ids),
        "attention_mask": torch.from_numpy(np.ones((B, L), dtype=np.int64)),
        "token_type_ids": torch.from_numpy(np.zeros((B, L), dtype=np.int64)),
    }


# ── Model state ───────────────────────────────────────────────────────────────

class _ModelState:
    def __init__(self) -> None:
        self.tokenizer: BertTokenizer | None = None
        self.model: BertForSequenceClassification | None = None
        self.device: str = "cpu"
        self.config: dict[str, Any] = {}
        # DeepPL uses P(LABEL_1) for voting: LABEL_1 = Lysogenic/Temperate.
        self.lysogenic_label_id: int = 1
        # Vectorized tokenizer arrays — built at startup from vocab.txt.
        self.nucl_arr: np.ndarray | None = None       # shape (256,) uint8
        self.token_id_arr: np.ndarray | None = None   # shape (4**k,) int32
        # Special token IDs — populated from the tokenizer at startup.
        # DNABERT uses CLS=2, SEP=3, UNK=1 (not the standard BERT 101/102/100).
        self.cls_id: int = 0
        self.sep_id: int = 0
        self.unk_id: int = 0


_state = _ModelState()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    _state.config = cfg

    model_path = cfg["model"]["path"]
    k: int = cfg["model"]["kmer"]

    logger.info(f"Loading tokenizer from {model_path}")
    _state.tokenizer = BertTokenizer.from_pretrained(model_path)

    # Read special token IDs directly from the tokenizer — do not hardcode.
    # DNABERT uses CLS=2, SEP=3, UNK=1 (different from standard BERT 101/102/100).
    _state.cls_id = _state.tokenizer.cls_token_id
    _state.sep_id = _state.tokenizer.sep_token_id
    _state.unk_id = _state.tokenizer.unk_token_id
    logger.info(
        f"Special token IDs — CLS={_state.cls_id}, SEP={_state.sep_id}, "
        f"UNK={_state.unk_id}"
    )

    # Build the vectorized tokenizer arrays from the loaded vocabulary.
    _state.nucl_arr, _state.token_id_arr = _build_fast_tokenizer(
        _state.tokenizer, k, _state.unk_id
    )
    n_vocab_kmers = int((_state.token_id_arr != _state.unk_id).sum())
    logger.info(
        f"Vectorized tokenizer built: nucl_arr(256,) + token_id_arr({4**k},) | "
        f"{n_vocab_kmers} k-mers from vocab | stride={cfg['model']['stride']}"
    )

    logger.info(f"Loading model from {model_path}")
    _state.model = BertForSequenceClassification.from_pretrained(model_path)
    _state.model.eval()

    _state.device = "cuda" if torch.cuda.is_available() else "cpu"
    if _state.device == "cpu":
        logger.warning("CUDA not available — running on CPU (slow for large genomes)")
    _state.model = _state.model.to(_state.device)

    # Determine which output neuron = Lysogenic/Temperate.
    # DeepPL's run_finetune.py saves softmax(logits)[:,1] as pred_results.npy,
    # and the predict script calls Lysogenic when the fraction of windows with
    # P(LABEL_1) > threshold is >= 0.016.  LABEL_1 = Lysogenic/Temperate.
    id2label: dict[int, str] = _state.model.config.id2label
    lysogenic_id = 1
    for label_id, label_str in id2label.items():
        if label_str.lower() in ("lysogenic", "temperate"):
            lysogenic_id = int(label_id)
            break
    _state.lysogenic_label_id = lysogenic_id
    logger.info(
        f"Model loaded | device={_state.device} | "
        f"lysogenic_label_id={lysogenic_id} | id2label={id2label}"
    )

    yield

    _state.model = None
    _state.tokenizer = None
    _state.nucl_arr = None
    _state.token_id_arr = None
    logger.info("Model unloaded")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="deeppl-service", version="0.2.0", lifespan=lifespan)


# ── Request / response models ─────────────────────────────────────────────────

class BatchPredictRequest(BaseModel):
    sequences: list[str]
    sequence_ids: list[str] | None = None


class PredictionResult(BaseModel):
    sequence_id: str
    predicted_lifestyle: Literal["Virulent", "Temperate"]
    virulent_probability: float
    temperate_probability: float
    windows_evaluated: int


class BatchPredictResponse(BaseModel):
    results: list[PredictionResult]


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_batch(
    genome_arr: np.ndarray,
    starts: np.ndarray,
    max_batch_size: int,
) -> np.ndarray:
    """Tokenize and run the model on a set of windows from one genome.

    Args:
        genome_arr:     uint8 ndarray of shape (genome_len,) — raw ASCII bytes
                        of the (already-encoded) genome sequence.
        starts:         int64 ndarray of shape (N,) — window start positions.
        max_batch_size: Maximum number of windows per GPU forward pass.

    Returns:
        Softmax probability array of shape (N, 2).

    Automatically halves the GPU batch size on CUDA OOM and retries from the
    failed position.  After a successful pass the batch size is gradually
    restored toward max_batch_size.

    Replaces the previous string-based pipeline:
        OLD: window_strings → _fast_tokenize (Python loop) → GPU
        NEW: genome_arr + starts → _vectorized_tokenize (pure numpy) → GPU
    """
    if len(starts) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    model = _state.model
    device = _state.device
    nucl_arr = _state.nucl_arr
    token_id_arr = _state.token_id_arr
    k = _state.config["model"]["kmer"]
    max_seq_length = _state.config["model"]["max_seq_length"]
    cls_id = _state.cls_id
    sep_id = _state.sep_id
    unk_id = _state.unk_id

    all_probs: list[np.ndarray] = []
    i = 0
    current_batch_size = min(max_batch_size, len(starts))

    while i < len(starts):
        batch_starts = starts[i: i + current_batch_size]
        try:
            # Fully vectorized tokenization — no Python loops over windows.
            inputs = _vectorized_tokenize(
                genome_arr, batch_starts,
                k, max_seq_length,
                nucl_arr, token_id_arr,
                cls_id, sep_id, unk_id,
            )
            inputs = {key: v.to(device) for key, v in inputs.items()}

            # fp16 autocast: uses fp16 matmuls with fp32 accumulation.
            # Produces the same classification result as fp32 within rounding
            # tolerance; logit differences are negligible for binary classification.
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device, dtype=torch.float16, enabled=(device == "cuda")
                ):
                    logits = model(**inputs).logits  # (B, 2)

            # Softmax in fp32 for numerical stability.
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            all_probs.append(probs)
            i += len(batch_starts)
            # Gradually restore original batch size after a successful call.
            current_batch_size = min(current_batch_size * 2, max_batch_size)

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and current_batch_size > 1:
                if device == "cuda":
                    torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                logger.warning(f"CUDA OOM — reduced batch size to {current_batch_size}")
            else:
                raise

    return np.concatenate(all_probs, axis=0).astype(np.float32)


def _predict_sequence(sequence: str) -> tuple[str, float, float, int]:
    """Predict lifestyle for a single sequence using DeepPL's sliding-window algorithm.

    Steps:
      1. Encode the genome as a uint8 numpy array (ASCII bytes) — done once.
      2. Compute all window start positions as an int64 numpy array.
      3. Process start positions in GPU-sized chunks via _run_batch.
      4. Apply DeepPL's two-threshold voting to the aggregated probabilities.

    Args:
        sequence: Raw DNA sequence string (any case; non-ACGT chars treated as N).

    Returns:
        (predicted_lifestyle, virulent_probability, temperate_probability,
         windows_evaluated)
    """
    cfg = _state.config["model"]
    window_bp: int = cfg["window_bp"]
    stride: int = cfg["stride"]
    max_batch_size: int = cfg["max_batch_size"]
    conf_threshold: float = cfg["confidence_threshold"]
    lysogenic_fraction_threshold: float = cfg["lysogenic_window_fraction"]
    lysogenic_id: int = _state.lysogenic_label_id

    # Encode genome as uint8 ASCII bytes once.  nucl_arr handles both upper
    # and lower case; non-ASCII characters are replaced with '?' (ASCII 63),
    # which nucl_arr maps to 4 (invalid), causing those k-mers to get unk_id.
    genome_arr = np.frombuffer(
        sequence.encode("ascii", errors="replace"), dtype=np.uint8
    )
    n = len(genome_arr)

    # Number of windows with the given stride.
    n_windows = max(0, (n - window_bp) // stride + 1) if n >= window_bp else 0
    if n_windows == 0:
        logger.debug(
            f"Sequence length {n} < window_bp {window_bp}; "
            "returning default Virulent prediction (no windows to evaluate)"
        )
        return "Virulent", 1.0, 0.0, 0

    # All window start positions in a single numpy array — no generator needed.
    starts = np.arange(n_windows, dtype=np.int64) * stride

    # Run model on all windows and collect (n_windows, 2) probability array.
    probs = _run_batch(genome_arr, starts, max_batch_size)  # (n_windows, 2)

    # DeepPL two-threshold voting:
    #   P(LABEL_1 = Lysogenic) per window → fraction above conf_threshold.
    lyso_probs: np.ndarray = probs[:, lysogenic_id]
    high_conf_lysogenic = int((lyso_probs > conf_threshold).sum())
    lysogenic_fraction = high_conf_lysogenic / n_windows
    predicted: Literal["Virulent", "Temperate"] = (
        "Temperate" if lysogenic_fraction >= lysogenic_fraction_threshold else "Virulent"
    )
    mean_lyso_prob = float(lyso_probs.mean())

    return predicted, float(1.0 - mean_lyso_prob), float(mean_lyso_prob), n_windows


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict[str, Any]:
    cfg = _state.config
    model_cfg = cfg.get("model", {})
    return {
        "service": "deeppl-service",
        "version": "0.2.0",
        "model_path": model_cfg.get("path", "unknown"),
        "port": cfg.get("server", {}).get("port", 8004),
        "algorithm": "DNABERT sliding-window with two-threshold voting",
        "stride": model_cfg.get("stride", "unknown"),
        "tokenizer": "vectorized_numpy",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _state.model is not None,
        "device": _state.device,
        "vectorized_tokenizer": _state.nucl_arr is not None,
        "stride": _state.config.get("model", {}).get("stride", "unknown"),
    }


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    if _state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sequences = request.sequences
    ids = request.sequence_ids

    if ids is None:
        ids = [f"seq_{i}" for i in range(len(sequences))]
    elif len(ids) != len(sequences):
        raise HTTPException(
            status_code=422,
            detail=(
                f"sequence_ids length ({len(ids)}) must match "
                f"sequences length ({len(sequences)})"
            ),
        )

    results: list[PredictionResult] = []
    for seq_id, seq in zip(ids, sequences):
        lifestyle, vir_prob, temp_prob, n_windows = _predict_sequence(seq)
        results.append(
            PredictionResult(
                sequence_id=seq_id,
                predicted_lifestyle=lifestyle,
                virulent_probability=round(vir_prob, 6),
                temperate_probability=round(temp_prob, 6),
                windows_evaluated=n_windows,
            )
        )

    return BatchPredictResponse(results=results)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    server = cfg.get("server", {})
    uvicorn.run(
        "service:app",
        host=server.get("host", "0.0.0.0"),
        port=server.get("port", 8004),
        log_level="info",
    )
