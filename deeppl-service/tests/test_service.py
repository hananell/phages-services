"""Tests for deeppl-service using the real DeepPL/DNABERT model.

Requires the downloaded model weights at:
  /home/user/python_projects/phages_services/deeppl-service/model/deeppl_ckpt-340000/

The tests load the real BertForSequenceClassification checkpoint and run actual
inference on GPU (falls back to CPU if CUDA is unavailable).

Runtime benchmarks
------------------
test_runtime_by_sequence_length is parametrized over lengths up to 200_000 bp.
Each test measures real end-to-end latency including tokenisation, model forward
passes, and HTTP serialisation.  Run with -s to see the timing table.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
import service  # noqa: E402

# ── Model path ────────────────────────────────────────────────────────────────

_MODEL_DIR = (
    Path(__file__).parent.parent / "model" / "deeppl_ckpt-340000"
)

_CFG = {
    "model": {
        "path": str(_MODEL_DIR),
        "kmer": 6,
        "window_bp": 105,
        "max_seq_length": 100,
        "stride": 1,
        "max_batch_size": 2048,
        "confidence_threshold": 0.9,
        "lysogenic_window_fraction": 0.016,
    },
    "server": {"host": "0.0.0.0", "port": 8004},
}

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Load real model once for the entire test module (slow startup)."""
    import unittest.mock as mock

    with mock.patch.object(service, "load_config", return_value=_CFG):
        with TestClient(service.app) as c:
            yield c


# ── Utility ───────────────────────────────────────────────────────────────────

def _seq(length: int) -> str:
    """Deterministic phage-like DNA sequence."""
    bases = "ATCGATCGATCGATCG"
    return (bases * (length // len(bases) + 1))[:length]


# ── Smoke tests ───────────────────────────────────────────────────────────────

def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    print(f"\n  device: {body['device']}")


def test_root(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "deeppl-service"


# ── Sequence-length edge cases ────────────────────────────────────────────────

def test_sequence_too_short(client: TestClient) -> None:
    """Sequence < 105 bp: 0 windows, default Virulent."""
    r = client.post("/predict/batch", json={"sequences": [_seq(50)]})
    assert r.status_code == 200
    result = r.json()["results"][0]
    assert result["windows_evaluated"] == 0
    assert result["predicted_lifestyle"] == "Virulent"


def test_exactly_one_window(client: TestClient) -> None:
    """105 bp sequence produces exactly 1 window."""
    r = client.post("/predict/batch", json={"sequences": [_seq(105)]})
    assert r.status_code == 200
    assert r.json()["results"][0]["windows_evaluated"] == 1


def test_window_count(client: TestClient) -> None:
    r = client.post("/predict/batch", json={"sequences": [_seq(500)]})
    assert r.status_code == 200
    assert r.json()["results"][0]["windows_evaluated"] == 500 - 105 + 1


# ── Response shape ────────────────────────────────────────────────────────────

def test_probabilities_sum_to_one(client: TestClient) -> None:
    r = client.post("/predict/batch", json={"sequences": [_seq(300)]})
    result = r.json()["results"][0]
    total = result["virulent_probability"] + result["temperate_probability"]
    assert abs(total - 1.0) < 1e-4


def test_lifestyle_is_valid(client: TestClient) -> None:
    r = client.post("/predict/batch", json={"sequences": [_seq(300)]})
    result = r.json()["results"][0]
    assert result["predicted_lifestyle"] in ("Virulent", "Temperate")


def test_batch_multiple_sequences(client: TestClient) -> None:
    seqs = [_seq(200), _seq(300), _seq(500)]
    r = client.post(
        "/predict/batch",
        json={"sequences": seqs, "sequence_ids": ["a", "b", "c"]},
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    assert [res["sequence_id"] for res in results] == ["a", "b", "c"]


def test_auto_sequence_ids(client: TestClient) -> None:
    r = client.post("/predict/batch", json={"sequences": [_seq(200), _seq(300)]})
    assert r.status_code == 200
    ids = [res["sequence_id"] for res in r.json()["results"]]
    assert ids == ["seq_0", "seq_1"]


def test_mismatched_ids_returns_422(client: TestClient) -> None:
    r = client.post(
        "/predict/batch",
        json={"sequences": [_seq(200), _seq(300)], "sequence_ids": ["only_one"]},
    )
    assert r.status_code == 422


def test_iupac_normalization(client: TestClient) -> None:
    """Non-ACGT IUPAC codes are silently normalised to N."""
    seq = "ATCGRYSWKMBDHVN" * 20
    r = client.post("/predict/batch", json={"sequences": [seq]})
    assert r.status_code == 200
    assert r.json()["results"][0]["predicted_lifestyle"] in ("Virulent", "Temperate")


# ── Runtime benchmarks ────────────────────────────────────────────────────────

_BENCHMARK_LENGTHS = [105, 1_000, 5_000, 10_000, 50_000, 100_000, 200_000]


@pytest.mark.parametrize("seq_length", _BENCHMARK_LENGTHS)
def test_runtime_by_sequence_length(client: TestClient, seq_length: int) -> None:
    """Measure real end-to-end prediction latency including GPU inference."""
    sequence = _seq(seq_length)
    expected_windows = max(0, seq_length - 105 + 1)

    t0 = time.perf_counter()
    r = client.post(
        "/predict/batch",
        json={"sequences": [sequence], "sequence_ids": [f"phage_{seq_length}bp"]},
    )
    elapsed = time.perf_counter() - t0

    assert r.status_code == 200, r.text
    result = r.json()["results"][0]

    assert result["sequence_id"] == f"phage_{seq_length}bp"
    assert result["windows_evaluated"] == expected_windows
    assert result["predicted_lifestyle"] in ("Virulent", "Temperate")
    assert 0.0 <= result["virulent_probability"] <= 1.0
    assert 0.0 <= result["temperate_probability"] <= 1.0

    print(
        f"\n  {seq_length:>7} bp | {expected_windows:>7} windows | "
        f"{elapsed:7.3f}s | {result['predicted_lifestyle']}"
    )
