"""Tests for the evo2 embedding service.

The real Evo2 model is always mocked — these tests cover:
- HTTP contract (valid/invalid inputs, response shape)
- Single-sequence full-pass embedding extraction
- Chunked (tiled) fallback embedding extraction and length-weighted mean-pooling
- CUDA OOM → chunked fallback path
- Base64 encoding / decoding of returned embeddings
"""
from __future__ import annotations

import base64
import struct

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import service
from service import Evo2EmbeddingModel, validate_sequence

EMBED_DIM = 8
LAYER_NAMES = ["blocks.28.mlp.l3", "blocks.31"]


def _make_fake_embedding(sequence: str, layer_names: list[str], embed_dim: int = EMBED_DIM) -> dict:
    """Fake embedding: value = sequence length (deterministic, easy to assert)."""
    return {layer: np.full(embed_dim, float(len(sequence)), dtype=np.float32) for layer in layer_names}


@pytest.fixture
def mock_model(monkeypatch):
    """Replace global embedding_model with a fake that doesn't need GPU."""
    fake = Evo2EmbeddingModel.__new__(Evo2EmbeddingModel)
    fake._model = object()  # truthy — marks as "loaded"
    fake.device = "cpu"

    def fake_get_embedding_single(sequence, layer_names, fallback_chunk_size=65_536):
        return _make_fake_embedding(sequence, layer_names)

    fake.get_embedding_single = fake_get_embedding_single
    monkeypatch.setattr(service, "embedding_model", fake)
    return fake


@pytest.fixture
def client(mock_model):
    with TestClient(service.app) as c:
        yield c


# ── health ───────────────────────────────────────────────────────────────────

def test_health_returns_healthy(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_health_reports_model_loaded(client):
    assert client.get("/health").json()["model_loaded"] is True


# ── happy path ────────────────────────────────────────────────────────────────

def test_single_sequence_returns_one_result(client):
    r = client.post("/embed/batch", json={"sequences": ["ATCG"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["sequence_length"] == 4


def test_batch_of_three_returns_three_results(client):
    seqs = ["ATCG", "GCTA", "TTTT"]
    r = client.post("/embed/batch", json={"sequences": seqs})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3


def test_default_layers_from_config(client):
    r = client.post("/embed/batch", json={"sequences": ["ATCG"]})
    embs = r.json()["results"][0]["embeddings"]
    assert set(embs.keys()) == set(LAYER_NAMES)


def test_custom_layer_names_override(client):
    r = client.post(
        "/embed/batch",
        json={"sequences": ["ATCG"], "layer_names": ["blocks.28.mlp.l3"]},
    )
    embs = r.json()["results"][0]["embeddings"]
    assert list(embs.keys()) == ["blocks.28.mlp.l3"]


def test_embedding_dimensions_in_response(client):
    r = client.post("/embed/batch", json={"sequences": ["ATCG"]})
    dims = r.json()["embedding_dimensions"]
    assert all(d == EMBED_DIM for d in dims.values())


def test_lowercase_sequence_is_uppercased(client):
    r = client.post("/embed/batch", json={"sequences": ["atcg"]})
    assert r.status_code == 200
    assert r.json()["results"][0]["sequence_length"] == 4


def test_result_order_matches_input_order(client):
    """Results must be returned in the same order as input sequences."""
    seqs = ["ATCG", "GCTAGCTA", "TT"]
    r = client.post("/embed/batch", json={"sequences": seqs})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["sequence_length"] == 4
    assert results[1]["sequence_length"] == 8
    assert results[2]["sequence_length"] == 2


def test_base64_embedding_decodes_correctly(client):
    """Embeddings are base64-encoded little-endian float32 — must round-trip."""
    r = client.post("/embed/batch", json={"sequences": ["ATCG"]})
    b64_str = r.json()["results"][0]["embeddings"][LAYER_NAMES[0]]
    raw_bytes = base64.b64decode(b64_str)
    values = list(struct.unpack(f"<{len(raw_bytes)//4}f", raw_bytes))
    # The fake model returns float(len("ATCG")) = 4.0 for every element
    assert values == pytest.approx([4.0] * EMBED_DIM)


# ── error cases ───────────────────────────────────────────────────────────────

def test_empty_sequence_returns_400(client):
    r = client.post("/embed/batch", json={"sequences": [""]})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_invalid_characters_returns_400(client):
    r = client.post("/embed/batch", json={"sequences": ["ATCGXYZ"]})
    assert r.status_code == 400
    assert "Invalid" in r.json()["detail"]


def test_sequence_too_long_returns_400(client, monkeypatch):
    monkeypatch.setattr(service.config.model, "max_sequence_length", 5)
    r = client.post("/embed/batch", json={"sequences": ["ATCGATCG"]})
    assert r.status_code == 400
    assert "too long" in r.json()["detail"].lower()


# ── validate_sequence (unit) ───────────────────────────────────────────────────

def test_validate_sequence_empty():
    assert validate_sequence("") is not None


def test_validate_sequence_invalid_chars():
    assert validate_sequence("ATCGX") is not None


def test_validate_sequence_valid():
    assert validate_sequence("ATCGN") is None


def test_validate_sequence_lowercase_accepted():
    # validate_sequence uppercases internally, so lowercase nucleotides are accepted
    assert validate_sequence("atcg") is None


# ── _embed_tokens_full (unit, no HTTP, no GPU) ────────────────────────────────

class _FakeEvo2:
    """Minimal stand-in for Evo2 that returns predictable embeddings."""

    class _Tokenizer:
        def tokenize(self, seq: str):
            return list(range(len(seq)))  # one token per character

    tokenizer = _Tokenizer()

    def __call__(self, input_ids, return_embeddings=False, layer_names=None):
        B, L = input_ids.shape
        embeddings = {
            layer: torch.full((B, L, EMBED_DIM), float(layer_names.index(layer) + 1))
            for layer in layer_names
        }
        return None, embeddings


def _make_real_model(monkeypatch, model_override=None):
    """Build a real Evo2EmbeddingModel whose ._model is replaced by a fake."""
    m = Evo2EmbeddingModel.__new__(Evo2EmbeddingModel)
    m._model = model_override or _FakeEvo2()
    m.device = "cpu"
    return m


def test_embed_tokens_full_shape():
    m = _make_real_model(None)
    result = m._embed_tokens_full([0, 1, 2, 3], LAYER_NAMES)
    for layer in LAYER_NAMES:
        assert layer in result
        assert result[layer].shape == (EMBED_DIM,)


def test_embed_tokens_full_mean_pooled_correctly():
    """Mean pooling over 4 identical tokens gives the same value."""
    m = _make_real_model(None)
    result = m._embed_tokens_full([0, 1, 2, 3], LAYER_NAMES)
    # _FakeEvo2 fills layer index 0 with 1.0, index 1 with 2.0
    assert result[LAYER_NAMES[0]] == pytest.approx([1.0] * EMBED_DIM)
    assert result[LAYER_NAMES[1]] == pytest.approx([2.0] * EMBED_DIM)


# ── _embed_tokens_chunked (unit, no HTTP, no GPU) ─────────────────────────────

def test_embed_tokens_chunked_single_chunk():
    """If all tokens fit in one chunk, result matches full-pass result."""
    m = _make_real_model(None)
    result = m._embed_tokens_chunked([0, 1, 2, 3], LAYER_NAMES, chunk_size=16)
    assert result[LAYER_NAMES[0]] == pytest.approx([1.0] * EMBED_DIM)


def test_embed_tokens_chunked_two_equal_chunks():
    """4 tokens split into two 2-token chunks — mean of equal chunks = same value."""
    m = _make_real_model(None)
    # 4 tokens, chunk_size=2 → chunk1=[0,1], chunk2=[2,3]; both give 1.0
    result = m._embed_tokens_chunked([0, 1, 2, 3], LAYER_NAMES, chunk_size=2)
    assert result[LAYER_NAMES[0]] == pytest.approx([1.0] * EMBED_DIM)


def test_embed_tokens_chunked_unequal_last_chunk_weighted_correctly():
    """Length-weighted mean: 4-token chunk + 2-token chunk → weight 4:2."""

    class _VaryingEvo2:
        """Returns distinct values per call so we can verify weighted averaging."""
        call_count = 0

        class _Tokenizer:
            def tokenize(self, seq):
                return list(range(len(seq)))

        tokenizer = _Tokenizer()

        def __call__(self, input_ids, return_embeddings=False, layer_names=None):
            B, L = input_ids.shape
            # First call: value 6.0; second call: value 3.0
            value = 6.0 if self.call_count == 0 else 3.0
            self.call_count += 1
            embeddings = {
                layer: torch.full((B, L, EMBED_DIM), value)
                for layer in layer_names
            }
            return None, embeddings

    m = _make_real_model(None, _VaryingEvo2())
    # 6 tokens, chunk_size=4 → chunk1=4 tokens (value=6.0), chunk2=2 tokens (value=3.0)
    # weighted mean = (4*6.0 + 2*3.0) / 6 = (24 + 6) / 6 = 5.0
    result = m._embed_tokens_chunked(list(range(6)), [LAYER_NAMES[0]], chunk_size=4)
    assert result[LAYER_NAMES[0]] == pytest.approx([5.0] * EMBED_DIM)


# ── OOM fallback (unit) ───────────────────────────────────────────────────────

def test_get_embedding_single_falls_back_on_oom(monkeypatch):
    """When _embed_tokens_full raises OOM, get_embedding_single retries chunked."""
    m = _make_real_model(None)
    chunked_called = {}

    def raise_oom(tokens, layer_names):
        raise torch.cuda.OutOfMemoryError("mock OOM")

    def fake_chunked(tokens, layer_names, chunk_size):
        chunked_called["chunk_size"] = chunk_size
        return {layer: np.ones(EMBED_DIM, dtype=np.float32) for layer in layer_names}

    monkeypatch.setattr(m, "_embed_tokens_full", raise_oom)
    monkeypatch.setattr(m, "_embed_tokens_chunked", fake_chunked)

    result = m.get_embedding_single("ATCG", LAYER_NAMES, fallback_chunk_size=32)
    assert chunked_called["chunk_size"] == 32
    assert result[LAYER_NAMES[0]] == pytest.approx([1.0] * EMBED_DIM)


def test_get_embedding_single_no_fallback_when_full_succeeds(monkeypatch):
    """When _embed_tokens_full succeeds, _embed_tokens_chunked is never called."""
    m = _make_real_model(None)
    chunked_called = []

    def fake_chunked(tokens, layer_names, chunk_size):
        chunked_called.append(True)

    monkeypatch.setattr(m, "_embed_tokens_chunked", fake_chunked)

    result = m.get_embedding_single("ATCG", LAYER_NAMES)
    assert len(chunked_called) == 0
    assert set(result.keys()) == set(LAYER_NAMES)
