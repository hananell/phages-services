"""Tests for the evo2 embedding service.

The real Evo2 model is always mocked — these tests cover HTTP contract,
batch splitting, mean-pooling, padding logic, and error handling.
"""
from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

import service
from service import Evo2EmbeddingModel, validate_sequence

EMBED_DIM = 8
LAYER_NAMES = ["blocks.28.mlp.l3", "blocks.31"]


def _make_fake_embeddings(sequences, layer_names, embed_dim=EMBED_DIM):
    """Return fake embeddings: use sequence length as the value (deterministic)."""
    results = []
    for seq in sequences:
        entry = {}
        for layer in layer_names:
            entry[layer] = [float(len(seq))] * embed_dim
        results.append(entry)
    return results


@pytest.fixture
def mock_model(monkeypatch):
    """Replace the global embedding_model with a fake that doesn't need GPU."""
    fake = Evo2EmbeddingModel.__new__(Evo2EmbeddingModel)
    fake._model = object()  # truthy — marks as "loaded"
    fake.device = "cpu"

    def fake_get_embeddings_batch(sequences, layer_names):
        return _make_fake_embeddings(sequences, layer_names)

    fake.get_embeddings_batch = fake_get_embeddings_batch
    monkeypatch.setattr(service, "embedding_model", fake)
    return fake


@pytest.fixture
def client(mock_model):
    with TestClient(service.app) as c:
        yield c


# --- health ---

def test_health_returns_healthy(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_health_reports_model_loaded(client):
    assert client.get("/health").json()["model_loaded"] is True


# --- happy path ---

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


def test_long_sequence_is_tiled_and_mean_pooled(client, monkeypatch):
    """Via HTTP: a sequence longer than tile_size is tiled; result is one embedding."""
    # Patch tile_size to 4 and batch_size large enough to process all tiles in one pass
    monkeypatch.setattr(service.config.embedding, "tile_size", 4)
    monkeypatch.setattr(service.config.embedding, "batch_size", 8)

    # "ATCGATCG" = 8 bp → 2 tiles of 4 bp each → both embeddings = [4.0]*EMBED_DIM → mean = [4.0]*EMBED_DIM
    r = client.post("/embed/batch", json={"sequences": ["ATCGATCG"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["sequence_length"] == 8
    emb = body["results"][0]["embeddings"][LAYER_NAMES[0]]
    assert len(emb) == EMBED_DIM
    assert emb == pytest.approx([4.0] * EMBED_DIM)


# --- error cases ---

def test_empty_sequence_returns_400(client):
    r = client.post("/embed/batch", json={"sequences": [""]})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_invalid_characters_returns_400(client):
    r = client.post("/embed/batch", json={"sequences": ["ATCGXYZ"]})
    assert r.status_code == 400
    assert "Invalid" in r.json()["detail"]


# --- mean-pool padding logic (unit test, no HTTP) ---

def test_mean_pool_masks_padding():
    """Verify that only real-length positions contribute to mean-pool."""
    batch_size = 2
    max_len = 4
    embed_dim = 2

    # seq A: all 1s (real); seq B: first 2 are 2s, last 2 are 0s (padding)
    raw = torch.zeros(batch_size, max_len, embed_dim)
    raw[0] = 1.0
    raw[1, :2] = 2.0

    lengths = [4, 2]
    for i, real_len in enumerate(lengths):
        seq_emb = raw[i, :real_len, :]
        mean_emb = seq_emb.float().mean(dim=0)
        if i == 0:
            assert torch.allclose(mean_emb, torch.ones(embed_dim))
        else:
            assert torch.allclose(mean_emb, torch.full((embed_dim,), 2.0))


# --- _embed_sequences_batched (unit tests, no HTTP) ---

def test_embed_sequences_batched_single_sequence_single_tile(monkeypatch):
    """Single short sequence → exactly one forward pass with that one tile."""
    calls = []

    def fake_get(seqs, layers):
        calls.append(list(seqs))
        return _make_fake_embeddings(seqs, layers)

    monkeypatch.setattr(service.embedding_model, "get_embeddings_batch", fake_get)

    results = service._embed_sequences_batched(
        sequences=["ATCG"],
        layer_names=LAYER_NAMES,
        tile_size=16,
        batch_size=8,
    )
    assert len(calls) == 1
    assert calls[0] == ["ATCG"]
    assert len(results) == 1
    assert set(results[0].keys()) == set(LAYER_NAMES)
    assert len(results[0][LAYER_NAMES[0]]) == EMBED_DIM


def test_embed_sequences_batched_multiple_short_sequences_one_pass(monkeypatch):
    """Multiple short sequences all fit in one chunk → exactly one forward pass."""
    calls = []

    def fake_get(seqs, layers):
        calls.append(list(seqs))
        return _make_fake_embeddings(seqs, layers)

    monkeypatch.setattr(service.embedding_model, "get_embeddings_batch", fake_get)

    seqs = ["ATCG", "GCTA", "TTTT"]
    results = service._embed_sequences_batched(
        sequences=seqs,
        layer_names=LAYER_NAMES,
        tile_size=16,
        batch_size=8,  # batch_size >= n_tiles (3), so one pass
    )
    assert len(calls) == 1
    assert len(results) == 3


def test_embed_sequences_batched_result_order_preserved(monkeypatch):
    """Result list matches input sequence order."""
    def fake_get(seqs, layers):
        return _make_fake_embeddings(seqs, layers)

    monkeypatch.setattr(service.embedding_model, "get_embeddings_batch", fake_get)

    seqs = ["ATCG", "GCTAGCTA", "TT"]  # lengths 4, 8, 2 → fake values 4.0, 8.0, 2.0
    results = service._embed_sequences_batched(
        sequences=seqs,
        layer_names=LAYER_NAMES,
        tile_size=16,
        batch_size=8,
    )
    assert results[0][LAYER_NAMES[0]] == pytest.approx([4.0] * EMBED_DIM)
    assert results[1][LAYER_NAMES[0]] == pytest.approx([8.0] * EMBED_DIM)
    assert results[2][LAYER_NAMES[0]] == pytest.approx([2.0] * EMBED_DIM)


def test_embed_sequences_batched_small_batch_size_multiple_chunks(monkeypatch):
    """batch_size < n_tiles → multiple forward pass chunks."""
    calls = []

    def fake_get(seqs, layers):
        calls.append(list(seqs))
        return _make_fake_embeddings(seqs, layers)

    monkeypatch.setattr(service.embedding_model, "get_embeddings_batch", fake_get)

    # 5 sequences × 1 tile each = 5 tiles; batch_size=2 → 3 chunks (2+2+1)
    seqs = ["ATCG", "GCTA", "TTTT", "AAAA", "CCCC"]
    results = service._embed_sequences_batched(
        sequences=seqs,
        layer_names=LAYER_NAMES,
        tile_size=16,
        batch_size=2,
    )
    assert len(calls) == 3
    assert len(calls[0]) == 2
    assert len(calls[1]) == 2
    assert len(calls[2]) == 1
    assert len(results) == 5


def test_embed_sequences_batched_long_sequence_mean_pooled(monkeypatch):
    """Long sequence is tiled and tile embeddings are averaged."""
    def fake_get(seqs, layers):
        return _make_fake_embeddings(seqs, layers)

    monkeypatch.setattr(service.embedding_model, "get_embeddings_batch", fake_get)

    # "ATCGATCGATCG" = 12 bp → 3 tiles of 4 bp each → fake [4.0]*dim each → mean [4.0]*dim
    results = service._embed_sequences_batched(
        sequences=["ATCGATCGATCG"],
        layer_names=LAYER_NAMES,
        tile_size=4,
        batch_size=8,
    )
    assert len(results) == 1
    assert results[0][LAYER_NAMES[0]] == pytest.approx([4.0] * EMBED_DIM)


def test_embed_sequences_batched_unequal_last_tile(monkeypatch):
    """Last tile is shorter than tile_size — mean is still element-wise correct."""
    def fake_get(seqs, layers):
        return _make_fake_embeddings(seqs, layers)

    monkeypatch.setattr(service.embedding_model, "get_embeddings_batch", fake_get)

    # "ATCGAT" = 6 bp → tiles ["ATCG"(4), "AT"(2)] → fake [4.0]*dim, [2.0]*dim → mean [3.0]*dim
    results = service._embed_sequences_batched(
        sequences=["ATCGAT"],
        layer_names=LAYER_NAMES,
        tile_size=4,
        batch_size=8,
    )
    assert results[0][LAYER_NAMES[0]] == pytest.approx([3.0] * EMBED_DIM)


def test_embed_sequences_batched_mixed_long_and_short(monkeypatch):
    """Mixed long/short sequences: tiles interleaved in chunks, correctly reassembled."""
    calls = []

    def fake_get(seqs, layers):
        calls.append(list(seqs))
        return _make_fake_embeddings(seqs, layers)

    monkeypatch.setattr(service.embedding_model, "get_embeddings_batch", fake_get)

    # seq0: "ATCGATCG" (8 bp) → 2 tiles of 4 bp each → fake [4.0]*dim → mean [4.0]*dim
    # seq1: "GG" (2 bp) → 1 tile → fake [2.0]*dim
    # Total: 3 tiles; batch_size=2 → 2 chunks: [tile0_seq0, tile1_seq0], [tile0_seq1]
    results = service._embed_sequences_batched(
        sequences=["ATCGATCG", "GG"],
        layer_names=LAYER_NAMES,
        tile_size=4,
        batch_size=2,
    )
    assert len(calls) == 2
    assert len(results) == 2
    # seq0: mean of two [4.0]*dim tiles = [4.0]*dim
    assert results[0][LAYER_NAMES[0]] == pytest.approx([4.0] * EMBED_DIM)
    # seq1: single [2.0]*dim tile
    assert results[1][LAYER_NAMES[0]] == pytest.approx([2.0] * EMBED_DIM)
