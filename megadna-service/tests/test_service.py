"""Integration tests for megadna-service.

These tests use a mock model to avoid requiring GPU/model weights.
They verify endpoint routing, request validation, and response shapes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_model():
    """Patch the embedding model to avoid loading real weights."""
    mock = MagicMock()
    # get_embedding returns a 512-dim vector
    mock.get_embedding.return_value = np.random.randn(512).astype(np.float32)
    # get_embeddings_batch returns (N, 512) array
    mock.get_embeddings_batch.side_effect = lambda seqs, **kw: np.random.randn(
        len(seqs), 512
    ).astype(np.float32)

    with patch("service.embedding_model", mock), patch("service.config") as mock_cfg:
        mock_cfg.model.max_sequence_length = 96000
        mock_cfg.embedding.layer_index = 0
        mock_cfg.server.host = "0.0.0.0"
        mock_cfg.server.port = 8000
        from service import app

        yield TestClient(app)


class TestHealth:
    def test_health(self, mock_model):
        client = mock_model
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestEmbedSingle:
    def test_embed_valid(self, mock_model):
        client = mock_model
        resp = client.post("/embed", json={"sequence": "ATCGATCG", "layer_index": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["embedding"]) == 512
        assert data["layer_index"] == 0

    def test_embed_empty_sequence(self, mock_model):
        client = mock_model
        resp = client.post("/embed", json={"sequence": "", "layer_index": 0})
        assert resp.status_code == 400


class TestEmbedBatch:
    def test_batch_valid(self, mock_model):
        client = mock_model
        resp = client.post(
            "/embed_batch",
            json={"sequences": ["ATCG", "GCTA", "AAAA"], "layer_index": 0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["embeddings"]) == 3
        assert all(len(e) == 512 for e in data["embeddings"])

    def test_batch_empty_list(self, mock_model):
        client = mock_model
        resp = client.post(
            "/embed_batch", json={"sequences": [], "layer_index": 0}
        )
        assert resp.status_code == 422  # validation error
