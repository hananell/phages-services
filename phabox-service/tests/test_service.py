"""Integration tests for phabox-service.

Uses mocked PhaBOX2 subprocess to avoid requiring conda env and database.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked PhaBOX2."""
    # Mock the config and startup checks
    mock_cfg = MagicMock()
    mock_cfg.phabox.dbdir = "/mock/phabox_db"
    mock_cfg.phabox.min_len = 3000
    mock_cfg.phabox.threads = 4
    mock_cfg.phabox.timeout = 7200
    mock_cfg.server.host = "0.0.0.0"
    mock_cfg.server.port = 8005

    with patch("service._state") as mock_state, \
         patch("service.load_config", return_value=mock_cfg):
        mock_state.config = mock_cfg

        from service import app
        yield TestClient(app)


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200


class TestRoot:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "PhaBOX" in data.get("service", data.get("name", ""))


class TestPredict:
    def test_predict_batch_validation(self, client):
        """Empty sequences should fail validation."""
        resp = client.post(
            "/predict/batch",
            json={"sequences": [], "sequence_ids": []},
        )
        assert resp.status_code == 422

    def test_predict_batch_mismatched_ids(self, client):
        """Mismatched sequence/ID counts should fail."""
        resp = client.post(
            "/predict/batch",
            json={
                "sequences": ["ATCG" * 1000],
                "sequence_ids": ["a", "b"],
            },
        )
        # Should be caught by validation or endpoint logic
        assert resp.status_code in (400, 422, 500)
