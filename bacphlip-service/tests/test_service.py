"""Integration tests for bacphlip-service.

Uses mocked bacphlip pipeline to avoid requiring HMMER3 binary.
"""

from __future__ import annotations

import shutil
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked HMMER3 and bacphlip."""
    # Mock HMMER3 availability
    with patch("service.check_hmmer_installed", return_value=True), \
         patch("service.bacphlip") as mock_bacphlip:

        # bacphlip.run_pipeline writes result files; we mock the file reading
        mock_bacphlip.run_pipeline = MagicMock()

        from service import app
        yield TestClient(app)


class TestHealth:
    def test_health_healthy(self, client):
        with patch("service.check_hmmer_installed", return_value=True):
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert data["hmmer_available"] is True


class TestRoot:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "BACPHLIP" in data["service"]


class TestPredict:
    def test_predict_short_sequence(self, client):
        """Sequence shorter than 100bp should fail validation."""
        resp = client.post(
            "/predict",
            json={"sequence": "ATCG", "sequence_id": "short"},
        )
        assert resp.status_code == 422  # pydantic min_length=100

    def test_predict_missing_sequence(self, client):
        """Missing sequence field."""
        resp = client.post("/predict", json={"sequence_id": "x"})
        assert resp.status_code == 422
