"""Integration tests for hmm-service.

Uses a mock HMM matcher to avoid requiring the PHROGs database.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fastapi.testclient import TestClient

from hmm_service.models import GenomeHMMResult


@pytest.fixture
def client():
    """Create test client with mocked HMM matcher."""
    mock_matcher = MagicMock()
    mock_matcher.is_initialized = True
    mock_matcher.database_info = {
        "loaded": True,
        "path": "/mock/all_phrogs.h3m",
        "profile_count": 38880,
        "alphabet": "amino",
    }
    mock_matcher.search.return_value = (
        [
            GenomeHMMResult(
                genome_id="test_genome",
                protein_count=10,
                hmm_hit_count=3,
                hmm_hit_count_normalized=0.3,
                unique_phrogs=["phrog_1", "phrog_42", "phrog_100"],
            )
        ],
        None,  # no detailed hits
    )

    with patch("hmm_service.main.hmm_matcher", mock_matcher):
        from hmm_service.main import app

        yield TestClient(app)


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["database_loaded"] is True


class TestDatabase:
    def test_database_info(self, client):
        resp = client.get("/database")
        assert resp.status_code == 200
        data = resp.json()
        assert data["loaded"] is True
        assert data["profile_count"] == 38880


class TestSearch:
    def test_search_valid(self, client):
        resp = client.post(
            "/search",
            json={
                "proteins": [
                    {
                        "protein_id": "prot1",
                        "sequence": "MKTLLLTGFGGAAAAAAAAAA",
                        "genome_id": "test_genome",
                    }
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["genome_results"]) == 1
        assert data["genome_results"][0]["genome_id"] == "test_genome"
        assert len(data["genome_results"][0]["unique_phrogs"]) == 3

    def test_search_empty_proteins(self, client):
        resp = client.post("/search", json={"proteins": []})
        assert resp.status_code == 422  # validation: min_length=1

    def test_search_short_protein(self, client):
        resp = client.post(
            "/search",
            json={
                "proteins": [
                    {
                        "protein_id": "short",
                        "sequence": "MKT",  # too short (min 10)
                        "genome_id": "g1",
                    }
                ]
            },
        )
        assert resp.status_code == 422
