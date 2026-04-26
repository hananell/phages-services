"""PhaBOX2 phage analysis service.

Wraps the phabox2 CLI to provide taxonomy (PhaGCN), lifestyle (PhaTYP), and
host prediction (CHERRY) via a REST API.  Incoming sequences are written to a
temporary FASTA, processed by ``phabox2 --task end_to_end --skip Y``, and
results are parsed from the output TSVs.

Parallelism
-----------
PhaBOX's bottleneck is DIAMOND/BLAST alignment, which is already well
multi-threaded internally.  One phabox2 process using all available threads is
more efficient than multiple concurrent processes splitting threads — it avoids
duplicate DB loading, prodigal initialization overhead, and CPU cache thrashing.

A semaphore (max_concurrent=1 by default) prevents concurrent requests from
overloading the system.  For multi-machine scaling, run service instances on
separate hosts and distribute work from the feature calculator.

Configuration: config.yaml (see README).
"""

from __future__ import annotations

import asyncio
import csv
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class BatchPredictRequest(BaseModel):
    sequences: list[str]
    sequence_ids: list[str] | None = None


class PhaboxResult(BaseModel):
    sequence_id: str
    # PhaTYP
    phatyp_lifestyle: str | None = None
    phatyp_score: float | None = None
    # PhaGCN
    lineage: str | None = None
    phagcn_score: str | None = None
    genus: str | None = None
    genus_cluster: str | None = None
    # CHERRY
    host: str | None = None
    cherry_score: float | None = None
    cherry_method: str | None = None
    host_ncbi_lineage: str | None = None
    host_gtdb_lineage: str | None = None
    # Meta
    skipped: bool = False


class BatchPredictResponse(BaseModel):
    results: list[PhaboxResult]


class HealthResponse(BaseModel):
    status: str
    phabox_available: bool
    db_available: bool
    threads: int


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


class _State:
    config: dict[str, Any]
    semaphore: asyncio.Semaphore


_state = _State()

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    _state.config = cfg

    phabox_cfg = cfg["phabox"]
    # Serialize requests: DIAMOND/BLAST scale well with threads, so one
    # phabox2 process using all cores is more efficient than splitting.
    _state.semaphore = asyncio.Semaphore(1)

    # Verify phabox2 CLI is available
    phabox_bin = shutil.which("phabox2")
    if not phabox_bin:
        logger.error("phabox2 CLI not found on PATH. Is the conda env active?")
        raise RuntimeError("phabox2 not found")
    logger.info("phabox2 found at {}", phabox_bin)

    # Verify database directory
    dbdir = Path(phabox_cfg["dbdir"])
    if not dbdir.is_dir():
        logger.error("PhaBOX database not found at {}", dbdir)
        raise RuntimeError(f"Database directory not found: {dbdir}")
    logger.info("Database directory: {}", dbdir.resolve())

    logger.info(
        "phabox-service ready: threads={}, min_len={}",
        phabox_cfg.get("threads", 24),
        phabox_cfg.get("min_len", 3000),
    )
    yield


app = FastAPI(
    title="phabox-service",
    version="0.1.0",
    description="PhaBOX2 taxonomy, lifestyle, and host prediction service",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fasta(path: Path, ids: list[str], sequences: list[str]) -> None:
    """Write sequences to a FASTA file."""
    with open(path, "w") as f:
        for sid, seq in zip(ids, sequences):
            f.write(f">{sid}\n{seq}\n")


def _sanitize(value: str | None) -> str | None:
    """Convert PhaBOX dash/empty placeholders to None."""
    if value is None:
        return None
    v = str(value).strip()
    if v in ("", "-", "nan", "NaN"):
        return None
    return v


def _safe_float(value: Any) -> float | None:
    """Parse a float, returning None for missing/invalid values."""
    s = _sanitize(str(value)) if value is not None else None
    if s is None:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _read_tsv(path: Path) -> list[dict[str, str]] | None:
    """Read a TSV file into a list of dicts. Returns None if file missing."""
    if not path.is_file():
        logger.debug("TSV not found: {}", path)
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def _parse_outputs(outpth: Path, expected_ids: list[str]) -> list[PhaboxResult]:
    """Parse phabox2 output TSVs into PhaboxResult objects."""
    final_dir = outpth / "final_prediction"
    if not final_dir.is_dir():
        logger.warning("No final_prediction directory in {}", outpth)
        return [PhaboxResult(sequence_id=sid) for sid in expected_ids]

    # Read individual task TSVs for full column coverage
    phatyp = _read_tsv(final_dir / "phatyp_prediction.tsv")
    phagcn = _read_tsv(final_dir / "phagcn_prediction.tsv")
    cherry = _read_tsv(final_dir / "cherry_prediction.tsv")

    # Build lookup by Accession
    phatyp_by_id = {row["Accession"]: row for row in phatyp} if phatyp else {}
    phagcn_by_id = {row["Accession"]: row for row in phagcn} if phagcn else {}
    cherry_by_id = {row["Accession"]: row for row in cherry} if cherry else {}

    results = []
    for sid in expected_ids:
        pt = phatyp_by_id.get(sid, {})
        gc = phagcn_by_id.get(sid, {})
        ch = cherry_by_id.get(sid, {})

        results.append(PhaboxResult(
            sequence_id=sid,
            # PhaTYP
            phatyp_lifestyle=_sanitize(pt.get("TYPE")),
            phatyp_score=_safe_float(pt.get("PhaTYPScore")),
            # PhaGCN
            lineage=_sanitize(gc.get("Lineage")),
            phagcn_score=_sanitize(gc.get("PhaGCNScore")),
            genus=_sanitize(gc.get("Genus")),
            genus_cluster=_sanitize(gc.get("GenusCluster")),
            # CHERRY
            host=_sanitize(ch.get("Host")),
            cherry_score=_safe_float(ch.get("CHERRYScore")),
            cherry_method=_sanitize(ch.get("Method")),
            host_ncbi_lineage=_sanitize(ch.get("Host_NCBI_lineage")),
            host_gtdb_lineage=_sanitize(ch.get("Host_GTDB_lineage")),
        ))

    return results


async def _run_phabox(
    ids: list[str],
    sequences: list[str],
) -> list[PhaboxResult]:
    """Run phabox2 end_to_end on a batch of sequences."""
    cfg = _state.config["phabox"]
    dbdir = str(Path(cfg["dbdir"]).resolve())
    threads = cfg.get("threads", 24)
    min_len = cfg.get("min_len", 3000)
    timeout = cfg.get("timeout", 7200)

    async with _state.semaphore:
        tmpdir = tempfile.mkdtemp(prefix="phabox_")
        try:
            tmppath = Path(tmpdir)
            fasta_path = tmppath / "input.fa"
            outpth = tmppath / "output"

            _write_fasta(fasta_path, ids, sequences)

            cmd = [
                "phabox2",
                "--task", "end_to_end",
                "--skip", "Y",
                "--contigs", str(fasta_path),
                "--outpth", str(outpth),
                "--dbdir", dbdir,
                "--midfolder", "mid",
                "--threads", str(threads),
                "--len", str(min_len),
            ]

            logger.info(
                "Running phabox2: {} sequences, threads={}",
                len(ids),
                threads,
            )

            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                logger.error(
                    "phabox2 failed (rc={}): stderr={}",
                    result.returncode,
                    result.stderr[-3000:],
                )
                raise RuntimeError(
                    f"phabox2 exited with code {result.returncode}: "
                    f"{result.stderr[-2000:]}"
                )

            return _parse_outputs(outpth, ids)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    cfg = _state.config
    return {
        "service": "phabox-service",
        "version": "0.1.0",
        "tool": "PhaBOX2",
        "tasks": "end_to_end (PhaTYP + PhaGCN + CHERRY)",
        "skip_phamer": True,
        "threads": cfg["phabox"].get("threads", 24),
        "min_len": cfg["phabox"].get("min_len", 3000),
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    cfg = _state.config["phabox"]
    return HealthResponse(
        status="ok",
        phabox_available=shutil.which("phabox2") is not None,
        db_available=Path(cfg["dbdir"]).is_dir(),
        threads=cfg.get("threads", 24),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    sequences = request.sequences
    ids = request.sequence_ids or [f"seq_{i}" for i in range(len(sequences))]

    if len(ids) != len(sequences):
        raise HTTPException(
            status_code=400,
            detail=f"Length mismatch: {len(sequences)} sequences vs {len(ids)} IDs",
        )

    if not sequences:
        return BatchPredictResponse(results=[])

    min_len = _state.config["phabox"].get("min_len", 3000)

    # Separate passable vs too-short sequences
    passable_ids: list[str] = []
    passable_seqs: list[str] = []
    short_ids: set[str] = set()

    for sid, seq in zip(ids, sequences):
        if len(seq) >= min_len:
            passable_ids.append(sid)
            passable_seqs.append(seq)
        else:
            short_ids.add(sid)

    # If nothing is passable, return all-skipped results
    if not passable_ids:
        return BatchPredictResponse(
            results=[PhaboxResult(sequence_id=sid, skipped=True) for sid in ids]
        )

    logger.info(
        "Batch: {} passable sequences, {} skipped (< {} bp)",
        len(passable_ids),
        len(short_ids),
        min_len,
    )

    # Run phabox2 (serialized via semaphore)
    try:
        results = await _run_phabox(passable_ids, passable_seqs)
    except Exception as e:
        logger.error("phabox2 batch failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))

    # Build result map
    result_map: dict[str, PhaboxResult] = {r.sequence_id: r for r in results}

    # Add skipped sequences
    for sid in short_ids:
        result_map[sid] = PhaboxResult(sequence_id=sid, skipped=True)

    # Preserve original input order
    ordered = [result_map[sid] for sid in ids]
    return BatchPredictResponse(results=ordered)
