"""Lazy HMM Profile Matcher using PyHMMER."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import pyhmmer
from pyhmmer.easel import Alphabet, DigitalSequenceBlock, TextSequence
from pyhmmer.plan7 import HMMFile

from hmm_service import logger
from hmm_service.config import settings
from hmm_service.models import GenomeHMMResult, HMMHit, ProteinInput


@dataclass
class HMMDatabase:
    """Container for loaded HMM database state."""

    hmms: list[pyhmmer.plan7.HMM] = field(default_factory=list)
    alphabet: Alphabet | None = None
    path: Path | None = None
    profile_count: int = 0


class LazyHMMMatcher:
    """
    Lazy-loading HMM profile matcher using PyHMMER.

    The HMM database is loaded only on the first search request,
    not at service startup.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        e_value_threshold: float = 1e-5,
        bit_score_threshold: float | None = None,
        cpus: int = 4,
    ) -> None:
        """
        Initialize the lazy HMM matcher.

        Args:
            db_path: Path to pressed HMM database (.h3m file)
            e_value_threshold: E-value cutoff for significant hits
            bit_score_threshold: Optional bit score cutoff
            cpus: Number of CPU threads for search
        """
        self._db_path: Path = db_path or settings.phrogs_db_path
        self._e_value_threshold: float = e_value_threshold
        self._bit_score_threshold: float | None = bit_score_threshold
        self._cpus: int = cpus

        self._database: HMMDatabase = HMMDatabase()
        self._initialized: bool = False
        self._init_lock: Lock = Lock()

    @property
    def is_initialized(self) -> bool:
        """Check if database is loaded."""
        return self._initialized

    @property
    def database_info(self) -> dict:
        """Get database information."""
        return {
            "loaded": self._initialized,
            "path": str(self._db_path) if self._initialized else None,
            "profile_count": self._database.profile_count if self._initialized else None,
            "alphabet": str(self._database.alphabet) if self._initialized else None,
        }

    def _ensure_initialized(self) -> None:
        """Load database if not already loaded (thread-safe)."""
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return
            self._load_database()

    def _load_database(self) -> None:
        """Load HMM profiles from database file (.hmm or .h3m)."""
        logger.info(f"Loading HMM database from {self._db_path}")

        if not self._db_path.exists():
            raise FileNotFoundError(f"HMM database not found: {self._db_path}")

        # HMMFile can read both raw .hmm and binary .h3m formats
        logger.info(f"Loading HMM database ({self._db_path.suffix} format)")
        with HMMFile(self._db_path) as hmm_file:
            self._database.hmms = list(hmm_file)
            if self._database.hmms:
                self._database.alphabet = self._database.hmms[0].alphabet

        self._database.profile_count = len(self._database.hmms)
        self._database.path = self._db_path
        self._initialized = True

        logger.info(
            f"Loaded {self._database.profile_count} HMM profiles "
            f"(alphabet: {self._database.alphabet})"
        )

    def _build_sequence_block(
        self, proteins: list[ProteinInput]
    ) -> tuple[DigitalSequenceBlock, dict[bytes, str]]:
        """
        Convert protein inputs to PyHMMER digital sequence block.

        Returns:
            Tuple of (sequence block, mapping from sequence name to genome_id)
        """
        alphabet = self._database.alphabet or Alphabet.amino()

        text_sequences: list[TextSequence] = []
        name_to_genome: dict[bytes, str] = {}

        for protein in proteins:
            name = protein.protein_id.encode()
            seq = TextSequence(name=name, sequence=protein.sequence)
            text_sequences.append(seq)
            name_to_genome[name] = protein.genome_id

        # Convert to digital sequences
        digital_sequences = [seq.digitize(alphabet) for seq in text_sequences]
        sequence_block = DigitalSequenceBlock(alphabet, digital_sequences)

        return sequence_block, name_to_genome

    def search(
        self,
        proteins: list[ProteinInput],
        return_detailed_hits: bool = False,
    ) -> tuple[list[GenomeHMMResult], list[HMMHit] | None]:
        """
        Search proteins against HMM database.

        Args:
            proteins: List of protein sequences to search
            return_detailed_hits: Whether to return per-protein hit details

        Returns:
            Tuple of (genome results, optional detailed hits)
        """
        self._ensure_initialized()

        if not proteins:
            return [], None

        logger.info(f"Searching {len(proteins)} proteins against HMM database")

        # Build sequence block
        sequence_block, name_to_genome = self._build_sequence_block(proteins)

        # Track hits per genome and per protein
        genome_phrogs: dict[str, set[str]] = defaultdict(set)
        genome_protein_count: dict[str, int] = defaultdict(int)
        detailed_hits: list[HMMHit] = []

        # Count proteins per genome
        for protein in proteins:
            genome_protein_count[protein.genome_id] += 1

        # Run hmmsearch for all HMM profiles at once (more efficient)
        all_hits = pyhmmer.hmmsearch(
            self._database.hmms, sequence_block, cpus=self._cpus
        )

        for top_hits in all_hits:
            # top_hits.query is the HMM object, get name from it
            hmm_name = top_hits.query.name.decode() if top_hits.query.name else "unknown"

            for hit in top_hits:
                # Apply E-value threshold
                if hit.evalue > self._e_value_threshold:
                    continue

                # Apply bit score threshold if set
                if (
                    self._bit_score_threshold is not None
                    and hit.score < self._bit_score_threshold
                ):
                    continue

                protein_name = hit.name
                genome_id = name_to_genome.get(protein_name, "unknown")

                # Track unique PHROGs per genome
                genome_phrogs[genome_id].add(hmm_name)

                if return_detailed_hits:
                    detailed_hits.append(
                        HMMHit(
                            protein_id=protein_name.decode(),
                            hmm_name=hmm_name,
                            e_value=hit.evalue,
                            bit_score=hit.score,
                        )
                    )

        # Build genome results
        genome_results: list[GenomeHMMResult] = []
        all_genomes = set(genome_protein_count.keys())

        for genome_id in all_genomes:
            phrogs = genome_phrogs.get(genome_id, set())
            protein_count = genome_protein_count[genome_id]
            hit_count = len(phrogs)

            genome_results.append(
                GenomeHMMResult(
                    genome_id=genome_id,
                    protein_count=protein_count,
                    hmm_hit_count=hit_count,
                    hmm_hit_count_normalized=(
                        hit_count / protein_count if protein_count > 0 else 0.0
                    ),
                    unique_phrogs=sorted(phrogs),
                )
            )

        logger.info(
            f"Search complete: {len(genome_results)} genomes, "
            f"{sum(len(p) for p in genome_phrogs.values())} total unique PHROG hits"
        )

        return genome_results, detailed_hits if return_detailed_hits else None


# Global singleton instance
hmm_matcher = LazyHMMMatcher(
    db_path=settings.phrogs_db_path,
    e_value_threshold=settings.e_value_threshold,
    bit_score_threshold=settings.bit_score_threshold,
    cpus=settings.cpus,
)
