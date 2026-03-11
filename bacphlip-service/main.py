"""Bacphlip service for phage lifestyle prediction.

This service predicts whether a phage genome is temperate (lysogenic) or
virulent (lytic) using the BACPHLIP machine learning classifier.

Requirements:
    - HMMER3 must be installed and available in PATH (hmmsearch command)
    - Install with: sudo apt-get install hmmer

Example usage:
    uv run python main.py data/lambda_phage.fasta
    uv run python main.py data/t4_phage.fasta
"""

import sys
import warnings
from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd
from loguru import logger

# Suppress deprecation warnings from bacphlip's use of pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, module="bacphlip")

import bacphlip

from config_model import Settings, load_settings


class PredictionResult(TypedDict):
    """Result of a phage lifestyle prediction."""

    genome_file: str
    predicted_lifestyle: Literal["Virulent", "Temperate"]
    virulent_probability: float
    temperate_probability: float


def check_hmmer_installed() -> bool:
    """Check if HMMER3 is installed and available."""
    import shutil

    return shutil.which("hmmsearch") is not None


def predict_lifestyle(
    fasta_path: Path,
    settings: Settings,
) -> PredictionResult:
    """Predict phage lifestyle from a FASTA genome file.

    Args:
        fasta_path: Path to the phage genome FASTA file.
        settings: Configuration settings.

    Returns:
        Dictionary containing prediction results with probabilities.

    Raises:
        FileNotFoundError: If the FASTA file doesn't exist.
        RuntimeError: If HMMER3 is not installed.
    """
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    if not check_hmmer_installed():
        raise RuntimeError(
            "HMMER3 is not installed. Install with: sudo apt-get install hmmer"
        )

    logger.info(f"Running bacphlip prediction on {fasta_path.name}")

    # Run the bacphlip pipeline (force_overwrite handles re-runs)
    bacphlip.run_pipeline(str(fasta_path), force_overwrite=True)

    # Read the prediction results
    results_file = fasta_path.with_suffix(".fasta.bacphlip")
    if not results_file.exists():
        raise RuntimeError(f"Prediction results file not found: {results_file}")

    results_df: pd.DataFrame = pd.read_csv(results_file, sep="\t")
    logger.debug(f"Raw results:\n{results_df}")

    # Extract probabilities
    virulent_prob: float = results_df["Virulent"].iloc[0]
    temperate_prob: float = results_df["Temperate"].iloc[0]

    # Determine predicted lifestyle
    predicted: Literal["Virulent", "Temperate"] = (
        "Temperate" if temperate_prob > virulent_prob else "Virulent"
    )

    result: PredictionResult = {
        "genome_file": fasta_path.name,
        "predicted_lifestyle": predicted,
        "virulent_probability": round(virulent_prob, 4),
        "temperate_probability": round(temperate_prob, 4),
    }

    # Cleanup intermediate files if configured
    if settings.cleanup_intermediate:
        cleanup_intermediate_files(fasta_path)

    return result


def cleanup_intermediate_files(fasta_path: Path) -> None:
    """Remove intermediate files created by bacphlip."""
    suffixes_to_remove = [
        ".fasta.6frame",
        ".fasta.hmmsearch",
        ".fasta.hmmsearch.tsv",
    ]
    for suffix in suffixes_to_remove:
        intermediate_file = fasta_path.parent / (fasta_path.stem + suffix)
        if intermediate_file.exists():
            intermediate_file.unlink()
            logger.debug(f"Removed intermediate file: {intermediate_file}")


def predict_lifestyle_batch(
    fasta_paths: list[Path],
    settings: Settings,
) -> list[PredictionResult]:
    """Predict lifestyle for multiple phage genomes.

    Args:
        fasta_paths: List of paths to phage genome FASTA files.
        settings: Configuration settings.

    Returns:
        List of prediction results.
    """
    results: list[PredictionResult] = []
    for fasta_path in fasta_paths:
        try:
            result = predict_lifestyle(fasta_path, settings)
            results.append(result)
            logger.info(
                f"{result['genome_file']}: {result['predicted_lifestyle']} "
                f"(V={result['virulent_probability']:.2%}, "
                f"T={result['temperate_probability']:.2%})"
            )
        except Exception as e:
            logger.error(f"Failed to process {fasta_path}: {e}")
    return results


def main() -> None:
    """Main entry point for the bacphlip service."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    settings = load_settings()
    logger.info("Bacphlip Phage Lifestyle Prediction Service")
    logger.info(f"Data directory: {settings.data_dir}")
    logger.info(f"Output directory: {settings.output_dir}")

    # Check HMMER3 installation
    if not check_hmmer_installed():
        logger.error("HMMER3 is not installed!")
        logger.error("Install with: sudo apt-get install hmmer")
        sys.exit(1)

    logger.info("HMMER3 is installed and available")

    # Process command line arguments or use sample data
    if len(sys.argv) > 1:
        fasta_files = [Path(arg) for arg in sys.argv[1:]]
    else:
        # Default: process sample files in data directory
        fasta_files = list(settings.data_dir.glob("*.fasta"))
        if not fasta_files:
            logger.warning("No FASTA files found in data directory")
            logger.info("Usage: uv run python main.py <fasta_file> [fasta_file ...]")
            sys.exit(0)

    logger.info(f"Processing {len(fasta_files)} file(s)")
    results = predict_lifestyle_batch(fasta_files, settings)

    # Summary
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    for result in results:
        logger.info(
            f"  {result['genome_file']}: {result['predicted_lifestyle']} "
            f"(V={result['virulent_probability']:.2%}, T={result['temperate_probability']:.2%})"
        )


if __name__ == "__main__":
    main()
