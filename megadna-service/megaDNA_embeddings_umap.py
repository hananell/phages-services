#!/usr/bin/env python3
"""
Calculate embeddings for WT and mutated start codon sequences using megaDNA model,
then visualize them using UMAP.

This script analyzes how start codon mutations affect the learned representations
of DNA sequences in the megaDNA transformer model. By comparing wild-type (WT)
sequences against versions with mutated start codons, we can visualize whether
the model learns meaningful representations that distinguish functional vs
disrupted gene sequences.

Workflow:
    1. Load a pre-trained megaDNA transformer model
    2. Parse a FASTA file containing genome sequence(s)
    3. Parse GFF3 annotations to identify CDS (coding sequence) features
    4. For each CDS, generate the WT embedding and a mutant embedding
       (with randomized nucleotides at the start codon position)
    5. Apply UMAP to reduce high-dimensional embeddings to 2D
    6. Visualize WT vs mutant embeddings as a scatter plot

Example Usage:
    python megaDNA_embeddings_umap.py \
        --fasta "sequence (1).fasta" \
        --gff "sequence (1).gff3" \
        --model ../external/megaDNA_phage_145M.pt \
        --output embeddings_umap.png

Dependencies:
    - torch: PyTorch for model loading and inference
    - numpy: Numerical operations on embeddings
    - umap-learn: UMAP dimensionality reduction
    - matplotlib: Visualization
    - biopython: FASTA parsing
    - bcbio-gff: GFF3 annotation parsing
"""

from __future__ import annotations

# Standard library imports
import argparse
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple, Optional

# Third-party imports
import torch
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt





from Bio import SeqIO
from BCBio import GFF

# =============================================================================
# Constants
# =============================================================================

# Vocabulary for encoding DNA sequences into integer tokens.
# Index 0: Start-of-sequence token ('**')
# Index 1-4: Nucleotides A, T, C, G
# Index 5: End-of-sequence token ('#')
NT_VOCAB: List[str] = ['**', 'A', 'T', 'C', 'G', '#']

# Maximum sequence length supported by the megaDNA model
MAX_SEQUENCE_LENGTH: int = 96000

# Indices for random nucleotide selection (A=1, T=2, C=3, G=4)
NUCLEOTIDE_INDICES: List[int] = [1, 2, 3, 4]

# UMAP default parameters
DEFAULT_UMAP_N_NEIGHBORS: int = 15
DEFAULT_UMAP_MIN_DIST: float = 0.1

# Device type alias
DeviceType = Literal["cpu", "cuda"]

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GeneAnnotation:
    """Represents a single CDS (coding sequence) annotation from a GFF3 file.
    
    Attributes:
        start: 0-based start position of the CDS in the genome
        end: 0-based end position (exclusive) of the CDS
        strand: Strand direction (1 for forward/+, -1 for reverse/-)
    """
    start: int
    end: int
    strand: int


# =============================================================================
# Sequence Encoding Functions
# =============================================================================

def encode_sequence(sequence: str, nt_vocab: List[str] = NT_VOCAB) -> List[int]:
    """Encode a DNA sequence to its numerical representation for the megaDNA model.

    The megaDNA model expects sequences to be tokenized as integers with special
    start and end tokens. This function converts a string DNA sequence into the
    required format.

    Token mapping:
        0 -> '**' (start-of-sequence token)
        1 -> 'A' (Adenine)
        2 -> 'T' (Thymine)  
        3 -> 'C' (Cytosine)
        4 -> 'G' (Guanine)
        5 -> '#' (end-of-sequence token)

    Args:
        sequence: Raw DNA sequence string (e.g., "ATCGATCG")
        nt_vocab: Vocabulary list mapping tokens to indices

    Returns:
        List of integers representing the encoded sequence with start/end tokens.
        Length will be len(sequence) + 2 (for the two special tokens).

    Example:
        >>> encode_sequence("ATG")
        [0, 1, 2, 4, 5]  # [start, A, T, G, end]
    """
    # Build encoded sequence: [start_token] + [nucleotides...] + [end_token]
    encoded = [0]  # Start token
    for nucleotide in sequence:
        if nucleotide in nt_vocab:
            encoded.append(nt_vocab.index(nucleotide))
        else:
            # Unknown characters default to 'A' (index 1) to maintain sequence length
            # This handles edge cases like N (ambiguous) or lowercase letters
            encoded.append(1)
    encoded.append(5)  # End token
    return encoded


# =============================================================================
# Model Inference Functions
# =============================================================================

def get_embedding_for_sequence(
    model: torch.nn.Module,
    sequence: List[int],
    device: DeviceType,
    layer_index: int = 0,
) -> np.ndarray:
    """Extract a fixed-size embedding vector from the megaDNA model for a sequence.

    The megaDNA transformer is a hierarchical model (MEGABYTE-based) that returns
    hidden states from multiple transformer stages. Each stage has different
    dimensions due to the hierarchical structure:
        - Layer 0: Coarsest level, shape (batch, positions, embed_dim)
        - Layer 1: Intermediate level
        - Layer 2: Finest level (may have different batch structure)

    We recommend using layer 0 (default) as it maintains proper batch dimensions
    and provides meaningful sequence-level representations.

    Args:
        model: Pre-trained megaDNA model in evaluation mode
        sequence: Encoded DNA sequence (list of token indices including start/end)
        device: Compute device ('cpu' or 'cuda')
        layer_index: Which layer's hidden states to use (0 for first/coarsest, recommended)

    Returns:
        1D numpy array of shape (embed_dim,) representing the sequence embedding.
        The embedding dimension depends on the selected layer (e.g., 512 for layer 0).

    Note:
        - Inference is performed with gradients disabled for efficiency
        - The model's 'embedding' return mode gives us hidden states from all
          transformer stages as a list of tensors
        - Layer 0 is recommended as it preserves batch structure and provides
          high-quality sequence representations
    """
    # Convert sequence to tensor and add batch dimension: (1, seq_len)
    input_seq: torch.Tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
    
    # Disable gradient computation for inference efficiency
    with torch.no_grad():
        # Forward pass through the model to get hidden states from all layers
        # Returns: List of tensors, one for each transformer stage
        hidden_states: List[torch.Tensor] = model(input_seq, return_value='embedding')
        
        # Select the specified layer
        # Layer 0 shape: (batch, positions, embed_dim) - cleanest structure
        # Other layers may have different batch structure due to hierarchical design
        layer_embeddings: torch.Tensor = hidden_states[layer_index]
        
        # Apply mean pooling across the position dimension (dim=1)
        # Shape: (batch, embed_dim)
        mean_embedding: torch.Tensor = layer_embeddings.mean(dim=1)
    
    # Remove batch dimension and convert to numpy for downstream processing
    # Result shape: (embed_dim,)
    return mean_embedding.squeeze(0).cpu().numpy()


# =============================================================================
# File I/O Functions
# =============================================================================

def validate_file_exists(file_path: str, file_type: str = "file") -> Path:
    """Validate that a file exists and return its Path object.
    
    Args:
        file_path: Path to the file to validate
        file_type: Description of the file type for error messages
        
    Returns:
        Path object for the validated file
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_type} not found: {file_path}")
    return path


def load_fasta_sequences(fasta_path: str) -> Tuple[List[str], List[str]]:
    """Load all sequences from a FASTA file.

    FASTA is a text-based format for representing nucleotide or protein sequences.
    Each sequence has a header line starting with '>' followed by the sequence ID,
    and subsequent lines contain the actual sequence.

    Args:
        fasta_path: Path to the FASTA file (.fasta, .fa, .fna)

    Returns:
        Tuple of (sequence_ids, sequences) where:
            - sequence_ids: List of sequence identifiers from headers
            - sequences: List of DNA sequences as uppercase strings

    Raises:
        FileNotFoundError: If the FASTA file doesn't exist
    """
    seq_ids: List[str] = []
    sequences: List[str] = []
    
    with open(fasta_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_ids.append(record.id)
            # Convert Seq object to string for easier manipulation
            sequences.append(str(record.seq))
    
    return seq_ids, sequences


def load_gff_annotations(gff_path: str) -> List[GeneAnnotation]:
    """Load CDS (coding sequence) annotations from a GFF3 file.

    GFF3 (General Feature Format version 3) is a standard format for describing
    genomic features. This function extracts only CDS features, which represent
    protein-coding regions of genes.

    Coordinate system notes:
        - GFF3 uses 1-based, fully-closed coordinates [start, end]
        - Biopython converts these to 0-based, half-open coordinates [start, end)
        - This matches Python's standard indexing conventions

    Args:
        gff_path: Path to the GFF3 annotation file

    Returns:
        List of GeneAnnotation objects containing start, end, and strand info
        for each CDS feature.

    Raises:
        FileNotFoundError: If the GFF3 file doesn't exist
    """
    # Filter to only parse CDS (coding sequence) features
    limit_info = {"gff_type": ["CDS"]}
    annotations: List[GeneAnnotation] = []
    
    with open(gff_path) as gff_handle:
        for record in GFF.parse(gff_handle, limit_info=limit_info):
            for feature in record.features:
                annotations.append(GeneAnnotation(
                    start=int(feature.location.start),
                    end=int(feature.location.end),
                    strand=int(feature.location.strand)
                ))
    
    return annotations


# =============================================================================
# Mutation Functions
# =============================================================================

def mutate_start_codon(encoded_sequence: List[int], positions: range) -> List[int]:
    """Create a mutated copy of an encoded sequence with randomized start codon.

    This function simulates start codon mutations by replacing the nucleotides
    at specified positions with randomly chosen nucleotides. Start codon mutations
    can disrupt translation initiation, making this useful for studying how the
    model represents functional vs non-functional gene sequences.

    Args:
        encoded_sequence: Original encoded DNA sequence (list of token indices)
        positions: Range of positions to mutate (typically 3 positions for a codon)

    Returns:
        New list with mutations applied (original sequence is not modified).
        Positions outside the valid range are silently skipped.

    Example:
        >>> original = [0, 1, 2, 4, 3, 5]  # [start, A, T, G, C, end] - "ATG" start codon
        >>> mutated = mutate_start_codon(original, range(1, 4))
        >>> mutated  # Positions 1,2,3 now have random nucleotides
        [0, 3, 1, 2, 3, 5]  # Could be any random combination
    """
    # Create a copy to avoid modifying the original sequence
    mutated = list(encoded_sequence)
    
    for pos in positions:
        # Bounds check: skip positions outside the sequence
        if pos < 0 or pos >= len(mutated):
            continue
        # Replace with a random nucleotide (A=1, T=2, C=3, G=4)
        mutated[pos] = random.choice(NUCLEOTIDE_INDICES)
    
    return mutated


def filter_annotations_by_length(
    annotations: List[GeneAnnotation], 
    max_length: int
) -> List[GeneAnnotation]:
    """Filter annotations to only include those fully within a length limit.

    When a sequence is trimmed to fit model constraints, we need to remove
    annotations that would extend beyond the trimmed region.

    Args:
        annotations: List of gene annotations to filter
        max_length: Maximum allowed position (features ending after this are excluded)

    Returns:
        Filtered list containing only annotations within bounds
    """
    return [
        ann for ann in annotations
        if ann.start < max_length and ann.end <= max_length
    ]


def get_start_codon_positions(annotation: GeneAnnotation) -> range:
    """Calculate the positions of the start codon for a gene annotation.

    The start codon position depends on the strand:
        - Forward strand (+1): First 3 nucleotides of the CDS
        - Reverse strand (-1): Last 3 nucleotides of the CDS
          (representing the reverse complement start codon)

    Note: Positions are adjusted +1 to account for the start token added
    during sequence encoding.

    Args:
        annotation: Gene annotation containing start, end, and strand info

    Returns:
        Range of 3 positions corresponding to the start codon in encoded space
    """
    if annotation.strand == 1:
        # Forward strand: start codon is at the beginning of the CDS
        # +1 offset accounts for the start-of-sequence token in encoded sequence
        return range(annotation.start + 1, annotation.start + 4)
    else:
        # Reverse strand: start codon is at the end of the CDS
        # (it's the reverse complement, so we mutate the 3' end)
        return range(annotation.end - 2, annotation.end + 1)


# =============================================================================
# Main Entry Point
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the embedding analysis pipeline.

    Args:
        argv: Command-line arguments (defaults to sys.argv if None)
    """
    import time
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # Parse command-line arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Calculate embeddings for WT and mutated sequences, visualize with UMAP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--fasta", 
        default="sequence (1).fasta", 
        help="FASTA file containing the genome sequence(s)"
    )
    parser.add_argument(
        "--gff", 
        default="sequence (1).gff3", 
        help="GFF3 file containing CDS annotations"
    )
    parser.add_argument(
        "--model", 
        default="../external/megaDNA_phage_145M.pt", 
        help="Path to the pre-trained megaDNA PyTorch model file"
    )
    parser.add_argument(
        "--device", 
        default=None, 
        choices=[None, "cpu", "cuda"],
        help="Compute device: 'cpu' or 'cuda' (auto-detects GPU if not specified)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility of mutations and UMAP"
    )
    parser.add_argument(
        "--seq-id", 
        type=int, 
        default=0, 
        help="Index of the sequence record to analyze (0-based)"
    )
    parser.add_argument(
        "--output-dir", 
        default="output", 
        help="Output directory for the UMAP visualizations"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=DEFAULT_UMAP_N_NEIGHBORS,
        help="UMAP n_neighbors parameter (controls local vs global structure)"
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=DEFAULT_UMAP_MIN_DIST,
        help="UMAP min_dist parameter (controls point clustering tightness)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    args = parser.parse_args(argv)

    # -------------------------------------------------------------------------
    # Setup: logging, device selection, and file validation
    # -------------------------------------------------------------------------
    # Configure logging level based on verbosity
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("umap").setLevel(logging.WARNING)
    
    # Validate input files exist before proceeding
    validate_file_exists(args.fasta, "FASTA file")
    validate_file_exists(args.gff, "GFF3 file")
    validate_file_exists(args.model, "Model file")
    
    # Auto-detect GPU if available, otherwise fall back to CPU
    device: DeviceType = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Load the pre-trained megaDNA model
    # -------------------------------------------------------------------------
    logger.info(f"Loading model from {args.model} to device {device}")
    
    # Load the full model (not just weights) - megaDNA models are saved this way
    # weights_only=False is required because the model architecture is included
    model = torch.load(args.model, map_location=torch.device(device), weights_only=False)
    
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    # -------------------------------------------------------------------------
    # Load input data: sequences and annotations
    # -------------------------------------------------------------------------
    logger.info(f"Loading sequences from {args.fasta}")
    seq_ids, sequences = load_fasta_sequences(args.fasta)
    
    if len(sequences) == 0:
        logger.error("No sequences found in FASTA file")
        raise SystemExit("Error: No sequences found in FASTA file")
    
    if args.seq_id >= len(sequences):
        logger.error(f"--seq-id {args.seq_id} out of range (file has {len(sequences)} sequences)")
        raise SystemExit(f"Error: --seq-id {args.seq_id} out of range (file has {len(sequences)} sequences)")

    logger.info(f"Found {len(sequences)} sequence(s), using index {args.seq_id}: {seq_ids[args.seq_id]}")

    logger.info(f"Loading gene annotations from {args.gff}")
    annotations = load_gff_annotations(args.gff)
    logger.info(f"Found {len(annotations)} CDS features")

    # -------------------------------------------------------------------------
    # Handle sequence length constraints
    # -------------------------------------------------------------------------
    # The megaDNA model has a maximum context length of 96,000 nucleotides.
    # If the input sequence exceeds this, we must truncate it and filter
    # out any annotations that would be outside the truncated region.
    
    original_length: int = len(sequences[args.seq_id])
    
    if original_length > MAX_SEQUENCE_LENGTH:
        logger.warning(f"Trimming sequence from {original_length:,} to {MAX_SEQUENCE_LENGTH:,} bp (model limit)")
        sequences[args.seq_id] = sequences[args.seq_id][:MAX_SEQUENCE_LENGTH]

        # Filter annotations to only include those within the trimmed region
        original_count = len(annotations)
        annotations = filter_annotations_by_length(annotations, MAX_SEQUENCE_LENGTH)
        logger.info(f"Filtered annotations: {original_count} -> {len(annotations)} features")

    # -------------------------------------------------------------------------
    # Set random seed for reproducibility
    # -------------------------------------------------------------------------
    # This affects both the random mutations and UMAP's random initialization
    random.seed(args.seed)
    np.random.seed(args.seed)  # Also seed numpy for UMAP reproducibility

    # -------------------------------------------------------------------------
    # Encode the wild-type sequence
    # -------------------------------------------------------------------------
    encoded_wt_sequence: List[int] = encode_sequence(sequences[args.seq_id])
    logger.info(f"Encoded sequence length: {len(encoded_wt_sequence):,} tokens (including start/end)")

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # -------------------------------------------------------------------------
    # Loop over all 3 embedding layers
    # -------------------------------------------------------------------------
    layer_names = ["Layer 0 (Coarsest)", "Layer 1 (Intermediate)", "Layer 2 (Finest)"]
    
    for layer_index in range(3):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {layer_names[layer_index]}")
        logger.info(f"{'='*60}")
        
        # -------------------------------------------------------------------------
        # Calculate embeddings for each gene (WT and mutant versions)
        # -------------------------------------------------------------------------
        logger.info(f"Calculating embeddings for {len(annotations)} genes...")
        logger.debug("For each gene: computing WT embedding and mutant (disrupted start codon) embedding")
        
        wt_embeddings: List[np.ndarray] = []
        mutant_embeddings: List[np.ndarray] = []

        # Pre-compute WT embedding once (optimization: avoid redundant computation)
        wt_embedding = get_embedding_for_sequence(model, encoded_wt_sequence, device, layer_index=layer_index)

        for gene_idx, annotation in enumerate(annotations):
            # Get the positions of the start codon in the encoded sequence
            # (accounts for strand direction and encoding offset)
            codon_positions = get_start_codon_positions(annotation)

            # For WT, we use the same embedding (it's the same sequence)
            wt_embeddings.append(wt_embedding)

            # Create a mutated version with randomized start codon nucleotides
            # This simulates a loss-of-function mutation that disrupts translation
            mutated_sequence = mutate_start_codon(encoded_wt_sequence, codon_positions)

            # Compute embedding for the mutated sequence
            mutant_embedding = get_embedding_for_sequence(model, mutated_sequence, device, layer_index=layer_index)
            mutant_embeddings.append(mutant_embedding)

            # Progress indicator (every 10 genes or at the end)
            if (gene_idx + 1) % 10 == 0 or gene_idx == len(annotations) - 1:
                logger.info(f"Processed {gene_idx + 1}/{len(annotations)} genes")

        logger.info(f"Completed embedding calculation for {len(wt_embeddings)} genes")

        # -------------------------------------------------------------------------
        # Prepare embeddings for UMAP dimensionality reduction
        # -------------------------------------------------------------------------
        # Convert lists to numpy arrays for efficient computation
        wt_embeddings_array = np.array(wt_embeddings)
        mutant_embeddings_array = np.array(mutant_embeddings)

        logger.debug(f"Embedding dimensions:")
        logger.debug(f"  WT embeddings: {wt_embeddings_array.shape} (genes × embedding_dim)")
        logger.debug(f"  Mutant embeddings: {mutant_embeddings_array.shape}")

        # Stack WT and mutant embeddings together for joint UMAP fitting
        # This ensures both are projected into the same 2D space for comparison
        all_embeddings = np.vstack([wt_embeddings_array, mutant_embeddings_array])
        logger.debug(f"  Combined for UMAP: {all_embeddings.shape}")

        # -------------------------------------------------------------------------
        # Apply UMAP dimensionality reduction
        # -------------------------------------------------------------------------
        logger.info(f"Applying UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...")
        reducer = umap.UMAP(
            n_components=2, 
            random_state=args.seed, 
            n_neighbors=args.n_neighbors, 
            min_dist=args.min_dist,
            metric='euclidean'
        )
        embeddings_2d = reducer.fit_transform(all_embeddings)

        # Split the 2D projections back into WT and mutant groups
        n_genes = len(wt_embeddings)
        wt_2d = embeddings_2d[:n_genes]
        mutant_2d = embeddings_2d[n_genes:]

        # -------------------------------------------------------------------------
        # Create visualization
        # -------------------------------------------------------------------------
        output_filename = output_dir / f"embeddings_umap_layer{layer_index}.png"
        logger.info(f"Creating visualization and saving to {output_filename}")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot WT embeddings in blue
        ax.scatter(
            wt_2d[:, 0], wt_2d[:, 1], 
            c='blue', 
            label=f'Wild Type (n={n_genes})', 
            alpha=0.7, 
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
        
        # Plot mutant embeddings in red  
        ax.scatter(
            mutant_2d[:, 0], mutant_2d[:, 1], 
            c='red', 
            label=f'Mutated Start Codon (n={n_genes})', 
            alpha=0.7, 
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
        
        # Labels and styling
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title(f'UMAP Visualization: WT vs Mutated Start Codon\n{layer_names[layer_index]}', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        
        # Add grid for easier reading
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close figure to free memory
        logger.info(f"Plot saved to {output_filename}")

    logger.info(f"\nAnalysis complete: compared {n_genes} WT vs mutant gene embeddings across 3 layers")
    logger.info(f"Results saved to: {output_dir}/")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()
