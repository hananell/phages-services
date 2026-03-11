#!/usr/bin/env python3
"""
Convert Jupyter notebook 'megaDNA_mutagenesis.ipynb' to a runnable Python script.

This script: 
- Loads a pre-trained model (Torch) and device config
- Reads a FASTA sequence and GFF/Coding sequence annotations
- Encodes the DNA sequence for model input
- Calculates model losses for wildtype and mutated start codons

Usage:
    python megaDNA_mutagenesis.py --fasta NC_001416.1.fasta --gff NC_001416.1.gff3 \
        --model ../external/megaDNA_phage_145M.pt

Notes:
- Requires Biopython (SeqIO) and BCBio for GFF parsing
- The model call assumes `model(input_seq, return_value='loss')` returns a loss tensor
"""

from __future__ import annotations
import argparse
import random
import sys
from typing import List, Tuple
import megaDNA

import torch
from Bio import SeqIO
from BCBio import GFF

# Vocabulary used to encode sequences
NT_VOCAB = ['**', 'A', 'T', 'C', 'G', '#']


def encode_sequence(sequence: str, nt_vocab: List[str] = NT_VOCAB) -> List[int]:
    """Encode a DNA sequence to its numerical representation.

    Adds a start token (0) and a terminal token (5) as used in the original notebook.
    Unknown characters are encoded as index 1 (A) by default to keep indexing stable.
    """
    return [0] + [nt_vocab.index(n) if n in nt_vocab else 1 for n in sequence] + [5]


def get_loss_for_sequence(
    model: torch.nn.Module,
    sequence: List[int],
    device: str,
    positions: range | None = None,
) -> torch.Tensor:
    """Get model loss for a given encoded sequence.

    Args:
        model: The megaDNA model.
        sequence: Encoded DNA sequence (with start/end tokens).
        device: Device to run inference on.
        positions: Optional range of positions to compute local loss for.
            If provided, returns the mean loss only at those positions.
            If None, returns the global mean loss over the entire sequence.

    Returns:
        Loss tensor (scalar).
    """
    input_seq: torch.Tensor = torch.tensor(sequence).unsqueeze(0).to(device)
    with torch.no_grad():
        # Get logits instead of loss so we can compute per-position loss
        logits: torch.Tensor = model(input_seq, return_value='logits')
        # logits shape: (batch, seq_len, vocab_size)
        # Shift for next-token prediction: predict position i from position i-1
        # labels are the input sequence shifted by 1
        labels: torch.Tensor = input_seq[:, 1:]  # (batch, seq_len - 1)
        preds: torch.Tensor = logits[:, :-1, :]  # (batch, seq_len - 1, vocab_size)

        if positions is not None:
            # Compute loss only at the specified positions
            # Adjust positions for the shift (position i in encoded seq predicts i+1)
            # So to get loss at position p, we need pred[p-1] predicting label[p-1]
            pos_indices: List[int] = [p - 1 for p in positions if 0 < p < len(sequence)]
            if not pos_indices:
                return torch.tensor(float('nan'), device=device)

            pos_preds: torch.Tensor = preds[0, pos_indices, :]  # (num_pos, vocab_size)
            pos_labels: torch.Tensor = labels[0, pos_indices]  # (num_pos,)
            loss: torch.Tensor = torch.nn.functional.cross_entropy(pos_preds, pos_labels)
        else:
            # Compute global mean loss
            loss = torch.nn.functional.cross_entropy(
                preds.reshape(-1, preds.shape[-1]),
                labels.reshape(-1),
                ignore_index=0,  # pad_id
            )
    return loss


def load_fasta_sequences(fasta_path: str) -> Tuple[List[str], List[str]]:
    """Load sequences from a FASTA file.

    Returns a tuple of (ids, sequences).
    """
    seq_ids: List[str] = []
    sequences: List[str] = []
    with open(fasta_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_ids.append(record.id)
            sequences.append(str(record.seq))
    return seq_ids, sequences


def load_gff_annotations(gff_path: str) -> Tuple[List[int], List[int], List[int]]:
    """Load CDS annotations (start,end,strand) from a GFF3 file.

    Returns three lists: starts, ends, strands.
    """
    limit_info = dict(gff_type=["CDS"])  # limit to CDS entries
    starts: List[int] = []
    ends: List[int] = []
    strands: List[int] = []
    with open(gff_path) as in_handle:
        for rec in GFF.parse(in_handle, limit_info=limit_info):
            for feature in rec.features:
                starts.append(int(feature.location.start))
                ends.append(int(feature.location.end))
                # some gff libraries encode strand as +1 or -1, so we keep that
                strands.append(int(feature.location.strand))
    return starts, ends, strands


def mutate_start_codon(encoded_sequence: List[int], positions: range) -> List[int]:
    """Mutate the start codon nucleotides to randomly chosen nucleotides.

    This mutates the positions in-place of the encoded_sequence. We pick random nucleotides
    from indices 1..4 which correspond to A,T,C,G in the provided vocabulary.
    """
    mutated = list(encoded_sequence)
    for pos in positions:
        if pos < 0 or pos >= len(mutated):
            continue
        mutated[pos] = random.choice([1, 2, 3, 4])
    return mutated


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mutate start codons and measure model loss")
    parser.add_argument("--fasta", default="sequence (1).fasta", help="FASTA file containing the sequences (default: trimmed to 96k bp)")
    parser.add_argument("--gff", default="sequence (1).gff3", help="GFF3 file containing CDS annotations (default: trimmed to 96k bp)")
    parser.add_argument("--model", default="../external/megaDNA_phage_145M.pt", help="PyTorch model file")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], nargs="?",
                        help="Force device: 'cpu' or 'cuda'; default: auto-detect")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--seq-id", type=int, default=0, help="Which record index to analyze from fasta")
    args = parser.parse_args(argv)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model} to device {device}")
    model = torch.load(args.model, map_location=torch.device(device), weights_only=False)
    model.eval()

    print(f"Loading sequences from {args.fasta}")
    seq_ids, sequences = load_fasta_sequences(args.fasta)
    if len(sequences) == 0:
        raise SystemExit("No sequences found in fasta file")

    print(f"Loading gene annotations from {args.gff}")
    start_positions, end_positions, strand_positions = load_gff_annotations(args.gff)

    # Trim sequences to 96k max model input length
    max_length: int = 96000
    original_length: int = len(sequences[args.seq_id])
    if original_length > max_length:
        print(f"Trimming sequence from {original_length} to {max_length} bp")
        sequences[args.seq_id] = sequences[args.seq_id][:max_length]

        # Filter GFF3 annotations to only include features within the trimmed region
        filtered_starts: List[int] = []
        filtered_ends: List[int] = []
        filtered_strands: List[int] = []

        for start, end, strand in zip(start_positions, end_positions, strand_positions):
            # Keep only CDS features that are entirely within the trimmed region
            if start < max_length and end <= max_length:
                filtered_starts.append(start)
                filtered_ends.append(end)
                filtered_strands.append(strand)

        print(f"Filtered GFF3 annotations: {len(start_positions)} -> {len(filtered_starts)} features")
        start_positions, end_positions, strand_positions = filtered_starts, filtered_ends, filtered_strands

    random.seed(args.seed)

    # encode the WT sequence
    encoded_wt_sequence: List[int] = encode_sequence(sequences[args.seq_id])
    print("WT encoded length:", len(encoded_wt_sequence))

    wt_loss: torch.Tensor = get_loss_for_sequence(model, encoded_wt_sequence, device)
    print("WT global loss:", wt_loss.item())

    print("Mutating start codons and computing losses at mutated positions...")
    wt_losses_at_positions: List[torch.Tensor] = []
    mutant_losses_at_positions: List[torch.Tensor] = []

    for j, (start, end, strand) in enumerate(zip(start_positions, end_positions, strand_positions)):
        # GFF3 uses 1-based coordinates, but Biopython converts them to 0-based half-open intervals.
        # encode_sequence() adds a start token at position 0, so we add +1 to map sequence positions to encoded positions.
        if strand == 1:
            positions: range = range(start + 1, start + 4)
        else:
            # negative strand; mutate the reverse start codon range
            positions = range(end - 2, end + 1)

        # Compute WT loss at these specific positions
        wt_loss_at_pos: torch.Tensor = get_loss_for_sequence(
            model, encoded_wt_sequence, device, positions=positions
        )
        wt_losses_at_positions.append(wt_loss_at_pos)

        # Extract WT nucleotides at positions
        wt_nucs: List[int] = [encoded_wt_sequence[p] for p in positions]

        # Create mutated sequence
        mutated: List[int] = mutate_start_codon(list(encoded_wt_sequence), positions)

        # Extract mutated nucleotides at positions
        mut_nucs: List[int] = [mutated[p] for p in positions]

        # Compute mutant loss at the same positions
        mutant_loss_at_pos: torch.Tensor = get_loss_for_sequence(
            model, mutated, device, positions=positions
        )
        mutant_losses_at_positions.append(mutant_loss_at_pos)

        # Convert indices to nucleotide strings
        wt_str: str = "".join(NT_VOCAB[n] for n in wt_nucs)
        mut_str: str = "".join(NT_VOCAB[n] for n in mut_nucs)

        if j < 5:
            print(f"  Gene {j}: positions {list(positions)}")
            print(f"    WT:  {wt_str}")
            print(f"    Mut: {mut_str}")
            print(
                f"    WT loss={wt_loss_at_pos.item():.4f}, "
                f"Mutant loss={mutant_loss_at_pos.item():.4f}, "
                f"Delta={mutant_loss_at_pos.item() - wt_loss_at_pos.item():.4f}"
            )
            print()

    print(f"\nProcessed {len(wt_losses_at_positions)} genes")
    wt_mean: float = sum(l.item() for l in wt_losses_at_positions) / len(wt_losses_at_positions)
    mut_mean: float = sum(l.item() for l in mutant_losses_at_positions) / len(mutant_losses_at_positions)
    print(f"Mean WT loss at start codons: {wt_mean:.4f}")
    print(f"Mean Mutant loss at start codons: {mut_mean:.4f}")


if __name__ == "__main__":
    main()
