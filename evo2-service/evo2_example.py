"""Evo2 usage examples.

Demonstrates:
1. Forward pass - score a DNA sequence and get per-token logits
2. Embedding extraction - get intermediate layer representations
3. DNA generation - generate novel DNA from a prompt sequence
"""

import torch
from evo2 import Evo2

MODEL_NAME = "evo2_7b"

print(f"Loading {MODEL_NAME}...")
model = Evo2(MODEL_NAME)

# ---------- Example 1: Forward Pass (Sequence Scoring) ----------
print("\n=== Example 1: Forward Pass ===")

sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
input_ids = torch.tensor(
    model.tokenizer.tokenize(sequence),
    dtype=torch.int,
).unsqueeze(0).to("cuda:0")

outputs, _ = model(input_ids)
logits = outputs[0]

print(f"Sequence length: {len(sequence)}")
print(f"Logits shape (batch, length, vocab): {logits.shape}")
print(f"Logits sample (first 5 positions):\n{logits[0, :5, :]}")

# ---------- Example 2: Embedding Extraction ----------
print("\n=== Example 2: Embedding Extraction ===")

# Intermediate layers give better representations than the final layer
layer_name = "blocks.28.mlp.l3"

outputs, embeddings = model(
    input_ids,
    return_embeddings=True,
    layer_names=[layer_name],
)

emb = embeddings[layer_name]
print(f"Embedding layer: {layer_name}")
print(f"Embedding shape: {emb.shape}")
print(f"Embedding mean: {emb.float().mean().item():.6f}")
print(f"Embedding std:  {emb.float().std().item():.6f}")

# ---------- Example 3: DNA Sequence Generation ----------
print("\n=== Example 3: DNA Generation ===")

prompt = "ATGCGATCGATCGATCGATCG"
generated = model.generate(
    prompt_seqs=[prompt],
    n_tokens=200,
    temperature=1.0,
    top_k=4,
)

full_seq = generated.sequences[0]
new_bases = full_seq[len(prompt):]

print(f"Prompt:    {prompt}")
print(f"Generated: {new_bases[:80]}...")
print(f"Total length: {len(full_seq)} bases")

# ---------- Example 4: Batch Scoring ----------
print("\n=== Example 4: Batch Comparison ===")

seq_a = "ATGAAAGCAATTTTCGTACTGAAACATCTTAATCATGC"  # typical start codon context
seq_b = "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"  # low-complexity repeat

for label, seq in [("Coding-like", seq_a), ("Low-complexity", seq_b)]:
    ids = torch.tensor(
        model.tokenizer.tokenize(seq), dtype=torch.int
    ).unsqueeze(0).to("cuda:0")
    out, _ = model(ids)
    log_probs = torch.log_softmax(out[0], dim=-1)
    # Average log-probability of each next token (a rough sequence "naturalness" score)
    mean_lp = log_probs[0, :-1].max(dim=-1).values.mean().item()
    print(f"  {label:15s} | mean max log-prob: {mean_lp:.4f}")

print("\nDone!")
