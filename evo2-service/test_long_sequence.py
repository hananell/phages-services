"""Test evo2-service with a long phage sequence using the updated layer names."""
import base64
import time

import httpx
import numpy as np
from Bio import SeqIO

FASTA_PATH = "/home/user/python_projects/phages_dataset/data/373/GCA_024674815.1_4104.fasta"
SERVICE_URL = "http://localhost:8001"
LAYER_NAMES = ["blocks.28.mlp.l3", "blocks.31.mlp.l3"]

record = next(SeqIO.parse(FASTA_PATH, "fasta"))
sequence = str(record.seq).upper()
print(f"Sequence ID  : {record.id}")
print(f"Sequence len : {len(sequence):,} bp")

print(f"\nSending to {SERVICE_URL}/embed/batch with layers: {LAYER_NAMES}")
t0 = time.time()
resp = httpx.post(
    f"{SERVICE_URL}/embed/batch",
    json={"sequences": [sequence], "layer_names": LAYER_NAMES},
    timeout=600.0,
)
elapsed = time.time() - t0

print(f"Status       : {resp.status_code}  ({elapsed:.1f}s)")
if resp.status_code != 200:
    print(f"Error: {resp.text[:500]}")
    raise SystemExit(1)

data = resp.json()
result = data["results"][0]
print(f"Seq length in response: {result['sequence_length']:,}")

for layer in LAYER_NAMES:
    arr = np.frombuffer(base64.b64decode(result["embeddings"][layer]), dtype=np.float32)
    print(f"Layer {layer}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")

print("\nTest passed.")
