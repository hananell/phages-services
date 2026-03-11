from megaDNA_mutagenesis import encode_sequence, mutate_start_codon


def test_encode_sequence():
    seq = "ATCG"
    enc = encode_sequence(seq)
    # expected: [0] + indices for A,T,C,G + [5]
    assert enc[0] == 0
    assert enc[-1] == 5
    assert len(enc) == len(seq) + 2


def test_mutate_start_codon():
    # build a simple encoded sequence with placeholders
    enc = [0, 1, 2, 3, 4, 5]
    # mutate positions 2-3
    mutated = mutate_start_codon(enc, range(2, 4))
    # ensure positions were changed to values between 1 and 4
    for i in range(2, 4):
        assert 1 <= mutated[i] <= 4
