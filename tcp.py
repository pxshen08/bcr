import numpy as np
import tensorflow as tf
import gpflow
def max_len(seq_list):
    """Returns the maximum sequence length within the given list of lists."""
    lmax=0
    for seq in seq_list:
        lmax=max( lmax, len(seq) )
    return lmax
def add_gap(tcr,l_max,gap_char='-'):
    """Add gap to given TCR. Returned tcr will have length l_max.
    If there is an odd number of letters in the sequence, one more letter is placed in the beginning."""
    l = len(tcr)
    if l<l_max:
        i_gap=np.int32(np.ceil(l/2))
        tcr = tcr[0:i_gap] + (gap_char*(l_max-l))+tcr[i_gap:l]
    return tcr
def align_gap(tcrs, l_max=None, gap_char='-'):
    """Align sequences by introducing a gap in the middle of the sequence.
    If there is an odd number of letters in the sequence, one more letter is placed in the beginning."""
    if l_max == None:
        l_max = max_len(tcrs)
    else:
        assert (l_max >= max_len(
            tcrs)), "Given max length must be greater than or equal to the max lenght of given sequences, " + str(
            max_len(tcrs))

    tcrs_aligned = []
    for tcr in tcrs:
        tcrs_aligned.append(add_gap(tcr, l_max, gap_char))
    return tcrs_aligned
