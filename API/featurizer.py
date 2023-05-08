import numpy as np
import torch

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
A2I = {a: i for i, a in enumerate(ALPHABET)}


def shuffle_subset(n, p):
    n_shuffle = np.random.binomial(n, p)
    ix = np.arange(n)
    ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
    ix_subset_shuffled = np.copy(ix_subset)
    np.random.shuffle(ix_subset_shuffled)
    ix[ix_subset] = ix_subset_shuffled
    return ix


def featurize_GTrans(batch: list, shuffle_fraction: float = 0.0) -> list:
    """Pack and pad batch into torch tensors. Output:
    * X: [B, L_max, C=4, 3] atom coordinates
    * S: [B, L_max] sequence labels
    * score: [B, L_max] ??? seems useless
    * mask: [B, L_max] present AA mask (even partially unresolved AAs are discarded)
    * lengths: [B,]
    """
    B = len(batch)
    lengths = np.array([len(b["seq"]) for b in batch], dtype=np.int32)
    L_max = max([len(b["seq"]) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    score = np.ones([B, L_max]) * 100.0

    # Build the batch
    for i, b in enumerate(batch):
        l = len(b["seq"])
        x = np.stack([b[c] for c in ["N", "CA", "C", "O"]], 1)  # [L, C=4 , 3]

        x_pad = np.pad(
            x, [[0, L_max - l], [0, 0], [0, 0]], "constant", constant_values=(np.nan,)
        )  # [#atom, 4, 3]
        X[i, ...] = x_pad

        # Convert to labels
        indices = np.asarray(list(map(A2I.get, b["seq"])), dtype=np.int32)
        if shuffle_fraction > 0.0:
            idx_shuffle = shuffle_subset(l, shuffle_fraction)
            S[i, :l] = indices[idx_shuffle]
        else:
            S[i, :l] = indices

    # TODO: hypnopump@ this mask sets partially unresolved AAs to 0
    # TODO: hypnopump@ consider passing positions of correct atoms + consider
    # TODO: hypnopump@ passing unresolved masks to the model (meaningless geometrical feats)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)  # [B, L_max] atom mask
    numbers = np.sum(mask, axis=1).astype(int)  # [B,] num atoms per seq
    S_new = np.zeros_like(S)
    X_new = np.full_like(X, fill_value=np.nan)
    # Group valid nodes to first N positions of graph
    # FIXME: hypnopump@ wtf! dihedrals of non-contiguous AAs dont have meaning!
    for i, n in enumerate(numbers):
        X_new[i, :n] = X[i, mask[i] == 1]
        S_new[i, :n] = S[i, mask[i] == 1]

    X, S = X_new, S_new
    isnan = np.isnan(X)
    # TODO: hypnopump@ see above for details on unresolved atoms
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    return X, S, score, mask, lengths
