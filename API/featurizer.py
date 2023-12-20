import numpy as np
import torch as th

from .frame import A2I, ALPHABET, Frame, aa_bb_positions, aa_bb_positions_mean


def encode_bb_af2(x: th.Tensor) -> Frame:
    """Inputs: (B, L, 4, D) -> Outputs: (B, L); 4 = N, CA, C, O"""
    frames = Frame.make_transform_from_reference(
        n_xyz=x[..., 0, :],
        ca_xyz=x[..., 1, :],
        c_xyz=x[..., 2, :],
        eps=1e-12,
    )
    return frames


def decode_bb_af2(frames: Frame, idxs: th.Tensor, mask: th.Tensor) -> th.Tensor:
    """Rebuilds the backbone for a given peptide sequence with AF2 frames.
    # TODO: check if AF2 standard params leak perfect information into pifold ???
    # FIXME: checked. yes it leaks and its separable in 1 epoch
    # TODO: fold all to same frame: mean
    Inputs:
    * frames: (B, L)
    * idxs: (B, L) aa idxs
    * mask: (B, L)
    Outputs: (B, L, 4, D); 4 = N, CA, C, O
    """
    # (B, L) -> (B, L, 4, D)
    # pos = aa_bb_positions[idxs]
    pos = aa_bb_positions_mean[idxs]
    # (B, L), (B, L, 4, D) -> (B, L, 4, D)
    x = frames[..., None].apply(pos)
    # (B, L, 4, D), (B, L) -> (B, L, 4, 3)
    x = x * mask.float()[..., None, None]
    return x


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
    S = th.from_numpy(S).to(dtype=th.long)
    score = th.from_numpy(score).float()
    X = th.from_numpy(X).to(dtype=th.float32)
    mask = th.from_numpy(mask).to(dtype=th.float32)

    # encode in AF2-like frames, then decode. AF2 has per-res params so might leak
    # noises af2 frames (orientations + global translations) a bit
    alpha = 0.035  # 0.0
    alpha = 0.0175
    # X: (b, l, 4, 3) -> (b, l, 4, 3)
    X = decode_bb_af2(encode_bb_af2(X + alpha * th.randn_like(X)), S, mask)

    # add noise to coordinates for robustness training
    # X = X + th.randn_like(X) * 0.1

    return X, S, score, mask, lengths
