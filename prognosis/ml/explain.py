import numpy as np
import pandas as pd


def shapley(predict_fn, x: pd.Series, background: pd.DataFrame, n_perm: int = 40, seed: int = 0):
    """Shapley values for one row by sampling feature orderings (interventional, background rows as the reference).

    Each sample takes a random background row and a random ordering, then swaps features over to the
    patient's values one at a time. The change in predicted probability at each swap is credited to that
    feature. Every ordering is also walked in reverse (antithetic pairs) to cut the noise.
    The credits add up exactly to f(patient) minus the average prediction over the sampled background rows.
    """
    cols = list(background.columns)
    F = len(cols)
    rng = np.random.default_rng(seed)
    orders, b_idx = [], []
    for _ in range(n_perm):
        perm = rng.permutation(F)
        b = int(rng.integers(len(background)))
        for o in (perm, perm[::-1]):
            orders.append(o)
            b_idx.append(b)
    S = len(orders)
    # mask[s, t, c] is True when feature c already holds the patient's value at step t of sample s
    mask = np.zeros((S, F + 1, F), dtype=bool)
    for s, o in enumerate(orders):
        for t in range(1, F + 1):
            mask[s, t, o[:t]] = True
    B = background.iloc[np.repeat(b_idx, F + 1)].reset_index(drop=True)
    flat = mask.reshape(S * (F + 1), F)
    data = {}
    for j, c in enumerate(cols):
        xv = x[c]
        numeric = pd.api.types.is_numeric_dtype(background[c])
        col = B[c].to_numpy(dtype=float if numeric else object).copy()
        col[flat[:, j]] = xv
        data[c] = col
    frame = pd.DataFrame(data)
    for c in cols:
        if pd.api.types.is_numeric_dtype(background[c]):
            frame[c] = frame[c].astype(float)
        else:
            frame[c] = frame[c].astype(object)
    p = np.asarray(predict_fn(frame)).reshape(S, F + 1)
    phi = np.zeros(F)
    for s, o in enumerate(orders):
        deltas = np.diff(p[s])
        phi[o] += deltas
    phi /= S
    return phi, float(p[:, 0].mean()), float(p[:, -1].mean())
