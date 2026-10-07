"""Loaders that apply the agreed restrictions.

Scope for the first search (agreed 2026-10-04): conductor N <= 400000, where every column
is determined. Classes with N in (300000, 400000] are the step 8 hold-out and are excluded
by default from the working set.
"""
import hashlib
import numpy as np
import pandas as pd

NMAX = 400000          # scope of the first search
HOLDOUT_FROM = 300000  # classes with N > HOLDOUT_FROM are the step 8 hold-out
DATA = "data"


def _restrict(df, nmax, include_holdout):
    df = df[df["N"] <= nmax]
    if not include_holdout:
        df = df[df["N"] <= HOLDOUT_FROM]
    return df.reset_index(drop=True)


def load_curves(columns=None, nmax=NMAX, include_holdout=False):
    df = pd.read_parquet(f"{DATA}/curves.parquet", columns=columns)
    return _restrict(df, nmax, include_holdout)


def load_classes(columns=None, nmax=NMAX, include_holdout=False):
    ap = pd.read_parquet(f"{DATA}/ap.parquet", columns=columns)
    return _restrict(ap, nmax, include_holdout)


def fold(class_labels, k=5):
    """Deterministic fold assignment by class label, so curves of one class share a fold."""
    def h(s):
        return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % k
    return np.array([h(s) for s in class_labels])


if __name__ == "__main__":
    c = load_classes(columns=["N", "class_label", "rank"])
    print("working-set classes (N <= %d):" % HOLDOUT_FROM, len(c))
    h = load_classes(columns=["N", "class_label"], include_holdout=True)
    print("hold-out classes (%d < N <= %d):" % (HOLDOUT_FROM, NMAX), (h["N"] > HOLDOUT_FROM).sum())
    f = fold(c["class_label"])
    print("fold sizes:", np.bincount(f).tolist())
