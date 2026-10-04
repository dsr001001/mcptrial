"""Build the per-bad-prime table for optimal curves (work and hold-out sets): one row per (curve, q | N).
Columns: label, q, e = v_q(N), kod (PARI code), c = Tamagawa number, mult, n = v_q(Delta) for multiplicative q,
type = Kodaira type name, defect = semistability defect for potentially good reduction (12 / gcd(v_q(Delta), 12)), qmod3."""
import re, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from load import load_curves, HOLDOUT_FROM, NMAX

cols = ["N", "label", "optimal", "kodaira", "tamagawa_list", "log_abs_disc"]
df = load_curves(columns=cols, include_holdout=True); df = df[df.optimal == 1].reset_index(drop=True)
M = NMAX + 1; spf = np.arange(M)
for i in range(2, int(M ** 0.5) + 1):
    if spf[i] == i:
        sl = spf[i * i::i]; sl[sl == np.arange(i * i, M, i)] = i
def factor(n):
    out = []
    while n > 1:
        p = int(spf[n]); e = 0
        while n % p == 0: n //= p; e += 1
        out.append((p, e))
    return out
def kname(k):
    if k == 1: return "I0"
    if k == 2: return "II"
    if k == 3: return "III"
    if k == 4: return "IV"
    if k >= 5: return "In"
    if k == -1: return "I0*"
    if k == -2: return "II*"
    if k == -3: return "III*"
    if k == -4: return "IV*"
    return "In*"
rows = []
for r in df.itertuples(index=False):
    primes = factor(int(r.N)); ks = list(map(int, re.findall(r"-?\d+", r.kodaira))); cs = list(map(int, re.findall(r"\d+", r.tamagawa_list)))
    for (q, e), k, c in zip(primes, ks, cs):
        rows.append((r.label, int(r.N), q, e, k, c, k >= 5, k - 4 if k >= 5 else (-k - 4 if k <= -5 else 0), kname(k), q % 3))
loc = pd.DataFrame(rows, columns=["label", "N", "q", "e", "kod", "c", "mult", "n", "type", "qmod3"])
loc["set"] = np.where(loc.N > HOLDOUT_FROM, "holdout", "work")
loc.to_parquet("results/local_primes.parquet", index=False)
print(loc.groupby("set").size().to_dict()); print(loc.type.value_counts().to_dict())
