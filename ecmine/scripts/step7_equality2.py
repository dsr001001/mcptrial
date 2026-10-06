"""Sharpness of the bounds (eq:main = Conjecture 1.1, eq:twist = Conjecture 1.3) from the checker's per-curve files,
and the odd part of the modular degree against the product of the bounds. Usage: python3 scripts/step7_equality2.py [work|holdout|ext]"""
import sys, numpy as np, pandas as pd
from collections import Counter
tag = sys.argv[1] if len(sys.argv) > 1 else "work"
per = pd.read_parquet(f"results/check_{tag}_perl.parquet", columns=["label", "l", "V", "bound1", "bound2", "eis", "cm"], filters=[("bound2", ">", 0)])
opt = pd.read_parquet(f"results/optimal_{tag}.parquet", columns=["label", "N", "moddeg", "max_isogeny_degree", "is_cm"]).set_index("label")
ne = per[~per.eis & ~per.cm & (per.l != 2)]
print(f"{tag}: equality rates among non-Eisenstein non-CM curves with a positive bound")
for l in (3, 5, 7, 11, 13):
    s = ne[ne.l == l]
    for name, b in (("eq:main ", "bound1"), ("eq:twist", "bound2")):
        t = s[s[b] > 0]; ex = t.V - t[b]
        print(f"  l={l:2d} {name}: bound>0 {len(t):8d}  equality {100*(ex == 0).mean():5.1f}%  excess 1: {100*(ex == 1).mean():5.1f}%  excess>=2: {100*(ex >= 2).mean():5.1f}%")
    if l == 3:
        t = s[s.bound2 > 0].join(opt.N, on="label")
        t = t.assign(bin=pd.cut(t.N, [0, 1000, 10000, 100000, 300000, 400000, 500000]))
        print("  equality rate for eq:twist at l=3 by conductor range:")
        for b, g in t.groupby("bin", observed=True): print(f"     {str(b):18s} {100*(g.V == g.bound2).mean():5.1f}% of {len(g)}")
mid = opt.max_isogeny_degree.astype("int64")
noodd = opt.index[(~opt.is_cm.astype(bool)) & ((mid & (mid - 1)) == 0)]          # isogeny degrees are powers of 2
p2 = ne[ne.label.isin(noodd)]
prod = p2.groupby("label").apply(lambda g: int(np.prod([int(l) ** int(b) for l, b in zip(g.l, g.bound2)], dtype=object))).reindex(noodd).fillna(1).astype(object)
deg = opt.moddeg.reindex(noodd).astype("int64")
oddpart = deg.map(lambda d: int(d) >> ((int(d) & -int(d)).bit_length() - 1))
quot = pd.Series([int(o) // int(p) for o, p in zip(oddpart, prod)], index=noodd)
assert all(int(o) % int(p) == 0 for o, p in zip(oddpart, prod)), "a bound exceeds the degree"
eq = (quot == 1)
print(f"\ncurves with no odd isogeny degree: {len(noodd)}; odd part of deg phi equals prod l^bound(eq:twist): {int(eq.sum())} ({100*eq.mean():.2f}%)")
print("most common quotients:", Counter(quot).most_common(10))
def spf(n):
    if n == 1: return None
    p = 3
    while p * p <= n:
        if n % p == 0: return p
        p += 2
    return n
s = quot[quot > 1].map(spf); Ns = opt.N.reindex(s.index)
print(f"quotient > 1: {len(s)}; smallest prime factor 3: {100*(s == 3).mean():.1f}%, 5: {100*(s == 5).mean():.1f}%, 7: {100*(s == 7).mean():.1f}%; divides N: {100*((Ns % s) == 0).mean():.1f}%")
