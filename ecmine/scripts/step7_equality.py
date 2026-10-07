"""Step 4 of the follow-up list: equality cases and what else is in the modular degree.
Reads results/check_<set>_perl.parquet (per curve and prime: V, bound1, bound2, eis, cm) and results/optimal_<set>.parquet."""
import sys, math
import numpy as np, pandas as pd
tag = sys.argv[1] if len(sys.argv) > 1 else "work"
per = pd.read_parquet(f"results/check_{tag}_perl.parquet", columns=["label", "l", "V", "bound1", "bound2", "eis", "cm"]); per = per[per.l.isin([3, 5, 7, 11, 13, 17, 19, 23])]; opt = pd.read_parquet(f"results/optimal_{tag}.parquet").set_index("label")
def vl(x, l):
    v = 0; x = int(x)
    while x > 0 and x % l == 0: x //= l; v += 1
    return v
print(f"=== {tag}: equality cases, non-Eisenstein non-CM curves")
for l in (3, 5, 7):
    d = per[(per.l == l) & ~per.eis & ~per.cm].copy(); b = d.bound2 if l == 3 else d.bound1
    d["excess"] = d.V - b; dz = d[b > 0]
    print(f"\n--- l = {l}: curves with a positive bound: {len(dz)}; excess = V - bound distribution:")
    print(dz.excess.clip(upper=6).value_counts().sort_index().to_string())
    print(f"    exact (excess 0): {(dz.excess == 0).mean():.3f}")
    nb = opt.reindex(dz.index if dz.index.name == 'label' else dz.label).N
    dz = dz.assign(N=nb.to_numpy()); dz["logNbin"] = (np.log10(dz.N) * 2).astype(int) / 2
    print("    P(excess = 0) by log10 N (half-decades):"); print(dz.groupby("logNbin").excess.apply(lambda s: round((s == 0).mean(), 3)).to_string())
# odd part exactness: curves with no rational isogeny of odd degree, not CM
w = per[~per.cm].copy()
md = opt.max_isogeny_degree.astype("int64")
noodd = md[((md & (md - 1)) == 0)].index           # power of 2 (including 1): no odd-degree isogeny
w = w[w.label.isin(noodd) & (w.l % 2 == 1)]
w["b"] = np.where(w.l == 3, w.bound2, w.bound1); w["lb"] = w.b * np.log(w.l)
pred = np.exp(w.groupby("label").lb.sum()).round().astype("int64")
deg = opt.moddeg.astype("int64").reindex(pred.index)
odd = deg.map(lambda x: x // (x & -x))
R = (odd // pred)
assert (odd % pred == 0).all()
print(f"\n=== odd part of deg phi versus the conjectured product P = prod l^bound (curves with no odd-degree isogeny, not CM): {len(R)} curves")
print(f"    odd part equals P exactly: {(R == 1).mean():.3f}   ratio R = oddpart/P distribution (top 12):")
print(R.value_counts().head(12).to_string())
Nn = opt.N.reindex(R.index).astype("int64")
def smallest_prime(x):
    x = int(x); p = 3
    while p * p <= x:
        if x % p == 0: return p
        p += 2
    return x
r1 = R[R > 1]
sp = r1.map(smallest_prime); divN = pd.Series([ (int(n) % int(p) == 0) for n, p in zip(Nn.reindex(r1.index), sp)], index=r1.index)
print(f"    among R > 1: smallest prime factor of R divides N in {divN.mean():.3f} of cases; smallest prime factor distribution:")
print(sp.value_counts().head(8).to_string())
nb = pd.Series([len([p for p in (2,3,5,7,11,13) if int(n) % p == 0]) for n in Nn], index=R.index)
print("    P(R = 1) by number of bad primes among 2,3,5,7,11,13 (proxy):"); print(pd.Series(R.to_numpy() == 1, index=R.index).groupby(nb).mean().round(3).to_string())
