"""Step 6, candidate U1: l-adic valuation of the modular degree against the l-adic valuations of the
Tamagawa numbers, per bad prime, for X_0(N)-optimal curves.

Quantities per optimal curve E and odd prime l:
  V      = v_l(deg phi_E)
  Tm     = sum over multiplicative q of v_l(c_q)
  Ta     = sum over additive q of v_l(c_q)
  RT     = the Ribet-Takahashi bound: the largest sum of v_l(c_q) over an even number of
           multiplicative primes (all of them if their number is even, else drop the smallest)
  eis    = E has a rational l-isogeny (mod-l representation reducible)
Usage: python3 scripts/step6_u1.py [--holdout]
"""
import re, sys, os, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from load import load_curves, HOLDOUT_FROM, NMAX

ap = argparse.ArgumentParser(); ap.add_argument("--holdout", action="store_true"); args = ap.parse_args()
cols = ["N", "label", "class_label", "optimal", "moddeg", "tamagawa_list", "kodaira", "max_isogeny_degree", "rank", "torsion", "is_cm", "n_split_mult", "n_nonsplit_mult", "n_additive"]
df = load_curves(columns=cols, include_holdout=True)
df = df[(df.N > HOLDOUT_FROM) & (df.N <= NMAX)] if args.holdout else df[df.N <= HOLDOUT_FROM]
df = df[df.optimal == 1].reset_index(drop=True)
print(f"{'hold-out' if args.holdout else 'working set'}: {len(df)} optimal curves", flush=True)

# smallest-prime-factor sieve for factoring N
M = NMAX + 1; spf = np.arange(M)
for i in range(2, int(M ** 0.5) + 1):
    if spf[i] == i: spf[i * i::i][spf[i * i::i] == np.arange(i * i, M, i)] = i
def factor(n):
    out = []
    while n > 1:
        p = spf[n]; e = 0
        while n % p == 0: n //= p; e += 1
        out.append((int(p), e))
    return out

def vl(x, l):
    v = 0
    while x % l == 0 and x > 0: x //= l; v += 1
    return v

rows = []
for r in df.itertuples(index=False):
    primes = factor(int(r.N)); ks = list(map(int, re.findall(r"-?\d+", r.kodaira))); cs = list(map(int, re.findall(r"\d+", r.tamagawa_list)))
    assert len(primes) == len(ks) == len(cs), (r.label, primes, ks, cs)
    for (q, e), k, c in zip(primes, ks, cs):
        rows.append((r.label, q, e, k, c, k >= 5))
loc = pd.DataFrame(rows, columns=["label", "q", "e", "kod", "c", "mult"])
print("bad-prime rows:", len(loc), flush=True)

# primes l dividing some Tamagawa number, odd
L = sorted({p for c in loc.c.unique() for p, _ in factor(int(c))} - {2})
print("odd primes dividing a Tamagawa number:", L, flush=True)

out = []
deg = df.set_index("label")["moddeg"].astype("int64"); mid = df.set_index("label")["max_isogeny_degree"].astype("int64")
nmult = (df.n_split_mult.astype(int) + df.n_nonsplit_mult.astype(int)).set_axis(df.label)
for l in L:
    loc["v"] = loc.c.map(lambda c: vl(int(c), l)).astype(int)
    m = loc.mult.to_numpy()
    t = loc.assign(vm=np.where(m, loc.v, 0), va=np.where(~m, loc.v, 0), vmin=np.where(m, loc.v, 10 ** 6),
                   nml=((loc.v > 0) & m).astype(int), nal=((loc.v > 0) & ~m).astype(int))
    g = t.groupby("label")
    d = pd.DataFrame({"Tm": g.vm.sum(), "Ta": g.va.sum(), "min_v_mult": g.vmin.min(), "n_mult_l": g.nml.sum(), "n_add_l": g.nal.sum(), "max_v": g.v.max()})
    d["n_mult"] = nmult.reindex(d.index).to_numpy()
    d["RT"] = d.Tm - np.where((d.n_mult % 2 == 1) & (d.n_mult > 0), d.min_v_mult, 0)
    d["V"] = [vl(int(x), l) for x in deg.reindex(d.index).to_numpy()]
    d["eis"] = (mid.reindex(d.index).to_numpy() % l == 0)
    d["l"] = l; d["T"] = d.Tm + d.Ta
    out.append(d.reset_index())
    print(f"  l={l} done", flush=True)
res = pd.concat(out, ignore_index=True)
res.to_parquet(f"results/step6_u1_{'holdout' if args.holdout else 'work'}.parquet", index=False)

def report(d, name):
    n = len(d); nz = d[d["T"] > 0]
    print(f"\n=== {name}: curves {n}, with l | some c_q: {len(nz)}")
    for cond, desc in [("V >= RT", "Ribet-Takahashi bound (even subset of multiplicative primes)"),
                       ("V >= Tm", "all multiplicative primes"),
                       ("V >= Tm + Ta", "all bad primes (multiplicative and additive)"),
                       ("V >= max_v", "largest single v_l(c_q)"),
                       ("V >= n_mult_l + n_add_l", "number of bad primes with l | c_q"),
                       ("V >= n_mult_l + n_add_l - 1", "number of bad primes with l | c_q, minus one")]:
        viol = nz[~nz.eval(cond)]
        print(f"  {desc:62s} [{cond:28s}] violations {len(viol):6d} of {len(nz):7d}" + (f"   e.g. {viol.label.head(5).tolist()}" if 0 < len(viol) <= 10**9 else ""))

for l in L:
    d = res[res.l == l]
    if (d["T"] > 0).sum() == 0: continue
    print(f"\n######## l = {l}: curves with l | some c_q: {(d['T'] > 0).sum()}  (Eisenstein among them: {(d.eis & (d['T'] > 0)).sum()})")
    report(d[~d.eis], f"l={l}, no rational {l}-isogeny")
    if (d.eis & (d["T"] > 0)).sum(): report(d[d.eis], f"l={l}, with a rational {l}-isogeny")

# cross-tabs for l = 3, 5, 7 (non-Eisenstein): V against T, and V against Ta for additive
for l in (3, 5, 7):
    d = res[(res.l == l) & (~res.eis)]
    print(f"\n--- l={l}, non-Eisenstein: rows V = v_{l}(deg), cols T = sum v_{l}(c_q) (clipped at 6)")
    print(pd.crosstab(d.V.clip(upper=8), d["T"].clip(upper=6)).to_string())
    if l == 3:
        print(f"\n--- l=3, non-Eisenstein, curves with Ta > 0 (additive primes with 3 | c_q): rows V, cols Ta")
        dd = d[d.Ta > 0]; print(pd.crosstab(dd.V.clip(upper=8), dd.Ta).to_string())
        print("    ... and with Tm = 0 (only additive contributions): rows V, cols Ta")
        dd = d[(d.Ta > 0) & (d.Tm == 0)]; print(pd.crosstab(dd.V.clip(upper=8), dd.Ta).to_string())
    print(f"\n--- l={l}, non-Eisenstein, exactly one multiplicative prime with {l} | c_q and no additive: rows V, cols Tm")
    dd = d[(d.n_mult_l == 1) & (d.Ta == 0)]; print(pd.crosstab(dd.V.clip(upper=8), dd.Tm.clip(upper=6)).to_string())
    print(f"\n--- l={l}, Eisenstein (rational {l}-isogeny), T > 0: rows V, cols T")
    dd = res[(res.l == l) & res.eis & (res["T"] > 0)]; print(pd.crosstab(dd.V.clip(upper=8), dd["T"].clip(upper=6)).to_string())
