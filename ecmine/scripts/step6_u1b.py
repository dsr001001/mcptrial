"""Step 6, candidate U1, second pass: the Tamagawa exponent instead of the Tamagawa number.

For a multiplicative prime q the mod l^t representation is unramified at q exactly when l^t divides
n_q = v_q(Delta_min) (the order of the geometric component group), whether the reduction is split or
not. So the quantity the theorems (Ribet-Takahashi, Pollack-Weston / Kim-Ota) speak about is
  S_l = sum over multiplicative q of v_l(n_q),
not the valuation of the Tamagawa number. Additive primes have Tamagawa exponent 0 for odd l.

Per optimal curve and odd prime l we record:
  V = v_l(deg phi), S = sum_mult v_l(n_q), S_even = best even subset (Ribet-Takahashi), n_mult,
  A3 = number of additive primes with 3 | c_q (types IV, IV*; only for l = 3), eis = rational l-isogeny,
  sqf = N squarefree, l_div_N = l | N, cm.
Usage: python3 scripts/step6_u1b.py [--holdout]
"""
import re, sys, os, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from load import load_curves, HOLDOUT_FROM, NMAX

ap = argparse.ArgumentParser(); ap.add_argument("--holdout", action="store_true"); args = ap.parse_args()
tag = "holdout" if args.holdout else "work"
cols = ["N", "label", "optimal", "moddeg", "tamagawa_list", "kodaira", "max_isogeny_degree", "is_cm", "n_split_mult", "n_nonsplit_mult", "n_additive", "rank"]
df = load_curves(columns=cols, include_holdout=True)
df = df[(df.N > HOLDOUT_FROM) & (df.N <= NMAX)] if args.holdout else df[df.N <= HOLDOUT_FROM]
df = df[df.optimal == 1].reset_index(drop=True)
print(f"{tag}: {len(df)} optimal curves", flush=True)

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
def vl(x, l):
    v = 0
    while x > 0 and x % l == 0: x //= l; v += 1
    return v

rows = []
for r in df.itertuples(index=False):
    primes = factor(int(r.N)); ks = list(map(int, re.findall(r"-?\d+", r.kodaira))); cs = list(map(int, re.findall(r"\d+", r.tamagawa_list)))
    assert len(primes) == len(ks) == len(cs), r.label
    for (q, e), k, c in zip(primes, ks, cs):
        mult = k >= 5
        rows.append((r.label, q, e, k, c, mult, k - 4 if mult else 0))
loc = pd.DataFrame(rows, columns=["label", "q", "e", "kod", "c", "mult", "n"])
L = sorted({p for n in loc.n.unique() if n > 0 for p, _ in factor(int(n))} - {2})
print("odd primes dividing some n_q:", L, flush=True)
deg = df.set_index("label")["moddeg"].astype("int64"); mid = df.set_index("label")["max_isogeny_degree"].astype("int64")
Nn = df.set_index("label")["N"].astype("int64"); cm = df.set_index("label")["is_cm"].astype(bool)
sqf = pd.Series([all(e == 1 for _, e in factor(int(n))) for n in Nn], index=Nn.index)
out = []
for l in L:
    loc["v"] = loc.n.map(lambda n: vl(int(n), l)).astype(int)
    loc["vc"] = loc.c.map(lambda c: vl(int(c), l)).astype(int)
    m = loc.mult.to_numpy()
    t = loc.assign(vm=np.where(m, loc.v, 0), vmin=np.where(m, loc.v, 10 ** 6), ism=m.astype(int),
                   a3=((~m) & (loc.vc > 0)).astype(int), vcm=np.where(m, loc.vc, 0))
    g = t.groupby("label")
    d = pd.DataFrame({"S": g.vm.sum(), "min_v": g.vmin.min(), "n_mult": g.ism.sum(), "A": g.a3.sum(), "S_tam": g.vcm.sum(), "max_v": g.vm.max()})
    d["S_even"] = d.S - np.where((d.n_mult % 2 == 1), d.min_v, 0)
    idx = d.index
    d["V"] = [vl(int(x), l) for x in deg.reindex(idx).to_numpy()]
    d["eis"] = (mid.reindex(idx).to_numpy() % l == 0); d["cm"] = cm.reindex(idx).to_numpy()
    d["sqf"] = sqf.reindex(idx).to_numpy(); d["l_div_N"] = (Nn.reindex(idx).to_numpy() % l == 0); d["l"] = l
    out.append(d.reset_index()); print(f"  l={l} done", flush=True)
res = pd.concat(out, ignore_index=True)
res.to_parquet(f"results/step6_u1b_{tag}.parquet", index=False)

def check(d, name):
    nz = d[d.S > 0]
    print(f"\n=== {name}: curves {len(d)}, with S > 0: {len(nz)}")
    for cond, desc in [("V >= S_even", "Ribet-Takahashi (even subset)"), ("V >= S", "all multiplicative primes"), ("V >= S - 1", "all multiplicative primes minus one")]:
        viol = nz[~nz.eval(cond)]
        print(f"  {desc:40s} [{cond:12s}] violations {len(viol):7d} of {len(nz):8d}" + (f"  e.g. {viol.label.head(8).tolist()}" if len(viol) else ""))
for l in L:
    d = res[res.l == l]
    if (d.S > 0).sum() == 0: continue
    print(f"\n######## l = {l}: curves with S > 0: {(d.S > 0).sum()}, Eisenstein among them {(d.eis & (d.S > 0)).sum()}, CM {(d.cm & (d.S > 0)).sum()}")
    ne = d[~d.eis & ~d.cm]
    check(ne, f"l={l} non-Eisenstein non-CM, all N")
    check(ne[ne.sqf & ~ne.l_div_N], f"l={l} non-Eisenstein, N squarefree, l not dividing N (theorem range for l>=5)")
    check(ne[~ne.sqf], f"l={l} non-Eisenstein, N not squarefree")
    check(ne[ne.l_div_N], f"l={l} non-Eisenstein, l divides N")
    check(ne[ne.n_mult == 1], f"l={l} non-Eisenstein, exactly one multiplicative prime")
    if (d.eis & (d.S > 0)).sum(): check(d[d.eis], f"l={l} Eisenstein (rational {l}-isogeny)")
for l in (3, 5, 7):
    d = res[(res.l == l) & ~res.eis & ~res.cm]
    print(f"\n--- l={l} non-Eisenstein non-CM: rows V = v_{l}(deg), cols S = sum_mult v_{l}(n_q)")
    print(pd.crosstab(d.V.clip(upper=9), d.S.clip(upper=6)).to_string())
    e = res[(res.l == l) & res.eis & (res.S > 0)]
    print(f"\n--- l={l} Eisenstein with S > 0: rows V, cols S")
    print(pd.crosstab(e.V.clip(upper=9), e.S.clip(upper=6)).to_string())
d = res[(res.l == 3) & ~res.eis & ~res.cm & (res.A > 0)]
print("\n--- l=3 non-Eisenstein non-CM with additive primes of type IV or IV* (3 | c_q): rows V - S (excess), cols A = number of such primes")
print(pd.crosstab((d.V - d.S).clip(lower=-2, upper=6), d.A).to_string())
d0 = d[d.S == 0]
print("    ... restricted to S = 0 (no multiplicative contribution): rows V, cols A")
print(pd.crosstab(d0.V.clip(upper=6), d0.A).to_string())
