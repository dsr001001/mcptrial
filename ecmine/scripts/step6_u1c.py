"""Step 6, U1, third pass: per-prime analysis of the inequality  v_l(deg phi) >= sum_{q || N} v_l(v_q(Delta)).

For each optimal curve, odd l, and multiplicative prime q with l | n_q = v_q(Delta), record q mod l.
Then compare V = v_l(deg phi) with
  S      = sum over all multiplicative q of v_l(n_q)
  S_ko   = the same sum restricted to q not congruent to +-1 mod l  (the Kim-Ota protected part)
and list violations by category: l = 3 / l >= 5, l | N or not, Eisenstein or not, N squarefree or not,
and whether every contributing prime q with v_l(n_q) > 0 satisfies q = +-1 mod l.
Usage: python3 scripts/step6_u1c.py [--holdout]
"""
import re, sys, os, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from load import load_curves, HOLDOUT_FROM, NMAX

ap = argparse.ArgumentParser(); ap.add_argument("--holdout", action="store_true"); args = ap.parse_args()
tag = "holdout" if args.holdout else "work"
cols = ["N", "label", "optimal", "moddeg", "kodaira", "max_isogeny_degree", "is_cm", "rank", "torsion", "class_size"]
df = load_curves(columns=cols, include_holdout=True)
df = df[(df.N > HOLDOUT_FROM) & (df.N <= NMAX)] if args.holdout else df[df.N <= HOLDOUT_FROM]
df = df[df.optimal == 1].reset_index(drop=True)
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
    primes = factor(int(r.N)); ks = list(map(int, re.findall(r"-?\d+", r.kodaira)))
    for (q, e), k in zip(primes, ks):
        if k >= 5: rows.append((r.label, q, k - 4))
loc = pd.DataFrame(rows, columns=["label", "q", "n"])
L = sorted({p for n in loc.n.unique() if n > 1 for p, _ in factor(int(n))} - {2})
info = df.set_index("label")
out = []
for l in L:
    t = loc[loc.n % l == 0].copy()
    if t.empty: continue
    t["v"] = t.n.map(lambda n: vl(int(n), l)); t["pm1"] = ((t.q % l == 1) | (t.q % l == l - 1))
    t["v_ko"] = np.where(t.pm1, 0, t.v)
    g = t.groupby("label")
    d = pd.DataFrame({"S": g.v.sum(), "S_ko": g.v_ko.sum(), "n_contrib": g.v.size(), "all_pm1": g.pm1.all(), "any_pm1": g.pm1.any(),
                      "primes": g.apply(lambda x: ";".join(f"{q}^{n}" for q, n in zip(x.q, x.n)), include_groups=False)})
    i = info.reindex(d.index)
    d["V"] = [vl(int(x), l) for x in i.moddeg.astype("int64")]
    d["eis"] = (i.max_isogeny_degree.astype("int64") % l == 0).to_numpy(); d["cm"] = i.is_cm.astype(bool).to_numpy()
    d["N"] = i.N.to_numpy(); d["l_div_N"] = (d.N % l == 0); d["sqf"] = [all(e == 1 for _, e in factor(int(n))) for n in d.N]
    d["rank"] = i["rank"].to_numpy(); d["torsion"] = i.torsion.to_numpy(); d["l"] = l
    out.append(d.reset_index())
res = pd.concat(out, ignore_index=True)
res.to_parquet(f"results/step6_u1c_{tag}.parquet", index=False)
print(f"{tag}: optimal curves {len(df)}; (curve, l) pairs with some multiplicative q having l | v_q(Delta): {len(res)}")

ne = res[~res.eis & ~res.cm]
print("\n=== Non-Eisenstein, non-CM pairs:", len(ne))
def show(d, name):
    v = d[d.V < d.S]; vk = d[d.V < d.S_ko]
    print(f"  {name:70s} pairs {len(d):7d}  V < S: {len(v):5d}   V < S_ko: {len(vk):5d}")
    return v
show(ne, "all")
show(ne[(ne.l >= 5) & ~ne.l_div_N], "l >= 5, l not dividing N   (Kim-Ota range: V >= S_ko is a theorem)")
show(ne[(ne.l >= 5) & ne.l_div_N], "l >= 5, l divides N")
show(ne[ne.l == 3], "l = 3")
show(ne[ne.sqf], "N squarefree"); show(ne[~ne.sqf], "N not squarefree")
v = ne[ne.V < ne.S]
print("\n--- all non-Eisenstein non-CM violations of V >= S, by (l, every contributing prime is +-1 mod l):")
print(pd.crosstab(v.l, v.all_pm1).to_string())
print("\n--- violations where NOT every contributing prime is +-1 mod l (these would contradict the conjecture even in its weakest form):")
w = v[~v.all_pm1]
print(w[["label", "l", "V", "S", "S_ko", "primes", "N", "rank", "torsion", "l_div_N", "sqf"]].head(40).to_string())
print("\n--- violations with every contributing prime +-1 mod l (first 40):")
w2 = v[v.all_pm1]
print(w2[["label", "l", "V", "S", "S_ko", "primes", "N", "rank", "torsion", "l_div_N", "sqf"]].head(40).to_string())
print("\n--- among pairs with a contributing prime q = +-1 mod l: how often does V >= S still hold?")
a = ne[ne.any_pm1]; print(pd.crosstab(a.l, a.V >= a.S).to_string())
print("\n=== Eisenstein pairs (rational l-isogeny):", int(res.eis.sum()))
e = res[res.eis]
print("  V - S distribution:"); print((e.V - e.S).clip(lower=-4, upper=4).value_counts().sort_index().to_string())
print(pd.crosstab(e.l, (e.V - e.S).clip(lower=-3, upper=2)).to_string())
