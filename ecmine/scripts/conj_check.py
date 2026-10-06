"""Consolidated checker for the three conjectures of STEP6_U1_REPORT.md (and Observation 4 at l = 2).

Input: a table of X_0(N)-optimal curves with columns
   label, N, moddeg, kodaira (PARI codes as a bracketed list, primes in increasing order),
   tamagawa_list (same order), max_isogeny_degree (1 if the isogeny class is trivial), is_cm, torsion (order)
   and optionally galrep_images (for the split-Cartan term of Conjecture 3; absent -> treated as 0 and reported).
For every odd prime l it computes
   V      = v_l(deg phi)
   G      = sum over bad q of v_l(#Phi_q(F_q-bar))          (Conjecture 1 bound; types IV, IV* count 1 at l = 3)
   TW     = twisted weight at l = 3: 1 for types II, II*, v_3(n) for I_n*, 0 otherwise
   bound2 = G + (TW - max single twisted weight)            (Conjecture 2 bound)
   eis    = E has a rational l-isogeny (max_isogeny_degree divisible by l)
   deficit bound for the Eisenstein case: v_l(#E(Q)_tors) + [E[3] = Z/3 + mu_3]  (Conjecture 3)
and at l = 2 the Observation-4 bound: v_2 of multiplicative n_q plus 1 for each III, III*, I_0*, I_n*.

Usage: python3 scripts/conj_check.py <table.parquet|table.tsv> [--out results/prefix]
"""
import re, sys, os, argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("table"); ap.add_argument("--out", default=None); ap.add_argument("--nmax", type=int, default=None)
args = ap.parse_args()
df = pd.read_parquet(args.table) if args.table.endswith(".parquet") else pd.read_csv(args.table, sep="\t")
df = df.reset_index(drop=True)
NM = int(args.nmax or df.N.max()) + 1
spf = np.arange(NM)
for i in range(2, int(NM ** 0.5) + 1):
    if spf[i] == i:
        sl = spf[i * i::i]; sl[sl == np.arange(i * i, NM, i)] = i
def factor(n):
    out = []
    while n > 1:
        p = int(spf[n]); e = 0
        while n % p == 0: n //= p; e += 1
        out.append((p, e))
    return out
def vl(x, l):
    v = 0; x = int(x)
    while x > 0 and x % l == 0: x //= l; v += 1
    return v
GEOM = {1: 1, 2: 1, -2: 1, 3: 2, -3: 2, 4: 3, -4: 3, -1: 4}
def geom_order(k):   # order of the geometric component group from the PARI Kodaira code
    if k >= 5: return k - 4
    if k <= -5: return 4
    return GEOM[k]
def kind(k):
    if k >= 5: return "In"
    if k <= -5: return "In*"
    return {1: "I0", 2: "II", -2: "II*", 3: "III", -3: "III*", 4: "IV", -4: "IV*", -1: "I0*"}[k]

rows = []
for r in df.itertuples(index=False):
    primes = factor(int(r.N)); ks = list(map(int, re.findall(r"-?\d+", str(r.kodaira)))); cs = list(map(int, re.findall(r"\d+", str(r.tamagawa_list))))
    assert len(primes) == len(ks) == len(cs), (r.label, primes, ks, cs)
    for (q, e), k, c in zip(primes, ks, cs):
        rows.append((r.label, q, k, c, kind(k), geom_order(k), k - 4 if k >= 5 else (-k - 4 if k <= -5 else 0)))
loc = pd.DataFrame(rows, columns=["label", "q", "kod", "c", "type", "geom", "n"])
has_galrep = "galrep_images" in df.columns
info = df.set_index("label")
split_cartan3 = info.galrep_images.fillna("").str.contains(r"\b3Cs\.1\.1\b", regex=True) if has_galrep else pd.Series(False, index=info.index)

L = sorted({p for g in loc.geom.unique() for p, _ in factor(int(g))} | {3})
report = []
out_rows = []
for l in L:
    loc["vg"] = loc.geom.map(lambda g: vl(g, l)); loc["vn"] = loc.n.map(lambda n: vl(n, l))
    if l == 3:
        loc["tw"] = np.where(loc.type.isin(["II", "II*"]), 1, np.where(loc.type == "In*", loc.vn, 0))
    else:
        loc["tw"] = 0
    if l == 2:
        loc["obs"] = np.where(loc.type == "In", loc.vn, np.where(loc.type.isin(["III", "III*", "I0*", "In*"]), 1, 0))
    g = loc.groupby("label")
    d = pd.DataFrame({"G": g.vg.sum(), "TW": g.tw.sum(), "TWmax": g.tw.max()})
    if l == 2: d["OBS"] = g.obs.sum()
    d = d.reindex(info.index).fillna(0).astype(int)
    d["V"] = [vl(x, l) for x in info.moddeg.astype("int64")]
    d["eis"] = (info.max_isogeny_degree.astype("int64") % l == 0).to_numpy()
    d["cm"] = info.is_cm.astype(bool).to_numpy()
    d["vT"] = [vl(x, l) for x in info.torsion.astype("int64")]
    d["bound1"] = d.G; d["bound2"] = d.G + (d.TW - d.TWmax).clip(lower=0)
    d["eisbound"] = d.vT + (split_cartan3.reindex(info.index).fillna(False).astype(int).to_numpy() if l == 3 else 0)
    d["l"] = l
    ne = d[~d.eis & ~d.cm]; e = d[d.eis & ~d.cm]
    rep = {"l": l, "curves": len(d), "non_eis": len(ne), "non_eis_with_bound": int((ne.bound1 > 0).sum()),
           "viol_conj1": int((ne.V < ne.bound1).sum()) if l != 2 else None, "viol_conj2": int((ne.V < ne.bound2).sum()) if l == 3 else None,
           "eis": len(e), "eis_with_bound": int((e.bound1 > 0).sum()), "viol_conj3": int((e.bound1 - e.V > e.eisbound).sum()) if l != 2 else None,
           "viol_obs4": int((ne.V < ne.OBS).sum()) if l == 2 else None}
    report.append(rep); out_rows.append(d.reset_index())
    v1 = ne[ne.V < ne.bound1] if l != 2 else ne.iloc[0:0]; v2 = ne[ne.V < ne.bound2] if l == 3 else ne.iloc[0:0]; v3 = e[e.bound1 - e.V > e.eisbound] if l != 2 else e.iloc[0:0]
    for name, v in [("Conjecture 1", v1), ("Conjecture 2", v2), ("Conjecture 3", v3)]:
        if len(v): print(f"  l={l} {name} VIOLATIONS ({len(v)}):\n" + v.head(15).to_string())
rep = pd.DataFrame(report)
print(f"table: {args.table}  curves: {len(df)}  galrep available: {has_galrep}")
print(rep.to_string(index=False))
if args.out:
    pd.concat(out_rows, ignore_index=True).to_parquet(args.out + "_perl.parquet", index=False); rep.to_csv(args.out + "_summary.csv", index=False)
