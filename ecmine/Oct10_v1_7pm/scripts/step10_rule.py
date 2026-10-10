"""The uniform Euler-factor rule with PARI's symmetric-square Euler factors at every bad prime q != l:
   e_l(E,q) = v_l( P_q(q^{-2}) / (1 - a_q^2 q^{-2}) ),  P_q from lfunsympow (results/sym2_*.parquet),
   t_l(E,q) = v_l(-v_q(j)) if v_q(j) < 0 (potentially multiplicative: the Tamagawa exponent of ad^0), else 0,
   at q = l: t_l(E,l) plus, for l = 3, the empirical weights w(IV)=1, w(IV*)=2, w(II)=1, w(II*)=2, w(III*)=1.
Variants: (full) all terms; (noIII) e-terms of types III, III* at q >= 5 dropped; (no_ql) nothing at q = l.
Usage: python3 scripts/step10_rule.py <work|holdout|ext|oot>"""
import sys, re, glob, numpy as np, pandas as pd, cypari2
from fractions import Fraction
pari = cypari2.Pari(); pari.allocatemem(400_000_000)
tag = sys.argv[1] if len(sys.argv) > 1 else "work"
L = (3, 5, 7, 11, 13)
opt = pd.read_parquet("results/oot_table.parquet" if tag == "oot" else f"results/optimal_{tag}.parquet").set_index("label")
ainvs = pd.Series(opt.index, index=opt.index) if tag == "oot" else pd.read_parquet("data/curves.parquet", columns=["label", "ainvs"]).set_index("label").ainvs.reindex(opt.index)
sym = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"results/sym2_{tag}_*.parquet"))])
NM = int(opt.N.max()) + 1; spf = np.arange(NM)
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
def kind(k):
    if k >= 5: return "In"
    if k <= -5: return "In*"
    return {1: "I0", 2: "II", -2: "II*", 3: "III", -3: "III*", 4: "IV", -4: "IV*", -1: "I0*"}[k]
rows = []
for lab, r in opt.iterrows():
    ks = list(map(int, re.findall(r"-?\d+", str(r.kodaira))))
    for (q, e), k in zip(factor(int(r.N)), ks):
        rows.append((lab, q, kind(k), k - 4 if k >= 5 else (-k - 4 if k <= -5 else 0)))
loc = pd.DataFrame(rows, columns=["label", "q", "type", "n"]).merge(sym, on=["label", "q"], how="left")
assert loc.P.notna().all()
# the valuation of j at the primes of type I_n* at 2 (at odd q, and for I_n, the index n is -v_q(j))
need = loc[(loc.type == "In*") & (loc.q == 2)].label.unique()
vj = {}
for lab in need:
    E = pari.ellinit(pari(ainvs[lab])); j = E.j(); vj[lab] = int(pari.valuation(j, 2)) if j != 0 else 10**6
def tam_index(row):
    if row.type == "In": return int(row.n)
    if row.type == "In*":
        if row.q != 2: return int(row.n)
        v = vj[row.label]; return -v if v < 0 else 0
    return 0
loc["m"] = [tam_index(r) for r in loc.itertuples()]
def euler_num(P, q):            # P_q(q^{-2}) * q^{2 deg} as an integer, and the naive factor's numerator
    c = [int(x) for x in P.split(",")]; d = len(c) - 1
    return sum(ci * q ** (2 * (d - i)) for i, ci in enumerate(c)), d
loc["num"], loc["deg"] = zip(*[euler_num(P, int(q)) for P, q in zip(loc.P, loc.q)])
nocm = ~opt.is_cm.astype(bool); mid = opt.max_isogeny_degree.astype("int64")
W3 = {"IV": 1, "IV*": 2, "II": 1, "II*": 2, "III*": 1}
summary = []
for l in L:
    keep = opt.index[nocm & (mid % l != 0)]
    t = loc[loc.label.isin(keep)].copy()
    q = t.q.to_numpy(); ty = t.type.to_numpy(); num = t.num.to_numpy(); m = t.m.to_numpy()
    e = np.zeros(len(t), dtype=int); tam = np.zeros(len(t), dtype=int); wl = np.zeros(len(t), dtype=int)
    for i in range(len(t)):
        qq = int(q[i])
        if qq != l:
            ve = vl(int(num[i]), l) if num[i] != 0 else 0
            if ty[i] == "In": ve -= vl(qq * qq - 1, l)                 # naive factor 1 - q^{-2} at a multiplicative prime
            e[i] = ve
        else:
            wl[i] = W3.get(ty[i], 0) if l == 3 else 0
        tam[i] = vl(m[i], l) if m[i] > 0 else 0
    t["e"] = e; t["tam"] = tam; t["wl"] = wl
    t["eIII"] = np.where(np.isin(ty, ["III", "III*"]) & (q >= 5), e, 0)
    t["eIIIall"] = np.where(np.isin(ty, ["III", "III*"]), e, 0)
    g = t.groupby("label")[["e", "tam", "wl", "eIII", "eIIIall"]].sum().reindex(keep).fillna(0).astype(int)
    g["V"] = [vl(x, l) for x in opt.moddeg.reindex(keep).astype("int64")]
    g["full"] = g.e + g.tam + g.wl; g["noIII"] = g.full - g.eIII; g["noIIIall"] = g.full - g.eIIIall; g["noql"] = g.e + g.tam - g.eIIIall
    rec = {"l": l, "curves": len(g)}
    for name in ("full", "noIII", "noIIIall", "noql"):
        b = g[name]; pos = int((b > 0).sum()); rec[f"{name}>0"] = pos; rec[f"viol_{name}"] = int((g.V < b).sum())
        rec[f"eq_{name}"] = round(float((g.V[b > 0] == b[b > 0]).mean()), 4) if pos else float("nan")
    summary.append(rec)
    if rec["viol_noIIIall"]:
        print(f"  l={l}: violations of the rule without III terms at any prime:\n", g[g.V < g.noIIIall].head(12).to_string())
    g.to_parquet(f"results/step10_rule_{tag}_l{l}.parquet")
s = pd.DataFrame(summary); print(tag); print(s.to_string(index=False)); s.to_csv(f"results/step10_rule_{tag}_summary.csv", index=False)
