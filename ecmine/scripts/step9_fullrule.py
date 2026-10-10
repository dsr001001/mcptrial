"""The Euler-factor rule as a complete bound, checked on a whole range.
For an optimal non-CM curve E with E[l] irreducible (l odd) the bound is
   B_l(E) = sum over additive q >= 5, q != l of e_l(q)  +  sum over q of type I_n (any q) or I_n* (odd q) of v_l(n)
            + wild terms at q in {2, 3} as in the twist rule ([l = 3] for IV, IV* at 2 or 3 and for II, II* at 3),
   e_l(q) = v_l(q - chi_{-3}(q)) for II, II*, IV, IV*;  v_l(q - chi_{-4}(q)) for III, III*;  v_l(q^2 - 1) for I_n*;
            v_l(q - 1) + v_l((q + 1)^2 - a_q(E')^2) for I_0*, E' the quadratic twist of E by q* (good at q),
i.e. e_l(q) is the l-adic valuation of the inverse Euler factor of L(Sym^2 f_E, s) at s = 2 (the one that the naive
symmetric square with a_q = 0 omits).  Usage: python3 scripts/step9_fullrule.py [work|holdout|ext|oot]"""
import sys, re, numpy as np, pandas as pd, cypari2
pari = cypari2.Pari(); pari.allocatemem(400_000_000)
tag = sys.argv[1] if len(sys.argv) > 1 else "work"
L = (3, 5, 7, 11, 13)
opt = pd.read_parquet("results/oot_table.parquet" if tag == "oot" else f"results/optimal_{tag}.parquet").set_index("label")
if tag == "oot":
    ainvs = pd.Series(opt.index, index=opt.index)           # the label of an out-of-table curve is its a-invariant list
else:
    ainvs = pd.read_parquet("data/curves.parquet", columns=["label", "ainvs"]).set_index("label").ainvs.reindex(opt.index)
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
loc = pd.DataFrame(rows, columns=["label", "q", "type", "n"])
# trace of Frobenius of the good twist at every I_0* prime q >= 5
i0 = loc[(loc.type == "I0*") & (loc.q >= 5)]
aq = {}
for lab, q in zip(i0.label, i0.q):
    q = int(q); qs = q if q % 4 == 1 else -q
    E = pari.ellinit(pari(ainvs[lab])); Et = pari.ellminimalmodel(pari.elltwist(E, qs)); aq[(lab, q)] = int(pari.ellap(Et, q))
loc["aq"] = [aq.get((lab, int(q)), 0) for lab, q in zip(loc.label, loc.q)]
print(f"{tag}: {len(opt)} curves, {len(loc)} bad primes, {len(i0)} I_0* primes >= 5 twisted")
nocm = ~opt.is_cm.astype(bool); mid = opt.max_isogeny_degree.astype("int64")
summary = []
for l in L:
    keep = opt.index[nocm & (mid % l != 0)]
    t = loc[loc.label.isin(keep)].copy()
    q = t.q.to_numpy(); ty = t.type.to_numpy(); n = t.n.to_numpy(); a = t.aq.to_numpy()
    chi3 = np.array([int(pari.kronecker(-3, int(x))) for x in q]); chi4 = np.array([int(pari.kronecker(-4, int(x))) for x in q])
    tame = (q >= 5) & (q != l)
    e = np.zeros(len(t), dtype=int)
    for i in range(len(t)):
        if not tame[i]: continue
        T = ty[i]; qq = int(q[i])
        if T in ("II", "II*", "IV", "IV*"): e[i] = vl(qq - chi3[i], l)
        elif T in ("III", "III*"): e[i] = vl(qq - chi4[i], l)
        elif T == "In*": e[i] = vl(qq * qq - 1, l)
        elif T == "I0*": e[i] = vl(qq - 1, l) + vl((qq + 1) ** 2 - int(a[i]) ** 2, l)
    tam = np.array([vl(nn, l) if (T == "In" or (T == "In*" and qq % 2 == 1)) else 0 for T, nn, qq in zip(ty, n, q)])
    wild = np.array([1 if (l == 3 and ((T in ("IV", "IV*") and qq in (2, 3)) or (T in ("II", "II*") and qq == 3))) else 0 for T, qq in zip(ty, q)])
    old_tw = np.array([ (1 if (l == 3 and T in ("IV", "IV*")) else 0) + (1 if (l == 3 and T in ("II", "II*") and qq % 2 == 1) else 0) for T, qq in zip(ty, q)])
    eIII = np.where(np.isin(ty, ["III", "III*"]), e, 0)
    m1 = np.array([1 if (T == "In" and vl(nn, l) > 0 and qq % l == l - 1) else 0 for T, nn, qq in zip(ty, n, q)])
    t["e"] = e; t["tam"] = tam; t["wild"] = wild; t["old"] = old_tw + tam; t["eIII"] = eIII; t["m1"] = m1
    g = t.groupby("label")[["e", "tam", "wild", "old", "eIII", "m1"]].sum().reindex(keep).fillna(0).astype(int)
    g["V"] = [vl(x, l) for x in opt.moddeg.reindex(keep).astype("int64")]
    g["new"] = g.e + g.tam + g.wild
    viol_new = int((g.V < g.new).sum()); viol_old = int((g.V < g.old).sum())
    pos_new = int((g.new > 0).sum()); pos_old = int((g.old > 0).sum())
    eq_new = float((g.V[g.new > 0] == g.new[g.new > 0]).mean()) if pos_new else float("nan"); eq_old = float((g.V[g.old > 0] == g.old[g.old > 0]).mean()) if pos_old else float("nan")
    g["noIII"] = g.new - g.eIII; g["cond"] = np.where(g.m1 > 0, g.new - g.eIII, g.new)
    viol_noIII = int((g.V < g.noIII).sum()); viol_cond = int((g.V < g.cond).sum()); eq_cond = float((g.V[g.cond > 0] == g.cond[g.cond > 0]).mean())
    strict = int((g.new > g.old).sum())
    summary.append({"l": l, "curves": len(g), "bound_new>0": pos_new, "viol_new": viol_new, "equality_new": round(eq_new, 4), "bound_twist>0": pos_old, "viol_twist": viol_old, "equality_twist": round(eq_old, 4), "new>twist": strict, "viol_noIII": viol_noIII, "viol_IIIcond": viol_cond, "equality_IIIcond": round(eq_cond, 4)})
    if viol_new:
        print(f"  l={l}: violations of the Euler-factor rule:\n", g[g.V < g.new].head(10).to_string())
    g.to_parquet(f"results/step9_fullrule_{tag}_l{l}.parquet")
s = pd.DataFrame(summary); print(s.to_string(index=False)); s.to_csv(f"results/step9_fullrule_{tag}_summary.csv", index=False)
