"""Step 11: the boundary of the Taylor-Wiles hypothesis at l = 3.
For each range, from the per-curve output of step10_rule.py (bounds with and without the type III terms) and the mod-3
image labels of the tables (galrep; for the out-of-table curves the factorisation pattern of the 3-division polynomial),
compute: the rule with the III terms and the correction term [image = 3Ns] (Conjecture 1.3 of the note), its violations
and equality rate; the 3Ns statistics (every exception has image 3Ns; failure rates; the length h of
H^0(Q, ad^0 rho(1) (x) Q_3/Z_3) read off the traces); the residue classes mod 12 of the III primes of the exceptions;
the type IV/IV* at 2 count for Proposition 1.4. Writes results/step11_summary.csv, results/step11_ns3.csv,
paper/tables/euler_uniform.tex and paper/tables/ns3.tex."""
import re, glob, pandas as pd, numpy as np, cypari2
from collections import Counter
pari = cypari2.Pari(); pari.allocatemem(400_000_000)
cur = pd.read_parquet("data/curves.parquet", columns=["label", "N", "galrep_images", "ainvs"]).set_index("label")
SETS = [("work", "working set"), ("holdout", "hold-out"), ("ext", "extension"), ("oot", "out of table")]
L = (3, 5, 7, 11, 13)
def vl(x, l):
    v = 0; x = int(x)
    while x > 0 and x % l == 0: x //= l; v += 1
    return v
def factor(n):
    out = []; p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0: n //= p
            out.append(p)
        p += 1
    if n > 1: out.append(n)
    return out
def pattern3(ai):   # degrees of the factors of the 3-division polynomial; (2, 2) iff image in the normaliser of a split Cartan
    E = pari.ellinit(pari(ai)); F = pari.factor(pari.elldivpol(E, 3))
    return tuple(sorted(int(pari.poldegree(F[i, 0])) for i in range(pari.matsize(F)[0])))
INERT = [int(p) for p in pari.primes(200) if int(p) % 3 == 2]
def h3(ai, N):   # largest k with a_p = 0 mod 3^k at the inert primes p < 1223 not dividing N (traces of rho and rho (x) chi_{-3})
    E = pari.ellinit(pari(ai)); k = 10
    for p in INERT:
        if N % p == 0: continue
        a = int(pari.ellap(E, p))
        if a: k = min(k, vl(abs(a), 3))
    return k
def pct(x, n): return "--" if n == 0 else f"{100*x:.1f}\\%"
summary = []; ns3rows = []; unif = []
for tag, name in SETS:
    opt = pd.read_parquet("results/oot_table.parquet" if tag == "oot" else f"results/optimal_{tag}.parquet").set_index("label")
    ai = pd.Series(opt.index, index=opt.index) if tag == "oot" else cur.ainvs.reindex(opt.index)
    g3 = pd.read_parquet(f"results/step10_rule_{tag}_l3.parquet")
    if tag == "oot":
        ns = pd.Series([pattern3(ai[lab]) == (2, 2) for lab in g3.index], index=g3.index)
    else:
        ns = cur.galrep_images.reindex(g3.index).fillna("").astype(str).str.contains("3Ns")
    first = True
    for l in L:
        g = pd.read_parquet(f"results/step10_rule_{tag}_l{l}.parquet")
        corr = (ns.reindex(g.index).fillna(False).astype(int) if l == 3 else pd.Series(0, index=g.index))
        b = g.full - corr                      # Conjecture 1.3: all types, minus the correction term at l = 3
        pos = b > 0; viol = int((g.V < b).sum()); eq = float((g.V[pos] == b[pos]).mean()) if pos.any() else float("nan")
        posf = g.noIIIall > 0; eqf = float((g.V[posf] == g.noIIIall[posf]).mean()) if posf.any() else float("nan")
        rec = dict(set=tag, l=l, curves=len(g), bound_pos=int(pos.sum()), viol=viol, eq=round(eq, 4),
                   viol_full=int((g.V < g.full).sum()), noIII_pos=int(posf.sum()), viol_noIII=int((g.V < g.noIIIall).sum()), eq_noIII=round(eqf, 4))
        summary.append(rec)
        unif.append(f"{name if first else ''} & {l} & {len(g):,} & {int(pos.sum()):,} & {viol} & {pct(eq, int(pos.sum()))} & {rec['viol_full']} & {int(posf.sum()):,} / {rec['viol_noIII']} / {pct(eqf, int(posf.sum()))} \\\\")
        first = False
    unif.append("\\midrule")
    # 3Ns statistics at l = 3
    g = g3; fail = g.V < g.full; exc = g.index[fail]; nsi = ns.reindex(g.index).fillna(False)
    N = opt.N.reindex(g.index).astype(int)
    hs = Counter(h3(ai[lab], int(N[lab])) for lab in g.index[nsi])
    pats = Counter(pattern3(ai[lab]) for lab in exc)
    sym = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"results/sym2_{tag}_*.parquet"))])
    sym = sym[sym.label.isin(exc)].set_index(["label", "q"])
    q12 = Counter(); tamp = 0; short = Counter()
    for lab in exc:
        ks = list(map(int, re.findall(r"-?\d+", str(opt.kodaira[lab])))); ps = factor(int(N[lab]))
        E = pari.ellinit(pari(ai[lab])); j = E.j(); hast = False
        for q, k in zip(ps, ks):
            if abs(k) == 3 and q != 3:
                c = [int(x) for x in sym.P[(lab, q)].split(",")]; d = len(c) - 1
                if vl(sum(ci * q ** (2 * (d - i)) for i, ci in enumerate(c)), 3) > 0: q12[q % 12 if q >= 5 else q] += 1
            vj = int(pari.valuation(j, q)) if j != 0 else 0
            if vj < 0 and vl(-vj, 3) > 0 and q % 3 == 2: hast = True
        tamp += hast; short[int(g.full[lab] - g.V[lab])] += 1
    row = dict(set=tag, curves=len(g), ns3=int(nsi.sum()), ns3_with_III_term=int((g.eIIIall > 0)[nsi].sum()), ns3_failing=int(fail[nsi].sum()),
               non_ns3_failing=int(fail[~nsi].sum()), exceptions=len(exc), exc_pattern_22=pats.get((2, 2), 0), h_eq_1=hs.get(1, 0), h_other=sum(v for k, v in hs.items() if k != 1),
               III_q_1mod12=q12.get(1, 0), III_q_11mod12=q12.get(11, 0), III_q_2=q12.get(2, 0), exc_with_tam_prime_2mod3=tamp, shortfall_1=short.get(1, 0), shortfall_other=sum(v for k, v in short.items() if k != 1))
    ns3rows.append(row); print(row)
    if tag == "work":   # type IV / IV* at 2 (Proposition 1.4) and calibration of the corrected rule at l = 3
        iv2 = []
        for lab in g.index:
            ks = list(map(int, re.findall(r"-?\d+", str(opt.kodaira[lab]))))
            if int(N[lab]) % 2 == 0 and abs(ks[0]) == 4: iv2.append(lab)
        sym2 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("results/sym2_work_*.parquet"))])
        sym2 = sym2[(sym2.q == 2) & sym2.label.isin(iv2)].set_index("label").P
        e2 = Counter(vl(sum(int(ci) * 2 ** (2 * (len(P.split(",")) - 1 - i)) for i, ci in enumerate(P.split(","))), 3) for P in sym2.reindex(iv2))
        print("type IV/IV* at 2 in the working set (E[3] irreducible, non-CM):", len(iv2), "e_3(E,2) distribution:", dict(e2))
        b = g.full - nsi.astype(int); zero = g[b == 0]; posc = g[b > 0]
        dist = zero.V.value_counts(normalize=True).sort_index(); cdf = dist.cumsum()
        expv = sum(float(cdf[cdf.index <= bb - 1].max()) if (cdf.index <= bb - 1).any() else 0.0 for bb in b[b > 0])
        print(f"calibration l=3 corrected rule: bound-0 curves {len(zero)}, frac V=0 {float((zero.V==0).mean()):.3f}, positive {len(posc)}, expected violations if independent {expv:,.0f}, mean excess {float((posc.V - b[b>0]).mean()):.2f}")
unif[-1] = "\\bottomrule"
with open("paper/tables/euler_uniform.tex", "w") as fh:
    fh.write("\\begin{tabular}{llrrrrrr}\n\\toprule\nset & $\\ell$ & curves & \\eqref{eq:rule} $>0$ & viol. & equality & viol.\\ without $h$ & III-free: $>0$ / viol. / equality\\\\\n\\midrule\n")
    fh.write("\n".join(unif) + "\n\\end{tabular}\n")
pd.DataFrame(summary).to_csv("results/step11_summary.csv", index=False)
ns = pd.DataFrame(ns3rows); ns.to_csv("results/step11_ns3.csv", index=False)
with open("paper/tables/ns3.tex", "w") as fh:
    fh.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\nset & $\\Ns$ & with III term & failing & others failing & $h_3=1$ & $q\\equiv1$ / $11\\ (12)$ / $q=2$ & shortfall $1$\\\\\n\\midrule\n")
    for _, r in ns.iterrows():
        nm = dict(SETS)[r.set]
        fh.write(f"{nm} & {r.ns3:,} & {r.ns3_with_III_term:,} & {r.ns3_failing} ({100*r.ns3_failing/max(r.ns3,1):.0f}\\%) & {r.non_ns3_failing} & {r.h_eq_1:,} of {r.ns3:,} & {r.III_q_1mod12} / {r.III_q_11mod12} / {r.III_q_2} & {r.shortfall_1} of {r.exceptions}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(pd.DataFrame(summary).to_string(index=False))
