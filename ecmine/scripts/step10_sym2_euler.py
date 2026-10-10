"""Symmetric-square Euler factors at the bad primes, from PARI's lfunsympow (which carries the correct factors at the
primes of bad reduction, wild ones included). For every optimal curve of a range: the list of bad primes q with the
polynomial P_q(X) such that L_q(Sym^2 f, s) = P_q(q^{-s})^{-1}, as integer coefficient lists.
Usage: python3 scripts/step10_sym2_euler.py <work|holdout|ext|oot> <shard> <nshards>   -> results/sym2_<set>_<shard>.parquet"""
import sys, pandas as pd, cypari2
pari = cypari2.Pari(); pari.allocatemem(600_000_000)
tag, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
opt = pd.read_parquet("results/oot_table.parquet" if tag == "oot" else f"results/optimal_{tag}.parquet").set_index("label")
if tag == "oot": ai = pd.Series(opt.index, index=opt.index)
else: ai = pd.read_parquet("data/curves.parquet", columns=["label", "ainvs"]).set_index("label").ainvs.reindex(opt.index)
labs = [l for i, l in enumerate(opt.index) if i % nsh == shard]
rows = []
for lab in labs:
    E = pari.ellinit(pari(ai[lab])); L = pari.lfunsympow(E, 2); N = int(opt.N[lab])
    for p in [int(x) for x in pari.factor(N)[0]]:
        P = 1 / pari.lfuneuler(L, p)                       # polynomial in x with integer coefficients
        coeffs = [int(c) for c in pari.Vec(pari.Pol(P))][::-1]   # ascending powers of x
        rows.append((lab, p, ",".join(map(str, coeffs))))
pd.DataFrame(rows, columns=["label", "q", "P"]).to_parquet(f"results/sym2_{tag}_{shard}.parquet", index=False)
print(tag, shard, "curves:", len(labs), "rows:", len(rows))
