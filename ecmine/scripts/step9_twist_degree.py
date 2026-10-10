"""Numerical check of twist monotonicity of the congruence number on the tables: for an optimal curve E and an odd
prime q >= 5 at which E has additive reduction, let E_q be the quadratic twist by q*. Then r_{E_q} | r_E when E has
type I_n* at q (N_q = N/q), and r_{E_q} = r_E when E has potentially good reduction at q (N_q = N). For every prime
l with l^2 not dividing N this gives v_l(deg phi_E) >= v_l(deg phi_{E_q}), resp. equality. Usage: ... [work|holdout]"""
import sys, re, numpy as np, pandas as pd, cypari2
pari = cypari2.Pari(); pari.allocatemem(400_000_000)
tag = sys.argv[1] if len(sys.argv) > 1 else "work"
opt = pd.read_parquet(f"results/optimal_{tag}.parquet").set_index("label")
allc = pd.read_parquet("data/curves.parquet", columns=["label", "N", "ainvs", "moddeg", "optimal"])
allc["cls"] = allc.label.str.replace(r"\d+$", "", regex=True)
optdeg = allc[allc.optimal == 1].set_index("cls").moddeg.astype("int64")
lookup = dict(zip(allc.ainvs.str.replace(" ", ""), allc.cls))
ainvs = allc.set_index("label").ainvs.reindex(opt.index)
def vl(x, l):
    v = 0; x = int(x)
    while x > 0 and x % l == 0: x //= l; v += 1
    return v
NM = int(opt.N.max()) + 1; spf = np.arange(NM)
for i in range(2, int(NM ** 0.5) + 1):
    if spf[i] == i:
        sl = spf[i * i::i]; sl[sl == np.arange(i * i, NM, i)] = i
def primes(n):
    out = []
    while n > 1:
        p = int(spf[n]); out.append(p)
        while n % p == 0: n //= p
    return out
L = (3, 5, 7, 11, 13)
stats = {"pg": {"pairs": 0, "found": 0, "viol_eq": 0, "viol_ge": 0}, "ins": {"pairs": 0, "found": 0, "viol_ge": 0}}
bad = []
for lab, r in opt.iterrows():
    ks = list(map(int, re.findall(r"-?\d+", str(r.kodaira)))); ps = primes(int(r.N)); N = int(r.N); d = int(r.moddeg)
    for q, k in zip(ps, ks):
        if q < 5 or k >= 5: continue
        kind = "ins" if k <= -5 else "pg"
        qs = q if q % 4 == 1 else -q
        E = pari.ellinit(pari(ainvs[lab])); Et = pari.ellminimalmodel(pari.elltwist(E, qs))
        a = "[" + ",".join(str(Et[i]) for i in range(5)) + "]"
        stats[kind]["pairs"] += 1
        c = lookup.get(a)
        if c is None: continue
        stats[kind]["found"] += 1
        dq = int(optdeg[c]); Nq = int(allc.N[allc.cls == c].iloc[0]) if False else None
        for l in L:
            if N % (l * l) == 0: continue
            vq, v = vl(dq, l), vl(d, l)
            if kind == "ins" and v < vq: stats["ins"]["viol_ge"] += 1; bad.append((lab, q, "In*", l, v, vq, c))
            if kind == "pg":
                if v < vq: stats["pg"]["viol_ge"] += 1; bad.append((lab, q, "pg", l, v, vq, c))
                if v != vq: stats["pg"]["viol_eq"] += 1
print(tag, stats)
print("examples of v_l(deg E) < v_l(deg E_q):", bad[:15])
pd.DataFrame(bad, columns=["label", "q", "kind", "l", "v", "vq", "twist_class"]).to_csv(f"results/step9_twist_degree_{tag}.csv", index=False)
