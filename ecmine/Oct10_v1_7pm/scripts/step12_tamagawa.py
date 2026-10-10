"""Step 12: test of the Tamagawa factor of ad^0 at the multiplicative primes (Lemma 2.7 of the note).
Theorem 1.2 (under (TW), l not dividing 2N) gives v_l(deg) >= sum e + sum_q v_l(Tam_q(ad^0)), and Lemma 2.7 computes
v_l(Tam_q) = v_l(n) + min(v_l(n), v_l(q-1), v_l(c_q)) at a prime q of type I_n, where c_q is the class of the unit part
u = q_E q^{-n} of the Tate parameter in F_q^x (x) Z_l: v_l(c_q) >= k iff u is an l^k-th power in F_q^x.
This script adds the extra terms min(...) to the bound of Conjecture 1.3 and checks the strengthened inequality.
Usage: python3 scripts/step12_tamagawa.py <work|holdout|ext> -> results/step12_<set>.csv"""
import sys, re, pandas as pd, numpy as np, cypari2
pari = cypari2.Pari(); pari.allocatemem(600_000_000)
TATE = pari("(E)->E.tate")
tag = sys.argv[1] if len(sys.argv) > 1 else "work"
cur = pd.read_parquet("data/curves.parquet", columns=["label", "N", "galrep_images", "ainvs"]).set_index("label")
opt = pd.read_parquet(f"results/optimal_{tag}.parquet").set_index("label")
def vl(x, l):
    v = 0; x = abs(int(x))
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
rows = []
for l in (3, 5):
    g = pd.read_parquet(f"results/step10_rule_{tag}_l{l}.parquet")
    ns = cur.galrep_images.reindex(g.index).fillna("").astype(str).str.contains("3Ns") if l == 3 else pd.Series(False, index=g.index)
    N = opt.N.reindex(g.index).astype(int)
    cnt = dict(curves_in_range=0, with_candidate_prime=0, with_extra=0, viol_old=0, viol_new=0, eq_old=0, eq_new=0, pos_old=0, pos_new=0, extra_total=0)
    for lab in g.index:
        n_ = int(N[lab])
        if n_ % l == 0 or (l == 3 and ns[lab]): continue
        cnt["curves_in_range"] += 1
        ks = list(map(int, re.findall(r"-?\d+", str(opt.kodaira[lab])))); ps = factor(n_)
        cand = [(q, k - 4) for q, k in zip(ps, ks) if k >= 5 and (k - 4) % l == 0 and q % l == 1]
        if not cand: continue
        cnt["with_candidate_prime"] += 1
        extra = 0
        for q, n in cand:
            Eq = pari.ellinit(pari(cur.ainvs[lab]), pari(f"O({q}^{n + 12})"))
            qE = TATE(Eq)[2]; nn = int(pari.valuation(qE, q)); assert nn == n, (lab, q, n, nn)
            u = int(pari.truncate(qE / pari(q) ** n)) % q
            vq = vl(q - 1, l); k = 0
            while k < vq and pow(u, (q - 1) // l ** (k + 1), q) == 1: k += 1
            extra += min(vl(n, l), vq, k)
        b_old = int(g.full[lab]); b_new = b_old + extra; V = int(g.V[lab])
        if extra > 0: cnt["with_extra"] += 1; cnt["extra_total"] += extra
        cnt["viol_old"] += V < b_old; cnt["viol_new"] += V < b_new
        cnt["pos_old"] += b_old > 0; cnt["pos_new"] += b_new > 0
        cnt["eq_old"] += (b_old > 0 and V == b_old); cnt["eq_new"] += (b_new > 0 and V == b_new)
        if V < b_new: print("VIOLATION", lab, l, V, b_old, b_new, cand, flush=True)
    cnt["l"] = l; rows.append(cnt); print(tag, cnt, flush=True)
pd.DataFrame(rows).to_csv(f"results/step12_{tag}.csv", index=False)
