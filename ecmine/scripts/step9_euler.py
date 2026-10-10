"""Local terms at additive primes: the l-adic valuation of the Euler factor of the adjoint L-function at s = 1.
For curves of the working set with E[l] irreducible, not CM, and a single additive prime q >= 5 (q != l) and no
multiplicative l-contribution, compare v_l(deg phi) with the Euler-factor term
   II, II*, IV, IV*: v_l(q - chi_{-3}(q));  III, III*: v_l(q - chi_{-4}(q));  I_n*: v_l(q^2 - 1) (+ v_l(n), to test);
   I_0*: v_l(q - 1) + v_l((q + 1)^2 - a^2), a = a_q of the quadratic twist of E by q*, which has good reduction at q.
Usage: python3 scripts/step9_euler.py [l]     (writes results/step9_euler_l{l}.txt)"""
import sys, pandas as pd, numpy as np, cypari2
pari = cypari2.Pari(); pari.allocatemem(400_000_000)
l = int(sys.argv[1]) if len(sys.argv) > 1 else 3
def vl(x):
    v = 0; x = int(x)
    while x > 0 and x % l == 0: x //= l; v += 1
    return v
def kron(a, q): return int(pari.kronecker(a, q))
d = pd.read_parquet(f"results/step7_twist2_work_l{l}.parquet")            # per curve: V, mult (multiplicative l-part), ...
lp = pd.read_parquet("results/local_primes.parquet"); lp = lp[(lp.set == "work") & lp.label.isin(d.index)]
add = lp[lp.type != "In"]; nadd = add.groupby("label").size()
one = nadd[nadd == 1].index
a1 = add[add.label.isin(one)].set_index("label")[["q", "type", "n"]]
j = a1.join(d[["V", "mult"]], how="inner"); j = j[(j.q >= 5) & (j.q != l) & (j.mult == 0)]
ai = pd.read_parquet("data/curves.parquet", columns=["label", "ainvs"]).set_index("label").ainvs
def euler_term(row):
    q, t, n = int(row.q), row.type, int(row.n)
    if t in ("II", "II*", "IV", "IV*"): return vl(q - kron(-3, q))
    if t in ("III", "III*"): return vl(q - kron(-4, q))
    if t == "In*": return vl(q * q - 1)
    if t == "I0*":
        qs = q if q % 4 == 1 else -q
        E = pari.ellinit(pari(ai[row.Index])); Et = pari.ellminimalmodel(pari.elltwist(E, qs))
        a = int(pari.ellap(Et, q)); return vl(q - 1) + vl((q + 1) ** 2 - a * a)
    return 0
j["euler"] = [euler_term(r) for r in j.itertuples()]
j["tam"] = np.where(j.type == "In*", j.n.map(vl), 0)
j["pred"] = j.euler + j.tam; j["eu"] = j.euler; j["tm"] = j.tam
out = []
out.append(f"l = {l}: working-set curves with E[l] irreducible, not CM, one additive prime q >= 5 (q != l), no multiplicative l-part: {len(j)}")
out.append("min and distribution of V - euler (Euler-factor term of ad^0 at s = 1) by type and euler term:")
g = j.groupby(["type", "euler"]).apply(lambda s: pd.Series({"n": len(s), "minV": s.V.min(), "P(V>=euler)": round((s.V >= s.eu).mean(), 4), "P(V>=euler+1)": round((s.V >= s.eu + 1).mean(), 3)}))
out.append(g.to_string())
out.append("\nI_n*: V - euler versus v_l(n) (additivity of the Tamagawa exponent of the twist):")
s = j[j.type == "In*"]; out.append(s.groupby(["euler", "tam"]).apply(lambda t: pd.Series({"n": len(t), "min(V-euler)": (t.V - t.eu).min(), "P(V>=pred)": round((t.V >= t.pred).mean(), 4)})).to_string())
out.append("\nviolations of V >= euler + tam over all single-additive-prime curves: " + str(int((j.V < j.pred).sum())) + " of " + str(len(j)))
if int((j.V < j.pred).sum()): out.append(j[j.V < j.pred].head(20).to_string())
# multiplicative primes with l | n and q = +-1 mod l: is there an extra term?
mm = lp[(lp.type == "In")].copy(); mm["vn"] = mm.n.map(vl)
cnt = mm[mm.vn > 0].groupby("label").size(); single = cnt[cnt == 1].index
m1 = mm[(mm.vn > 0) & mm.label.isin(single)].set_index("label")[["q", "n", "vn"]]
m1 = m1.join(d[["V", "mult"]], how="inner"); m1 = m1[~m1.index.isin(add.label)]          # no additive primes at all
m1 = m1[(m1.q >= 5) & (m1.q != l)]; m1["e"] = m1.q.map(lambda q: vl(q * q - 1))
out.append(f"\nmultiplicative: curves with no additive prime and exactly one multiplicative prime q >= 5 with l | n: {len(m1)}")
out.append("V - v_l(n) by v_l(q^2-1):"); out.append(m1.groupby("e").apply(lambda t: pd.Series({"n": len(t), "min(V-vn)": (t.V - t.vn).min(), "P(V>vn)": round((t.V > t.vn).mean(), 3)})).to_string())
m0 = mm[mm.vn == 0].groupby("label").size(); m0 = m0[m0 == 1].index
m0 = mm[(mm.vn == 0) & mm.label.isin(m0)].set_index("label")[["q", "n"]].join(d[["V", "mult"]], how="inner")
m0 = m0[(m0.mult == 0) & ~m0.index.isin(add.label) & (m0.q >= 5) & (m0.q != l)]; m0["e"] = m0.q.map(lambda q: vl(q * q - 1))
out.append(f"multiplicative, l not dividing n, no additive prime, single bad prime >= 5 besides 2,3: {len(m0)}; V by v_l(q^2-1):")
out.append(m0.groupby("e").apply(lambda t: pd.Series({"n": len(t), "minV": t.V.min(), "P(V>=1)": round((t.V >= 1).mean(), 3)})).to_string())
txt = "\n".join(out); print(txt); open(f"results/step9_euler_l{l}.txt", "w").write(txt + "\n")
