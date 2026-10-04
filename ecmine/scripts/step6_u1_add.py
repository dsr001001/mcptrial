"""Step 6, U1: additive primes and the 3-adic (and 2-adic) valuation of the modular degree.

For optimal curves with E[l] irreducible (no rational l-isogeny) and no CM:
   X_l = v_l(deg phi) - sum over multiplicative q of v_l(v_q(Delta))      (the excess over the multiplicative part)
and counts of additive primes by Kodaira type, Tamagawa number and q mod 3.
Prediction of the level-lowering mechanism at l = 3: each prime of type IV or IV* with c_q = 3 adds at least 1
to X_3 (the mod 3 representation at q has Steinberg shape and the level can be lowered from q^2 to q).
Usage: python3 scripts/step6_u1_add.py [work|holdout]
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from load import load_curves
tag = sys.argv[1] if len(sys.argv) > 1 else "work"
loc = pd.read_parquet("results/local_primes.parquet"); loc = loc[loc.set == tag]
cur = load_curves(columns=["label", "N", "moddeg", "max_isogeny_degree", "is_cm", "torsion", "rank", "optimal"], include_holdout=True)
cur = cur[cur.optimal == 1].set_index("label")
def vl(x, l):
    v = 0
    while x > 0 and x % l == 0: x //= l; v += 1
    return v

for l in (3, 2):
    print(f"\n################################ l = {l}  ({tag}) ################################")
    ok = cur[(cur.max_isogeny_degree.astype("int64") % l != 0) & (~cur.is_cm.astype(bool))]
    t = loc[loc.label.isin(ok.index)].copy()
    t["v"] = np.where(t.mult, t.n.map(lambda n: vl(int(n), l)), 0)
    t["cv"] = t.c.map(lambda c: vl(int(c), l))
    g = t.groupby("label")
    d = pd.DataFrame({"S": g.v.sum()})
    d["V"] = [vl(int(x), l) for x in ok.moddeg.reindex(d.index).astype("int64")]
    d["X"] = d.V - d.S
    add = t[~t.mult]
    def count(mask, name):
        d[name] = add[mask].groupby("label").size().reindex(d.index).fillna(0).astype(int)
    if l == 3:
        iv = add.type.isin(["IV", "IV*"])
        count(iv & (add.c == 3) & (add.q != 3), "IV_c3"); count(iv & (add.c == 1) & (add.q != 3), "IV_c1"); count(iv & (add.q == 3), "IV_q3")
        count(iv & (add.c == 3) & (add.qmod3 == 1), "IV_c3_q1"); count(iv & (add.c == 3) & (add.qmod3 == 2), "IV_c3_q2")
        count(iv & (add.c == 1) & (add.qmod3 == 1), "IV_c1_q1"); count(iv & (add.c == 1) & (add.qmod3 == 2), "IV_c1_q2")
        count(add.type.isin(["II", "II*"]), "II"); count(add.type.isin(["III", "III*"]), "III"); count(add.type == "I0*", "I0s"); count(add.type == "In*", "Ins")
        print(f"curves: {len(d)}; with some IV/IV* prime: {int(((d.IV_c3 + d.IV_c1 + d.IV_q3) > 0).sum())}")
        for col in ["IV_c3", "IV_c1", "IV_q3", "IV_c3_q1", "IV_c3_q2", "IV_c1_q1", "IV_c1_q2"]:
            others = [c for c in ["IV_c3", "IV_c1", "IV_q3"] if c != col and not col.startswith(c)]
            sub = d[(d[col] > 0)]
            print(f"\n--- X = V - S against {col} (all curves having such a prime): rows X (clipped), cols count")
            print(pd.crosstab(sub.X.clip(lower=-1, upper=5), sub[col].clip(upper=4)).to_string())
            iso = d[(d[col] > 0) & (d[["IV_c3", "IV_c1", "IV_q3"]].sum(1) == d[col]) & (d.S == 0)]
            print(f"    isolated (only this kind of IV/IV* prime, S = 0): rows V, cols count")
            print(pd.crosstab(iso.V.clip(upper=5), iso[col].clip(upper=4)).to_string())
        print("\n--- controls: curves whose additive primes are all of one other type and S = 0: distribution of V")
        for col in ["II", "III", "I0s", "Ins"]:
            iso = d[(d[col] > 0) & (d.IV_c3 + d.IV_c1 + d.IV_q3 == 0) & (d.S == 0)]
            print(f"    {col}: n={len(iso)}, V distribution {iso.V.clip(upper=4).value_counts().sort_index().to_dict()}")
        # the conjectured bound
        d["pred"] = d.IV_c3
        viol = d[d.V < d.S + d.pred]
        print(f"\n=== conjecture  v_3(deg) >= S + #(IV or IV* with c_q = 3, q != 3):  violations {len(viol)} of {len(d)}")
        if len(viol): print(viol.head(20).to_string())
        d["pred2"] = d.IV_c3 + d.IV_c1 + d.IV_q3
        viol2 = d[d.V < d.S + d.pred2]
        print(f"=== stronger   v_3(deg) >= S + #(all IV or IV*):                      violations {len(viol2)} of {len(d)}")
        if len(viol2):
            print(pd.crosstab(viol2.IV_c1.clip(upper=3), viol2.IV_q3.clip(upper=3)).to_string())
            print(viol2.head(10).to_string())
        d.to_parquet(f"results/step6_u1_add3_{tag}.parquet")
    else:
        for name, mask in [("III", add.type.isin(["III", "III*"])), ("I0s", add.type == "I0*"), ("Ins", add.type == "In*"), ("II", add.type.isin(["II", "II*"])), ("IV", add.type.isin(["IV", "IV*"]))]:
            count(mask, name); count(mask & (add.cv > 0), name + "_ceven")
        print(f"curves (no 2-isogeny, no CM): {len(d)}")
        for col in ["III", "III_ceven", "I0s", "I0s_ceven", "Ins", "Ins_ceven", "II", "IV"]:
            sub = d[d[col] > 0]
            print(f"\n--- X = V - S against {col}: rows X (clipped), cols count")
            print(pd.crosstab(sub.X.clip(lower=-1, upper=6), sub[col].clip(upper=4)).to_string())
        print("\n--- multiplicative part alone: rows V, cols S (curves with no additive primes)")
        sub = d[d[["III", "I0s", "Ins", "II", "IV"]].sum(1) == 0]
        print(pd.crosstab(sub.V.clip(upper=8), sub.S.clip(upper=5)).to_string())
        d.to_parquet(f"results/step6_u1_add2_{tag}.parquet")
