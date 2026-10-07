"""Step 6, U1: (a) per-type contributions at l = 2 and l = 3 from isolated configurations, and the unified
conjecture  v_l(deg phi) >= sum_{q | N} v_l(#Phi_q(F_q-bar))  with geometric component group orders
   In: n,  II, II*: 1,  III, III*: 2,  IV, IV*: 3,  I0*: 4,  In*: 4;
(b) the Eisenstein deficit at l = 3 recomputed with the additive term.
Usage: python3 scripts/step6_u1_fit.py [work|holdout]
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from load import load_curves
tag = sys.argv[1] if len(sys.argv) > 1 else "work"
loc = pd.read_parquet("results/local_primes.parquet"); loc = loc[loc.set == tag].copy()
cur = load_curves(columns=["label", "N", "moddeg", "max_isogeny_degree", "is_cm", "torsion", "torsion_structure", "galrep_images", "rank", "optimal", "class_size"], include_holdout=True)
cur = cur[cur.optimal == 1].set_index("label")
def vl(x, l):
    v = 0
    while x > 0 and x % l == 0: x //= l; v += 1
    return v
GEOM = {"In": None, "II": 1, "II*": 1, "III": 2, "III*": 2, "IV": 3, "IV*": 3, "I0*": 4, "In*": 4, "I0": 1}
loc["geom"] = np.where(loc.mult, loc.n, loc.type.map(GEOM))

for l in (3, 2):
    print(f"\n################ l = {l} ({tag}) ################")
    loc["vg"] = loc.geom.map(lambda g: vl(int(g), l)); loc["vm"] = np.where(loc.mult, loc.vg, 0)
    ok = cur[(cur.max_isogeny_degree.astype("int64") % l != 0) & (~cur.is_cm.astype(bool))]
    t = loc[loc.label.isin(ok.index)]
    g = t.groupby("label"); d = pd.DataFrame({"S": g.vm.sum(), "G": g.vg.sum()})
    d["V"] = [vl(int(x), l) for x in ok.moddeg.reindex(d.index).astype("int64")]
    types = ["II", "II*", "III", "III*", "IV", "IV*", "I0*", "In*"]
    add = t[~t.mult]
    for ty in types: d[ty] = add[add.type == ty].groupby("label").size().reindex(d.index).fillna(0).astype(int)
    d["n_add"] = d[types].sum(1)
    print("isolated configurations (S = 0, all additive primes of a single type): min V and distribution by count")
    for ty in types:
        iso = d[(d[ty] > 0) & (d.n_add == d[ty]) & (d.S == 0)]
        for k in (1, 2, 3):
            s = iso[iso[ty] == k]
            if len(s): print(f"   {ty:4s} x{k}: n={len(s):7d}  min V = {s.V.min()}  V counts {s.V.clip(upper=6).value_counts().sort_index().to_dict()}")
    viol = d[d.V < d.G]
    print(f"\n=== unified conjecture v_{l}(deg) >= sum v_{l}(#Phi_geom):  violations {len(viol)} of {len(d)}")
    if len(viol):
        print("   violations by type counts:"); print(viol[types + ["S", "G", "V"]].head(15).to_string())
        for ty in types:
            s = viol[viol[ty] > 0]; print(f"   involving {ty}: {len(s)}")
    # per-type maximal constants c_T with V >= S + sum c_T #T on all curves (greedy check of candidate vectors)
    if l == 2:
        for cand in [dict(III=1, I0s=1, Ins=1), dict(III=1, I0s=2, Ins=2), dict(III=1, I0s=1, Ins=2), dict(III=1, I0s=2, Ins=1), dict(III=0, I0s=1, Ins=1)]:
            pred = cand["III"] * (d["III"] + d["III*"]) + cand["I0s"] * d["I0*"] + cand["Ins"] * d["In*"]
            print(f"   V >= S + {cand}: violations {int((d.V < d.S + pred).sum())}")
    d.to_parquet(f"results/step6_u1_fit{l}_{tag}.parquet")

# Eisenstein deficit at l = 3 with the additive term
l = 3
loc["vg"] = loc.geom.map(lambda g: vl(int(g), l))
eis = cur[(cur.max_isogeny_degree.astype("int64") % l == 0)]
t = loc[loc.label.isin(eis.index)]; g = t.groupby("label"); d = pd.DataFrame({"G": g.vg.sum()})
d["V"] = [vl(int(x), l) for x in eis.moddeg.reindex(d.index).astype("int64")]; d["D"] = d.V - d.G
i = eis.reindex(d.index)
d["vT"] = [vl(int(x), l) for x in i.torsion]; d["lab"] = [" ".join(s for s in str(x).split() if s.startswith("3")) for x in i.galrep_images]
d["cm"] = i.is_cm.astype(bool).to_numpy(); d["rank"] = i["rank"].to_numpy(); d["tors"] = i.torsion_structure.to_numpy()
print(f"\n################ l = 3 Eisenstein with the additive term ({tag}): curves {len(d)} ################")
print("D = V - G by (vT, label):"); print(pd.crosstab([d.vT, d.lab], d.D.clip(lower=-3, upper=2)).to_string())
print("CM among Eisenstein:", int(d.cm.sum()), " D for CM:", d[d.cm].D.clip(lower=-3, upper=2).value_counts().sort_index().to_dict())
d["bound"] = -(d.vT + d.lab.str.contains("Cs").astype(int))
print("violations of D >= -(vT + [split Cartan]):", int((d.D < d.bound).sum()))
print(d[d.D < d.bound].head(20).to_string())
print("violations of D >= -2:", int((d.D < -2).sum()), " of D >= -1 among curves without 3-torsion point:", int(((d.vT == 0) & (d.D < 0)).sum()))
d.to_parquet(f"results/step6_u1_eis3_{tag}.parquet")
