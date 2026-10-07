"""Step 7 of the follow-up list: a principled predictor for the Eisenstein deficit at l = 3.
For optimal curves with a rational 3-isogeny (not CM): deficit D = bound1 - v_3(deg phi) (positive means the
Conjecture 1 bound fails by D). Candidate predictors built from the whole isogeny class."""
import sys, re
import numpy as np, pandas as pd
tag = sys.argv[1] if len(sys.argv) > 1 else "work"
per = pd.read_parquet(f"results/check_{tag}_perl.parquet"); per = per[(per.l == 3) & per.eis & ~per.cm].set_index("label")
cur = pd.read_parquet("data/curves.parquet", columns=["label", "N", "class_label", "torsion", "torsion_structure", "galrep_images", "num", "optimal", "rank", "class_size", "max_isogeny_degree", "tamagawa"])
cur = cur[cur.label.isin(per.index) | cur.class_label.isin(cur[cur.label.isin(per.index)].class_label)]
def v3(x):
    v = 0; x = int(x)
    while x > 0 and x % 3 == 0: x //= 3; v += 1
    return v
cur["vT"] = cur.torsion.map(v3)
cls = cur.groupby("class_label").agg(vTmax=("vT", "max"), vTsum=("vT", "sum"), n3=("vT", lambda s: int((s > 0).sum())), size=("vT", "size"))
e = per.drop(columns=["vT"]).join(cur.set_index("label")[["class_label", "vT", "torsion_structure", "galrep_images", "rank", "class_size", "max_isogeny_degree", "tamagawa", "N"]])
e = e.join(cls, on="class_label")
e["D"] = (e.bound1 - e.V).clip(lower=0)
e["lab3"] = [" ".join(s for s in str(g).split() if s.startswith("3")) for g in e.galrep_images]
e["Cs"] = e.lab3.str.contains("3Cs.1.1").astype(int)
e["v3tam"] = e.tamagawa.map(v3)
print(f"{tag}: Eisenstein-at-3 optimal curves (not CM): {len(e)}; with a positive bound: {int((e.bound1 > 0).sum())}; with deficit: {int((e.D > 0).sum())}")
print("\nD by (vT of the optimal curve, image label):"); print(pd.crosstab([e.vT, e.lab3], e.D).to_string())
print("\nD by vTmax over the class:"); print(pd.crosstab(e.vTmax, e.D).to_string())
print("\nD by n3 (number of curves in the class with a rational 3-torsion point):"); print(pd.crosstab(e.n3, e.D).to_string())
print("\nD by vTsum over the class:"); print(pd.crosstab(e.vTsum.clip(upper=5), e.D).to_string())
for name, bound in [("vT + [Cs]", e.vT + e.Cs), ("vTmax", e.vTmax), ("vTmax + [Cs]", e.vTmax + e.Cs), ("vTsum", e.vTsum), ("n3", e.n3), ("n3 - [n3>0] + vT", e.n3 - (e.n3 > 0).astype(int) + e.vT),
                    ("v3(class size)", e.class_size.map(v3)), ("vT + [Cs] restricted to vT>0", e.vT + e.Cs)]:
    viol = int((e.D > bound).sum()); tight = int(((e.D == bound) & (e.D > 0)).sum())
    print(f"   D <= {name:32s}: violations {viol:5d}; attained with D > 0: {tight:5d}")
print("\nthe D = 2 cases with their class torsion profile:")
print(e[e.D >= 2][["N", "V", "bound1", "vT", "lab3", "torsion_structure", "vTmax", "vTsum", "n3", "class_size", "max_isogeny_degree", "rank", "v3tam"]].to_string())
