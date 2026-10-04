"""Step 6, U1: structure of the Eisenstein deficit  V - S  when E has a rational l-isogeny."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from load import load_curves
tag = sys.argv[1] if len(sys.argv) > 1 else "work"
res = pd.read_parquet(f"results/step6_u1c_{tag}.parquet")
cur = load_curves(columns=["N", "label", "galrep_images", "torsion_structure", "class_size", "max_isogeny_degree", "n_split_mult", "n_nonsplit_mult", "sha_an", "tamagawa", "disc_sign"], include_holdout=True).set_index("label").drop(columns=["N"])
e = res[res.eis].copy()
e = e.join(cur, on="label")
e["D"] = e.V - e.S
e["l_torsion"] = (e.torsion % e.l == 0)
e["l_label"] = [" ".join(s for s in str(g).split() if s.startswith(str(l))) for g, l in zip(e.galrep_images, e.l)]
print(f"{tag}: Eisenstein (curve, l) pairs: {len(e)}; deficit D = V - S")
print("\n--- D by l-torsion point present (True) vs l-isogeny only (False):")
print(pd.crosstab([e.l, e.l_torsion], e.D.clip(lower=-3, upper=2)).to_string())
print("\n--- D by number of contributing multiplicative primes (n_contrib):")
print(pd.crosstab(e.n_contrib.clip(upper=4), e.D.clip(lower=-3, upper=2)).to_string())
print("\n--- D by mod-l image label (top labels):")
top = e.l_label.value_counts().head(12).index
print(pd.crosstab(e.l_label.where(e.l_label.isin(top), "other"), e.D.clip(lower=-3, upper=2)).to_string())
print("\n--- D by whether every contributing prime is +-1 mod l:")
print(pd.crosstab([e.l, e.all_pm1], e.D.clip(lower=-3, upper=2)).to_string())
print("\n--- D by rank:")
print(pd.crosstab(e["rank"], e.D.clip(lower=-3, upper=2)).to_string())
print("\n--- the D <= -2 cases:")
cols = ["label", "l", "V", "S", "primes", "N", "rank", "torsion_structure", "l_label", "class_size", "max_isogeny_degree", "sha_an", "tamagawa"]
print(e[e.D <= -2][cols].to_string())
print("\n--- 30 of the D = -1 cases (smallest conductor):")
print(e[e.D == -1].sort_values("N")[cols].head(30).to_string())
# candidate correction terms
for name, corr in [("v_l(torsion)", e.torsion.map(lambda t: 0) ), ]:
    pass
def vl(x, l):
    v = 0
    while x > 0 and x % l == 0: x //= l; v += 1
    return v
e["vT"] = [vl(int(t), int(l)) for t, l in zip(e.torsion, e.l)]
e["vc"] = [vl(int(t), int(l)) for t, l in zip(e.tamagawa, e.l)]
print("\n--- does D >= -vT hold (vT = v_l(|E(Q)_tors|))?  violations:", int((e.D < -e.vT).sum()))
print("--- does D >= -1 - [l == 3] hold?  violations:", int((e.D < -1 - (e.l == 3)).sum()))
print("--- D by vT:"); print(pd.crosstab(e.vT, e.D.clip(lower=-3, upper=2)).to_string())
print("--- D by v_l(Tamagawa product) - S (Tamagawa valuation not explained by the geometric exponents):")
e["extra"] = e.vc - e.S
print(pd.crosstab(e.extra.clip(lower=-3, upper=3), e.D.clip(lower=-3, upper=2)).to_string())
