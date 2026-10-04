"""Step 6, U1: 'twisted' contributions at l = 3.

Quadratic twisting at an odd prime q swaps Kodaira types II <-> IV*, IV <-> II*, In <-> In*, I0 <-> I0*,
III <-> III*. The mod 3 representation at a type II or II* prime is (ramified quadratic character) x
(nontrivial unipotent), and at an In* prime it is (ramified quadratic character) x (unipotent trivial mod 3^t
iff 3^t | n). The isolated-type table shows two type II primes force 3 | deg while one does not.
Here: for non-Eisenstein non-CM optimal curves, excess Y = v_3(deg) - S - A (S = multiplicative exponents,
A = number of IV/IV* primes) against the twisted data: T = number of II/II* primes, W = multiset of v_3(n_q)
over In* primes.
Usage: python3 scripts/step6_u1_twist.py [work|holdout]
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from load import load_curves
tag = sys.argv[1] if len(sys.argv) > 1 else "work"
loc = pd.read_parquet("results/local_primes.parquet"); loc = loc[loc.set == tag].copy()
cur = load_curves(columns=["label", "N", "moddeg", "max_isogeny_degree", "is_cm", "optimal"], include_holdout=True)
cur = cur[cur.optimal == 1].set_index("label")
def vl(x, l):
    v = 0
    while x > 0 and x % l == 0: x //= l; v += 1
    return v
l = 3
ok = cur[(cur.max_isogeny_degree.astype("int64") % l != 0) & (~cur.is_cm.astype(bool))]
t = loc[loc.label.isin(ok.index)].copy()
t["v3n"] = t.n.map(lambda n: vl(int(n), l))
t["s"] = np.where(t.mult, t.v3n, 0)
t["a"] = t.type.isin(["IV", "IV*"]).astype(int)
t["tw2"] = t.type.isin(["II", "II*"]).astype(int)                     # twisted IV-shape, weight 1
t["w"] = np.where(t.type == "In*", t.v3n, 0)                           # twisted Steinberg exponent
t["twsum"] = t.tw2 + t.w
t["twmax"] = t.twsum
g = t.groupby("label")
d = pd.DataFrame({"S": g.s.sum(), "A": g.a.sum(), "T2": g.tw2.sum(), "W": g.w.sum(), "Wmax": g.w.max(), "TW": g.twsum.sum(), "TWmax": g.twmax.max(),
                  "nIns": (t.type == "In*").groupby(t.label).sum(), "nIns3": ((t.type == "In*") & (t.v3n > 0)).groupby(t.label).sum()})
d["V"] = [vl(int(x), l) for x in ok.moddeg.reindex(d.index).astype("int64")]
d["Y"] = d.V - d.S - d.A
print(f"{tag}: curves {len(d)}")
print("\n--- Y = V - S - A against T2 (number of II/II* primes), curves with W = 0:")
s = d[d.W == 0]; print(pd.crosstab(s.Y.clip(upper=5), s.T2.clip(upper=4)).to_string())
print("\n--- Y against W (sum of v_3(n) over In* primes), curves with T2 = 0:")
s = d[d.T2 == 0]; print(pd.crosstab(s.Y.clip(upper=5), s.W.clip(upper=4)).to_string())
print("\n--- Y against nIns3 (number of In* primes with 3 | n), curves with T2 = 0:")
print(pd.crosstab(s.Y.clip(upper=5), s.nIns3.clip(upper=4)).to_string())
print("\n--- Y against TW = T2 + W (all twisted weight), all curves:")
print(pd.crosstab(d.Y.clip(upper=6), d.TW.clip(upper=6)).to_string())
print("\n--- minimum of Y for each (T2, W) cell (counts in brackets):")
m = d.groupby([d.T2.clip(upper=3), d.W.clip(upper=3)]).Y.agg(["min", "size"]); print(m.to_string())
for name, pred in [("TW - TWmax (sum minus the largest twisted weight)", (d.TW - d.TWmax).clip(lower=0)),
                   ("TW - 1 if TW > 0", (d.TW - 1).clip(lower=0)),
                   ("2*floor(TW/2) (even subset)", 2 * (d.TW // 2)),
                   ("T2 - 1 if T2 > 0 (II/II* only)", (d.T2 - 1).clip(lower=0))]:
    print(f"   conjecture Y >= {name:55s}: violations {int((d.Y < pred).sum())}")
d.to_parquet(f"results/step6_u1_twist_{tag}.parquet")
