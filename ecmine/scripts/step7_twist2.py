"""Re-examination of the twisted primes with q = 2 separated from odd q, for every odd l.

For an odd prime q, quadratic twisting exchanges II <-> IV*, IV <-> II*, I_n <-> I_n*, so the l-part of the
geometric component group of the better of E and its twist at q is
   v_l(n_q) for I_n and I_n*,   1 at l = 3 for II, II*, IV, IV*,   0 otherwise.
Candidate rule R_odd:  v_l(deg) >= sum over odd q of that quantity  +  (contribution of q = 2 to be read off).
Usage: python3 scripts/step7_twist2.py [work|holdout|ext]
"""
import sys, re
import numpy as np, pandas as pd
tag = sys.argv[1] if len(sys.argv) > 1 else "work"
opt = pd.read_parquet(f"results/optimal_{tag}.parquet").set_index("label")
NM = int(opt.N.max()) + 1; spf = np.arange(NM)
for i in range(2, int(NM ** 0.5) + 1):
    if spf[i] == i:
        sl = spf[i * i::i]; sl[sl == np.arange(i * i, NM, i)] = i
def factor(n):
    out = []
    while n > 1:
        p = int(spf[n]); e = 0
        while n % p == 0: n //= p; e += 1
        out.append((p, e))
    return out
def vl(x, l):
    v = 0; x = int(x)
    while x > 0 and x % l == 0: x //= l; v += 1
    return v
def kind(k):
    if k >= 5: return "In"
    if k <= -5: return "In*"
    return {1: "I0", 2: "II", -2: "II*", 3: "III", -3: "III*", 4: "IV", -4: "IV*", -1: "I0*"}[k]
rows = []
for lab, r in opt.iterrows():
    ks = list(map(int, re.findall(r"-?\d+", str(r.kodaira))))
    for (q, e), k in zip(factor(int(r.N)), ks):
        rows.append((lab, q, kind(k), k - 4 if k >= 5 else (-k - 4 if k <= -5 else 0)))
loc = pd.DataFrame(rows, columns=["label", "q", "type", "n"])
nocm = ~opt.is_cm.astype(bool); mid = opt.max_isogeny_degree.astype("int64")
print(f"{tag}: optimal curves {len(opt)}")
for l in (3, 5, 7, 11, 13):
    keep = opt.index[nocm & (mid % l != 0)]
    t = loc[loc.label.isin(keep)].copy()
    odd = t.q != 2
    t["mult"] = np.where(t.type == "In", t.n.map(lambda n: vl(n, l)), 0)                 # Conjecture 1, multiplicative part (any q)
    t["iv"] = ((t.type.isin(["IV", "IV*"])) & (l == 3)).astype(int)                        # Conjecture 1, additive part (any q)
    t["ii_odd"] = ((t.type.isin(["II", "II*"])) & odd & (l == 3)).astype(int)
    t["ii_2"] = ((t.type.isin(["II", "II*"])) & ~odd & (l == 3)).astype(int)
    t["ins_odd"] = np.where((t.type == "In*") & odd, t.n.map(lambda n: vl(n, l)), 0)
    t["ins_2"] = np.where((t.type == "In*") & ~odd, t.n.map(lambda n: vl(n, l)), 0)
    t["iv_2"] = ((t.type.isin(["IV", "IV*"])) & ~odd & (l == 3)).astype(int)
    g = t.groupby("label"); d = g[["mult", "iv", "ii_odd", "ii_2", "ins_odd", "ins_2", "iv_2"]].sum().reindex(keep).fillna(0).astype(int)
    d["V"] = [vl(x, l) for x in opt.moddeg.reindex(keep).astype("int64")]
    d["base"] = d.mult + d.iv                                   # Conjecture 1 as stated
    d["Y"] = d.V - d.base
    print(f"\n######## l = {l}: curves {len(d)}; with ii_odd>0: {(d.ii_odd>0).sum()}, ii_2>0: {(d.ii_2>0).sum()}, ins_odd>0: {(d.ins_odd>0).sum()}, ins_2>0: {(d.ins_2>0).sum()}")
    for name, pred in [("R_odd: base + ii_odd + ins_odd", d.base + d.ii_odd + d.ins_odd),
                       ("R_odd + ii_2", d.base + d.ii_odd + d.ins_odd + d.ii_2),
                       ("R_odd + ins_2", d.base + d.ii_odd + d.ins_odd + d.ins_2),
                       ("old Conjecture 2 (all but largest, any q)", d.base + ((d.ii_odd + d.ii_2 + d.ins_odd + d.ins_2) - np.maximum(np.maximum(d.ii_odd + d.ii_2 > 0, 0), 0)).clip(lower=0))]:
        print(f"   {name:48s} violations {int((d.V < pred).sum()):7d}")
    if l == 3:
        print("   min Y by (ii_odd, ins_odd) for curves with no twisted prime at 2:")
        s = d[(d.ii_2 == 0) & (d.ins_2 == 0)]; print(s.groupby([s.ii_odd.clip(upper=3), s.ins_odd.clip(upper=3)]).Y.agg(["min", "size"]).to_string())
        print("   min Y - (ii_odd + ins_odd) by (ii_2, ins_2) (contribution of the prime 2 beyond the odd rule):")
        d["Y2"] = d.Y - d.ii_odd - d.ins_odd; print(d.groupby([d.ii_2.clip(upper=2), d.ins_2.clip(upper=3)]).Y2.agg(["min", "size"]).to_string())
        print("   isolated: single odd II/II* prime, nothing else: V distribution", d[(d.ii_odd == 1) & (d.base == 0) & (d.ins_odd == 0) & (d.ii_2 == 0) & (d.ins_2 == 0)].V.clip(upper=4).value_counts().sort_index().to_dict())
        print("   isolated: single II/II* prime at 2, nothing else: V distribution", d[(d.ii_2 == 1) & (d.base == 0) & (d.ins_odd == 0) & (d.ii_odd == 0) & (d.ins_2 == 0)].V.clip(upper=4).value_counts().sort_index().to_dict())
        print("   isolated: single odd In* prime with 3|n, nothing else: V distribution", d[(d.ins_odd >= 1) & (d.base == 0) & (d.ii_odd == 0) & (d.ii_2 == 0) & (d.ins_2 == 0)].groupby("ins_odd").V.apply(lambda s: s.clip(upper=4).value_counts().sort_index().to_dict()).to_dict())
        print("   isolated: single In* prime at 2 with 3|n, nothing else: V distribution", d[(d.ins_2 >= 1) & (d.base == 0) & (d.ii_odd == 0) & (d.ii_2 == 0) & (d.ins_odd == 0)].groupby("ins_2").V.apply(lambda s: s.clip(upper=4).value_counts().sort_index().to_dict()).to_dict())
    else:
        print("   isolated: single odd In* prime with l|n and no multiplicative l-contribution: V distribution", d[(d.ins_odd >= 1) & (d.base == 0) & (d.ins_2 == 0)].groupby("ins_odd").V.apply(lambda s: s.clip(upper=3).value_counts().sort_index().to_dict()).to_dict())
        print("   isolated: In* at 2 with l|n and nothing else:", d[(d.ins_2 >= 1) & (d.base == 0) & (d.ins_odd == 0)].groupby("ins_2").V.apply(lambda s: s.clip(upper=3).value_counts().sort_index().to_dict()).to_dict())
    d.to_parquet(f"results/step7_twist2_{tag}_l{l}.parquet")
