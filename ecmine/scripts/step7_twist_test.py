"""Step 5 of the follow-up list: test the candidate mechanism for Conjecture 2.
(a) Is the newform of a curve with two type II primes q1, q2 congruent mod 3 to its own twist by the quadratic
    character of Q(sqrt(q1* q2*))?  Test: a_p = 0 mod 3 for every good p < 1000 with chi(p) = -1.
(b) For curves with a single type II/II* prime (odd q), which ones have 3 | deg phi?  Split by q mod 3, by the
    Tamagawa number at q of the quadratic twist by q* (type IV* there), and by the mod 3 image label."""
import sys, re
import numpy as np, pandas as pd
import cypari2
pari = cypari2.Pari(); pari.allocatemem(256 * 10 ** 6)
loc = pd.read_parquet("results/local_primes.parquet"); loc = loc[loc.set == "work"]
opt = pd.read_parquet("results/optimal_work.parquet").set_index("label")
ap = pd.read_parquet("data/ap.parquet"); ap = ap[ap.N <= 300000].set_index("class_label")
P = np.array([int(c[1:]) for c in ap.columns if c.startswith("a") and c[1:].isdigit()])
cur = pd.read_parquet("data/curves.parquet", columns=["label", "class_label", "ainvs"]).set_index("label")
def v3(x):
    v = 0; x = int(x)
    while x > 0 and x % 3 == 0: x //= 3; v += 1
    return v
ne = opt[(opt.max_isogeny_degree.astype("int64") % 3 != 0) & (~opt.is_cm.astype(bool))]
t = loc[loc.label.isin(ne.index)]
hasII = set(t[t.type.isin(["II", "II*"])].label)          # only curves with a type II or II* prime are needed
t = t[t.label.isin(hasII)]
byl = {lab: grp for lab, grp in t.groupby("label")}
g = t.groupby("label")
prof = pd.DataFrame({"nII": g.type.apply(lambda s: int(s.isin(["II", "II*"]).sum())), "nIV": g.type.apply(lambda s: int(s.isin(["IV", "IV*"]).sum())),
                     "S": g.apply(lambda x: int(sum(v3(n) for n, m in zip(x.n, x.mult) if m)), include_groups=False),
                     "W": g.apply(lambda x: int(sum(v3(n) for n, ty in zip(x.n, x.type) if ty == "In*")), include_groups=False),
                     "IIodd": g.apply(lambda x: all(q % 2 == 1 for q, ty in zip(x.q, x.type) if ty in ("II", "II*")), include_groups=False)})
prof["V"] = [v3(x) for x in ne.moddeg.reindex(prof.index).astype("int64")]
qstar = lambda q: q if q % 4 == 1 else -q

# (a) two type II primes, both odd, nothing else twisted or IV, S = 0
two = prof[(prof.nII == 2) & prof.IIodd & (prof.nIV == 0) & (prof.S == 0) & (prof.W == 0)]
print(f"(a) curves with exactly two odd type II/II* primes and no other 3-contribution: {len(two)}; v3(deg) distribution: {two.V.value_counts().sort_index().to_dict()}")
hits = 0; tested = 0; frac = []
for lab in two.index:
    gg = byl[lab]; qs = [int(q) for q, ty in zip(gg.q, gg.type) if ty in ("II", "II*")]
    Dd = qstar(qs[0]) * qstar(qs[1]); N = int(ne.N[lab])
    row = ap.loc[cur.class_label[lab]]; a = row[[f"a{p}" for p in P]].to_numpy().astype(int)
    mask = np.array([(N % p != 0) and (int(pari.kronecker(Dd, int(p))) == -1) for p in P])
    if mask.sum() < 20: continue
    tested += 1; z = (a[mask] % 3 == 0); frac.append(z.mean()); hits += int(z.all())
print(f"    tested {tested}: f congruent to its twist mod 3 (all a_p = 0 mod 3 at inert primes) in {hits} cases; mean fraction of inert primes with 3 | a_p: {np.mean(frac):.3f} (1/3 expected at random)")

# (b) single odd type II/II* prime, nothing else
one = prof[(prof.nII == 1) & prof.IIodd & (prof.nIV == 0) & (prof.S == 0) & (prof.W == 0)].copy()
print(f"\n(b) curves with exactly one odd type II/II* prime and no other 3-contribution: {len(one)}; P(3 | deg) = {(one.V > 0).mean():.3f}")
rows = []
for lab in one.index:
    gg = byl[lab]; sub = gg[gg.type.isin(["II", "II*"])].iloc[0]; q = int(sub.q)
    E = pari.ellinit(pari(cur.ainvs[lab])); Et = pari.elltwist(E, qstar(q)); lr = pari.elllocalred(Et, q)
    rows.append((lab, q, q % 3, sub.type, int(lr[1]), int(lr[3])))
b = pd.DataFrame(rows, columns=["label", "q", "qmod3", "type", "twist_kod", "twist_c"]).set_index("label")
b["div3"] = (one.V.reindex(b.index) > 0)
b["lab3"] = ["" if not isinstance(x, str) else " ".join(s for s in x.split() if s.startswith("3")) for x in opt.galrep_images.reindex(b.index)]
print("   twisted Kodaira code at q (expect IV* = -4 for II, IV = 4 for II*):", b.twist_kod.value_counts().to_dict())
print("   P(3 | deg) by q mod 3:"); print(b.groupby("qmod3").div3.agg(["mean", "size"]).to_string())
print("   P(3 | deg) by Tamagawa number of the twist at q:"); print(b.groupby("twist_c").div3.agg(["mean", "size"]).to_string())
print("   P(3 | deg) by (q mod 3, twist Tamagawa):"); print(b.groupby(["qmod3", "twist_c"]).div3.agg(["mean", "size"]).to_string())
print("   P(3 | deg) by mod 3 image label ('' = surjective):"); print(b.groupby("lab3").div3.agg(["mean", "size"]).to_string())
print("   P(3 | deg) by type:"); print(b.groupby("type").div3.agg(["mean", "size"]).to_string())
b.to_parquet("results/step7_twist_single.parquet")
