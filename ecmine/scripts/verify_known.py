"""Step 3 support: verify known exact relations on the table (N <= 400000).
Every one of these is a theorem or an identity; a nonzero violation count means the TABLE is wrong."""
import re, numpy as np, pandas as pd
NMAX = 400000
df = pd.read_parquet("data/curves.parquet"); df = df[df.N <= NMAX].reset_index(drop=True)
ap = pd.read_parquet("data/ap.parquet"); ap = ap[ap.N <= NMAX].reset_index(drop=True)
P = np.array([int(c[1:]) for c in ap.columns if c.startswith("a") and c[1:].isdigit()])
A = ap[[f"a{p}" for p in P]].to_numpy()                      # classes x 168, int16
cls_index = pd.Series(np.arange(len(ap)), index=ap.class_label)
Nc = ap.N.to_numpy().astype(np.int64)
print(f"curves {len(df)}  classes {len(ap)}  primes {len(P)} (max {P.max()})")

# A. root number from local data for semistable curves: w = (-1)^(1 + #split multiplicative primes)
ss = df.n_additive == 0
viol = (df.loc[ss, "root_number"] != (-1) ** (1 + df.loc[ss, "n_split_mult"])).sum()
print(f"[A] semistable curves {ss.sum()}: root number != (-1)^(1+n_split): {viol}")

# B. a_p at bad primes p|N is in {0,+1,-1}; +1 split, -1 nonsplit, 0 additive
bad = np.zeros_like(A, dtype=bool)
for s in range(0, len(ap), 200000):
    bad[s:s+200000] = (Nc[s:s+200000, None] % P[None, :]) == 0
print(f"[B] bad-prime a_p outside {{0,+-1}}: {int((np.abs(A[bad]) > 1).sum())}  (bad-prime entries: {int(bad.sum())})")
opt = df[df.num == 1].set_index("class_label").loc[ap.class_label]
# count split primes < 1000 from a_p and compare with n_split_mult when all bad primes are < 1000
nsplit_ap = ((A == 1) & bad).sum(1); nns_ap = ((A == -1) & bad).sum(1); nadd_ap = ((A == 0) & bad).sum(1)
nbad_lt1000 = bad.sum(1)
allsmall = nbad_lt1000 == opt.n_bad_primes.to_numpy()
print(f"[B] classes with all bad primes < 1000: {allsmall.sum()}; split/nonsplit/additive counts mismatch vs PARI: "
      f"{int((nsplit_ap[allsmall] != opt.n_split_mult.to_numpy()[allsmall]).sum())} / "
      f"{int((nns_ap[allsmall] != opt.n_nonsplit_mult.to_numpy()[allsmall]).sum())} / "
      f"{int((nadd_ap[allsmall] != opt.n_additive.to_numpy()[allsmall]).sum())}")

# C. torsion injects into E(F_p): |T| divides p+1-a_p for good p not dividing |T|
tot_viol = 0; tot_tests = 0
for t in sorted(df.torsion.unique()):
    if t == 1: continue
    sub = df[df.torsion == t]
    rows = cls_index.loc[sub.class_label].to_numpy()
    a = A[rows]; b = bad[rows]
    ok = (~b) & ((t % P) != 0)[None, :]   # good p not dividing |T|
    r = (P[None, :] + 1 - a) % t
    v = int(((r != 0) & ok).sum()); tot_viol += v; tot_tests += int(ok.sum())
    if v: print(f"   torsion {t}: {v} violations")
print(f"[C] torsion congruence a_p = p+1 mod |T|: violations {tot_viol} of {tot_tests} tests")

# D. full rational 2-torsion forces three real roots, i.e. positive discriminant
full2 = df.torsion_structure.str.startswith("[2,")
print(f"[D] curves with Z/2xZ/2 torsion {full2.sum()}: discriminant negative: {int((df.loc[full2,'disc_sign'] < 0).sum())}")

# E. N divides the minimal discriminant
print(f"[E] log|D| < log N: {int((df.log_abs_disc + 1e-9 < df.log_conductor).sum())}")

# F. period relation: log Omega ~ -(1/12) log max(|c4|^3, |D|) + const, with a factor 2 for two real components
y = df.omega.map(np.log) - np.log(2) * (df.disc_sign > 0)
x = df.log_j_height
for name, m in [("all", np.ones(len(df), bool)), ("semistable", ss.to_numpy())]:
    X = np.c_[np.ones(m.sum()), x[m]]; coef, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    res = y[m] - X @ coef
    print(f"[F] log(Omega/2^[D>0]) = {coef[0]:.3f} + {coef[1]:.4f} * log_j_height  ({name}; expected slope -1/12 = {-1/12:.4f}; "
          f"R2 {1 - res.var()/y[m].var():.4f}, resid sd {res.std():.3f}, resid range [{res.min():.2f},{res.max():.2f}])")

# G. Watkins' conjecture: 2^rank divides the modular degree
v2 = np.array([int(m) & -int(m) for m in df.moddeg]); v2 = np.log2(v2).astype(int)
print(f"[G] Watkins 2^rank | moddeg: violations all curves {int((v2 < df['rank']).sum())}, optimal {int((v2 < df['rank'])[df.optimal == 1].sum())}")

# H. Lorenzini: torsion of order divisible by 5, 7 or 9 forces 5, 7 or 3 to divide the Tamagawa product (finitely many exceptions)
for tdiv, cdiv in [(5, 5), (7, 7), (9, 3), (3, 3), (4, 2), (2, 2)]:
    sub = df[df.torsion % tdiv == 0]
    exc = sub[sub.tamagawa % cdiv != 0]
    print(f"[H] {tdiv} | |T| ({len(sub)} curves): Tamagawa not divisible by {cdiv}: {len(exc)}" + (f"  e.g. {list(exc.label[:6])}" if 0 < len(exc) <= 60 else ""))

# I. Tate's algorithm constraints: Kodaira symbol -> allowed Tamagawa number
def allowed(kod, c):
    if kod >= 5:   return c == kod - 4 or c in (1, 2)       # I_n: split c=n, nonsplit c in {1,2}
    if kod == 1:   return c == 1
    if kod in (2, -2): return c == 1
    if kod in (3, -3): return c == 2
    if kod in (4, -4): return c in (1, 3)
    if kod == -1:  return c in (1, 2, 4)                     # I_0*
    if kod <= -5:  return c in (2, 4)                         # I_n*
    return False
bad_rows = 0; pairs = 0
for k, t in zip(df.kodaira, df.tamagawa_list):
    ks = list(map(int, re.findall(r"-?\d+", k))); ts = list(map(int, re.findall(r"\d+", t)))
    pairs += len(ks)
    if not all(allowed(a, b) for a, b in zip(ks, ts)): bad_rows += 1
print(f"[I] Kodaira/Tamagawa pairs {pairs}: curves violating Tate's algorithm constraints: {bad_rows}")

# J. L-value (hence Omega*Reg*Tam*Sha/|T|^2) is constant within an isogeny class (Cassels)
g = df.groupby("class_label").lvalue.agg(["min", "max"]); spread = ((g["max"] - g["min"]) / g["max"]).max()
print(f"[J] max relative spread of L-value within a class: {spread:.2e}")

# K. CM curves have a_p = 0 at every inert prime (half of all primes)
good = ~bad
frac0 = ((A == 0) & good).sum(1) / good.sum(1)
cm = ap.is_cm.to_numpy().astype(bool)
print(f"[K] fraction of good p<1000 with a_p=0: CM classes min/median {frac0[cm].min():.3f}/{np.median(frac0[cm]):.3f}; "
      f"non-CM max/median {frac0[~cm].max():.3f}/{np.median(frac0[~cm]):.3f}")

# L. modular degree vs conductor and period (optimal curves): log deg ~ a + b log N + c log Omega
o = df[df.optimal == 1]
X = np.c_[np.ones(len(o)), o.log_conductor, np.log(o.omega)]; yy = np.log(o.moddeg.astype(float))
coef, *_ = np.linalg.lstsq(X, yy, rcond=None); res = yy - X @ coef
print(f"[L] log moddeg = {coef[0]:.2f} + {coef[1]:.3f} log N + {coef[2]:.3f} log Omega  (optimal curves; R2 {1-res.var()/yy.var():.4f}, resid sd {res.std():.3f})")
X = np.c_[np.ones(len(o)), o.log_conductor, np.log(o.omega), o.log_j_height]
coef, *_ = np.linalg.lstsq(X, yy, rcond=None); res = yy - X @ coef
print(f"[L] ... + {coef[3]:.3f} log_j_height: coefs N {coef[1]:.3f}, Omega {coef[2]:.3f}; R2 {1-res.var()/yy.var():.4f}, resid sd {res.std():.3f}")

# M. baselines that the literature already reports on this very data
bins = pd.cut(ap.N, [0, 1000, 10000, 50000, 100000, 200000, 300000, 400000])
print("[M] average rank by conductor bin (classes):\n" + ap.groupby(bins, observed=True)["rank"].mean().round(3).to_string())
print("[M] P(Sha>1) by rank (curves):\n" + df.groupby("rank").sha_an.apply(lambda s: (s > 1).mean()).round(4).to_string())
print("[M] mean rank by torsion structure (curves, N<=400000):\n" + df.groupby("torsion_structure")["rank"].agg(["mean", "size"]).round(3).to_string())
