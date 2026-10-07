"""Second build stage (step 5 preparation): add the columns KNOWN_RELATIONS.md section 6 asks for.

Reads data/curves.parquet and data/ap.parquet (from build_features.py) plus data/pari/periods.*.tsv
(from pari_periods.gp), writes them back with extra columns, and writes data/ap_summary.parquet.

Added to curves.parquet
  c4_sign, log_abs_c4, c6_sign, log_abs_c6      from the minimal model
  omega1, omega2_im, area, log_area              periods and lattice covolume (N <= 400000 only)
  petersson_resid                                log moddeg - log N + log area, optimal curves only (T12)
  v2_moddeg, v3_moddeg, v5_moddeg, v7_moddeg, log_odd_moddeg
  v2/v3/v5/v7 of tamagawa and sha_an, log_odd_tamagawa, log_odd_sha, v2_torsion, odd_torsion
  n_kod_In, n_kod_II, n_kod_III, n_kod_IV, n_kod_I0s, n_kod_Ins, n_kod_IIs, n_kod_IIIs, n_kod_IVs   Kodaira type counts
  has_2_isog, has_3_isog, has_5_isog, has_7_isog   from the isogeny degrees of the class
  exceptional_image                              non-CM curve with a mod-l image not of Borel or split-Cartan type
  nagao_100, nagao_300, nagao_1000, frac_even, frac_zero, st_mean, st_m2, mur_lo, mur_mid, mur_hi
                                                 a_p summaries of the class (also in ap_summary.parquet)
"""
import glob, re, sys
import numpy as np, pandas as pd

df = pd.read_parquet("data/curves.parquet")
ap = pd.read_parquet("data/ap.parquet")
print("loaded", df.shape, ap.shape, file=sys.stderr)

# periods
def read_periods(f):
    # PARI prints small reals as "1.234 E-5" (space before the exponent); remove the space before parsing
    t = pd.read_csv(f, sep="\t", header=None, names=["N", "iso", "num", "c4", "c6", "omega1", "omega2_im", "area"],
                    dtype={"N": "int64", "iso": str, "num": "int64", "c4": float, "c6": float, "omega1": str, "omega2_im": str, "area": str})
    for c in ("omega1", "omega2_im", "area"):
        t[c] = t[c].str.replace(" E", "E", regex=False).astype(float)
    return t
per = pd.concat([read_periods(f) for f in sorted(glob.glob("data/pari/periods.*.tsv"))], ignore_index=True)
per["label"] = per["N"].astype(str) + per["iso"] + per["num"].astype(str)
per = per.drop(columns=["N", "iso", "num"])
assert not per["label"].duplicated().any(), "duplicate labels in period files"
print("periods", per.shape, file=sys.stderr)
for c in ["c4", "c6", "omega1", "omega2_im", "area", "log_abs_c4", "log_abs_c6", "c4_sign", "c6_sign", "log_area", "petersson_resid"]:
    if c in df.columns: df = df.drop(columns=[c])
df = df.merge(per, on="label", how="left")
df["c4_sign"] = np.sign(df["c4"]); df["c6_sign"] = np.sign(df["c6"])
df["log_abs_c4"] = np.log(np.maximum(df["c4"].abs(), 1.0)); df["log_abs_c6"] = np.log(np.maximum(df["c6"].abs(), 1.0))
df["log_area"] = np.log(df["area"])
# check the table's real period against PARI: Omega = omega1 * (2 if D > 0 else 1)
chk = df["omega1"] * np.where(df["disc_sign"] > 0, 2.0, 1.0)
rel = ((chk - df["omega"]).abs() / df["omega"])[df["omega1"].notna()]
print("real period check: max rel err", rel.max(), file=sys.stderr)
assert rel.max() < 1e-6
df["petersson_resid"] = np.where((df["optimal"] == 1).fillna(False).to_numpy(), np.log(df["moddeg"].astype(float)) - df["log_conductor"] + df["log_area"], np.nan)

# valuations and odd parts
def vp(x, p):
    x = x.astype("int64").to_numpy().copy(); v = np.zeros(len(x), dtype=np.int16)
    m = x > 0
    while True:
        d = m & (x % p == 0)
        if not d.any(): break
        v[d] += 1; x[d] //= p
    return v
for p in (2, 3, 5, 7): df[f"v{p}_moddeg"] = vp(df["moddeg"], p)
df["log_odd_moddeg"] = np.log(df["moddeg"].astype(float) / 2.0 ** df["v2_moddeg"])
for p in (2, 3, 5, 7):
    df[f"v{p}_tamagawa"] = vp(df["tamagawa"], p); df[f"v{p}_sha"] = vp(df["sha_an"], p)
df["log_odd_tamagawa"] = np.log(df["tamagawa"].astype(float) / 2.0 ** df["v2_tamagawa"])
df["log_odd_sha"] = np.log(df["sha_an"].astype(float) / 2.0 ** df["v2_sha"])
# Kodaira type counts per curve (PARI codes: 1 I0, 2 II, 3 III, 4 IV, n+4 I_n, -1 I0*, -2 II*, -3 III*, -4 IV*, -n-4 I_n*)
def kod_counts(s):
    ks = list(map(int, re.findall(r"-?\d+", s)))
    return (sum(k >= 5 for k in ks), sum(k == 2 for k in ks), sum(k == 3 for k in ks), sum(k == 4 for k in ks),
            sum(k == -1 for k in ks), sum(k <= -5 for k in ks), sum(k == -2 for k in ks), sum(k == -3 for k in ks), sum(k == -4 for k in ks))
kc = np.array([kod_counts(s) for s in df["kodaira"]], dtype=np.int16)
for i, name in enumerate(["n_kod_In", "n_kod_II", "n_kod_III", "n_kod_IV", "n_kod_I0s", "n_kod_Ins", "n_kod_IIs", "n_kod_IIIs", "n_kod_IVs"]):
    df[name] = kc[:, i]
df["v2_torsion"] = vp(df["torsion"], 2); df["odd_torsion"] = (df["torsion"].astype("int64") // 2 ** df["v2_torsion"].astype("int64")).astype("int16")

# isogeny flags from the maximal cyclic isogeny degree of the class (Kenku's list makes this exact)
m = df["max_isogeny_degree"].astype("int64")
df["has_2_isog"] = (m % 2 == 0); df["has_3_isog"] = (m % 3 == 0); df["has_5_isog"] = (m % 5 == 0); df["has_7_isog"] = (m % 7 == 0)

# exceptional mod-l image: any label that is not Borel (lB...) or split Cartan (lCs...), on a non-CM curve
lab = df["galrep_images"].fillna("").str.split()
df["exceptional_image"] = lab.map(lambda L: any(not re.match(r"\d+(B|Cs)", s) for s in L)) & ~df["is_cm"].astype(bool)

# a_p summaries per class
P = np.array([int(c[1:]) for c in ap.columns if c.startswith("a") and c[1:].isdigit()])
A = ap[[f"a{p}" for p in P]].to_numpy().astype(np.float64)
Nc = ap["N"].to_numpy().astype(np.int64)
good = np.zeros(A.shape, dtype=bool)
for s in range(0, len(ap), 200000): good[s:s + 200000] = (Nc[s:s + 200000, None] % P[None, :]) != 0
logp = np.log(P); sq = np.sqrt(P)
summ = pd.DataFrame({"class_label": ap["class_label"]})
for B in (100, 300, 1000):
    w = (P <= B)
    summ[f"nagao_{B}"] = (A[:, w] * (logp[w] / P[w])[None, :]).sum(1) / np.log(B)
ngood = good.sum(1)
summ["frac_even"] = ((A % 2 == 0) & good).sum(1) / ngood
summ["frac_zero"] = ((A == 0) & good).sum(1) / ngood
An = A / sq[None, :]
summ["st_mean"] = (An * good).sum(1) / ngood
summ["st_m2"] = (An ** 2 * good).sum(1) / ngood
for name, lo, hi in (("mur_lo", 0, 100), ("mur_mid", 100, 400), ("mur_hi", 400, 1000)):
    w = (P > lo) & (P <= hi); g = good[:, w]
    summ[name] = (An[:, w] * g).sum(1) / np.maximum(g.sum(1), 1)
summ.to_parquet("data/ap_summary.parquet", index=False)
for c in summ.columns:
    if c != "class_label" and c in df.columns: df = df.drop(columns=[c])
df = df.merge(summ, on="class_label", how="left")
df.to_parquet("data/curves.parquet", index=False)
print("curves.parquet", df.shape, file=sys.stderr)
print("petersson residual (optimal, N<=400000): mean %.3f sd %.3f" % (df.petersson_resid.mean(), df.petersson_resid.std()), file=sys.stderr)
