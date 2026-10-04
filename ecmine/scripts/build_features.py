"""Build the flat feature table (step 2) from the frozen ecdata snapshot + PARI pass.

Outputs
  data/curves.parquet : one row per curve (3,064,705 rows), class-level columns joined in
  data/ap.parquet     : one row per isogeny class, a_p for the 168 primes p < 1000 (int16)
"""
import glob, re, sys, os
import numpy as np, pandas as pd

RAW = "data/raw"; PARI = "data/pari"

LIMIT = int(os.environ.get("ECMINE_LIMIT", "0"))  # debug: use only the first LIMIT range files
def files(d):
    fs = sorted(glob.glob(f"{RAW}/{d}/{d}.*"))
    return fs[:LIMIT] if LIMIT else fs

def read_split(d, names, usecols=None, sep=r"\s+", **kw):
    parts = [pd.read_csv(f, sep=sep, header=None, names=names, usecols=usecols,
                         engine="python" if len(sep) > 1 else "c", **kw) for f in files(d)]
    return pd.concat(parts, ignore_index=True)

def key(df): return df["N"].astype(str) + df["iso"] + df["num"].astype(str)

# allcurves: N iso num ainvs rank torsion
cur = read_split("allcurves", ["N", "iso", "num", "ainvs", "rank", "torsion"])
cur["label"] = key(cur)
print("allcurves", len(cur), file=sys.stderr)

# allgens: N iso num ainvs rank [torsion structure] gens...  -> keep torsion structure
rows = []
for f in files("allgens"):
    with open(f) as fh:
        for line in fh:
            v = line.split()
            rows.append((v[0] + v[1] + v[2], v[5]))
tg = pd.DataFrame(rows, columns=["label", "torsion_structure"])
tg["n_torsion_gens"] = tg["torsion_structure"].map(lambda s: 0 if s == "[]" else s.count(",") + 1)

# allbsd: N iso num ainvs rank torsion tamagawa omega lvalue reg sha
bsd = read_split("allbsd", ["N", "iso", "num", "ainvs", "rank", "torsion",
                            "tamagawa", "omega", "lvalue", "regulator", "sha_an"],
                 usecols=["N", "iso", "num", "tamagawa", "omega", "lvalue", "regulator", "sha_an"])
bsd["label"] = key(bsd); bsd = bsd.drop(columns=["N", "iso", "num"])
bsd["sha_an"] = bsd["sha_an"].round().astype("int64")

# alldegphi: N iso num ainvs moddeg
dp = read_split("alldegphi", ["N", "iso", "num", "ainvs", "moddeg"], usecols=["N", "iso", "num", "moddeg"])
dp["label"] = key(dp); dp = dp.drop(columns=["N", "iso", "num"])

# opt_man: N iso num ainvs optimal manin
om = read_split("opt_man", ["N", "iso", "num", "ainvs", "optimal", "manin"], usecols=["N", "iso", "num", "optimal", "manin"])
om["label"] = key(om); om = om.drop(columns=["N", "iso", "num"])

# intpts: label ainvs [x1,x2,...]
rows = []
for f in files("intpts"):
    with open(f) as fh:
        for line in fh:
            v = line.split(); xs = v[2]
            rows.append((v[0], 0 if xs == "[]" else xs.count(",") + 1))
ip = pd.DataFrame(rows, columns=["label", "n_intpts_x"])

# 2adic: N iso num ainvs index level gens label
ta = read_split("2adic", ["N", "iso", "num", "ainvs", "two_adic_index", "two_adic_level", "two_adic_gens", "two_adic_label"],
                usecols=["N", "iso", "num", "two_adic_index", "two_adic_level", "two_adic_label"])
ta["label"] = key(ta); ta = ta.drop(columns=["N", "iso", "num"])

# galrep: label [image labels for non-surjective primes]
rows = []
for f in files("galrep"):
    with open(f) as fh:
        for line in fh:
            v = line.split()
            imgs = v[1:]
            ps = sorted({int(re.match(r"(\d+)", s).group(1)) for s in imgs})
            rows.append((v[0], len(imgs), " ".join(imgs), " ".join(map(str, ps))))
gr = pd.DataFrame(rows, columns=["label", "n_nonsurj_primes", "galrep_images", "nonsurj_primes"])

# allisog (per class): N iso num ainvs [curves] [[isogeny matrix]]
rows = []
for f in files("allisog"):
    with open(f) as fh:
        for line in fh:
            v = line.split()
            mat = v[5]
            degs = [int(x) for x in re.findall(r"\d+", mat)]
            size = int(round(len(degs) ** 0.5))
            rows.append((int(v[0]), v[1], size, max(degs)))
isog = pd.DataFrame(rows, columns=["N", "iso", "class_size", "max_isogeny_degree"])

# PARI per-curve pass
pc = pd.concat([pd.read_csv(f, sep="\t", header=None,
                            names=["N", "iso", "num", "root_number", "disc_sign", "log_abs_disc", "log_j_height",
                                   "n_bad_primes", "n_split_mult", "n_nonsplit_mult", "n_additive",
                                   "kodaira", "tamagawa_list"])
                for f in (sorted(glob.glob(f"{PARI}/curves.*.tsv"))[:LIMIT] if LIMIT else sorted(glob.glob(f"{PARI}/curves.*.tsv")))], ignore_index=True)
pc["label"] = key(pc); pc = pc.drop(columns=["N", "iso", "num"])

# Merge everything per curve
df = cur.merge(tg, on="label", how="left").merge(bsd, on="label", how="left") \
        .merge(dp, on="label", how="left").merge(om, on="label", how="left") \
        .merge(ip, on="label", how="left").merge(ta, on="label", how="left") \
        .merge(gr, on="label", how="left").merge(pc, on="label", how="left") \
        .merge(isog, on=["N", "iso"], how="left")
df["class_label"] = df["N"].astype(str) + df["iso"]
df["log_conductor"] = np.log(df["N"].astype(float))
# The 2adic table records an infinite index/level for CM curves: use that as the CM flag.
df["is_cm"] = np.isinf(df["two_adic_index"].astype(float))
df.loc[df["is_cm"], ["two_adic_index", "two_adic_level"]] = np.nan
df["n_conductor_primes"] = df["n_bad_primes"]
for c in ["rank", "torsion", "sha_an", "moddeg", "optimal", "manin", "n_intpts_x", "two_adic_index", "two_adic_level",
          "n_nonsurj_primes", "root_number", "disc_sign", "n_bad_primes", "n_split_mult", "n_nonsplit_mult",
          "n_additive", "class_size", "max_isogeny_degree", "n_torsion_gens", "tamagawa"]:
    df[c] = df[c].astype("Int64")
missing = df.isna().sum(); print(missing[missing > 0].to_string(), file=sys.stderr)
df.to_parquet("data/curves.parquet", index=False)
print("curves.parquet", df.shape, file=sys.stderr)

# a_p per class
P = [p for p in range(2, 1000) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
assert len(P) == 168
rows = []
for f in (sorted(glob.glob(f"{PARI}/ap.*.tsv"))[:LIMIT] if LIMIT else sorted(glob.glob(f"{PARI}/ap.*.tsv"))):
    with open(f) as fh:
        for line in fh:
            N, iso, ap = line.rstrip("\n").split("\t")
            rows.append((int(N), iso, *[int(x) for x in ap.strip("[]").split(",")]))
ap = pd.DataFrame(rows, columns=["N", "iso", *[f"a{p}" for p in P]])
for p in P: ap[f"a{p}"] = ap[f"a{p}"].astype("int16")
ap["class_label"] = ap["N"].astype(str) + ap["iso"]
cls = df[df["num"] == 1][["class_label", "rank", "root_number", "torsion", "sha_an", "log_conductor", "class_size", "is_cm"]]
ap = ap.merge(cls, on="class_label", how="left")
ap.to_parquet("data/ap.parquet", index=False)
print("ap.parquet", ap.shape, file=sys.stderr)
