"""Sanity checks on the flat table: known identities must hold exactly, or the table is wrong."""
import numpy as np, pandas as pd
df = pd.read_parquet("data/curves.parquet"); ap = pd.read_parquet("data/ap.parquet")
print("curves", df.shape, "classes", ap.shape)
print("rows with any NA in core columns:", df[["rank","torsion","sha_an","moddeg","root_number","tamagawa","omega"]].isna().any(axis=1).sum())
# 1. parity: root number = (-1)^rank  (theorem for all curves in the table, since rank is proven there)
bad = df[df["root_number"] != (-1) ** df["rank"]]
print("parity violations:", len(bad))
# 2. BSD identity as stored: lvalue = omega * reg * tamagawa * sha / torsion^2  (analytic Sha is defined by this)
lhs = df["lvalue"]; rhs = df["omega"] * df["regulator"] * df["tamagawa"].astype(float) * df["sha_an"].astype(float) / df["torsion"].astype(float) ** 2
rel = ((lhs - rhs).abs() / lhs.abs().clip(lower=1e-12))
print("BSD identity max rel err:", rel.max(), " n > 1e-6:", (rel > 1e-6).sum())
# 3. Tamagawa product equals product of the per-prime list
prod = df["tamagawa_list"].map(lambda s: int(np.prod([int(x) for x in s.strip("[]").split(",")])) if s.strip("[]") else 1)
print("tamagawa product mismatches:", (prod != df["tamagawa"]).sum())
# 4. bad prime counts add up
print("reduction-type count mismatches:", ((df["n_split_mult"] + df["n_nonsplit_mult"] + df["n_additive"]) != df["n_bad_primes"]).sum())
# 5. exactly one optimal curve per class; class sizes agree with row counts
per = df.groupby("class_label").agg(n=("label","size"), opt=("optimal","sum"), cs=("class_size","first"))
g = df.groupby("class_label").agg(allknown=("optimal_known","all"), s=("optimal","sum")); print("classes with known optimality and optimal count != 1:", (g[g.allknown].s != 1).sum(), " class_size mismatches:", (per["n"] != per["cs"]).sum())
# 6. Hasse bound on a_p
P = [int(c[1:]) for c in ap.columns if c.startswith("a") and c[1:].isdigit()]
viol = sum(int((ap[f"a{p}"].abs() > 2 * np.sqrt(p)).sum()) for p in P)
print("Hasse bound violations:", viol)
# 7. Sha is a square (Cassels-Tate), for rank-known curves
sq = np.sqrt(df["sha_an"].astype(float)); print("non-square Sha:", (sq != sq.round()).sum())
print("rank distribution:\n", df["rank"].value_counts().sort_index().to_string())
print("Sha>1 curves:", (df["sha_an"] > 1).sum(), " CM curves:", df["is_cm"].sum())
