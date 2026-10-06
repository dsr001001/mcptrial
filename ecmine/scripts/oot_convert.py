"""Convert the out-of-table generator output into the checker's table format (deduplicated)."""
import pandas as pd, glob
rows = []
for f in sorted(glob.glob("results/oot_*.tsv")):
    for line in open(f):
        a, N, kod, tam, deg, fam, t = line.rstrip("\n").split("\t")
        rows.append((a, int(N), int(deg), kod, tam, 1, False, 1, fam, float(t)))
d = pd.DataFrame(rows, columns=["label", "N", "moddeg", "kodaira", "tamagawa_list", "max_isogeny_degree", "is_cm", "torsion", "family", "seconds"])
d = d.drop_duplicates(subset=["label"]).reset_index(drop=True)
d.to_parquet("results/oot_table.parquet", index=False)
print("out-of-table curves (unique):", len(d), " by family:", d.family.value_counts().to_dict(), " conductor range:", d.N.min(), "-", d.N.max(), " median seconds per degree:", round(d.seconds.median(), 2))
