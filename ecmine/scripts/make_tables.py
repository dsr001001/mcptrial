"""Generate the LaTeX tables of the note from the result files (no hand transcription of numbers)."""
import pandas as pd, os, glob, re
os.makedirs("paper/tables", exist_ok=True)
sets = [("work", r"working set, $N \le 300000$"), ("holdout", r"hold-out, $300000 < N \le 400000$"), ("ext", r"extension, $400000 < N \le 500000$"), ("oot", r"out of table, $500000 < N \le 4\cdot 10^6$")]
S = {k: pd.read_csv(f"results/check_{k}_summary.csv") for k, _ in sets if os.path.exists(f"results/check_{k}_summary.csv")}
# Table 1: curves per set and violations of Conjectures 1-3 by prime
with open("paper/tables/counts.tex", "w") as f:
    f.write(r"\begin{tabular}{llrrrrr}" "\n" r"\hline" "\n")
    f.write(r"set & $\ell$ & curves & $E[\ell]$ irreducible, bound $>0$ & violations C1 & violations C2 & Eisenstein, bound $>0$ / violations C3 \\" "\n" r"\hline" "\n")
    for k, name in sets:
        if k not in S: continue
        d = S[k]
        for l in (3, 5, 7, 11, 13):
            r = d[d.l == l]
            if r.empty: continue
            r = r.iloc[0]
            c2 = "" if pd.isna(r.viol_conj2) else f"{int(r.viol_conj2)}"
            f.write(f"{name if l == 3 else ''} & {l} & {int(r.curves):,} & {int(r.non_eis_with_bound):,} & {int(r.viol_conj1)} & {c2} & {int(r.eis_with_bound):,} / {int(r.viol_conj3)} \\\\\n")
        f.write(r"\hline" "\n")
    f.write(r"\end{tabular}" "\n")
# Table 2: isolated configurations at l = 3 (from the fit output of the working set)
iso = {}
for line in open("results/step6_u1_fit_work.txt"):
    m = re.match(r"^\s*(\S+)\s+x(\d): n=\s*(\d+)\s+min V = (\d+)", line)
    if m:
        ty, k, n, mn = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        iso.setdefault(ty, {})[k] = (n, mn)
    if line.startswith("################ l = 2"): break
with open("paper/tables/isolated3.tex", "w") as f:
    f.write(r"\begin{tabular}{lccc}" "\n" r"\hline" "\n" r"type & one prime & two primes & three primes \\" "\n" r"\hline" "\n")
    for ty in ["IV", "IV*", "II", "II*", "III", "III*", "I0*", "In*"]:
        cells = []
        for k in (1, 2, 3):
            if k in iso.get(ty, {}): n, mn = iso[ty][k]; cells.append(f"{mn} ({n:,})")
            else: cells.append("--")
        name = {"IV": "IV", "IV*": "IV$^*$", "II": "II", "II*": "II$^*$", "III": "III", "III*": "III$^*$", "I0*": "$\\mathrm{I}_0^*$", "In*": "$\\mathrm{I}_n^*$"}[ty]
        f.write(name + " & " + " & ".join(cells) + " \\\\\n")
    f.write(r"\hline" "\n" r"\end{tabular}" "\n")
# Table 3: out-of-table families
if os.path.exists("results/oot_table.parquet"):
    t = pd.read_parquet("results/oot_table.parquet")
    with open("paper/tables/oot.tex", "w") as f:
        f.write(r"\begin{tabular}{lrrr}" "\n" r"\hline" "\n" r"family & curves & largest conductor & median seconds per degree \\" "\n" r"\hline" "\n")
        for fam, g in t.groupby("family"):
            f.write(f"{fam} & {len(g):,} & {int(g.N.max()):,} & {g.seconds.median():.1f} \\\\\n")
        f.write(f"all & {len(t):,} & {int(t.N.max()):,} & {t.seconds.median():.1f} \\\\\n" r"\hline" "\n" r"\end{tabular}" "\n")
print("tables written:", sorted(os.listdir("paper/tables")))
