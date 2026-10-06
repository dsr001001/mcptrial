"""Generate the LaTeX tables of the note from the result files (no hand transcription of numbers).
Inputs: results/check_{work,holdout,ext,oot}_summary.csv (scripts/conj_check.py), results/step6_u1_fit_work.txt,
results/step7_twist2_{work,holdout,ext}_l{l}.parquet (scripts/step7_twist2.py), results/oot_table.parquet."""
import pandas as pd, os, re
os.makedirs("paper/tables", exist_ok=True)
sets = [("work", r"working set, $N \le 300000$"), ("holdout", r"hold-out, $300000 < N \le 400000$"), ("ext", r"extension, $400000 < N \le 500000$"), ("oot", r"out of table, $500000 < N \le 4\cdot 10^6$")]
S = {k: pd.read_csv(f"results/check_{k}_summary.csv") for k, _ in sets if os.path.exists(f"results/check_{k}_summary.csv")}
def n(x): return f"{int(x):,}"
# Table 1: curves per set, positive bounds and violations of Conjectures 1-3 by prime
with open("paper/tables/counts.tex", "w") as f:
    f.write(r"\begin{tabular}{llrrrrrr}" "\n" r"\hline" "\n")
    f.write(r"set & $\ell$ & curves & \eqref{eq:main} $>0$ & \eqref{eq:twist} $>0$ & viol.\ \eqref{eq:main} & viol.\ \eqref{eq:twist} & Eis.\ $>0$ / viol. \\" "\n" r"\hline" "\n")
    for k, name in sets:
        if k not in S: continue
        d = S[k]
        for l in (3, 5, 7, 11, 13):
            r = d[d.l == l]
            if r.empty: continue
            r = r.iloc[0]
            f.write(f"{name if l == 3 else ''} & {l} & {n(r.curves)} & {n(r.non_eis_with_bound)} & {n(r.non_eis_with_bound2)} & {int(r.viol_conj1)} & {int(r.viol_conj2)} & {n(r.eis_with_bound)} / {int(r.viol_conj3)} \\\\\n")
        f.write(r"\hline" "\n")
    f.write(r"\end{tabular}" "\n")
# Table 2: isolated configurations at l = 3. Untwisted additive types (any q) from the step 6 fit output;
# twisted types from the step 7 re-examination, separated into odd q and q = 2.
iso = {}
for line in open("results/step6_u1_fit_work.txt"):
    m = re.match(r"^\s*(\S+)\s+x(\d): n=\s*(\d+)\s+min V = (\d+)", line)
    if m:
        ty, k, cnt, mn = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        iso.setdefault(ty, {})[k] = (cnt, mn)
    if line.startswith("################ l = 2"): break
d3 = pd.read_parquet("results/step7_twist2_work_l3.parquet")
zero = (d3.base == 0)
def cell(mask):
    s = d3.V[mask]
    return f"{int(s.min())} ({len(s):,})" if len(s) else "--"
tw_rows = [("II, II$^*$ at odd $q$", lambda k: zero & (d3.ii_odd == k) & (d3.ins_odd == 0) & (d3.ii_2 == 0) & (d3.ins_2 == 0)),
           ("$\\mathrm{I}_n^*$ at odd $q$, weight $v_3(n)$", lambda k: zero & (d3.ins_odd == k) & (d3.ii_odd == 0) & (d3.ii_2 == 0) & (d3.ins_2 == 0)),
           ("II, II$^*$ at $q=2$", lambda k: zero & (d3.ii_2 == k) & (d3.ii_odd == 0) & (d3.ins_odd == 0) & (d3.ins_2 == 0)),
           ("$\\mathrm{I}_n^*$ at $q=2$, weight $v_3(n)$", lambda k: zero & (d3.ins_2 == k) & (d3.ii_odd == 0) & (d3.ins_odd == 0) & (d3.ii_2 == 0))]
with open("paper/tables/isolated3.tex", "w") as f:
    f.write(r"\begin{tabular}{lccc}" "\n" r"\hline" "\n" r"type & weight 1 & weight 2 & weight 3 \\" "\n" r"\hline" "\n")
    for ty in ["IV", "IV*", "III", "III*", "I0*"]:
        cells = []
        for k in (1, 2, 3):
            if k in iso.get(ty, {}): cnt, mn = iso[ty][k]; cells.append(f"{mn} ({cnt:,})")
            else: cells.append("--")
        name = {"IV": "IV", "IV*": "IV$^*$", "III": "III", "III*": "III$^*$", "I0*": "$\\mathrm{I}_0^*$"}[ty]
        f.write(name + " (any $q$) & " + " & ".join(cells) + " \\\\\n")
    for name, fn in tw_rows:
        f.write(name + " & " + " & ".join(cell(fn(k)) for k in (1, 2, 3)) + " \\\\\n")
    f.write(r"\hline" "\n" r"\end{tabular}" "\n")
# Table 3: the twisted primes, every odd l, all three ranges (from the step 7 per-curve files)
with open("paper/tables/twist.tex", "w") as f:
    f.write(r"\begin{tabular}{llrrrr}" "\n" r"\hline" "\n")
    f.write(r"set & $\ell$ & odd twisted weight $>0$ & viol.\ \eqref{eq:twist} & twisted prime $2$ & viol.\ if $2$ counted \\" "\n" r"\hline" "\n")
    for k, name in sets[:3]:
        for l in (3, 5, 7, 11, 13):
            p = f"results/step7_twist2_{k}_l{l}.parquet"
            if not os.path.exists(p): continue
            d = pd.read_parquet(p)
            odd = d.ii_odd + d.ins_odd; two = d.ii_2 + d.ins_2
            f.write(f"{name if l == 3 else ''} & {l} & {n((odd > 0).sum())} & {int((d.V < d.base + odd).sum())} & {n((two > 0).sum())} & {n((d.V < d.base + odd + two).sum())} \\\\\n")
        f.write(r"\hline" "\n")
    f.write(r"\end{tabular}" "\n")
# Table 4: out-of-table families
if os.path.exists("results/oot_table.parquet"):
    t = pd.read_parquet("results/oot_table.parquet")
    with open("paper/tables/oot.tex", "w") as f:
        f.write(r"\begin{tabular}{lrrr}" "\n" r"\hline" "\n" r"family & curves & largest conductor & median seconds per degree \\" "\n" r"\hline" "\n")
        for fam, g in t.groupby("family"):
            f.write(f"{fam} & {len(g):,} & {int(g.N.max()):,} & {g.seconds.median():.1f} \\\\\n")
        f.write(f"all & {len(t):,} & {int(t.N.max()):,} & {t.seconds.median():.1f} \\\\\n" r"\hline" "\n" r"\end{tabular}" "\n")
print("tables written:", sorted(os.listdir("paper/tables")))
