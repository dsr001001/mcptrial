"""Build paper/tables/euler_uniform.tex from results/step10_rule_{tag}_summary.csv.

Columns: set, l, curves, rule>0 (III terms dropped at every prime), violations,
equality rate; violations with the III terms added at every prime; and the
variant without the weights at q = l (>0 / violations / equality rate).
"""
import pandas as pd

SETS = [("work", "working set"), ("holdout", "hold-out"), ("ext", "extension"), ("oot", "out of table")]

def pct(x, n):
    return "--" if n == 0 else f"{100*x:.1f}\\%"

rows = []
for tag, name in SETS:
    df = pd.read_csv(f"results/step10_rule_{tag}_summary.csv")
    first = True
    for r in df.to_dict("records"):
        rows.append(
            f"{name if first else ''} & {r['l']} & {r['curves']:,} & {r['noIIIall>0']:,} & {r['viol_noIIIall']} & "
            f"{pct(r['eq_noIIIall'], r['noIIIall>0'])} & {r['viol_full']} & "
            f"{r['noql>0']:,} / {r['viol_noql']} / {pct(r['eq_noql'], r['noql>0'])} \\\\"
        )
        first = False
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"

with open("paper/tables/euler_uniform.tex", "w") as fh:
    fh.write("\\begin{tabular}{llrrrrrr}\n\\toprule\n")
    fh.write("set & $\\ell$ & curves & \\eqref{eq:rule} $>0$ & viol. & equality & viol.\\ $+\\mathrm{III}$ & no $q=\\ell$: $>0$ / viol. / equality\\\\\n\\midrule\n")
    fh.write("\n".join(rows) + "\n\\end{tabular}\n")
print(open("paper/tables/euler_uniform.tex").read())
