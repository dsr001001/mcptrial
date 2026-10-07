#!/bin/sh
# Step 3: reproducibility run on a clean clone. Re-downloads the pinned ecdata snapshot, recomputes the PARI
# features, rebuilds the tables and reruns the consolidated checker; then diffs the summaries against the
# committed ones. Usage: scripts/repro_run.sh <clean-clone-dir> <reference-results-dir>
set -e
CLONE=$1; REF=$2; cd "$CLONE/ecmine"
echo "[repro] start $(date)"
sh scripts/download_ecdata.sh data/raw > /dev/null 2>&1
echo "[repro] download done $(date)  files: $(ls data/raw/allcurves | wc -l)"
mkdir -p data/pari
ls data/raw/allcurves/ | xargs -P 2 -I{} sh -c 'f={}; r=${f#allcurves.}; echo "infile=\"data/raw/allcurves/$f\"; outc=\"data/pari/curves.$r.tsv\"; outa=\"data/pari/ap.$r.tsv\"; read(\"scripts/pari_features.gp\");" | gp -q -s 256M 2>&1 | grep -v Warning; true'
echo "[repro] pari features done $(date)  rows: $(cat data/pari/curves.*.tsv | wc -l)"
python3 scripts/build_features.py 2>&1 | grep -v Warning
echo "[repro] build done $(date)"
python3 - <<'EOF'
import pandas as pd
cols = ["label","N","moddeg","kodaira","tamagawa_list","max_isogeny_degree","is_cm","torsion","galrep_images","optimal","optimal_known"]
df = pd.read_parquet("data/curves.parquet", columns=cols); opt = df[df.optimal == 1]
import os; os.makedirs("results", exist_ok=True)
for name, d in [("work", opt[opt.N <= 300000]), ("holdout", opt[(opt.N > 300000) & (opt.N <= 400000)]), ("ext", opt[(opt.N > 400000) & (opt.N <= 500000) & opt.optimal_known])]:
    d.drop(columns=["optimal","optimal_known"]).to_parquet(f"results/optimal_{name}.parquet", index=False); print("[repro]", name, len(d))
EOF
for s in work holdout ext; do python3 scripts/conj_check.py results/optimal_$s.parquet --out results/check_$s 2>&1 | grep -v Warning | tail -3; done
echo "[repro] checker done $(date)"
for s in work holdout ext; do
  if diff -q results/check_${s}_summary.csv "$REF/check_${s}_summary.csv" > /dev/null; then echo "[repro] $s summary IDENTICAL to reference"; else echo "[repro] $s summary DIFFERS from reference"; diff results/check_${s}_summary.csv "$REF/check_${s}_summary.csv" | head -20; fi
done
echo "[repro] end $(date)"
