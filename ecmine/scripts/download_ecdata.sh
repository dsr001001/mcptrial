#!/bin/sh
# Download Cremona's ecdata tables at a pinned commit (frozen snapshot).
# Usage: scripts/download_ecdata.sh [outdir]
set -e
SHA=25cec5ecfec8b9f016eb1631ac633194c2bed39f
OUT=${1:-data/raw}
B=https://raw.githubusercontent.com/JohnCremona/ecdata/$SHA
mkdir -p "$OUT"; cd "$OUT"; echo "$SHA" > ECDATA_COMMIT
for d in allcurves allgens allbsd alldegphi allisog galrep opt_man intpts 2adic iwasawa allbigsha; do
  mkdir -p $d
  for i in $(seq 0 49); do
    lo=$(printf "%05d" $((i*10000))); hi=$(printf "%05d" $((i*10000+9999)))
    echo "$B/$d/$d.$lo-$hi"
  done
done > urls.txt
# iwasawa only exists upstream for conductor < 150000; its 404s are expected.
xargs -P 16 -n 1 sh -c 'u="$0"; d=$(echo "$u" | awk -F/ "{print \$(NF-1)}"); f=$(basename "$u"); [ -s "$d/$f" ] || curl -sS -f -o "$d/$f" --max-time 600 --retry 3 "$u" || echo "FAIL $u"' < urls.txt
