# ecmine: regression on invariant tables of elliptic curves over Q

Workstream for "item 9" (symbolic regression across invariant databases), following the
ten-step plan. This directory holds steps 1 and 2.

## Step 1: frozen snapshot

Source: John Cremona's `ecdata` repository (the upstream of the LMFDB elliptic curve tables),
pinned to commit `25cec5ecfec8b9f016eb1631ac633194c2bed39f` (see `data/raw/ECDATA_COMMIT`).
Coverage: every elliptic curve over Q with conductor below 500000 (3,064,705 curves in
2,164,260 isogeny classes). The LMFDB web API was not used: it serves a CAPTCHA page to this
environment, and the git commit is a cleaner freeze anyway.

Reproduce with `scripts/download_ecdata.sh` (1.2 GB, not committed; see `.gitignore`).

Tables used, with the column order as verified against 11a1, 37a1 and 389a1:

| table | columns |
|---|---|
| allcurves | N, class, number, a-invariants, rank, torsion order |
| allgens | ..., torsion structure, generators |
| allbsd | ..., Tamagawa product, real period, L^(r)(1)/r!, regulator, analytic Sha |
| alldegphi | ..., modular degree |
| opt_man | ..., optimal flag, Manin constant |
| intpts | label, a-invariants, x-coordinates of integral points |
| 2adic | ..., 2-adic image index, level, generators, label |
| galrep | label, images of mod-p Galois representations at non-surjective p |
| allisog | per class: curves and isogeny matrix |
| iwasawa | p-adic invariants, upstream only for N < 150000 (downloaded, not yet used) |
| allbigsha | curves with nontrivial Sha (redundant with allbsd, not used) |

## Step 2: flat feature table

`scripts/pari_features.gp` computes with PARI/GP 2.15 what the tables lack: root number,
discriminant sign and size, j-invariant height, reduction types and Kodaira symbols at bad
primes, and a_p for the 168 primes below 1000 (one row per isogeny class).

`scripts/build_features.py` merges everything into:

- `data/curves.parquet`: one row per curve, with class-level columns joined in
- `data/ap.parquet`: one row per isogeny class, 168 int16 a_p columns plus rank, root number,
  torsion, Sha, log conductor, class size and CM flag

Notes found while building:

- The 2adic table marks CM curves with an infinite index; that is the source of `is_cm` (5,892 curves).
- For conductor above 400000 the opt_man optimality flag is undetermined for 64,249 classes;
  `optimal` is masked to NA there and `optimal_known` records it.

`scripts/sanity_checks.py` passes on the full table: no parity violations, the BSD identity
holds to 1e-5, Tamagawa products and reduction-type counts agree, Hasse bound holds for all
a_p, every analytic Sha is a square, every class with known optimality has exactly one optimal
curve. Rank counts: 1,170,876 / 1,535,669 / 348,672 / 9,487 / 1 for ranks 0 to 4.

Both parquet files are build products and not committed.

## Scope of the first search

Agreed on 2026-10-04: conductor N <= 400000 only, where every column is determined
(1,741,002 classes, 2,483,649 curves). Classes with N in (300000, 400000] are reserved as
the step 8 hold-out. `scripts/load.py` applies both restrictions by default.

## Step 3: exclusion list

`KNOWN_RELATIONS.md` lists every identity, theorem, conjecture and heuristic known to
relate pairs of columns, with each exact relation verified on the table by
`scripts/verify_known.py` (all verified with zero violations), the pair matrix, and the
remaining open cells that form the search space.

## Step 4: null design (draft, not yet agreed)

`NULL_DESIGN.md` proposes the unit of analysis, the fixed splits, the nested-model test
with conditional permutation null, the hit criterion, the pre-registered test list and
the positive and negative controls. Step 5 does not start until it is agreed.

## Step 5: cheap models (done, 2026-10-04)

Two test families on the working set (N <= 300000, hold-out untouched):

- regression family (`scripts/step5.py`): 112 pre-registered nested-model tests with a
  conditional permutation null, after 22 controls (15 negative, all clean; 7 positive, 6 hits,
  Watkins failed as a variance-explained statistic cannot see a constraint);
- family B (`scripts/step5_cells.py`): forbidden, depleted and enriched cells for 2278 column
  pairs inside conductor strata, added because of the failed Watkins control.

Results: `results/STEP5_REPORT.md` (assembled by `scripts/step5_report.py`), with the raw
tables in `results/step5_*.csv`. Interpretation is step 6 and has not started.
