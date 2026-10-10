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

## Step 6 to 9 on candidate U1 (done, 2026-10-04)

`STEP6_U1_REPORT.md`: the l-adic valuation of the modular degree of the optimal curve is at least the
sum over bad primes of the l-adic valuation of the order of the geometric component group, for every
odd l with E[l] irreducible (Conjecture 1; the multiplicative part at l >= 5 is a theorem of Kim and
Ota with Agashe, Ribet and Stein; the Kodaira type IV and IV* contribution at l = 3 is new), with a
"twisted" supplement (Conjecture 2, since replaced by the sharper twist-symmetrised bound, Conjecture 1.3 of the note)
and a bound on the Eisenstein deficit (Conjecture 3). All three
hold with zero exceptions on the working set and on the hold-out; the degree column was recomputed
independently with PARI on fifteen examples.

## Verify one curve yourself

PARI/GP (any version from 2.13; the modular degree takes seconds for conductors up to a few million):

```
E = ellinit([0,0,0,-99,360]);  \\ any curve; use the optimal curve of its class for the sharp statement
N = ellglobalred(E)[1]; L = ellglobalred(E)[5];      \\ L[i] = [f_q, Kodaira code, [u,r,s,t], c_q], primes increasing
d = ellmoddegree(E);
\\ Conjecture 1 at l = 3: v_3(d) >= sum over multiplicative q of v_3(v_q(Delta)) + #{q : type IV or IV*}
\\ PARI codes: n+4 = I_n, 4 = IV, -4 = IV*
b = sum(i=1,#L, my(k=L[i][2]); if(k>=5, valuation(k-4,3), if(k==4||k==-4, 1, 0)));
print([N, d, valuation(d,3), b, valuation(d,3) >= b])
\\ Sharper bound (Conjecture 1.3 of the note): at an odd q also count 1 for types II, II* (codes 2, -2)
\\ and v_3(n) for type I_n* (code -(n+4)); at q = 2 these types count nothing
P = ellglobalred(E)[4][,1];
b2 = sum(i=1,#L, my(k=L[i][2], q=P[i]); if(k>=5, valuation(k-4,3), if(k<=-5, if(q>2, valuation(-k-4,3), 0), if(abs(k)==4 || (abs(k)==2 && q>2), 1, 0))));
print([valuation(d,3), b2, valuation(d,3) >= b2])
```

Sage:

```
E = EllipticCurve([0,0,0,-99,360]).optimal_curve()
d = E.modular_degree(); b = 0
for ld in E.local_data():
    k = ld.kodaira_symbol(); s = str(k)
    if s.startswith('I') and not s.startswith('I0') and not s.endswith('*'): b += ZZ(int(s[1:])).valuation(3)
    if s in ('IV', 'IV*'): b += 1
print(E.conductor(), d, d.valuation(3), b, d.valuation(3) >= b)
```

For a prime l >= 5 the sharper bound is: v_l(n) for every multiplicative I_n and for every I_n* at an odd prime.

Both statements assume E has no rational 3-isogeny (`E.isogeny_class()` has a single curve, or no degree divisible
by 3) and is not CM. For a prime l >= 5 replace the type IV term by nothing and the valuation by v_l.

## Follow-up (2026-10-06): steps 1 to 11 of the continuation plan

1. Extension to conductor 500000 (359,009 further optimal curves): zero violations of all statements (`results/check_ext_summary.csv`).
2. Out-of-table verification: `scripts/out_of_table.gp` generates curves with prescribed local types and conductor up to 4 million,
   trivial isogeny class, not CM, and computes their modular degrees with `ellmoddegree`; `scripts/oot_convert.py` collects them
   into `results/oot_table.parquet`; `scripts/conj_check.py results/oot_table.parquet --nmax 4000000 --out results/check_oot`
   checks them (zero violations; counts in `results/check_oot_summary.csv` and in the note, Table 1 and Table 4).
3. Reproducibility: `scripts/repro_run.sh` re-downloads the pinned data and rebuilds everything on a clean clone, then diffs the checker
   summaries against the committed ones (`results/repro.log`, `results/repro_check.log`, `results/repro_check2.log`: identical counts).
4. Equality cases: `scripts/step7_equality.py` (old bounds) and `scripts/step7_equality2.py` (final bounds; `results/step7_equality2_work.txt`).
5. Mechanism tests: `scripts/step7_twist_test.py` (`results/step7_twist_test.txt`: no congruence of f with its own twist; a single odd
   type II/II* prime always forces 3 | deg) and `scripts/step7_twist2.py` (`results/step7_twist2_{work,holdout,ext}.txt`), which replaced
   the "all but the largest" rule by the clean odd-prime rule of Conjecture 1.3 of the note; see the correction at the top of
   `STEP6_U1_REPORT.md`.
6. Literature at l = 2: recorded in the note, section 5.
7. Eisenstein deficit predictors: `scripts/step7_eis_proxy.py`.
8. The qualitative theorems with proofs: note, Theorem 1.4 and section 3 (types IV, IV* by level lowering; odd primes of types II, II*
   and I_n* by twisting, level lowering and twisting back).
9. Open problems: note, section 6.
10. The note: `paper/note.tex`, `paper/refs.bib`, tables generated by `scripts/make_tables.py` from the result files; compiled `paper/note.pdf`.
    Placeholders in double brackets (author, acknowledgement, two reference checks, the exact form of the level-lowering statement) are for you.
11. Outreach texts: `paper/outreach.md`.

Also: `scripts/twist_lemma_check.gp` (output `results/twist_lemma_check.txt`) checks Lemma 2.4 of the note (what a quadratic twist by q*
does to the Kodaira types) on random curves; the three worked examples of the note (121a1, 539c1, 980i1) can be rechecked with `ellap`.

The consolidated checker is `scripts/conj_check.py`; its summary columns are `viol_conj1` (Conjecture 1.1 of the note), `viol_conj2`
(Conjecture 1.3, the twist-symmetrised bound, every odd l) and `viol_conj3` (Conjecture 1.5, Eisenstein deficit against the bound of 1.1).

## Referee responses (2026-10-10)

Two reports were received on the note (an AI referee report and a research note assessing it). Settled in this commit:

- Theorem 1.4 now assumes l does not divide N (the case l | N is left to the data), and its proof is complete: Lemma 3.1
  (the congruence criterion, with the full Hecke algebra and the Abbes-Ullmo/Ribet duality) and Lemma 3.2 (the oldform
  eigenvector with the right U_p eigenvalues at every p | N and the right a_l) replace the former appeal to "a congruence at
  almost all primes"; the optimal-level theorem is quoted in the form of Darmon-Diamond-Taylor, Theorem 3.15, with the level
  taken to be the Serre conductor, which is what makes Lemma 3.2 applicable.
- Agashe-Ribet-Stein: the result is their Theorem 2.2, and 99A1 is in their Table 1 (modular degree 4, congruence number
  12); Kim-Ota is Res. Math. Sci. 10 (2023), Paper 22, and their congruence ideal is the full-space one; Pasten is
  J. Number Theory 254 (2024); Cesnavicius-Neururer-Saha (JEMS 26, 2024) is cited for additive primes and the Manin constant;
  Pollack-Weston are credited with the square-free weight-2 case.
- The growth sentence in the introduction is replaced by the abc statement of ARS; the twist weight at q = 3 is flagged as
  data-driven; Lemma 2.4(ii) now proves I_0 <-> I_0* at every odd q; the indicator bracket in Conjecture 1.5 is defined and
  the bound D <= 2 is justified from the tables (no curve has both a point of order 9 and image 3Cs.1.1).
- Calibration rates are reported (Section 4.3 of the note) and the control rows of Table 2 are named as such.

Not yet done: the local-factor memorandum (adjoint / congruence-ideal framing, M1 of the first report), the Zenodo archive
and the AI-use statement.

## Referee item 1 (2026-10-10): the local-factor memorandum

`paper/memo_local_factors.tex` (compiled: `paper/memo_local_factors.pdf`). Main finding: the local term of an additive prime
q >= 5 is the l-adic valuation of the inverse Euler factor of the symmetric square L-function at s = 2 (the factor that the
naive symmetric square omits), plus the Tamagawa exponent v_l(n) for I_n and I_n*: v_l(q - chi_{-3}(q)) for II, II*, IV, IV*;
v_l(q - chi_{-4}(q)) for III, III*; v_l(q^2 - 1) (+ v_l(n)) for I_n*; v_l(q - 1) + v_l((q+1)^2 - a_q(E')^2) for I_0* with E'
the good twist. The component-group and twist-symmetrised bounds of the note are its shadows. Zero violations on all four
ranges at l = 3, 5, 7, 11, 13 when the type III terms are left out; with them, 264 exceptions at l = 3, all with a prime
= -1 mod 3 of type I_n or I_n* with 3 | n (Kim-Ota's excluded case). Scripts `scripts/step9_euler.py` (single-prime minima),
`scripts/step9_fullrule.py` (the rule on whole ranges), `scripts/step9_additivity.py`, `scripts/step9_twist_degree.py`
(twist monotonicity of the degree); outputs `results/step9_*`. The memorandum also proves twist monotonicity of the congruence
number and the I_0* and I_n* (l not dividing n) Euler terms from Darmon-Diamond-Taylor Theorem 4.20 and Kim-Ota Lemma 2.7,
and states what remains open (types II, IV, III: integrality of the primitive adjoint L-value; the wild primes).
