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

**Current statement (2026-10-10).** The Euler-factor rule (Conjecture 1.2 of the note) is checked for one curve by
`scripts/euler_rule.gp`: in gp 2.15, `read("scripts/euler_rule.gp"); rule(ellinit([1,1,1,-30,-76]), 3)` returns `[1, 1]`
(121a1: `[v_3(deg), bound]`); start gp with `-D parisizemax=400000000 -D threadsizemax=200000000`. The two older statements
below are its corollaries (Proposition 1.3 of the note).

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

`paper/memo_local_factors.tex` (compiled: `paper/memo_local_factors.pdf`, 9 pages). Main finding: the local term of a bad
prime q != l is e_l(E,q) = v_l of the inverse Euler factor of the adjoint L-function at s = 1 that the naive adjoint
L-function of Diamond-Flach-Guo omits (in the symmetric-square normalisation, P_q(q^-2) / (1 - a_q^2 q^-2) with
L_q(Sym^2 f, s) = P_q(q^-s)^-1), plus the Tamagawa exponent t_l(E,q) = v_l(-v_q(j)) of the adjoint when the reduction is
potentially multiplicative. For tame q >= 5 this reads: 0 for I_n; v_l(q^2 - 1) for I_n*; v_l(q - chi_{-3}(q)) for
II, II*, IV, IV*; v_l(q - chi_{-4}(q)) for III, III*; v_l(q - 1) + v_l((q+1)^2 - a_q(E')^2) for I_0* with E' the good
twist. The component-group and twist-symmetrised bounds of the note are its shadows.

Verification (`scripts/step10_sym2_euler.py`: PARI's `lfunsympow(E,2)` / `lfuneuler` factors at every bad prime, the wild
primes 2 and 3 included, into `results/sym2_<set>_<shard>.parquet`; `scripts/step10_rule.py`: the rule on the four ranges,
summaries `results/step10_rule_<set>_summary.csv`, log `results/step10_rule.log`; `scripts/step10_make_table.py` builds
`paper/tables/euler_uniform.tex`): with the type III terms dropped at every prime, zero violations on all four ranges at
l = 3, 5, 7, 11, 13; equality rate at l = 3 of 56.8% (working set), against 39.4% for the twist rule of the note. With the
type III terms added: 475 exceptions, all at l = 3, all of one shape (a III/III* prime with nontrivial Euler factor and
trivial Frobenius on the invariant line, i.e. q = 1 mod 12 or q = 2 with factor 1 + 2X, together with a prime p = -1 mod 3
with t_3 >= 1; shortfall exactly 1). The weights of the prime l = 3 itself (w_3) remain empirical. PARI's factor at 2
resolves the irregular I_n* behaviour of the note: a symbol I_n* at 2 can be potentially good (v_2(j) >= 0), and then no
Tamagawa term is due. Earlier Kodaira-type scripts (q >= 5 only): `scripts/step9_euler.py` (single-prime minima),
`scripts/step9_fullrule.py`, `scripts/step9_additivity.py`, `scripts/step9_twist_degree.py` (twist monotonicity of the
degree); outputs `results/step9_*`, table `paper/tables/euler_full.tex`.

Proved in the memorandum: twist monotonicity of the congruence number (Theorem 4.1); the I_0* and I_n* Euler terms for every
odd l with l not dividing 2N and E[l] irreducible, from Diamond-Flach-Guo Proposition 1.4(c), the inequality of
Darmon-Diamond-Taylor Lemma 4.17 and multiplicity one at the lower level (Theorem 5.1); the Tamagawa terms for l >= 5 from
Kim-Ota (Theorem 5.2); their sum (Corollary 5.3). The potentially good types II, IV, III are reduced to a statement about
the Sigma-imprimitive adjoint Selmer group (Proposition 6.1: Bloch-Kato Selmer group plus local H^0 terms minus a dual
Selmer correction), which is the question for an expert; the type III anomaly is the case where the dual correction is 1.
The citations to DFG, DDT and Ribet-Stein in the proofs still need checking by someone who knows these papers.

## Note restructured around the Euler-factor rule (2026-10-10)

`paper/note.tex` (compiled: `paper/note.pdf`, 20 pages) now has the title "Local factors of the adjoint L-function and the
l-adic valuations of the modular degree". Structure: Section 1 (Definition 1.1 of the local terms e_l, t_l; Conjecture 1.2,
the Euler-factor rule; Proposition 1.3, the component-group inequality (1.2) and the twist-symmetrised inequality (1.3) as its
consequences; Theorem 1.4 Kim-Ota/ARS; Theorem 1.5, the qualitative additive statements at l = 3; Theorem 1.6, the quantitative
theorem for I_0* and I_n*; Conjecture 1.7, the Eisenstein case); Section 2 local invariants (Lemma 2.5: the omitted factors at
tame primes); Section 3 proof of Theorem 1.5 (unchanged); Section 4 the quantitative theorems (4.1 DFG's formula, 4.2 the
DDT Lemma 4.17 inequality with proof, 4.3 twist monotonicity, 4.4 the Euler terms, 4.5 Tamagawa terms, 4.6 their sum);
Section 5 the Selmer-group reformulation (Conjecture 5.1); Section 6 computations (Table 1 the rule on the four ranges,
Table 2 each term in isolation, Table 3 the corollary forms, Table 4 the wild primes, Table 5 out of table; 6.6 the type III
anomaly; 6.9 Eisenstein); Section 7 the prime 2; Section 8 open questions; Appendix with the PARI function. New tables:
`paper/tables/euler_uniform.tex` (from `scripts/step10_make_table.py`), `paper/tables/euler_iso3.tex` (from
`results/step9_euler_l3.txt`, copied from the memorandum). New references in `paper/refs.bib`: DFG04, Flach92, Hida81,
Tilouine97 (added from memory; check page numbers). The memorandum `paper/memo_local_factors.tex` now refers to the new
numbering, and its last section records the changes made. `paper/outreach.md` was rewritten for the rule.

Snapshot of today's state of all documents: `Oct10/` (note, memorandum, bibliography, tables, outreach texts, this README and
`HANDOFF.md`). To resume in a new session read `HANDOFF.md` first. Author name, email and ORCID are filled in
in the note, the memorandum and the outreach signature (live and snapshot copies). The statement on the use of an AI system at every stage (an unnumbered section at the end of the
note, 21 pages; a short version at the end of the memorandum, 10 pages; a sentence in the outreach message) replaces the
bracketed placeholder; the author takes responsibility for all statements, proofs, computations and errors.

## Version of 2026-10-10, evening: the rule is a theorem under the Taylor-Wiles hypothesis

Two reports on the morning version (an AI referee report and an independent assessment) asked whether the rule follows
from Diamond-Flach-Guo and found that every type III exception has mod 3 image 3Ns. Both points were verified and settled:

- `paper/note.tex` (24 pages) now has Theorem 1.2: for l not dividing 2N with rho-bar absolutely irreducible on
  Q(sqrt(l*)) (DFG's hypothesis l not in S_f), v_l(deg) >= length H^1_f(Q, ad^0 T (x) Q_l/Z_l) + sum_q v_l(Tam_q(ad^0 T))
  + sum_q e_l(E,q), equality under multiplicity one, hence v_l(deg) >= sum e_l + sum t_l for ALL Kodaira types, type III
  included. Chain (Section 4): Lemma 4.3 (DDT Lemma 4.17 inequality, with DFG's pairing cited precisely), DFG Prop 1.4(c)
  (Theorem 4.1), DFG Theorem 3.7 (4.5), DFG Theorem 2.7 (4.6), DFG Lemma 2.1 (4.7), Fontaine-Perrin-Riou local formula
  (4.8), Lemma 4.9 (vanishing of H^0(Q, ad^0 T(1) (x) Q_l/Z_l) iff TW), Lemma 2.7 (Tamagawa factor of ad^0 at a
  multiplicative prime: l^{v(n) + min(v(n), v(q-1), v(c_q))}, c_q the unit part of the Tate parameter).
- Conjecture 1.3 (all types, all odd l with E[l] irreducible, l | N allowed) includes the correction term h_l(E) =
  length H^0(Q, ad^0 T(1) (x) Q_l/Z_l), which is 1 exactly for the 3Ns curves (Proposition 1.7, Lemma 4.9) and 0 otherwise.
  Zero violations on all four ranges (Table 1; `scripts/step11_tw_boundary.py`, `results/step11_summary.csv`,
  `results/step11_ns3.csv`, `paper/tables/euler_uniform.tex`, `paper/tables/ns3.tex`).
- The 3Ns boundary (Section 6.7, Table 4): all 475 exceptions of the rule without correction are 3Ns, their 3-division
  polynomial factors as two quadratics, h_3 = 1 for all 1,697 3Ns curves of the tables, shortfall always exactly 1,
  31% / 23% / 18% of 3Ns curves fail. The morning version's claim "q = 1 mod 12" was wrong (140 / 160 / 72 for q = 1, 11 mod
  12, q = 2 on the working set).
- Tamagawa factor test (Section 6.4, `scripts/step12_tamagawa.py`, `results/step12_work.csv`): the predicted extra term
  (Steinberg q = 1 mod l, l | n, unit part of the Tate parameter an l-th power) holds with zero violations on the working
  set; equality among affected curves rises 45% -> 61% at l = 3 (46,222 curves, 12,095 with extra term) and 67% -> 79% at
  l = 5 (10,940 / 1,646).
- Other changes: Lemma 3.2 and Theorem 4.11's Step 1 say "generalised eigenform" where U_p is not semisimple; Zagier 1985
  credited for the degree formula; Kim-Ota's hypotheses stated (Remark 4.10); Proposition 1.4 gives the count 92,812 of
  type IV/IV* at 2 curves with e_3(E,2) = 1; the calibration method is stated; the AI-use statement names DFG and the
  reports; no statement in the paper defers to an outside review.
- `paper/memo_local_factors.tex` got a postscript recording the resolution and marking its superseded parts.
  `paper/outreach.md` rewritten as an optional notification to Diamond/Flach. New bib entries: Zagier85, FPR94, Zywina15.
- Snapshot of this version: `Oct10_v1_7pm/` (pdf, tex and plain-text versions of the note and the memorandum, the
  tables, the outreach texts, the README, HANDOFF.md, the scripts of steps 10-12).

Verified today from the sources: DFG published text (Prop 1.4(c), Theorem 3.7, Lemma 2.1, Theorem 2.7, 2.15, the
definition of eta^Sigma via the pairing delta-hat composed with w, pages 663-727); the arXiv long version 2512.02348
(different numbering, not cited); ARS Table 1 (99A1: degree 4, congruence number 12; 54B1: 2, 6).
