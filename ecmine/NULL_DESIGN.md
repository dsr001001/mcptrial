# Step 4 (draft for sign-off): the null, the hit criterion and the test budget

Nothing below has been run. This is the pre-registration: once agreed, step 5 runs
exactly these tests and nothing else, and anything found outside them is exploratory.

## 1. Unit of analysis and dependence

- Class-level targets (rank, root_number, is_cm, class_size, max_isogeny_degree): one
  row per isogeny class, 1,741,002 rows.
- Curve-level targets (torsion, tamagawa, sha_an, moddeg, regulator, n_intpts_x,
  two_adic_index, galrep, disc_sign, Kodaira counts): all 2,483,649 curves, but
  resampling and cross-validation folds are assigned by class, never by curve, because
  curves in a class share N, a_p, rank and the BSD quotient.

## 2. Splits (fixed now, never changed)

- Hold-out for step 8: every class with N in (300000, 400000] is untouched until step 8.
  Working set: N <= 300000, which is 1,311,066 classes; the hold-out has 429,936 classes (confirmed by `scripts/load.py`).
- Within the working set: 5-fold cross-validation with folds assigned by a hash of the
  class label (deterministic, documented in `scripts/load.py`).
- No model ever sees the hold-out, and no threshold is tuned on the test folds.

## 3. The question each test asks

For a target Y and a candidate feature block X, the test is not "does X predict Y" but
"does X predict Y beyond what is already known". So every test is a nested comparison:

- K(Y): the known-model features for Y, taken from KNOWN_RELATIONS.md (section 4). They
  always include log_conductor. Example for rank: log_conductor, root_number, torsion
  structure, n_bad_primes, and the Mestre-Nagao sum from the a_p block.
- Model A: fit Y from K(Y). Model B: fit Y from K(Y) plus X.
- Statistic: D = score(B) - score(A) on the held-out folds, where score is out-of-sample
  log-likelihood per row for categorical targets and R^2 for continuous targets.

Where a theorem gives an exact formula (I1, I3, T2, T4, T5, T11, T12), the target is
replaced by its residual from the formula before testing, so that the exact part cannot
produce a hit. In particular omega, lvalue and root_number are never targets; moddeg is
tested as its Petersson residual; sha_an, tamagawa and moddeg are also tested as their
odd parts and l-adic valuations.

## 4. The null

Conditional permutation: X is permuted among rows that share a stratum of K(Y), so the
X to Y link is destroyed while the K(Y) to Y link and the X to K(Y) link are kept. Strata
are defined by binning log_conductor in steps of 0.25 and crossing with the categorical
known features (for rank: root_number and torsion structure). 200 permutations per
test, giving a null distribution for D and a permutation p-value with resolution 1/201.

A test is a **hit** only if both hold:

1. permutation p < 0.001 after Bonferroni over the number of pre-registered tests
   (section 6), which with 200 permutations means D must exceed every permuted value
   and the test must be repeated with 2000 permutations before it is reported;
2. the effect is not negligible: D >= 0.01 nats per row for categorical targets, or
   D >= 0.01 in R^2 for continuous targets.

With 1.3 million rows almost any real effect is significant, so criterion 2 is the one
that does the work. Tests that pass 1 but fail 2 are logged as "small" and looked at only
in step 6 if a structural reason appears.

## 5. Models

- Primary: regularised linear or logistic regression (fast, so permutations are
  affordable; 200 refits of a 1.3M x 40 problem run in minutes).
- Secondary: gradient-boosted trees (scikit-learn HistGradientBoosting) fitted once per
  test, without permutations, to rank candidate blocks and to catch non-linear effects
  the linear model would miss. A block that is a hit under the trees but not under the
  linear model goes through the permutation test with a spline expansion of X.

## 6. Pre-registered test list

Targets (14): rank, torsion structure, tamagawa (odd part, 2-adic valuation), sha_an
(odd part, 2-adic valuation), moddeg (Petersson residual, 2-adic, 3-adic, 5-adic, 7-adic
valuations), regulator (residual after log|Delta| and rank), n_intpts_x, two_adic_index,
n_nonsurj_primes, class_size, max_isogeny_degree, disc_sign, Kodaira type counts,
log_abs_disc (residual after log_conductor).

Feature blocks (6): EQ, LOC, GAL, AN, AP, DIO as in KNOWN_RELATIONS.md, with the target's
own block and the identity partners of the target removed.

That gives at most 14 x 6 = 84 block tests, fewer in practice because identity partners
are removed; the exact count is fixed by the script before it runs. Bonferroni is over
that count. Per-column follow-ups are run only inside a block that is a hit and are
reported as exploratory.

Controls, run first, before any candidate test:

- Negative: root_number from EQ, GAL, DIO given LOC (must not be a hit); a_997 from EQ,
  GAL, DIO given the torsion congruence and the CM flag (must not be a hit); a
  shuffled copy of rank (must not be a hit from anything).
- Positive: Watkins (rank from 2-adic valuation of moddeg), Lorenzini (tamagawa from
  torsion), murmurations (rank from a_p in a conductor window), the torsion congruence,
  and the discriminant-sign effect on the 2-part of Sha. Each must be a hit.

If a negative control is a hit, the stratification is too coarse and the design is
revised before any candidate is run. If a positive control is not a hit, the pipeline
is not sensitive enough and the design is revised.

## 7. Compute

Linear model, 1.3M rows, 40 to 60 features: about 2 to 5 seconds per fit on this
machine. 84 tests x 5 folds x 201 fits is about 84,000 fits, roughly 2 to 4 days on 4
cores if done naively; with permutations run only on the two best folds and the
linear algebra shared across permutations (X changes, K(Y) does not, so the K(Y)
projection is computed once), this comes down to hours. Tree models: one fit per test
and fold, about 1 to 3 minutes each, under a day in total. The budget is acceptable and
no sampling of rows is needed.

## 8. Open choices for discussion

1. Whether to test the a_p block as 168 raw columns, as summary statistics (Nagao sums,
   mean a_p in p/N bins, parity fractions), or both. I favour summaries first, because
   168 raw columns with 1.3M rows will find the murmuration signal in every target that
   correlates with rank, and that is a known pattern.
2. Whether curve-level targets should use all curves or one curve per class. I favour all
   curves with class-level folds, because within-class variation of Sha, Tamagawa and
   torsion is itself a candidate signal (U4, U6).
3. The effect-size threshold of 0.01. It is a judgement call; it is meant to be the level
   at which a pattern would be visible in a plot in step 6.

## 9. Agreed choices and amendments (2026-10-04, before any test was run)

1. The a_p block enters the candidate tests as summary statistics only: Mestre-Nagao sums
   at 100, 300 and 1000, the fraction of even traces, the fraction of zero traces, the mean
   and second moment of a_p / sqrt p, and the mean of a_p / sqrt p in three prime bins.
   The raw 168-column block is used only in the murmuration positive control.
2. Curve-level targets use all curves with folds assigned by class.
3. The effect-size threshold stays at 0.01.
4. Amendment on the statistic for categorical targets: the permutation null needs 200
   refits per test, which logistic regression on 1.3M rows makes too slow. The primary
   statistic for a categorical target is therefore the gain in R^2 of the regression of
   each class indicator (a Brier-score gain), with the same 0.01 threshold, taken as the
   maximum over classes. Log-loss gain is reported from the tree model as a secondary
   check.
5. Continuous candidate features enter the linear model as the feature plus 8
   quantile-bin indicators, so that non-monotone effects are detectable without trees.
6. Permutation strata: log_conductor in bins of width 0.25 crossed with the categorical
   members of K(Y); the continuous members of K(Y) are handled by residualisation only.
   The negative controls decide whether this is fine enough.
7. Working set: N <= 300000 gives 1,311,066 classes and 1,869,000 or so curves; the
   hold-out (300000, 400000] has 429,936 classes.
