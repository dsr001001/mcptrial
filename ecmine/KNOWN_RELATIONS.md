# Step 3: what is already known (the exclusion list)

Scope: all curves with conductor N <= 400000 in the frozen ecdata snapshot
(commit 25cec5e): 1,741,002 isogeny classes, 2,483,649 curves. Every column is
determined for every row in this range.

Purpose: a regression in step 5 that rediscovers anything on this list is a sanity
check, not a result. Each relation is tagged:

- **I** identity by construction (the column is defined by it);
- **T** theorem;
- **C** conjecture with substantial evidence;
- **H** heuristic or empirical regularity reported in the literature;
- **?** nothing known to me; this is the search space.

Where a relation is exact it was verified on the table with `scripts/verify_known.py`
(run log at the end). A verified "0 violations" is also a check that the column
semantics in `build_features.py` are right.

## 0. Column groups

| group | columns | nature |
|---|---|---|
| EQ | log_abs_disc, disc_sign, log_j_height, (ainvs) | elementary functions of the equation |
| LOC | N, log_conductor, n_bad_primes, n_split_mult, n_nonsplit_mult, n_additive, kodaira, tamagawa_list, tamagawa, a_p for p dividing N | local data, Tate's algorithm |
| GAL | torsion, torsion_structure, n_torsion_gens, galrep_images, nonsurj_primes, n_nonsurj_primes, two_adic_index, two_adic_level, two_adic_label, class_size, max_isogeny_degree, is_cm | Galois structure of torsion |
| AN | rank, root_number, regulator, lvalue, omega, sha_an, moddeg | global analytic and arithmetic invariants |
| AP | a_p for good p < 1000 (per class) | Frobenius traces |
| DIO | n_intpts_x | integral points on the minimal model |
| BK | label, iso, num, optimal, optimal_known, manin, class_label | bookkeeping |

Class-level columns (constant on an isogeny class): N, a_p, rank, root_number,
lvalue, is_cm, class_size, max_isogeny_degree. All others vary within a class.

## 1. Identities by construction (never test these)

| code | relation |
|---|---|
| I1 | `sha_an` is defined by the BSD formula: lvalue = omega * regulator * tamagawa * sha_an / torsion^2. Any model containing lvalue together with the other four is computing this identity. Verified: relative error at most 1.5e-5 (file rounding). |
| I2 | tamagawa = product of tamagawa_list; n_bad_primes = n_split_mult + n_nonsplit_mult + n_additive; n_conductor_primes duplicates n_bad_primes. |
| I3 | omega is an elementary function of the a-invariants (the real period, computed by the AGM from c4 and c6). It is never a target; it is an input carrying height information. |
| I4 | log_abs_disc, disc_sign, log_j_height are functions of the a-invariants; log_j_height = log max(numerator, denominator) of j = c4^3 / Delta in lowest terms. |
| I5 | Within a class, `num` and `optimal` are labels; manin = 1 for every optimal curve in range (verified), and takes values in {1,2,3,4,5} on non-optimal curves by Cremona's convention. |
| I6 | The a_p row of a class is shared by all its curves (a_p is an isogeny invariant). |

## 2. Theorems (exact; all verified on the table)

| code | relation | verification |
|---|---|---|
| T1 | Parity: root_number = (-1)^rank. (Ranks in the table are established; parity is then a theorem for these curves.) | 0 violations in 2,483,649 curves |
| T2 | Root number from local data: w = -prod w_p over bad p. For semistable curves w = (-1)^(1 + n_split_mult). For additive p the local root number is a known function of the local type (Rohrlich; Halberstadt's tables at 2 and 3). So root_number is a function of LOC, and rank parity is a function of LOC. | semistable (643,264 curves): 0 violations |
| T3 | At bad p: a_p = +1 split, -1 nonsplit, 0 additive. So the AP block at p dividing N encodes the reduction types. | 0 violations in 6,395,420 bad-prime entries; split/nonsplit/additive counts agree with PARI on all 1,398,385 classes whose bad primes are all below 1000 |
| T4 | Tate's algorithm: Kodaira symbol constrains the Tamagawa number (I_n split: c = n; I_n nonsplit: c in {1,2}; II, II*: 1; III, III*: 2; IV, IV*: {1,3}; I_0*: {1,2,4}; I_n*: {2,4}). Ogg's formula f_p = v_p(Delta) + 1 - m_p gives N divides Delta, hence log_abs_disc >= log_conductor. | 0 violations in 9,892,790 (Kodaira, Tamagawa) pairs; 0 curves with log|Delta| < log N |
| T5 | Torsion injects into reduction: for good p not dividing |T|, |T| divides p + 1 - a_p. So a_p mod |T| is determined by p. This is the strongest exact link between AP and GAL and will dominate any linear model from a_p to torsion. | 0 violations in 185,677,784 tests |
| T6 | Mazur: 15 torsion structures (all 15 occur in range). Mazur and Kenku: rational cyclic isogenies have degree in {1..19, 21, 25, 27, 37, 43, 67, 163} (all occur), class sizes in {1,2,3,4,6,8}. | counts in verify log |
| T7 | Full rational 2-torsion (Z/2 x Z/2n) forces three real roots, so disc_sign = +1. | 0 violations in 85,164 curves |
| T8 | Cassels: the BSD quotient omega * regulator * tamagawa * sha_an / torsion^2 (= lvalue) is constant on an isogeny class, while each factor varies by controlled powers of the isogeny degree. Within-class ratios of omega, torsion, tamagawa, regulator, sha_an are therefore not independent patterns. | max relative spread of lvalue within a class: 0 |
| T9 | Galois images (galrep, 2adic) encode torsion and isogenies exactly: a rational l-torsion point or l-isogeny means the mod-l image lies in a Borel (labels lB, lCs); class_size and max_isogeny_degree are functions of the Borel primes; two_adic_label (Rouse and Zureick-Brown classification) determines the 2-torsion structure, the 2-isogeny graph and whether Delta is a square. CM curves have non-surjective image at every prime (Cartan normalizers), and the galrep file treats them specially (mostly empty, otherwise large-prime entries). Non-surjective primes for non-CM curves in range: 2, 3, 5, 7, 11, 13, 17, 37 (Borel) plus exceptional images 2Cn, 3Nn, 3Ns, 5S4, 5Ns, 5Nn. | tabulated |
| T10 | CM: 13 j-invariants; integral j, so potentially good reduction everywhere and no multiplicative primes; a_p = 0 at every inert prime of the CM field (half of all primes). is_cm is trivially separable from AP and from LOC. | 0 CM curves with multiplicative reduction; fraction of good p < 1000 with a_p = 0 is at least 0.46 for CM classes and at most 0.16 otherwise |
| T11 | Period size: log omega = -(1/12) log max(|c4|^3, |Delta|) + O(1), with an extra log 2 when Delta > 0 (two real components). The O(1) term is the exact AGM correction, not noise. | semistable fit: slope -0.078 (expected -0.083), R^2 0.81, residual sd 0.50 |
| T12 | Modular degree and Petersson norm: for the optimal curve, moddeg * vol(E) = 4 pi^2 <f,f> (Manin constant 1), where vol(E) is the covolume of the period lattice and <f,f> = N^(1+o(1)) up to a symmetric-square L-value. Consequences: moddeg >> N^(7/6 - eps) (Watkins 2004); the degree conjecture moddeg << N^(2+eps) is equivalent to a Szpiro-type statement (Frey; Mai and Murty). Non-optimal curves in a class have modular degrees tied to the optimal one through the isogeny graph. | optimal curves: log moddeg = -1.26 + 1.08 log N - 1.62 log omega, R^2 0.85, residual sd 0.92. The exact version needs the imaginary period, not yet in the table. |
| T13 | Lorenzini (2011): a rational point of order 5, 7 or 9 forces 5, 7 or 3 to divide the Tamagawa product, with finitely many listed exceptions. Extensions by Krumm, Barrios and Roy, Melistas. For order 3 there is an infinite family of exceptions; for order 2 no such statement. | order 5: 1 exception (11a3, the known one); order 7: 0; order 9: 0; order 3: 232 exceptions; order 2: 1575 curves with odd Tamagawa product |
| T14 | Visibility (Cremona and Mazur 2000; Agashe and Stein; Agashe, Ribet and Stein 2012): a prime dividing a visible part of Sha divides the congruence number, which shares its odd prime factors with moddeg. So primes dividing sha_an tend to divide moddeg. Cremona and Mazur established this empirically on these very tables. | not separately verified |
| T15 | Integral points: finite (Siegel). #E(Z) is bounded by C^(rank + Szpiro-type term) (Silverman 1987; Hindry and Silverman 1988). Torsion points of order not a prime power are integral on an integral model (Lutz, Nagell). So n_intpts_x grows with rank and with torsion, and falls with the size of the model. | not separately verified |
| T16 | Descent bounds: rank <= 2-Selmer rank, and for curves with a rational 2-torsion point the 2-Selmer rank is bounded by a linear function of the number of bad primes; the parity of the 2-Selmer rank equals the root-number parity (Monsky). So rank correlates with n_bad_primes among curves with 2-torsion. | not separately verified |
| T17 | Torsion and reduction at 2 and 3: #E(F_2) <= 5 and #E(F_3) <= 7, so an odd torsion order of 7 or more forces bad reduction at 2, and the Kubert families impose reduction types. In range, no curve with torsion order 6, 7, 9, 10, 12 or 16 has good reduction at 2; orders 2, 3, 4, 5, 8 do occur with good reduction at 2. | tabulated |
| T18 | Frobenius traces: Hasse |a_p| <= 2 sqrt p; a_p is even exactly when the 2-division cubic has a root mod p, so the parity pattern of a_p over p is the mod-2 image; more generally a_p mod l is determined by the mod-l image, which is constrained at every non-surjective l. Sato-Tate (horizontal, Taylor and coauthors) and Birch's vertical distribution describe the a_p distribution. a_p and a_q for different p, q are asymptotically independent. | Hasse: 0 violations |

## 3. Conjectures and heuristics (exclude as known patterns)

| code | relation |
|---|---|
| C1 | Watkins' conjecture: 2^rank divides moddeg. Partial results by Dummigan, Caro and Pasten, Kazalicki and Kohen. Verified: 0 violations in 2,483,649 curves; this is expected, Watkins formulated it on these tables. |
| C2 | Szpiro: log|Delta| <= (6 + eps) log N. In range the Szpiro ratio has maximum 8.90 (Nitaj's record curve) and 99.9th percentile 6.70. |
| C3 | Goldfeld and Szpiro: |Sha| << N^(1/2 + eps) (de Weger showed Sha can be as large as a power of N). |
| C4 | Delaunay's Cohen-Lenstra heuristics for Sha; Poonen and Rains, and Bhargava, Kane, Lenstra, Poonen and Rains for Selmer groups. Prediction and data: P(Sha > 1) is 18.5% at rank 0, 1.2% at rank 1, 0.01% at rank 2. Delaunay and Watkins: the distribution of Sha depends on the Tamagawa product and torsion through the integrality of the BSD quotient. |
| C5 | Average rank tends to 1/2 (Goldfeld; Katz and Sarnak), ranks are bounded (Park, Poonen, Voight and Wood predict at most 21). Bektemirov, Mazur, Stein and Watkins (2007) documented on these tables that the average rank rises through this range; verified: 0.47 below N = 1000 to 0.79 in (300000, 400000]. |
| C6 | Torsion and rank: mean rank by torsion structure is 0.79 (trivial), 0.67 (Z/2), 0.61 (Z/2 x Z/2) in range; expected from 2-Selmer heuristics for curves with 2-torsion (Klagsbrun and Lemke Oliver; Kane and Klagsbrun) and the conductor distribution of Kubert families. Isogeny and rank: 3-isogeny Selmer results of Bhargava, Klagsbrun, Lemke Oliver and Shnidman. |
| C7 | Rank from a_p. Mestre and Nagao sums estimate rank (Nagao's conjecture). Machine-learned rank from a_p (He, Lee and Oliver 2022; Kazalicki and Vlah 2023; Alessandretti, Baronchelli and He 2019 from a-invariants). Murmurations (He, Lee, Oliver and Pozdnyakov 2022): the mean of a_p over curves of fixed rank or root number with conductor in [X, 2X] oscillates in p/X; proved for weight-2 newforms by Zubrilina (2023), extended by Bober, Booker, Lee and Lowry-Duda, and by Lee, Oliver and Pozdnyakov for Dirichlet characters. Any AP to rank or root_number model is a rediscovery of these. |
| C8 | Lang's height conjecture: the canonical height of a non-torsion point is >> log|Delta| (Hindry and Silverman for bounded Szpiro ratio), so regulator grows with log|Delta|; the elliptic Brauer-Siegel heuristic (Hindry 2007; Hindry and Pacheco 2016) controls Sha * regulator. Through BSD these are the same as bounds on L-values (convexity L^(r)(1) << N^(1/4+eps), Lindelöf N^eps). |
| C9 | Watkins (2008, "Some heuristics about elliptic curves"): the probability of rank 0 depends on the Tamagawa product, torsion and the real period through the integrality of L(1)/omega. |
| C10 | Serre's uniformity question: no non-CM curve over Q has a non-surjective mod-l image for l > 37 (Borel case is Mazur's theorem; nonsplit Cartan case open for general l). In range the non-CM non-surjective primes are at most 37. |
| H1 | Sign of the discriminant: in range, mean rank is 0.735 (Delta < 0) against 0.726 (Delta > 0); P(Sha > 1 | rank 0) is 16.8% against 20.5%. The second is expected from the archimedean condition in 2-Selmer computations and the factor 2 in omega; treat as known unless the odd part of Sha shows the same dependence. |
| H2 | Counts: the number of curves with conductor at most X grows like X^(5/6) (Brumer and McGuinness; Watkins). This fixes the sampling density and matters for the null, not for any pair. |

## 4. The pair matrix

Rows are targets, columns are feature blocks. A cell gives the strongest known
status; "?" means I know of no theorem, conjecture or published heuristic.

| target \ features | EQ | LOC | GAL | AN (others) | AP | DIO |
|---|---|---|---|---|---|---|
| rank | C5, C8 | T2 (parity), T16 | C6, T5 | T1, I1, C1 | C7 | T15 |
| root_number | T2 via LOC | **T2 (exact)** | none expected | T1 | T3 | ? |
| torsion / structure | T7, T17 | T13, T17 | T9 (exact) | I1, T8 | **T5 (exact)** | T15 |
| tamagawa | T4 | **T4 (exact)** | T13 | I1, T8, C9 | T3 | ? |
| sha_an | H1 | C4, C9 | C4 (via torsion) | I1, T14, C3 | via I1 only | ? |
| moddeg | T12 | T12 | C1 (via rank), T14 | T12, C1, T14 | T12 (Petersson norm is an L-value) | ? |
| regulator | C8 | I1 | I1 | I1, C8 | via I1 | T15 |
| lvalue | I1 | I1 | I1 | I1 | computable in principle | none |
| omega | **I3 (exact)** | T11 | T7 | I1 | none | none |
| n_intpts_x | T15 (model size) | ? | T15 (torsion) | T15 (rank) | ? | n/a |
| two_adic_index / label | T9 | ? | T9 (exact) | ? | T18 | ? |
| galrep (non-surjective l) | T10 (CM) | ? | T9 (exact) | ? | T18 | ? |
| class_size / max_isogeny_degree | T10 | T17 | T9 (exact) | C6 | T18 | ? |
| is_cm | T10 | T10 | T9 | ? | T10 (exact) | ? |
| disc_sign | I4 | ? | T7 | H1 | T18 (mod-2 image) | ? |
| log_abs_disc | I4 | T4, C2 | T17 | T11, T12, C8 | ? | T15 |
| kodaira / reduction counts | T4 | I2 | T17 | T2 | T3 | ? |

## 5. What is left: the search space for step 5

These are the "?" cells, ordered by how surprising a real effect would be, with a
note on why it is not obviously a disguised entry of sections 1 to 3.

- **U1. moddeg, odd part, against GAL and tamagawa.** The 2-adic valuation is Watkins'
  conjecture and primes dividing Sha are visibility; the 3-, 5-, 7-adic valuations against
  torsion, isogeny degrees and Tamagawa numbers have no statement I know of.
- **U2. The Petersson residual of moddeg** (log moddeg - log N + log vol, which is the
  symmetric-square L-value in disguise) against rank, Sha, torsion, GAL. I know of no
  theorem linking L(Sym^2 f, 1) to the rank or to Sha. Needs the imaginary period added
  to the table first.
- **U3. n_intpts_x against tamagawa, sha_an, moddeg, class_size, disc_sign, Kodaira types,**
  controlling rank, torsion and log|Delta| (which are T15).
- **U4. Odd part of sha_an against disc_sign, Kodaira types, class_size, isogeny degrees
  and exceptional Galois images**, controlling rank and the known BSD factors. The 2-part
  is largely explained by 2-Selmer local conditions; the odd part is not.
- **U5. rank against class_size, max_isogeny_degree, exceptional images and two_adic_label**,
  controlling conductor and torsion structure. Selmer theory predicts the 2- and 3-isogeny
  cases; the 5-, 7-, 13-isogeny and exceptional-image cases have no prediction I know of.
- **U6. tamagawa (odd part) against non-surjective primes that do not divide the torsion
  order.** Lorenzini is about torsion points; the isogeny-only case is open to me.
- **U7. Kodaira types against rank and sha_an**, controlling conductor. Additive types
  II, III, IV, I_0*, I_n* carry no rank information in any theorem I know.
- **U8. disc_sign against rank, odd Sha and n_intpts**, controlling conductor (H1 shows a
  2-part effect; the question is whether anything survives beyond it).

Negative controls (must show nothing beyond the listed theorems): root_number against
EQ, GAL, DIO after controlling LOC; a fixed good a_p (say a_997) against EQ, GAL, DIO
after controlling T5 and T10.

Positive controls (the pipeline must recover them): C1, T13, C7 (murmurations in a
conductor window), T5, H1.

## 6. Additions needed before step 5

1. Imaginary period and lattice covolume from PARI (`E.omega`), to make T12 exact and
   define the Petersson residual (U2).
2. c4, c6 and the exact AGM period, to make I3 and T11 exact instead of a regression.
3. Odd-part and l-adic valuation columns for moddeg, tamagawa and sha_an (U1, U4, U6).
4. A clean "known-model" feature list per target, written down in NULL_DESIGN.md before
   any model is fitted.

## 7. Verification log

```
curves 2483649  classes 1741002  primes 168 (max 997)
[A] semistable curves 643264: root number != (-1)^(1+n_split): 0
[B] bad-prime a_p outside {0,+-1}: 0  (bad-prime entries: 6395420)
[B] classes with all bad primes < 1000: 1398385; split/nonsplit/additive counts mismatch vs PARI: 0 / 0 / 0
[C] torsion congruence a_p = p+1 mod |T|: violations 0 of 185677784 tests
[D] curves with Z/2xZ/2 torsion 85164: discriminant negative: 0
[E] log|D| < log N: 0
[F] log(Omega/2^[D>0]) = 1.997 + -0.0778 * log_j_height  (semistable; expected slope -0.0833; R2 0.8139, resid sd 0.501)
[G] Watkins 2^rank | moddeg: violations all curves 0, optimal 0
[H] 5 | |T| (1351 curves): Tamagawa not divisible by 5: 1  e.g. ['11a3']
[H] 7 | |T| (76 curves): Tamagawa not divisible by 7: 0
[H] 9 | |T| (18 curves): Tamagawa not divisible by 3: 0
[H] 3 | |T| (49658 curves): Tamagawa not divisible by 3: 232
[H] 2 | |T| (1089418 curves): Tamagawa not divisible by 2: 1575
[I] Kodaira/Tamagawa pairs 9892790: curves violating Tate's algorithm constraints: 0
[J] max relative spread of L-value within a class: 0.00e+00
[K] fraction of good p<1000 with a_p=0: CM classes min/median 0.464/0.518; non-CM max/median 0.159/0.036
[L] log moddeg = -1.26 + 1.078 log N + -1.616 log Omega  (optimal curves; R2 0.8509, resid sd 0.919)
[M] average rank by conductor bin: 0.471 (<=1000), 0.630, 0.710, 0.746, 0.769, 0.782, 0.792 (300000-400000)
[M] P(Sha>1) by rank: 0.1847 / 0.0124 / 0.0001 / 0 / 0
torsion order with good reduction at 2 (curves): 2: 147400, 3: 7445, 4: 23177, 5: 348, 8: 433; orders 6, 7, 9, 10, 12, 16: none
non-surjective primes (non-CM): 2, 3, 5, 7, 11, 13, 17, 37; exceptional images 2Cn, 3Nn, 3Ns, 5S4, 5Ns, 5Nn
Szpiro ratio: max 8.904, 99.9th percentile 6.698
mean rank by disc sign: -1: 0.7348, +1: 0.7262; P(Sha>1 | rank 0): -1: 0.1678, +1: 0.2047
```
