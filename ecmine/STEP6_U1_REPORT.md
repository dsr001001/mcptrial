# Step 6 result: l-adic valuations of the modular degree and local component groups

Superseded in part by the note `paper/note.tex` (compiled: `paper/note.pdf`), which contains the final statements, the proof of the qualitative type IV theorem, the extension to conductor 500000, the out-of-table checks and the corrected references. This file is kept as the working record.

Candidate U1 from step 5 ("the l-adic valuation of the modular degree against the l-adic valuation
of the Tamagawa product") was taken through steps 6, 8 and 9: the known mechanism was identified
in the literature, the quantity was corrected from the Tamagawa number to the geometric component
group, the statements were sharpened on the working set (conductor at most 300000) and then tested
once on the untouched hold-out (conductor in (300000, 400000]). Nothing below was tuned on the
hold-out.

Notation. E is the X_0(N)-optimal curve of its isogeny class (Cremona's optimal curve), deg phi_E its
modular degree, N its conductor. For a bad prime q, Phi_q is the component group of the Néron model
over F_q-bar (its order is n_q = v_q(Delta_min) for multiplicative reduction I_n, and 1, 2, 3, 4, 4 for
Kodaira types II/II*, III/III*, IV/IV*, I_0*, I_n*). "E[l] irreducible" means E has no rational
l-isogeny. All counts are of optimal curves, one per isogeny class.

| set | optimal curves | conductor range |
|---|---|---|
| working set | 1,311,066 | N <= 300000 |
| hold-out | 429,936 | 300000 < N <= 400000 |

## 1. Statements

**Conjecture 1 (all odd l, E[l] irreducible, E not CM).**

    v_l(deg phi_E)  >=  sum over q | N of  v_l( #Phi_q(F_q-bar) ).

For l >= 5 only multiplicative primes contribute (v_l(n_q)); for l = 3 every prime of Kodaira type IV
or IV* contributes 1 in addition. Verified with zero exceptions:

| | working set | hold-out |
|---|---|---|
| (curve, l) pairs with some multiplicative q having l dividing n_q | 913,244 | 314,592 |
| of which l = 3 | 455,707 | 159,808 |
| of which l divides N | 175,798 (l >= 5) + 320,310 (l = 3) | 61,595 (l >= 5) |
| of which some contributing q is congruent to +-1 mod l | 410,074 | not tabulated |
| curves at l = 3 (additive term included), all | 1,233,729 | 409,577 |
| violations | 0 | 0 |

**Conjecture 2 (l = 3, E[3] irreducible, E not CM; the twisted primes).** Give a prime q of type II
or II* the weight w_q = 1, and a prime of type I_n* the weight w_q = v_3(n); all other primes weight 0.
Then

    v_3(deg phi_E)  >=  [bound of Conjecture 1]  +  ( sum of the w_q  minus  the largest single w_q ).

Zero violations on both sets. The pattern "all but the largest" is exact at the minimum: two type II
primes force 3 | deg phi, one does not; three force 9 | deg phi. I have no mechanism for this
statement (see section 4).

**Conjecture 3 (the Eisenstein case: E has a rational l-isogeny, l odd).** Let D be the deficit,
that is the right-hand side of Conjecture 1 minus v_l(deg phi_E). Then

    D  <=  v_l( #E(Q)_tors )  +  [ E[3] is isomorphic to Z/3 + mu_3 ],

in particular D <= 0 whenever E has no rational l-torsion point (an l-isogeny alone costs nothing),
D <= 1 for l >= 5, and D <= 2 for l = 3. Zero violations on both sets (67,991 working-set pairs,
all odd l; 76,737 + 20,216 curves at l = 3 with the additive term).

**Observation 4 (l = 2, no rational 2-isogeny, not CM).** With weight 1 for each prime of type III,
III*, I_0* or I_n* and v_2(n_q) for multiplicative primes, the inequality holds with zero violations
on both sets (964,432 and 325,510 curves), and weight 2 for I_0* fails (123 and 26 violations). The
isolated-configuration minima suggest more (types III, III* and I_n* appear to contribute 3 each, type
II and IV 1 each), but the prime 2 is the subject of a large literature (Watkins, Dummigan, Calegari
and Emerton, Yazdani, Caro and Pasten, Kazalicki and Kohen) that I have not reconciled with these
numbers, so this is recorded as an observation only.

## 2. What is a theorem and what is new

- **Multiplicative primes, l >= 5.** Kim and Ota (arXiv 1905.02926, published 2023) proved the
  Pollack-Weston conjecture: for a newform f of level N = N+ N- with (N, p) = 1, rho-bar absolutely
  irreducible over Q(sqrt(p*)), N- squarefree, and every q | N- with q = +-1 mod p ramified for
  rho-bar, the p-adic valuation of the congruence ideal of f at level N equals that of its N- new
  congruence ideal plus the sum over q | N- of the Tamagawa exponent t_f(q), the largest t with rho
  mod p^t unramified at q. For an elliptic curve with multiplicative reduction at q, t_f(q) = v_p(n_q)
  (split or not). Agashe, Ribet and Stein (2012) proved that deg phi divides the congruence number
  with quotient supported on primes whose square divides 4N. Together: Conjecture 1 for l >= 5 is a
  theorem when l does not divide N, restricted to the sum over multiplicative q not congruent to +-1
  mod l (and N+ may be anything, so additive primes elsewhere are allowed). Ribet and Takahashi
  (1997) and Takahashi (2001) give the sum over any even number of multiplicative primes for
  squarefree N, up to primes where E[l] is reducible; Pasten (2023) controls that error term.
- **New in Conjecture 1:** the prime l = 3 (excluded by the Taylor-Wiles hypotheses of Kim and
  Ota), primes l dividing N, the primes q = +-1 mod l (where condition (6) of Kim and Ota would drop
  the term), and, above all, the additive contribution of Kodaira types IV and IV*, which is not a
  Tamagawa exponent at all: the mod 3 representation is ramified at such a prime, so no theorem of
  the level-lowering-exponent type applies to it.
- **Conjectures 2 and 3** have no counterpart in the literature I found. The closest statements are
  Watkins's conjecture and the Dummigan (J. Théor. Nombres Bordeaux 18, 2006) and Caro-Pasten (Proc. AMS 150, 2022) results at l = 2 ("number of nonsplit primes minus one"), which have the
  same "minus one" shape as Conjecture 2.
- **Checked against:** Jeon and Kwon (arXiv 2608.06054, 2026) on degree divisibility between levels
  (different question); Papikian and Rabinoff (arXiv 1212.3574) on non-surjectivity of component
  group maps; Hamidi (arXiv 2607.23244) on D-new modular degrees; Moakher (arXiv 2408.15410) on
  quantitative level lowering for Hilbert modular forms. None states the additive-prime inequality.

## 3. Why the type IV and IV* contribution should be true (sketch, not a proof)

At a prime q (q different from 3) of type IV or IV*, E has potentially good reduction with
semistability defect 3: inertia acts on the 3-adic Tate module through a cyclic group of order 3,
with eigenvalues the primitive cube roots of unity. The Tate module is then a free rank-one module
over Z_3[zeta_3], and reduction modulo 3 sends the generator zeta_3 to a nontrivial unipotent matrix
(zeta_3 = 1 - pi with pi the uniformiser, and pi is nonzero modulo 3). So the mod 3 representation is
ramified at q with unipotent inertia image, which is exactly the local shape of a Steinberg
representation: its Serre conductor at q is q^1, not q^2. By the level-lowering theorems of Ribet,
Carayol and Diamond (applied with E[3] irreducible), rho-bar_3 arises from a newform g of level N/q
with q exactly dividing N/q, and f is congruent to the old forms of g modulo 3. That is a mod 3
congruence between f and the orthogonal complement of f in S_2(Gamma_0(N)), so 3 divides the
congruence number, and by Agashe, Ribet and Stein, 3 divides deg phi unless 9 divides N (the data
show the inequality even then, and even at q = 3 itself, which the sketch does not cover). The sketch
gives one factor of 3 per such prime but not the additivity with the multiplicative part; that is the
quantitative statement of Conjecture 1 and would need a Kim-Ota style argument with q^2 in the level.

For the other additive types the sketch predicts nothing at l = 3: defect 2 (I_0*) and 4 (III, III*)
give inertia images that are not unipotent mod 3, and defect 6 (II, II*) gives minus a unipotent,
which is the shape of a Steinberg twisted by the ramified quadratic character. The data agree: those
types force nothing on their own, and the twisted shape is what Conjecture 2 is about.

## 4. Evidence beyond the counts

Isolated configurations at l = 3 (working set; no multiplicative prime with 3 | n_q, all additive
primes of one type): minimum of v_3(deg phi) by type and multiplicity.

| type | one prime | two primes | three primes |
|---|---|---|---|
| IV | 1 (n = 25,414) | 2 (969) | 3 (11) |
| IV* | 1 (30,168) | 2 (2,268) | 3 (45) |
| II | 0 (34,611) | 1 (2,073) | 2 (24) |
| II* | 0 (19,608) | 1 (968) | 2 (8) |
| III, III*, I_0* | 0 | 0 | 0 |
| I_n* | 0 (133,322) | 0 (28,125) | 1 (1,483) |

The hold-out reproduces every minimum in this table. The IV and IV* rows are Conjecture 1; the II,
II* and I_n* rows are Conjecture 2 (for I_n* the weight is v_3(n), so two such primes with 3 | n
give 1, three give 2, and the "0, 0, 1" row mixes primes with and without 3 | n).

Minimum of the excess over Conjecture 1 by (number of II/II* primes, total I_n* weight), working set:

| II/II* count \ I_n* weight | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 1 |
| 1 | 0 | 1 | 2 | 2 |
| 2 | 1 | 2 | 3 | 3 |
| 3 | 2 | 3 | | |

Eisenstein deficit at l = 3 (additive term included), working set, by 3-part of the torsion and
mod 3 image label:

| torsion 3-part | image | deficit 2 | deficit 1 | deficit 0 or less |
|---|---|---|---|---|
| 0 | 3B, 3B.1.2, 3Cs | 0 | 0 | 50,059 |
| 1 | 3B.1.1 (rational 3-torsion point) | 0 | 2,767 | 24,846 |
| 1 | 3Cs.1.1 (E[3] = Z/3 + mu_3) | 17 | 5 | 206 |
| 2 | 3B.1.1 (point of order 9) | 5 | 0 | 8 |

Twenty-one of the twenty-two deficit-2 curves have rank 0, torsion Z/3 and the split-Cartan image
(14a1, 26a1, 35a1, 38a1 and so on); the five with a point of order 9 have Tamagawa products divisible
by 3^5 or more.

Independent check of the degree column: `ellmoddegree` in PARI/GP 2.15 reproduces Cremona's modular
degree for all fifteen curves tried, covering each configuration (121c1, 1369b1, 1849b1, 52a1, 75a1,
76a1, 33a1, 77c1, 114c1, 200e1, 216c1, 675f1, 11a1, 37a1, 389a1). Two of them show that the
geometric exponent and not the Tamagawa number is the right quantity: 33a1 has I_6 at 3 with Tamagawa
number 2 and degree 3; 77c1 has I_3 at 7 with Tamagawa number 1 and degree 6.

Smallest examples of Conjecture 1's additive term with nothing else: 121c1 (N = 11^2, type IV,
degree 6), 1369b1 (37^2, IV*, degree 1332 = 4 * 9 * 37), 1849b1 (43^2, IV, degree 210), 2209b1 (47^2,
IV*, degree 5640). Of Conjecture 2: 200e1 (II* at 2, II at 5, degree 24), 216c1 (II* at 2 and 3,
degree 72), 675f1 (II* at 3, II at 5, degree 108).

## 5. Status against the ten-step plan

- Step 6 (structure in the residuals): done; the structure is in sections 1 and 4.
- Step 7 (symbolic regression): not needed, the statements are exact inequalities.
- Step 8 (hold-out): passed for every statement with zero exceptions.
- Step 9 (precise statement): Conjectures 1 to 3 above.
- Step 10 (proof or hand-off): section 3 is a proof sketch for the qualitative half of the additive
  term; the quantitative statements and Conjectures 2 and 3 are open. The natural next steps are
  (a) extend the check to the 2.1 million classes with known optimality up to conductor 500000,
  (b) compute a few hundred modular degrees of curves of conductor above 500000 with additive
  primes of type IV or IV* (PARI's `ellmoddegree` takes seconds per curve at that size) as an
  out-of-table test, (c) find the mechanism of Conjecture 2, starting from the quadratic twist that
  turns a type II prime into type IV* and an I_n* prime into I_n.

## 6. Files

- `scripts/step6_u1c.py`, `step6_u1b.py`: the per-prime tabulation and the Kim-Ota residue check.
- `scripts/step6_local.py`: the per-bad-prime table `results/local_primes.parquet`.
- `scripts/step6_u1_add.py`, `step6_u1_fit.py`, `step6_u1_twist.py`, `step6_u1_eis.py`: the additive,
  unified, twisted and Eisenstein analyses; outputs in `results/step6_u1_*.txt` for both sets.
