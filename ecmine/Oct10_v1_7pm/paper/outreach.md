# Outreach texts (updated 2026-10-10, evening)

These are optional. The note is written to stand on its own for a journal submission; the texts below are for the case
where you want to inform the authors whose theorems the note assembles, or ask whether the consequence has been noted
before. Suggested recipients, in order: Fred Diamond or Matthias Flach (Theorem 1.2 of the note is assembled from their
paper with Li Guo), Chan-Ho Kim or Kazuto Ota (the Tamagawa exponent and the congruence-ideal comparison), Hector Pasten
and Amod Agashe (modular degree versus congruence number), Mark Watkins (the computations).

## A. Message to Diamond or Flach (about 300 words)

Subject: a consequence of your Tamagawa number theorem for the modular degree of elliptic curves

Dear Professor [Name],

I am writing about a consequence of your paper with Li Guo on the Tamagawa number conjecture of adjoint motives
(Ann. Sci. ENS 37, 2004), which I have not found stated anywhere and would like to make sure is not already known.

Let E be the X_0(N)-optimal curve of its isogeny class, f its newform, and l an odd prime not dividing 2N such that the
mod l representation of E is absolutely irreducible on Gal(Qbar/Q(sqrt(l*))), i.e. l is not in your set S_f. Combining
your Theorem 3.7 (for Sigma the set of bad primes), your Proposition 1.4(c), your Lemma 2.1 with Theorem 2.7, the local
formula of Fontaine and Perrin-Riou as quoted in your proof of Theorem 2.15, the inequality half of Lemma 4.17 of
Darmon-Diamond-Taylor, and the theorem of Agashe, Ribet and Stein comparing the modular degree with the congruence
number, one gets

    v_l(deg phi_E) >= length H^1_f(Q, ad^0 T (x) Q_l/Z_l) + sum_q v_l(Tam_q(ad^0 T)) + sum_q e_l(E,q),

with equality under multiplicity one at level N, where e_l(E,q) is the l-adic valuation of the Euler factor of the
adjoint L-function at s = 1 that the naive adjoint L-function omits (zero unless q is additive). For elliptic curves the
omitted factors are explicit (v_l(q - chi_{-3}(q)) for types II, II*, IV, IV*, v_l(q - chi_{-4}(q)) for III, III*,
v_l(q^2-1) for I_n*, and the level-raising factor for I_0*), and the Tamagawa factor of ad^0 at a Steinberg prime is
l^{v_l(n) + min(v_l(n), v_l(q-1), v_l(c_q))}, c_q the class of the unit part of the Tate parameter. So the modular degree
is divisible by the omitted Euler factors and by the Tamagawa exponents.

Two things make me think this is worth recording. First, the hypothesis is sharp: for l = 3 the curves whose mod 3
image is the normaliser of a split Cartan subgroup, for which rho-bar is isomorphic to its twist by chi_{-3}, violate the
inequality in about a third of the cases, always by exactly one, and the one is the length of H^0(Q, ad^0 T(1) (x)
Q_3/Z_3), the term of your Lemma 2.1; no other curve of conductor up to 500000 violates it. Second, the Tamagawa factor
formula is visible in the data: where it predicts an extra unit of divisibility (a Steinberg prime q = 1 mod l whose
Tate parameter has an l-th power as unit part), the modular degree has it, on all 12,095 such curves at l = 3.

A note with the proofs and the tables is attached / at [link]. If this consequence of your theorem is known, or if I have
misread one of the statements I assemble, I would be grateful to be told.

With best regards,
Dharmaj Soni
ORCID 0009-0007-6397-3351, dharmaj.soni@gmail.com

I should add that the computations, the literature search and the drafts, this message included, were produced with
an AI system under my direction; the note says so in a statement at the end, and I take responsibility for everything
stated.

## B. Short announcement (for a mailing list or a talk abstract)

For an elliptic curve E/Q of conductor N and an odd prime l not dividing 2N with E[l] absolutely irreducible over
Q(sqrt(l*)), the l-adic valuation of the modular degree of E is at least the sum, over the bad primes, of the l-adic
valuations of the Euler factors of the adjoint L-function at s = 1 that the naive adjoint L-function omits, plus the
Tamagawa exponents at the multiplicative primes. This follows from the Tamagawa number theorem of Diamond, Flach and Guo
for the adjoint motive, read for elliptic curves; the local terms are explicit in the Kodaira types. The hypothesis is
sharp: at l = 3 exactly the curves with mod 3 image the normaliser of a split Cartan subgroup violate the inequality, by
one. The inequality, with that correction, holds for every optimal curve of conductor up to 500000 and 893 curves beyond,
including primes l dividing N where no theorem applies.
