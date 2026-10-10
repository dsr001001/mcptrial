# Outreach texts (updated 2026-10-10 for the Euler-factor rule)

Both texts are drafts for you to send under your own name. Suggested recipients, in order of how directly
their work is extended: Chan-Ho Kim or Kazuto Ota (the quantitative level-lowering theorem and the
Sigma-imprimitive adjoint Selmer groups; Section 5 of the note is the question for them), Fred Diamond or
Matthias Flach (the naive congruence ideal and the refined structures of DFG section 1.8), Hector Pasten
(refinement of Ribet-Takahashi; Watkins at 2), Amod Agashe (modular degree versus congruence number),
Mark Watkins (the computations and the symmetric-square formula), Neil Dummigan (Watkins at 2).

## A. Message to an expert (about 350 words)

Subject: the l-adic valuation of the modular degree and the omitted Euler factors of the adjoint: known?

Dear Professor [Name],

I am writing because a computation I have been doing touches on your work on [the Sigma-imprimitive
adjoint Selmer groups and congruence ideals / the Pollack-Weston conjecture / the modular degree], and I
would value a quick opinion on whether the following is known before I circulate it.

For E the X_0(N)-optimal curve of its class, f its newform and an odd prime l with E[l] irreducible, let
e_l(E,q) be the l-adic valuation of the Euler factor of L(ad^0 f, s) at s = 1 that the naive adjoint
L-function (the one of Diamond-Flach-Guo, with trivial factor at the additive primes) omits, and let
t_l(E,q) be the Tamagawa exponent of ad^0 at a potentially multiplicative prime (the l-part of the valuation
of the Tate parameter). For a tame prime q >= 5 the omitted factor gives v_l(q - chi_{-3}(q)) for the types
II, II*, IV, IV*, v_l(q^2 - 1) for I_n* and v_l(q - 1) + v_l((q+1)^2 - a_q(E')^2) for I_0*, with E' the twist
of good reduction. The data (every optimal curve of conductor up to 500000, about 2.1 million, and 893 curves
of conductor up to 4 million outside the tables, with PARI's Euler factors at every bad prime) satisfy

    v_l(deg phi_E) >= sum over bad q != l of e_l(E,q) + sum over bad q of t_l(E,q)

with the type III terms left out, with no exception at l = 3, 5, 7, 11, 13, and with equality in 57% of the
cases with a positive bound at l = 3. The type III terms are attained in isolation like the others, but they
fail to add, by exactly one and at l = 3 only, when a III prime with trivial Frobenius on its invariant line
meets a Steinberg prime p = -1 mod 3, which is the configuration excluded in your hypotheses.

I can prove the terms of the types I_0* and I_n* for every odd l not dividing 2N with E[l] irreducible: the
twist by the characters of those primes lowers the level, f is the depleted form of the twist, your
Proposition 1.4(c) [DFG] / Lemma 2.7 [Kim-Ota] says that depletion multiplies the naive congruence ideal by
the Euler factor, and an elementary twisting argument carries the congruences back to level N. The Tamagawa
terms for l >= 5 follow from the Kim-Ota formula applied to the twist. For the potentially good types II, IV,
III the twist does not lower the level, and the statement becomes, under DFG's Theorem 0.2, an inequality
between the length of the Sigma-imprimitive adjoint Selmer group and the local terms, i.e. the Bloch-Kato
Selmer group plus the local H^0(Q_p, ad rho(1) (x) K/O) minus a dual Selmer correction, and the data say the
correction vanishes except in the type III configuration above, where it is one.

A twenty-page note with the proofs and the tables, and a short memorandum with the Selmer-group
reformulation, are attached / at [link]. If the Selmer inequality is a special case of something you know,
or if the refined structures of DFG section 1.8 settle it, a one-line pointer would save me a great deal of
time; if not, I would be glad to hear whether it looks provable by the methods of [your paper].

With best regards,
Dharmaj Soni
ORCID 0009-0007-6397-3351, dharmaj.soni@gmail.com

## B. MathOverflow question (title and body)

Title: Do the Euler factors omitted by the naive adjoint L-function divide the modular degree?

Body:

Let $E/\mathbb{Q}$ be the $X_0(N)$-optimal curve in its isogeny class, $f$ its newform, $\deg\varphi_E$ its
modular degree, and $\ell$ an odd prime with $E[\ell]$ irreducible. Write $L(\operatorname{ad}^0 f,s)$ for the
adjoint $L$-function and $L^{\mathrm{nv}}$ for its naive version (Diamond-Flach-Guo), whose Euler product has the
trivial factor at the additive primes. For a bad prime $q\ne\ell$ let $e_\ell(E,q)$ be the $\ell$-adic valuation
of the omitted factor $L_q(\operatorname{ad}^0 f,1)^{-1}/L_q^{\mathrm{nv}}(\operatorname{ad}^0 f,1)^{-1}$, and let
$t_\ell(E,q)=v_\ell(-v_q(j_E))$ if $v_q(j_E)<0$ and $0$ otherwise (the Tamagawa exponent of $\operatorname{ad}^0$).

**Observation.** For every optimal non-CM curve of conductor at most 500000, at $\ell=3,5,7,11,13$, and with
PARI's symmetric-square Euler factors at every bad prime,
$$v_\ell(\deg\varphi_E)\ \ge\ \sum_{q\mid N,\ q\ne\ell,\ \text{type}\ne \mathrm{III},\mathrm{III}^*} e_\ell(E,q)\ +\ \sum_{q\mid N} t_\ell(E,q),$$
with no exception, and with equality in 57% of the cases with a positive bound at $\ell=3$. For a tame prime
$q\ge5$ the first term is $v_\ell(q-\chi_{-3}(q))$ for the types II, II*, IV, IV*, $v_\ell(q^2-1)$ for
$\mathrm{I}_n^*$ and $v_\ell(q-1)+v_\ell((q+1)^2-a_q(E')^2)$ for $\mathrm{I}_0^*$ ($E'$ the twist of good
reduction). Example: $1369b1$ has type IV* at $37$ and degree $1332=4\cdot 9\cdot 37$, and $v_3(37-1)=2$.
The type III terms are attained in isolation but fail to add, by one, at $\ell=3$ only, when the III prime has
trivial Frobenius on its invariant line and there is a Steinberg prime $p\equiv-1\pmod 3$.

**What I can prove.** For the types $\mathrm{I}_0^*$ and $\mathrm{I}_n^*$ and every odd $\ell\nmid 2N$ with
$E[\ell]$ irreducible: twisting by the characters of those primes lowers the level, $f$ is the depleted form
of the twist, Diamond-Flach-Guo's Proposition 1.4(c) says that depleting at a prime multiplies the naive
congruence ideal by the Euler factor, and an elementary twisting argument (twists of congruent forms are
congruent, and the congruence module is cyclic) carries the congruences back to level $N$. The Tamagawa terms
for $\ell\ge5$ follow from Kim-Ota's quantitative level lowering applied to the twist. For the potentially
good types the twist does not lower the level, and under DFG's Theorem 0.2 the statement is an inequality
between the length of the $\Sigma$-imprimitive adjoint Selmer group and the sum of the local terms.

**Questions.** (1) Is the inequality for the types II, IV, III (or its Selmer-group form) known, or a
consequence of a known result on $\Sigma$-imprimitive adjoint Selmer groups? (2) Is there a conceptual reason
for the type III exception, i.e. for the dual Selmer correction to be nonzero exactly when a type III prime
meets a Steinberg prime $p\equiv-1\pmod\ell$? (3) What is the right local term at $q=\ell=3$ (empirically:
one for II, IV, III*, two for II*, IV*, nothing for III, $\mathrm{I}_0^*$)?

Tags: nt.number-theory, elliptic-curves, modular-forms, galois-representations
