# Outreach texts (step 11)

Both texts are drafts for you to send under your own name. Suggested recipients, in order of how directly
their work is extended: Hector Pasten (refinement of Ribet-Takahashi; Watkins at 2), Chan-Ho Kim or Kazuto Ota
(the quantitative level-lowering theorem the multiplicative part rests on), Amod Agashe (modular degree versus
congruence number), Mark Watkins (the computations and the 2002 conjecture), Neil Dummigan (Watkins at 2).

## A. Message to an expert (about 250 words)

Subject: a divisibility of the modular degree by local component groups: known?

Dear Professor [Name],

I am writing because a computation I have been doing touches on your work on [the Ribet-Takahashi formula /
the Pollack-Weston conjecture / the modular degree], and I would value a quick opinion on whether the
following is known before I write it up.

For E the X_0(N)-optimal curve of its class and an odd prime l with E[l] irreducible, the data up to
conductor 500000 (every optimal curve, about 2.1 million) satisfy

    v_l(deg phi_E) >= sum over bad q of v_l(#Phi_q(F_q-bar)),

with no exception, where Phi_q is the geometric component group. For l >= 5 the right-hand side only sees
multiplicative primes through v_l(v_q(Delta)); I understand that part to follow, for l not dividing N and
primes q not congruent to +-1 mod l, from Kim and Ota's proof of the Pollack-Weston conjecture together with
Agashe-Ribet-Stein, and I would be grateful to be corrected if an earlier reference states the inequality
itself. What I have not found anywhere is the prime 3: every prime of Kodaira type IV or IV* contributes one
to v_3(deg phi_E), whatever its Tamagawa number, and the contributions add up with the multiplicative ones.
The qualitative statement (3 divides the degree) follows from level lowering, because the mod 3 inertia
action at such a prime is a nontrivial unipotent; the additivity I can only verify numerically. There is
also a curious "all but the largest" rule for the twisted types II, II* and I_n*, and a sharp description
of how the inequality fails in the Eisenstein case.

A two-page summary with the tables is attached / at [link]. If this is a special case of something you
know, a one-line pointer would save me a great deal of time; if it is not, I would be glad to hear whether
the additive statement looks provable by the methods of [your paper].

With best regards,
[Name]

## B. MathOverflow question (title and body)

Title: Does each Kodaira type IV or IV* prime force a factor of 3 in the modular degree?

Body:

Let $E/\mathbb{Q}$ be the $X_0(N)$-optimal curve in its isogeny class, $\deg\varphi_E$ its modular degree,
and for a bad prime $q$ let $\Phi_q$ be the component group of the Néron model over $\overline{\mathbb{F}}_q$
(order $n$ for $\mathrm{I}_n$, $3$ for $\mathrm{IV}$ and $\mathrm{IV}^*$, and so on).

**Observation.** For every optimal non-CM curve of conductor at most 500000 and every odd prime $\ell$ for
which $E[\ell]$ is irreducible,
$$v_\ell(\deg\varphi_E)\ \ge\ \sum_{q\mid N} v_\ell(\#\Phi_q(\overline{\mathbb{F}}_q)).$$
For $\ell \ge 5$ only multiplicative primes contribute, and I believe that part is a consequence of the
Pollack-Weston conjecture (proved by Kim and Ota) plus Agashe-Ribet-Stein, at least for $\ell\nmid N$ and
$q\not\equiv\pm1\pmod\ell$. For $\ell=3$ the primes of type IV and IV* contribute one each, independently of
the Tamagawa number, and the contributions add up with the multiplicative ones. Example: $121c1$ has type IV
at $11$ and modular degree $6$; $1369b1$ has type IV* at $37$ and degree $1332 = 4\cdot 9\cdot 37$.

The qualitative statement at a single type IV prime $q\ge 5$ seems to follow from level lowering: the inertia
image on $T_3E$ is cyclic of order 3, $T_3E$ is free of rank one over $\mathbb{Z}_3[\zeta_3]$, so mod 3 the
inertia acts as a nontrivial unipotent and the Serre conductor has exponent 1 at $q$; lowering the level from
$q^2$ to $q$ gives a mod 3 congruence, hence 3 divides the congruence number, hence (ARS) the degree.

**Questions.** (1) Is the inequality, in particular its additive term at 3, known or a consequence of a known
result? (2) Is there a reason to expect the additivity (the quantitative statement), for instance a version
of the Kim-Ota formula with $q^2$ in the level? (3) The data also show that primes of type II, II* and
$\mathrm{I}_n^*$ with $3\mid n$ contribute "all but the largest" of their weights (two type II primes force
$3\mid\deg\varphi_E$, one does not, three force $9$). Is there a mechanism for that?

Tags: nt.number-theory, elliptic-curves, modular-forms, galois-representations
