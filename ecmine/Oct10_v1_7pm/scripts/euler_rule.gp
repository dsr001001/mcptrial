\\ The Euler-factor rule for one curve (the note, Conjecture 1.3 and Theorem 1.2).
\\   rule(E, l)     returns [v_l(deg phi_E), bound] with the type III terms left out (the form that holds without (TW));
\\   rule(E, l, 1)  includes the type III terms and, at l = 3, subtracts 1 when the mod 3 image is the normaliser of a
\\                  split Cartan subgroup (ns3(E) = 1), the correction term of Conjecture 1.3.
\\ Usage:  gp -q -D parisizemax=400000000 -D threadsizemax=200000000
\\         read("scripts/euler_rule.gp"); rule(ellinit([1,1,1,-30,-76]), 3)      \\ 121a1: [1, 1]
\\ E should be the optimal curve of its class, without CM and without a rational l-isogeny.
rule(E, l, {iii = 0}) =
{
  my(R = ellglobalred(E), P = R[4][,1], K = R[5], S = lfunsympow(E, 2), b = 0, j = E.j);
  for(i = 1, #P,
    my(q = P[i], k = K[i][2], e = 0, t = 0, vj = valuation(j, q));
    if(q != l,
      my(F = 1/lfuneuler(S, q), r = subst(F, x, 1/q^2));   \\ P_q(q^{-2}): the omitted Euler factor at s = 2
      if(k >= 5, r /= (1 - 1/q^2));                       \\ multiplicative prime: remove the naive factor
      if(abs(k) != 3 || iii, e = valuation(r, l)),        \\ types III, III* (Kodaira codes 3, -3) only if iii = 1
      if(l == 3, e = if(k == 4 || k == 2, 1, if(k == -4 || k == -2, 2, if(k == -3, 1, 0)))));  \\ w_3 at q = 3
    if(vj < 0, t = valuation(-vj, l));                     \\ Tamagawa exponent: v_l of the valuation of the Tate parameter
    b += e + t);
  if(iii && l == 3, b -= ns3(E));
  [valuation(ellmoddegree(E), l), b];
}
\\ 1 if the 3-division polynomial factors as two quadratics, i.e. (for E[3] irreducible) the image of the mod 3
\\ representation is the normaliser of a split Cartan subgroup, where the Taylor-Wiles hypothesis fails.
ns3(E) = my(F = factor(elldivpol(E, 3))); (#F[,1] == 2 && poldegree(F[1,1]) == 2);
