\\ The Euler-factor rule for one curve (Conjecture 1.2 of the note): rule(E, l) returns [v_l(deg phi_E), bound].
\\ Usage:  gp -q -D parisizemax=400000000 -D threadsizemax=200000000
\\         read("scripts/euler_rule.gp"); rule(ellinit([1,1,1,-30,-76]), 3)      \\ 121a1: [1, 1]
\\ The curve should be the optimal curve of its class, without CM and without a rational l-isogeny.
rule(E, l) =
{
  my(R = ellglobalred(E), P = R[4][,1], K = R[5], S = lfunsympow(E, 2), b = 0, j = E.j);
  for(i = 1, #P,
    my(q = P[i], k = K[i][2], e = 0, t = 0, vj = valuation(j, q));
    if(q != l,
      my(F = 1/lfuneuler(S, q), r = subst(F, x, 1/q^2));   \\ F = P_q(X); r = P_q(q^{-2}), the omitted factor at s = 2
      if(k >= 5, r /= (1 - 1/q^2));                       \\ multiplicative prime: divide by the naive factor
      if(abs(k) != 3, e = valuation(r, l)),               \\ types III, III* (Kodaira codes 3, -3) are not counted
      if(l == 3, e = if(k == 4 || k == 2, 1, if(k == -4 || k == -2, 2, if(k == -3, 1, 0)))));  \\ w_3 at q = 3
    if(vj < 0, t = valuation(-vj, l));                     \\ Tamagawa exponent of the adjoint: v_l of the Tate parameter valuation
    b += e + t);
  [valuation(ellmoddegree(E), l), b];
}
