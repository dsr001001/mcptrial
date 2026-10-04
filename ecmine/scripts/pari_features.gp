\\ Usage: echo 'infile="..."; outc="..."; outa="..."; read("scripts/pari_features.gp")' | gp -q -s 256M
\\ Per curve: label fields, root number, disc sign, log|disc|, log j-height, n bad primes,
\\ counts of split/nonsplit/additive bad primes, Kodaira type list, Tamagawa list.
\\ Per isogeny class (optimal curve, number 1): a_p for the 168 primes p < 1000.
{
f = readstr(infile);
P = primes(168);
for(i=1,#f,
  v = strsplit(f[i], " ");
  if(#v < 6, next);
  N = eval(v[1]); iso = v[2]; num = v[3];
  E = ellinit(eval(v[4]));
  w = ellrootno(E);
  D = E.disc; j = E.j;
  jh = log(max(abs(numerator(j)), abs(denominator(j))) + 0.0);
  gr = ellglobalred(E);
  fa = gr[4]; nb = matsize(fa)[1];
  sm = 0; nm = 0; ad = 0; kod = []; tam = [];
  for(k=1, nb, p = fa[k,1]; lr = elllocalred(E, p);
    kod = concat(kod, [lr[2]]); tam = concat(tam, [lr[4]]);
    if(fa[k,2] == 1, if(ellap(E,p) == 1, sm++, nm++), ad++));
  write(outc, Str(N, "\t", iso, "\t", num, "\t", w, "\t", sign(D), "\t", log(abs(D)+0.0), "\t", jh, "\t",
        nb, "\t", sm, "\t", nm, "\t", ad, "\t", kod, "\t", tam));
  if(num == "1",
    ap = vector(#P, k, ellap(E, P[k]));
    write(outa, Str(N, "\t", iso, "\t", ap))));
}
