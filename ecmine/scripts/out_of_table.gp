\\ Out-of-table verification (step 2). Generates curves with prescribed local types, conductor in (NLO, NHI],
\\ trivial isogeny class (so the curve is optimal and E[l] is irreducible for every l), not CM, and computes the
\\ modular degree independently with ellmoddegree. One TSV line per curve:
\\   ainvs  N  kodaira_codes  tamagawa_list  moddeg  family  seconds
\\ Usage: echo 'seed=1; mode=1; NLO=500000; NHI=4000000; out="results/oot_1.tsv"; budget=10800; read("scripts/out_of_table.gp")' | gp -q -s 1G
\\ mode 1: families IV, IV*, pairs and triples of type II, pairs of I_n* with 3 | n (by twisting);  mode 2: random curves.
setrand(seed); t0 = getabstime();
CMJ = [0, 1728, -3375, 8000, -32768, 54000, 287496, -884736, -12288000, 16581375, -884736000, -147197952000, -262537412640768000];
isprime_(q) = isprime(q);
randprime(lo, hi) = my(q); until(isprime(q), q = lo + random(hi - lo + 1)); q;
nz(b) = my(x); until(x != 0, x = random(2*b+1) - b); x;
{
report(a, fam) =
  my(E, N, L, codes, tams, d, t, j);
  E = ellinit(a); if(E.disc == 0, return(0));
  E = ellminimalmodel(E); a = vector(5, i, E[i]);
  N = ellglobalred(E)[1]; if(N <= NLO || N > NHI, return(0));
  j = E.j; for(i=1,#CMJ, if(j == CMJ[i], return(0)));
  if(#ellisomat(E)[1] > 1, return(0));
  L = ellglobalred(E)[5]; codes = vector(#L, i, L[i][2]); tams = vector(#L, i, L[i][4]);
  t = getabstime(); d = ellmoddegree(E); t = (getabstime() - t)/1000.;
  write(out, Str(a, "\t", N, "\t", codes, "\t", tams, "\t", d, "\t", fam, "\t", t));
  1;
}
{
while((getabstime() - t0)/1000. < budget,
  if(mode == 1,
    my(f = random(5));
    if(f == 0, my(q = randprime(5, 400)); report([0, 0, 0, q^2 * nz(40), q^2 * nz(40)], "IV"));
    if(f == 1, my(q = randprime(5, 150)); report([0, 0, 0, q^4 * nz(12), q^4 * nz(12)], "IV*"));
    if(f == 2, my(q1 = randprime(5, 45), q2 = randprime(5, 45)); if(q1 != q2, report([0, 0, 0, q1*q2 * nz(25), q1*q2 * nz(25)], "II,II")));
    if(f == 3, my(q1 = randprime(5, 17), q2 = randprime(5, 17), q3 = randprime(5, 17)); if(q1 != q2 && q2 != q3 && q1 != q3, report([0, 0, 0, q1*q2*q3 * nz(12), q1*q2*q3 * nz(12)], "II,II,II")));
    if(f == 4, my(q = randprime(5, 400)); report([0, 0, 0, q^2 * nz(40), q^2 * nz(40) + q^3 * nz(3)], "IVmixed")),
    report([random(2), random(3) - 1, random(2), nz(60), nz(500)], "random")));
}
