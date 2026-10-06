\\ Numerical check of Lemma 2.4 of the note: for random curves and every odd bad prime q, the quadratic twist by q*
\\ exchanges the Kodaira types I_n <-> I_n* (same n), II <-> IV*, III <-> III*, IV <-> II*, I_0 <-> I_0* at q, with the
\\ conductor exponents 1 <-> 2 (resp. unchanged), and changes nothing at the other primes.
\\ Usage: gp -q -s 200M < scripts/twist_lemma_check.gp   (output: results/twist_lemma_check.txt)
setrand(5);
tname(k) = if(k>=5, Str("I", k-4), if(k<=-5, Str("I", -k-4, "*"), if(k==1,"I0", if(k==2,"II", if(k==3,"III", if(k==4,"IV", if(k==-1,"I0*", if(k==-2,"II*", if(k==-3,"III*", "IV*")))))))));
bad = 0; tested = 0; pairs = Map();
{
for(trial = 1, 1500,
  my(E = ellinit([random(2), random(3)-1, random(2), random(2001)-1000, random(200001)-100000]));
  if(#E == 0 || E.disc == 0, next);
  E = ellminimalmodel(E);
  my(R = ellglobalred(E), P = R[4][,1], L = R[5]);
  for(i = 1, #P,
    my(q = P[i]); if(q == 2, next);
    my(qs = if(q % 4 == 1, q, -q), Et = ellminimalmodel(elltwist(E, qs)), Rt = ellglobalred(Et), Pt = Rt[4][,1], Lt = Rt[5]);
    tested++;
    my(j = select(x -> x == q, Pt, 1), kt = if(#j, Lt[j[1]][2], 1), ft = if(#j, Lt[j[1]][1], 0));
    my(key = Str(if(q==3, "q=3 ", "q>=5 "), tname(L[i][2]), "->", tname(kt), " f=", L[i][1], "->", ft));
    mapput(pairs, key, iferr(mapget(pairs, key), e, 0) + 1);
    for(m = 1, #P, if(P[m] == q, next); my(jm = select(x -> x == P[m], Pt, 1)); if(!#jm || Lt[jm[1]][2] != L[m][2] || Lt[jm[1]][1] != L[m][1], bad++; print("MISMATCH at p=", P[m], " ", vector(5,i,E[i]), " ", qs)));
    for(m = 1, #Pt, if(Pt[m] == q, next); if(!#select(x -> x == Pt[m], P, 1), bad++; print("EXTRA PRIME ", Pt[m], " in twist of ", vector(5,i,E[i]), " by ", qs)))));
print("tested (curve, odd q) pairs: ", tested, "   mismatches away from q: ", bad);
my(K = Vec(pairs)); for(i = 1, #K, print(K[i], "   ", mapget(pairs, K[i])));
}
quit;
