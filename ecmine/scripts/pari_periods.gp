\\ Usage: echo 'infile="..."; out="..."; read("scripts/pari_periods.gp")' | gp -q -s 256M
\\ Per curve: N iso num, c4, c6, real half-period omega1, imaginary part of omega2, lattice area.
{
f = readstr(infile);
for(i=1,#f,
  v = strsplit(f[i], " ");
  if(#v < 6, next);
  E = ellinit(eval(v[4]));
  om = E.omega;
  write(out, Str(v[1], "\t", v[2], "\t", v[3], "\t", E.c4, "\t", E.c6, "\t", real(om[1]), "\t", imag(om[2]), "\t", E.area)));
}
