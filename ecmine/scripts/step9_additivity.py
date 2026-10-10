"""Additivity of the Euler-factor term of a single additive prime with the Tamagawa terms of the multiplicative primes,
split by the residue classes of the multiplicative primes carrying a Tamagawa term. Usage: python3 scripts/step9_additivity.py l"""
import sys, pandas as pd, numpy as np, cypari2
pari = cypari2.Pari()
l = int(sys.argv[1]) if len(sys.argv) > 1 else 3
def vl(x):
    v = 0; x = int(x)
    while x > 0 and x % l == 0: x //= l; v += 1
    return v
g = pd.read_parquet(f"results/step9_fullrule_work_l{l}.parquet")        # per curve: e, tam, wild, old, V, new
lp = pd.read_parquet("results/local_primes.parquet"); lp = lp[(lp.set == "work") & lp.label.isin(g.index)]
add = lp[lp.type != "In"]; nadd = add.groupby("label").size(); one = nadd[nadd == 1].index
a1 = add[add.label.isin(one)].set_index("label")[["q", "type", "n"]]
j = a1.join(g, how="inner"); j = j[(j.q >= 5) & (j.q != l) & (j.e >= 1) & (j.tam >= 1)]
mult = lp[(lp.type == "In")].copy(); mult["vn"] = mult.n.map(vl); mult = mult[mult.vn > 0]
cls = mult.groupby("label").apply(lambda s: "all +-1 mod l" if all((int(q) % l in (1, l - 1)) for q in s.q) else ("none +-1 mod l" if not any((int(q) % l in (1, l - 1)) for q in s.q) else "mixed"), include_groups=False)
cls1 = mult.groupby("label").apply(lambda s: "-1 only" if all(int(q) % l == l - 1 for q in s.q) else ("+1 only" if all(int(q) % l == 1 for q in s.q) else "other"), include_groups=False)
j["cls"] = cls.reindex(j.index); j["cls1"] = cls1.reindex(j.index); j["Y"] = j.V - j.e - j.tam
print(f"l = {l}: single additive prime q >= 5 with Euler term e >= 1, and multiplicative Tamagawa terms (tam >= 1): {len(j)} curves")
print("min(V - e - tam) and P(V >= e + tam) by type and residue class of the multiplicative primes with l | n:")
print(j.groupby(["type", "cls"]).Y.agg(["min", "size", lambda s: round((s >= 0).mean(), 4)]).rename(columns={"<lambda_0>": "P(additive)"}).to_string())
print("\n... by finer class (-1 only / +1 only / other):")
print(j.groupby(["type", "cls1"]).Y.agg(["min", "size", lambda s: round((s >= 0).mean(), 4)]).rename(columns={"<lambda_0>": "P(additive)"}).to_string())
print("\nfor the violating curves: V - tam (is the Euler term simply absent?) and V - e:")
vv = j[j.Y < 0]; print(pd.DataFrame({"V-tam": vv.V - vv.tam, "V-e": vv.V - vv.e, "e": vv.e, "tam": vv.tam}).describe().loc[["min", "max", "mean"]].to_string())
