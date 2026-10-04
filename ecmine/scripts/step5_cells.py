"""Step 5, family B: forbidden, depleted and enriched cells.

The regression family (step5.py) measures mean shifts. Number theory also produces constraints
(divisibility, congruences, inequalities) that occupy a sliver of the rows and explain no variance:
Watkins' 2^rank | moddeg and Lorenzini's torsion-to-Tamagawa divisibility are of that kind, and both
failed the regression family's positive control. This family tests, for every pair of discrete (or
binned) columns, whether some cell of the contingency table is far emptier or fuller than conditional
independence within conductor strata predicts.

Usage: python3 scripts/step5_cells.py
Writes results/step5_cells.csv (flagged cells) and results/step5_cells_pairs.csv (one row per pair).
"""
import os, sys, time, itertools, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from load import load_curves
from scipy.special import gammaln

t0 = time.time()
df = load_curves()
df["log_omega"] = np.log(df["omega"]); df["log_regulator"] = np.log(df["regulator"])
df["two_adic_log_index"] = np.log2(df["two_adic_index"].astype(float))
df["bad_at_2"] = (df["N"] % 2 == 0); df["bad_at_3"] = (df["N"] % 3 == 0)
df["rank_c"] = df["rank"].clip(upper=3).astype(int)
df["is_opt"] = (df["optimal"] == 1).fillna(False).to_numpy(); df["is_rep"] = (df["num"] == 1).to_numpy()
df["petersson_resid"] = df["class_label"].map(df.loc[df["is_opt"]].set_index("class_label")["petersson_resid"])
KOD = ["n_kod_In", "n_kod_II", "n_kod_III", "n_kod_IV", "n_kod_I0s", "n_kod_Ins", "n_kod_IIs", "n_kod_IIIs", "n_kod_IVs"]
print(f"loaded {len(df)} curves in {time.time()-t0:.0f}s", flush=True)

# column -> (group, class_level, discretisation)
CAT = {}
def cat(col, group, cl, cap=None):
    CAT[col] = (group, cl, ("cap", cap))
def binned(col, group, cl, nb=8):
    CAT[col] = (group, cl, ("bin", nb))
for c in ["torsion_structure", "two_adic_level", "two_adic_label", "n_nonsurj_primes", "n_torsion_gens"]: cat(c, "GAL", False)
cat("class_size", "GAL", True); cat("max_isogeny_degree", "GAL", True)
for c in ["has_2_isog", "has_3_isog", "has_5_isog", "has_7_isog", "is_cm"]: cat(c, "GAL", True)
cat("exceptional_image", "GAL", False); cat("v2_torsion", "GAL", False); cat("odd_torsion", "GAL", False); cat("two_adic_log_index", "GAL", False, cap=8)
cat("rank_c", "AN", True); cat("root_number", "AN", True)
for c in ["v2_sha", "v3_sha", "v5_sha", "v7_sha"]: cat(c, "AN", False, cap=3)
for c in ["v2_moddeg", "v3_moddeg", "v5_moddeg", "v7_moddeg"]: cat(c, "AN", False, cap=8)
for c in ["disc_sign", "c4_sign", "c6_sign"]: cat(c, "EQ", False)
cat("n_bad_primes", "LOC", True, cap=6); cat("n_split_mult", "LOC", True, cap=5); cat("n_nonsplit_mult", "LOC", True, cap=5); cat("n_additive", "LOC", True, cap=4)
for c in KOD: cat(c, "LOC", False, cap=3)
for c in ["v2_tamagawa", "v3_tamagawa", "v5_tamagawa", "v7_tamagawa"]: cat(c, "LOC", False, cap=4)
cat("bad_at_2", "LOC", True); cat("bad_at_3", "LOC", True)
cat("n_intpts_x", "DIO", False, cap=10); cat("is_opt", "BK", False)
for c in ["log_abs_disc", "log_j_height", "log_abs_c4"]: binned(c, "EQ", False)
for c in ["log_omega", "log_area", "log_regulator", "log_odd_tamagawa", "log_odd_sha", "log_odd_moddeg"]: binned(c, "AN" if c in ("log_omega", "log_area", "log_regulator", "log_odd_sha", "log_odd_moddeg") else "LOC", False)
binned("petersson_resid", "AN", True)
for c in ["nagao_1000", "frac_even", "frac_zero", "st_mean", "st_m2", "mur_lo", "mur_mid", "mur_hi"]: binned(c, "AP", True)

# pairs whose exact relation is on the exclusion list (unordered); everything else in the same group is "same-group"
KNOWN = {
    ("rank_c", "root_number"): "T1", ("rank_c", "v2_moddeg"): "C1", ("rank_c", "log_regulator"): "I1 (rank 0 has regulator 1)",
    ("torsion_structure", "disc_sign"): "T7", ("torsion_structure", "v5_tamagawa"): "T13", ("torsion_structure", "v7_tamagawa"): "T13",
    ("torsion_structure", "v3_tamagawa"): "T13", ("torsion_structure", "bad_at_2"): "T17", ("torsion_structure", "bad_at_3"): "T17",
    ("torsion_structure", "frac_even"): "T5/T18", ("torsion_structure", "n_torsion_gens"): "I", ("torsion_structure", "v2_torsion"): "I",
    ("torsion_structure", "odd_torsion"): "I", ("is_cm", "frac_zero"): "T10", ("is_cm", "n_split_mult"): "T10", ("is_cm", "n_nonsplit_mult"): "T10",
    ("is_cm", "log_j_height"): "T10", ("root_number", "n_split_mult"): "T2", ("rank_c", "n_split_mult"): "T1+T2", ("rank_c", "nagao_1000"): "C7",
    ("root_number", "nagao_1000"): "C7", ("rank_c", "mur_lo"): "C7", ("rank_c", "mur_mid"): "C7", ("rank_c", "mur_hi"): "C7", ("rank_c", "st_mean"): "C7",
    ("disc_sign", "c4_sign"): "I4", ("disc_sign", "c6_sign"): "I4", ("disc_sign", "log_omega"): "T11", ("log_abs_disc", "log_omega"): "T11",
    ("log_j_height", "log_omega"): "T11", ("log_abs_c4", "log_omega"): "T11", ("log_area", "log_omega"): "I", ("rank_c", "v2_sha"): "C4", ("rank_c", "v3_sha"): "C4",
    ("rank_c", "log_odd_sha"): "C4", ("rank_c", "n_intpts_x"): "T15", ("torsion_structure", "n_intpts_x"): "T15", ("n_bad_primes", "n_split_mult"): "I2",
    ("n_bad_primes", "n_nonsplit_mult"): "I2", ("n_bad_primes", "n_additive"): "I2", ("v2_sha", "v2_moddeg"): "T14", ("v3_sha", "v3_moddeg"): "T14",
    ("v5_sha", "v5_moddeg"): "T14", ("v7_sha", "v7_moddeg"): "T14", ("disc_sign", "v2_sha"): "H1", ("rank_c", "petersson_resid"): "U2 (candidate)",
}
TORS_ISOG = {"torsion_structure", "n_torsion_gens", "v2_torsion", "odd_torsion", "has_2_isog", "has_3_isog", "has_5_isog", "has_7_isog",
             "max_isogeny_degree", "class_size", "two_adic_level", "two_adic_label", "two_adic_log_index", "n_nonsurj_primes", "exceptional_image"}
LOCAL = {"n_split_mult", "n_nonsplit_mult", "n_additive", "n_bad_primes", "bad_at_2", "bad_at_3", "v2_tamagawa", "v3_tamagawa", "v5_tamagawa", "v7_tamagawa",
         "log_odd_tamagawa", "c4_sign", "c6_sign", "disc_sign", "log_abs_disc", "log_j_height", "log_abs_c4"} | set(KOD)
MODDEG_V = {"v2_moddeg", "v3_moddeg", "v5_moddeg", "v7_moddeg", "log_odd_moddeg"}
SIZE = {"log_omega", "log_area", "log_regulator", "log_abs_disc", "log_j_height", "log_abs_c4"}
KNOWN.update({("n_bad_primes", "petersson_resid"): "T12 (index of Gamma_0(N) carries prod(1+1/p))", ("bad_at_2", "petersson_resid"): "T12 (index of Gamma_0(N))",
              ("bad_at_3", "petersson_resid"): "T12 (index of Gamma_0(N))", ("n_bad_primes", "frac_zero"): "T10 (CM curves have few bad primes)",
              ("is_cm", "n_kod_In"): "T10", ("is_cm", "n_additive"): "T10", ("is_cm", "n_bad_primes"): "T10", ("class_size", "is_opt"): "I5", ("max_isogeny_degree", "is_opt"): "I5",
              ("is_opt", "v2_moddeg"): "T12 (non-optimal degrees carry the isogeny)", ("is_opt", "log_odd_moddeg"): "T12", ("is_opt", "log_omega"): "T8", ("is_opt", "log_area"): "T8",
              ("is_opt", "log_regulator"): "T8", ("is_opt", "v2_sha"): "T8", ("is_opt", "v2_tamagawa"): "T8", ("is_opt", "log_odd_tamagawa"): "T8", ("is_opt", "torsion_structure"): "T8"})
def known(a, b):
    if (a, b) in KNOWN: return KNOWN[(a, b)]
    if (b, a) in KNOWN: return KNOWN[(b, a)]
    ga, gb = CAT[a][0], CAT[b][0]
    if ga == gb: return f"same-group {ga}"
    if ga in ("LOC", "EQ") and gb in ("LOC", "EQ"): return "T4 (local data and equation)"
    if (a in TORS_ISOG and b in LOCAL) or (b in TORS_ISOG and a in LOCAL): return "family L: torsion/isogeny constrains local data (T4/T13/T17 type)"
    if (a in TORS_ISOG and b in MODDEG_V) or (b in TORS_ISOG and a in MODDEG_V): return "family E: Eisenstein congruence, rational torsion/isogeny divides the congruence number"
    if (a in TORS_ISOG and b in SIZE) or (b in TORS_ISOG and a in SIZE): return "family S: large isogeny degree forces special j and small models (T9/T10/T8)"
    if "is_cm" in (a, b) and (a in SIZE or b in SIZE or a in LOCAL or b in LOCAL): return "T10 (CM: integral j, no multiplicative reduction)"
    if "is_cm" in (a, b) and (a.startswith(("frac", "st_", "mur", "nagao")) or b.startswith(("frac", "st_", "mur", "nagao"))): return "T10 (CM trace pattern)"
    if {a, b} <= (MODDEG_V | {"petersson_resid"}) : return "same quantity (moddeg)"
    AP_ = {"nagao_1000", "frac_even", "frac_zero", "st_mean", "st_m2", "mur_lo", "mur_mid", "mur_hi"}
    SHA = {"v2_sha", "v3_sha", "v5_sha", "v7_sha", "log_odd_sha"}
    TAM = {"v2_tamagawa", "v3_tamagawa", "v5_tamagawa", "v7_tamagawa", "log_odd_tamagawa"}
    RANKLIKE = {"rank_c", "root_number", "log_regulator"}
    GALX = TORS_ISOG | {"is_cm", "c4_sign", "c6_sign"}
    SIZE2 = SIZE | {"log_omega", "log_area"}
    MODD = MODDEG_V | {"petersson_resid"}
    LOCC = {"n_split_mult", "n_nonsplit_mult", "n_additive", "n_bad_primes", "bad_at_2", "bad_at_3"} | set(KOD)
    if "is_opt" in (a, b): return "I5/T8 (optimal vs non-optimal curves differ by the isogeny)"
    if ({a, b} & {"c4_sign", "c6_sign"}) and ({a, b} & (MODD | TAM | SHA)): return "T10 (c4=0 or c6=0 is a CM curve with j=0 or 1728)"
    if ({a, b} & TAM) and "petersson_resid" in (a, b): return "T4+T12 (Tamagawa tracks v_p(Delta), hence the covolume in the residual)"
    if ("frac_even" in (a, b) or "frac_zero" in (a, b)) and (a in GALX or b in GALX): return "T18/T5 (mod-l image fixes a_p mod l; c4=0 or c6=0 is CM)"
    if (a in AP_ and b in (RANKLIKE | SHA | {"n_intpts_x"})) or (b in AP_ and a in (RANKLIKE | SHA | {"n_intpts_x"})): return "C7 (trace summaries are rank proxies; Sha and integral points follow rank by C4, T15)"
    if (a in AP_ and b in GALX) or (b in AP_ and a in GALX): return "T18/T5 (mod-l image fixes a_p mod l)"
    if (a in AP_ and b in (SIZE2 | LOCC | TAM | MODD)) or (b in AP_ and a in (SIZE2 | LOCC | TAM | MODD)): return "C7/T3 via conductor and bad primes"
    if "log_area" in (a, b) and (a in SIZE or b in SIZE): return "T11 (area is the covolume, a function of c4, c6)"
    if "n_intpts_x" in (a, b) and (a in TORS_ISOG or b in TORS_ISOG): return "T15 (torsion points are integral)"
    if "n_intpts_x" in (a, b) and (a == "log_regulator" or b == "log_regulator"): return "T15 (small generators are integral)"
    if "n_intpts_x" in (a, b) and (a in RANKLIKE or b in RANKLIKE): return "T15"
    if "n_intpts_x" in (a, b) and (a in SHA or b in SHA): return "U3 (likely via rank and period: T15, I1)"
    if "n_intpts_x" in (a, b) and (a in TAM or b in TAM): return "U3 (likely via torsion: T13, T15)"
    if "n_intpts_x" in (a, b) and (a in (MODD | LOCC | {"is_cm", "c4_sign", "c6_sign"}) or b in (MODD | LOCC | {"is_cm", "c4_sign", "c6_sign"})): return "U3 (likely via model size: T15)"
    if (a in SHA and b in SIZE2) or (b in SHA and a in SIZE2): return "I1+T11 (large Sha needs a small period, hence large c4 and Delta)"
    if (a in SHA and b in (TAM | LOCC)) or (b in SHA and a in (TAM | LOCC)): return "C9/I1 (Sha and Tamagawa enter BSD as a product)"
    if (a in SHA and b in GALX) or (b in SHA and a in GALX): return "U4 (partly C4: Selmer heuristics with torsion; Fisher for isogenies)"
    if (a in MODDEG_V and b in TAM) or (b in MODDEG_V and a in TAM): return "U1 (check Agashe-Ribet-Stein, Emerton: component groups and the congruence number)"
    if (a == "v2_moddeg" and b in LOCC) or (b == "v2_moddeg" and a in LOCC): return "family P: Dummigan, Caro-Pasten bounds on v2(moddeg) from multiplicative primes"
    if (a in MODD and b in (SIZE2 | LOCC | {"disc_sign"})) or (b in MODD and a in (SIZE2 | LOCC | {"disc_sign"})): return "family M: valuations track the size of moddeg (T12)"
    if (a in TAM and b in SIZE2) or (b in TAM and a in SIZE2): return "T4+T11 (Tamagawa tracks v_p(Delta), hence the period)"
    if (a in LOCC and b in SIZE2) or (b in LOCC and a in SIZE2): return "T4+T11 (reduction types fix v_p(Delta), hence the period)"
    if (a in RANKLIKE and b in SIZE) or (b in RANKLIKE and a in SIZE): return "C5/C8 (rank and height)"
    if (a in RANKLIKE and b in {"torsion_structure", "v2_torsion", "odd_torsion", "n_torsion_gens", "two_adic_label", "two_adic_log_index", "two_adic_level", "has_2_isog", "has_3_isog"}) or (b in RANKLIKE and a in {"torsion_structure", "v2_torsion", "odd_torsion", "n_torsion_gens", "two_adic_label", "two_adic_log_index", "two_adic_level", "has_2_isog", "has_3_isog"}): return "C6 (rank and 2-, 3-torsion structure)"
    if (a in RANKLIKE and b in {"class_size", "max_isogeny_degree", "n_nonsurj_primes", "has_5_isog", "has_7_isog", "exceptional_image"}) or (b in RANKLIKE and a in {"class_size", "max_isogeny_degree", "n_nonsurj_primes", "has_5_isog", "has_7_isog", "exceptional_image"}): return "U5 (rank and isogeny structure beyond 2 and 3)"
    if (a in RANKLIKE and b in LOCC) or (b in RANKLIKE and a in LOCC): return "U7 (rank and reduction types; T16 for n_bad_primes)"
    if (a in RANKLIKE and b in TAM) or (b in RANKLIKE and a in TAM): return "C9 (Tamagawa and the probability of rank 0)"
    if (a == "petersson_resid" and b in GALX) or (b == "petersson_resid" and a in GALX): return "family E/T10 (Eisenstein congruences; CM symmetric square)"
    if (a in SIZE2 and b in GALX) or (b in SIZE2 and a in GALX): return "family S"
    return ""

# discretise
codes, levels = {}, {}
for col, (group, cl, (how, par)) in CAT.items():
    s = df[col]
    if how == "cap":
        if s.dtype == bool: v = s.astype(int).astype(str)
        elif not pd.api.types.is_numeric_dtype(s): v = s.astype(str)
        else:
            x = s.astype(float); xi = x.fillna(0).astype(int).astype(str)
            v = pd.Series(np.where(x < par, xi, f"{par}+"), index=s.index) if par else xi
        v = v.where(s.notna(), None)
    else:
        x = s.astype(float); q = np.unique(np.nanquantile(x, np.linspace(0, 1, par + 1)[1:-1]))
        v = pd.Series(np.searchsorted(q, x).astype(str), index=s.index).where(x.notna(), None); v = "b" + v
    vc = v.value_counts(); keep = vc.index[:40]
    v = v.where(v.isin(keep), None)
    c, lv = pd.factorize(v, use_na_sentinel=True)
    codes[col] = c; levels[col] = list(lv)
strata = np.floor(df["log_conductor"].to_numpy() / 0.5).astype(int); strata -= strata.min(); S = strata.max() + 1
rep = df["is_rep"].to_numpy()

rows_cells, rows_pairs = [], []
cols = list(CAT)
for a, b in itertools.combinations(cols, 2):
    both_class = CAT[a][1] and CAT[b][1]
    m = rep if both_class else np.ones(len(df), bool)
    ca, cb = codes[a], codes[b]; m = m & (ca >= 0) & (cb >= 0)
    ka, kb = len(levels[a]), len(levels[b])
    idx = (strata[m] * ka + ca[m]) * kb + cb[m]
    T = np.bincount(idx, minlength=S * ka * kb).reshape(S, ka, kb).astype(float)
    O = T.sum(0); ns = T.sum((1, 2)); ra = T.sum(2); cb_ = T.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        E = np.nansum(ra[:, :, None] * cb_[:, None, :] / ns[:, None, None], 0)
    flagged = []
    for i in range(ka):
        for j in range(kb):
            o, e = O[i, j], E[i, j]
            if e >= 30 and o <= 0.1 * e:
                kind = "forbidden" if o == 0 else "depleted"; lp = (-e + o * math.log(e) - gammaln(o + 1)) / math.log(10)
            elif e >= 5 and o >= 30 and o >= 10 * e:
                kind = "enriched"; lp = (-e + o * math.log(e) - gammaln(o + 1) + math.log(o / (o - e))) / math.log(10)
            else: continue
            flagged.append((a, levels[a][i], b, levels[b][j], int(o), round(e, 1), kind, round(lp, 1)))
    if flagged:
        kn = known(a, b)
        for f in flagged: rows_cells.append((*f, int(m.sum()), kn))
        rows_pairs.append((a, b, len(flagged), min(f[-1] for f in flagged), sum(f[6] == "forbidden" for f in flagged), int(m.sum()), kn))
os.makedirs("results", exist_ok=True)
cells = pd.DataFrame(rows_cells, columns=["A", "a", "B", "b", "observed", "expected", "kind", "log10_p", "n_rows", "known"]).sort_values("log10_p")
pairs = pd.DataFrame(rows_pairs, columns=["A", "B", "n_flagged", "min_log10_p", "n_forbidden", "n_rows", "known"]).sort_values("min_log10_p")
cells.to_csv("results/step5_cells.csv", index=False); pairs.to_csv("results/step5_cells_pairs.csv", index=False)
print(f"pairs tested {len(list(itertools.combinations(cols, 2)))}, pairs with flags {len(pairs)}, cells flagged {len(cells)}, {time.time()-t0:.0f}s")
print("\nunknown-pair flags (not on the exclusion list, not same-group, not a known family):")
u = pairs[pairs.known == ""]
print(u.to_string())
print("\ncells of the unknown pairs:")
print(cells[cells.known == ""].to_string())
