"""Step 5: pre-registered nested-model tests with a conditional permutation null.

For each (target Y, candidate block X): fit Y ~ K(Y) and Y ~ K(Y) + X by ridge regression on
the working set (N <= 300000, folds by isogeny class), report the out-of-sample R^2 gain D
(for a categorical Y: per class indicator, max reported), local gains inside indicator levels,
and a permutation null in which X is permuted within strata of K(Y) on a fixed subsample.

Usage: python3 scripts/step5.py [--perms 50] [--sub 500000] [--only NAME] [--controls-only] [--gbt]
Writes results/step5_results.csv, results/step5_local.csv, results/step5_log.txt.
"""
import argparse, json, os, sys, time, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from load import load_curves, load_classes, fold, HOLDOUT_FROM

ap_ = argparse.ArgumentParser()
ap_.add_argument("--perms", type=int, default=50)
ap_.add_argument("--sub", type=int, default=500000)
ap_.add_argument("--only", type=str, default=None)
ap_.add_argument("--controls-only", action="store_true")
ap_.add_argument("--gbt", action="store_true")
ap_.add_argument("--seed", type=int, default=20261004)
args = ap_.parse_args()
os.makedirs("results", exist_ok=True)
LOG = open("results/step5_log.txt", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# ----------------------------------------------------------------------------- data
t0 = time.time()
df = load_curves()                       # working set, N <= 300000
df["log_omega"] = np.log(df["omega"]); df["log_tamagawa"] = np.log(df["tamagawa"].astype(float))
df["log_regulator"] = np.log(df["regulator"]); df["two_adic_log_index"] = np.log2(df["two_adic_index"].astype(float))
df["bad_at_2"] = (df["N"] % 2 == 0); df["bad_at_3"] = (df["N"] % 3 == 0)
df["has_2_torsion"] = (df["torsion"] % 2 == 0)
df["rank_c"] = df["rank"].clip(upper=3).astype(int)
df["fold"] = fold(df["class_label"].to_numpy())
df["is_opt"] = (df["optimal"] == 1).fillna(False).to_numpy()
df["is_rep"] = (df["num"] == 1).to_numpy()
# the Petersson residual is a class-level quantity (defined through the optimal curve): give it to every curve of the class
df["petersson_resid"] = df["class_label"].map(df.loc[df["is_opt"]].set_index("class_label")["petersson_resid"])
rng0 = np.random.default_rng(args.seed)
df["rank_shuffled"] = df["rank_c"].to_numpy()[rng0.permutation(len(df))]
KOD = ["n_kod_In", "n_kod_II", "n_kod_III", "n_kod_IV", "n_kod_I0s", "n_kod_Ins", "n_kod_IIs", "n_kod_IIIs", "n_kod_IVs"]
log(f"loaded working set: {len(df)} curves, {df.class_label.nunique()} classes, {time.time()-t0:.0f}s")

# ----------------------------------------------------------------------------- feature expansion
def kind_of(col):
    s = df[col]
    if s.dtype == bool: return "b"
    if not pd.api.types.is_numeric_dtype(s): return "c"
    if s.nunique(dropna=True) <= 12: return "c"
    return "n"

QBINS = {}
def expand(col, rows):
    """Return (matrix float64 (n,k), names, indicator-level names) for column col on the given row mask."""
    s = df.loc[rows, col]; k = kind_of(col)
    if k == "b":
        m = s.to_numpy(dtype=float)[:, None]; return m, [col], [col]
    if k == "c":
        vc = s.astype(str).value_counts()
        levels = [l for l in vc.index if vc[l] >= 200][:40]
        if len(levels) == vc.shape[0]: levels = levels[:-1]      # drop one level (intercept absorbs it)
        m = np.stack([(s.astype(str) == l).to_numpy(dtype=float) for l in levels], 1) if levels else np.zeros((len(s), 0))
        names = [f"{col}={l}" for l in levels]; return m, names, names
    x = s.to_numpy(dtype=float); x = np.where(np.isfinite(x), x, np.nanmedian(x))
    if col not in QBINS: QBINS[col] = np.unique(np.quantile(df[col].dropna().to_numpy(dtype=float), np.linspace(0, 1, 9)[1:-1]))
    q = QBINS[col]; b = np.searchsorted(q, x)
    cols = [(x - x.mean()) / (x.std() + 1e-12)]; names = [col]; inds = []
    for j in range(1, len(q) + 1):
        cols.append((b == j).astype(float)); names.append(f"{col}@bin{j}"); inds.append(f"{col}@bin{j}")
    return np.stack(cols, 1), names, inds

def build(cols, rows):
    mats, names, inds = [], [], []
    for c in cols:
        m, n, i = expand(c, rows); mats.append(m); names += n; inds += i
    return (np.concatenate(mats, 1) if mats else np.zeros((rows.sum(), 0))), names, inds

# ----------------------------------------------------------------------------- ridge
LAM = 1e-3
def fit(Z, Y, lam=LAM):
    """Ridge with unpenalised intercept (column 0). Y may have several columns."""
    G = Z.T @ Z; d = np.full(len(G), lam * len(Z)); d[0] = 0.0; G[np.diag_indices_from(G)] += d
    return np.linalg.solve(G, Z.T @ Y)

def target_matrix(spec, rows):
    s = df.loc[rows, spec["target"]]
    if spec["ttype"] == "cat":
        vc = s.astype(str).value_counts(); levels = [l for l in vc.index if vc[l] >= 1000][:8]
        return np.stack([(s.astype(str) == l).to_numpy(dtype=float) for l in levels], 1), [f"{spec['target']}={l}" for l in levels]
    y = s.to_numpy(dtype=float); return y[:, None], [spec["target"]]

def r2_gain(K, X, Y, tr, te):
    """Out-of-sample R^2 of K and K+X on test rows; returns (r2A, r2B) per target column and residuals."""
    one = np.ones((len(Y), 1)); ZA = np.concatenate([one, K], 1); ZB = np.concatenate([ZA, X], 1)
    bA = fit(ZA[tr], Y[tr]); bB = fit(ZB[tr], Y[tr])
    rA = Y[te] - ZA[te] @ bA; rB = Y[te] - ZB[te] @ bB
    sst = ((Y[te] - Y[te].mean(0)) ** 2).sum(0) + 1e-12
    return 1 - (rA ** 2).sum(0) / sst, 1 - (rB ** 2).sum(0) / sst, rA, rB

def local_gains(X, xnames, xinds, rA, rB, te, min_n=500):
    """Within each indicator level of X (on test rows): fraction of model-A error removed by model B."""
    out = []
    for name in xinds:
        j = xnames.index(name); m = X[te][:, j] > 0.5
        if m.sum() < min_n: continue
        sa = (rA[m] ** 2).sum(0); sb = (rB[m] ** 2).sum(0)
        tot = ((rA ** 2).sum(0) / len(rA)) * m.sum()            # error the known model makes on an average level of this size
        g = np.where(sa >= 1e-3 * tot, 1 - sb / (sa + 1e-12), 0.0)  # ignore levels the known model already fits almost exactly
        out.append((name, int(m.sum()), float(np.max(g))))
    return out

def strata_ids(spec, rows):
    parts = [np.floor(df.loc[rows, "log_conductor"].to_numpy() / 0.25).astype(int).astype(str)]
    for c in spec["K"]:
        if c != "log_conductor" and kind_of(c) in ("c", "b"): parts.append(df.loc[rows, c].astype(str).to_numpy())
    key = parts[0]
    for p in parts[1:]: key = np.char.add(np.char.add(key, "|"), p)
    return pd.factorize(key)[0]

def permute_within(strata, rng):
    n = len(strata); r1 = rng.random(n); r2 = rng.random(n)
    o1 = np.lexsort((r1, strata)); o2 = np.lexsort((r2, strata))
    perm = np.empty(n, dtype=np.int64); perm[o1] = o2; return perm

# ----------------------------------------------------------------------------- one test
def run_test(spec, bname, bcols, known, nperm, sub):
    rows = spec["rows"](df).to_numpy() if callable(spec["rows"]) else spec["rows"]
    rows = rows & df[spec["target"]].notna().to_numpy()
    for c in spec["K"] + bcols: rows = rows & df[c].notna().to_numpy()
    n = int(rows.sum())
    Y, ynames = target_matrix(spec, rows)
    if spec.get("ap_block"):   # raw a_p block for the murmuration control
        cl = df.loc[rows, "class_label"]; apm = AP_RAW.loc[cl].to_numpy(); X = apm; xnames = list(AP_RAW.columns); xinds = []
    else:
        X, xnames, xinds = build(bcols, rows)
    K, knames, _ = build(spec["K"], rows)
    f = df.loc[rows, "fold"].to_numpy()
    res = {"test": spec["name"], "target": spec["target"], "block": bname, "n_rows": n, "k_K": K.shape[1], "k_X": X.shape[1],
           "known": known, "control": spec.get("control", "")}
    gains = []; locs = []
    for tf in (0, 1):
        te = f == tf; tr = ~te
        r2A, r2B, rA, rB = r2_gain(K, X, Y, tr, te)
        gains.append(r2B - r2A)
        if tf == 0:
            res["r2_K"] = float(r2A.max()); res["r2_KX"] = float(r2B.max())
            for name, cnt, g in local_gains(X, xnames, xinds, rA, rB, te): locs.append((spec["name"], bname, name, cnt, g))
    D = np.mean(gains, 0); res["D_full"] = float(D.max()); res["D_full_by_class"] = json.dumps(dict(zip(ynames, np.round(D, 5).tolist())))
    res["local_max"] = max([g for *_, g in locs], default=0.0); res["local_max_level"] = max(locs, key=lambda t: t[-1])[2] if locs else ""
    # permutation null on a subsample (fold 0 as test)
    rng = np.random.default_rng(int(hashlib.md5(f"{spec['name']}|{bname}|{args.seed}".encode()).hexdigest()[:8], 16))
    idx = np.arange(n)
    if n > sub: idx = np.sort(rng.choice(n, sub, replace=False))
    Ks, Xs, Ys, fs = K[idx], X[idx], Y[idx], f[idx]; te = fs == 0; tr = ~te
    strata = strata_ids(spec, rows)[idx]
    r2A, r2B, rA, rB = r2_gain(Ks, Xs, Ys, tr, te); Dobs = float((r2B - r2A).max())
    lobs = max([g for *_, g in local_gains(Xs, xnames, xinds, rA, rB, te)], default=0.0)
    null, lnull = [], []
    for _ in range(nperm):
        p = permute_within(strata, rng); Xp = Xs[p]
        r2A, r2B, rA, rB = r2_gain(Ks, Xp, Ys, tr, te); null.append(float((r2B - r2A).max()))
        lnull.append(max([g for *_, g in local_gains(Xp, xnames, xinds, rA, rB, te)], default=0.0))
    null = np.array(null); lnull = np.array(lnull)
    res.update({"n_sub": len(idx), "D_sub": Dobs, "null_mean": float(null.mean()), "null_max": float(null.max()),
                "p_perm": float((1 + (null >= Dobs).sum()) / (1 + nperm)), "local_sub": lobs, "local_null_max": float(lnull.max()),
                "p_local": float((1 + (lnull >= lobs).sum()) / (1 + nperm)), "nperm": nperm, "n_strata": int(strata.max() + 1)})
    res["hit"] = bool(res["D_full"] >= 0.01 and res["p_perm"] <= 1 / (1 + nperm))
    res["local_hit"] = bool(res["local_max"] >= 0.05 and res["p_local"] <= 1 / (1 + nperm))
    return res, locs

# ----------------------------------------------------------------------------- GBT secondary
def run_gbt(spec, bname, bcols, sub):
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = spec["rows"](df).to_numpy() if callable(spec["rows"]) else spec["rows"]
    rows = rows & df[spec["target"]].notna().to_numpy()
    for c in spec["K"] + bcols: rows = rows & df[c].notna().to_numpy()
    Y, ynames = target_matrix(spec, rows); y = Y[:, 0]
    def raw(cols):
        out = []
        for c in cols:
            s = df.loc[rows, c]
            out.append(pd.factorize(s.astype(str))[0].astype(float) if kind_of(c) in ("c", "b") else s.to_numpy(dtype=float))
        return np.stack(out, 1)
    K = raw(spec["K"]); X = raw(bcols); f = df.loc[rows, "fold"].to_numpy()
    rng = np.random.default_rng(1); n = len(y); idx = np.sort(rng.choice(n, min(sub, n), replace=False))
    te = f[idx] == 0; tr = ~te
    def score(Z):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31).fit(Z[idx][tr], y[idx][tr])
        p = m.predict(Z[idx][te]); return 1 - ((y[idx][te] - p) ** 2).sum() / ((y[idx][te] - y[idx][te].mean()) ** 2).sum()
    a = score(K); b = score(np.concatenate([K, X], 1))
    return {"test": spec["name"], "block": bname, "gbt_r2_K": a, "gbt_r2_KX": b, "gbt_gain": b - a}

# ----------------------------------------------------------------------------- configuration
EQ_ALL = ["log_abs_disc", "log_j_height", "disc_sign", "log_abs_c4", "c4_sign", "c6_sign"]
LOC_ALL = ["n_bad_primes", "n_split_mult", "n_nonsplit_mult", "n_additive"] + KOD + ["v2_tamagawa", "v3_tamagawa", "v5_tamagawa", "v7_tamagawa", "log_odd_tamagawa", "log_tamagawa"]
LOC_TYPES = ["n_split_mult", "n_nonsplit_mult", "n_additive"] + KOD
GAL_REST = ["class_size", "max_isogeny_degree", "has_5_isog", "has_7_isog", "two_adic_log_index", "two_adic_level", "n_nonsurj_primes", "exceptional_image", "v2_torsion", "odd_torsion"]
GAL_ALL = ["torsion_structure", "has_2_isog", "has_3_isog", "is_cm"] + GAL_REST
AN_ALL = ["rank_c", "root_number", "v2_sha", "v3_sha", "v5_sha", "v7_sha", "log_odd_sha", "v2_moddeg", "v3_moddeg", "v5_moddeg", "v7_moddeg", "log_odd_moddeg", "petersson_resid", "log_regulator", "log_omega"]
MODDEG = ["v2_moddeg", "v3_moddeg", "v5_moddeg", "v7_moddeg", "log_odd_moddeg", "petersson_resid"]
AP_ALL = ["nagao_100", "nagao_300", "nagao_1000", "frac_even", "frac_zero", "st_mean", "st_m2", "mur_lo", "mur_mid", "mur_hi"]
AP_NOIMG = ["nagao_100", "nagao_300", "nagao_1000", "st_mean", "st_m2", "mur_lo", "mur_mid", "mur_hi"]
DIO = ["n_intpts_x"]
ALL = lambda d: pd.Series(True, index=d.index)
REP = lambda d: d["is_rep"]
OPT = lambda d: d["is_opt"]
NONCM = lambda d: ~d["is_cm"].astype(bool)
POS = lambda d: d["rank"] >= 1

def T(name, target, ttype, rows, K, blocks, control=""):
    return {"name": name, "target": target, "ttype": ttype, "rows": rows, "K": K, "blocks": blocks, "control": control}

SPECS = [
 T("rank", "rank_c", "cat", REP,
   ["log_conductor", "n_bad_primes", "root_number", "torsion_structure", "is_cm", "nagao_100", "nagao_1000", "has_2_isog", "has_3_isog", "log_tamagawa", "log_omega", "v2_moddeg", "v2_sha", "log_odd_sha", "n_split_mult"],
   {"EQ": (EQ_ALL, ""), "LOC": (["n_nonsplit_mult", "n_additive"] + KOD + ["v3_tamagawa", "v5_tamagawa", "v7_tamagawa", "log_odd_tamagawa"], ""),
    "GAL": (GAL_REST, "U5"), "AN": (["v3_moddeg", "v5_moddeg", "v7_moddeg", "log_odd_moddeg", "petersson_resid"], "U2"),
    "AP": (["nagao_300", "frac_even", "frac_zero", "st_mean", "st_m2", "mur_lo", "mur_mid", "mur_hi"], "known C7"), "DIO": (DIO, "known T15")}),
 T("torsion", "torsion_structure", "cat", ALL,
   ["log_conductor", "n_bad_primes", "disc_sign", "log_abs_disc", "log_j_height", "log_tamagawa", "v2_tamagawa", "bad_at_2", "bad_at_3"],
   {"AN": (["rank_c", "root_number", "v2_sha", "log_odd_sha", "v2_moddeg", "v3_moddeg", "log_odd_moddeg", "petersson_resid", "log_regulator", "log_omega"], ""),
    "LOC": (LOC_TYPES + ["v3_tamagawa", "v5_tamagawa", "v7_tamagawa", "log_odd_tamagawa"], "known T13 for orders 5,7,9"),
    "EQ": (["log_abs_c4", "c4_sign", "c6_sign"], ""), "DIO": (DIO, "known T15")}),
 T("tamagawa_v2", "v2_tamagawa", "num", ALL,
   ["log_conductor"] + LOC_TYPES + ["log_abs_disc", "torsion_structure", "v2_torsion", "odd_torsion", "n_bad_primes"],
   {"GAL": (["class_size", "max_isogeny_degree", "has_2_isog", "has_3_isog", "has_5_isog", "has_7_isog", "two_adic_log_index", "two_adic_level", "n_nonsurj_primes", "exceptional_image", "is_cm"], "U6"),
    "AN": (["rank_c", "root_number", "v2_sha", "log_odd_sha"] + MODDEG + ["log_omega"], "partly I1"),
    "EQ": (["disc_sign", "log_j_height", "log_abs_c4", "c4_sign", "c6_sign"], ""), "AP": (AP_ALL, ""), "DIO": (DIO, "")}),
 T("tamagawa_odd", "log_odd_tamagawa", "num", ALL,
   ["log_conductor"] + LOC_TYPES + ["log_abs_disc", "torsion_structure", "v2_torsion", "odd_torsion", "n_bad_primes"],
   {"GAL": (["class_size", "max_isogeny_degree", "has_2_isog", "has_3_isog", "has_5_isog", "has_7_isog", "two_adic_log_index", "two_adic_level", "n_nonsurj_primes", "exceptional_image", "is_cm"], "U6"),
    "AN": (["rank_c", "root_number", "v2_sha", "log_odd_sha"] + MODDEG + ["log_omega"], "partly I1"),
    "EQ": (["disc_sign", "log_j_height", "log_abs_c4", "c4_sign", "c6_sign"], ""), "AP": (AP_ALL, ""), "DIO": (DIO, "")}),
 T("sha_v2", "v2_sha", "num", ALL,
   ["log_conductor", "rank_c", "root_number", "torsion_structure", "log_tamagawa", "v2_tamagawa", "log_omega", "disc_sign", "n_bad_primes", "is_cm"],
   {"EQ": (["log_abs_disc", "log_j_height", "log_abs_c4", "c4_sign", "c6_sign"], ""), "LOC": (LOC_TYPES + ["v3_tamagawa", "v5_tamagawa", "v7_tamagawa", "log_odd_tamagawa"], ""),
    "GAL": (GAL_REST + ["has_2_isog", "has_3_isog"], "U4"), "AN": (MODDEG, "known T14 for 2-part"), "AP": (AP_ALL, ""), "DIO": (DIO, "")}),
 T("sha_odd", "log_odd_sha", "num", ALL,
   ["log_conductor", "rank_c", "root_number", "torsion_structure", "log_tamagawa", "v2_tamagawa", "log_omega", "disc_sign", "n_bad_primes", "is_cm"],
   {"EQ": (["log_abs_disc", "log_j_height", "log_abs_c4", "c4_sign", "c6_sign"], "U4"), "LOC": (LOC_TYPES + ["v3_tamagawa", "v5_tamagawa", "v7_tamagawa", "log_odd_tamagawa"], "U4"),
    "GAL": (GAL_REST + ["has_2_isog", "has_3_isog"], "U4"), "AN": (MODDEG, "known T14 (visibility)"), "AP": (AP_ALL, ""), "DIO": (DIO, "")}),
 T("petersson", "petersson_resid", "num", OPT,
   ["log_conductor", "log_abs_disc", "log_j_height", "torsion_structure", "n_bad_primes", "is_cm"],
   {"AN": (["rank_c", "root_number", "v2_sha", "v3_sha", "log_odd_sha", "log_regulator"], "U2"), "LOC": (LOC_ALL, ""),
    "GAL": (GAL_REST + ["has_2_isog", "has_3_isog"], ""), "EQ": (["disc_sign", "log_abs_c4", "c4_sign", "c6_sign"], ""), "AP": (AP_ALL, ""), "DIO": (DIO, "")}),
 T("moddeg_v2", "v2_moddeg", "num", OPT,
   ["log_conductor", "rank_c", "torsion_structure", "n_bad_primes", "v2_tamagawa", "v2_sha", "has_2_isog", "two_adic_log_index"],
   {"GAL": (["class_size", "max_isogeny_degree", "has_3_isog", "has_5_isog", "has_7_isog", "two_adic_level", "n_nonsurj_primes", "exceptional_image", "is_cm"], ""),
    "LOC": (LOC_TYPES + ["v3_tamagawa", "v5_tamagawa", "v7_tamagawa", "log_odd_tamagawa"], ""),
    "AN": (["root_number", "v3_sha", "v5_sha", "v7_sha", "log_odd_sha", "petersson_resid", "log_regulator", "log_omega"], ""), "EQ": (EQ_ALL, ""), "AP": (AP_ALL, ""), "DIO": (DIO, "")}),
] + [
 T(f"moddeg_v{l}", f"v{l}_moddeg", "num", OPT,
   ["log_conductor", "rank_c", "torsion_structure", "n_bad_primes", f"v{l}_sha"],
   {"GAL": (["class_size", "max_isogeny_degree", "has_2_isog", "has_3_isog", "has_5_isog", "has_7_isog", "two_adic_log_index", "two_adic_level", "n_nonsurj_primes", "exceptional_image", "is_cm", "v2_torsion", "odd_torsion"], "U1"),
    "LOC": (LOC_TYPES + ["v2_tamagawa", "v3_tamagawa", "v5_tamagawa", "v7_tamagawa", "log_odd_tamagawa"], "U1"),
    "AN": (["root_number"] + [f"v{m}_sha" for m in (2, 3, 5, 7) if m != l] + ["log_odd_sha", "petersson_resid", "log_regulator", "log_omega"], ""),
    "EQ": (EQ_ALL, ""), "AP": (AP_ALL, ""), "DIO": (DIO, "")}) for l in (3, 5, 7)
] + [
 T("regulator", "log_regulator", "num", POS,
   ["log_conductor", "rank_c", "log_abs_disc", "log_j_height", "log_omega", "log_tamagawa", "torsion_structure", "n_bad_primes", "is_cm"],
   {"EQ": (["disc_sign", "log_abs_c4", "c4_sign", "c6_sign"], ""), "LOC": (LOC_TYPES + ["v2_tamagawa", "v3_tamagawa", "v5_tamagawa", "v7_tamagawa", "log_odd_tamagawa"], ""),
    "GAL": (GAL_REST + ["has_2_isog", "has_3_isog"], ""), "AN": (MODDEG, ""), "AP": (AP_ALL, ""), "DIO": (DIO, "known T15")}),
 T("intpts", "n_intpts_x", "num", ALL,
   ["log_conductor", "rank_c", "torsion_structure", "log_abs_disc", "log_j_height", "log_regulator", "log_abs_c4", "n_bad_primes"],
   {"LOC": (LOC_TYPES + ["v2_tamagawa", "v3_tamagawa", "v5_tamagawa", "v7_tamagawa", "log_odd_tamagawa", "log_tamagawa"], "U3"),
    "GAL": (GAL_REST + ["has_2_isog", "has_3_isog", "is_cm"], "U3"), "AN": (["root_number", "v2_sha", "log_odd_sha"] + MODDEG + ["log_omega"], "U3"),
    "EQ": (["disc_sign", "c4_sign", "c6_sign"], "U3"), "AP": (AP_ALL, "")}),
 T("two_adic", "two_adic_log_index", "num", NONCM,
   ["torsion_structure", "class_size", "max_isogeny_degree", "has_2_isog", "disc_sign", "log_conductor", "v2_torsion"],
   {"AN": (["rank_c", "root_number", "v2_sha", "log_odd_sha", "v2_moddeg", "log_odd_moddeg", "petersson_resid", "log_regulator", "log_omega"], ""), "LOC": (LOC_ALL, ""),
    "EQ": (["log_abs_disc", "log_j_height", "log_abs_c4", "c4_sign", "c6_sign"], ""), "AP": (["nagao_100", "nagao_300", "nagao_1000", "frac_zero", "st_mean", "st_m2", "mur_lo", "mur_mid", "mur_hi"], ""),
    "DIO": (DIO, ""), "GAL": (["has_3_isog", "has_5_isog", "has_7_isog", "n_nonsurj_primes", "exceptional_image"], "")}),
 T("nonsurj", "n_nonsurj_primes", "cat", NONCM,
   ["torsion_structure", "class_size", "max_isogeny_degree", "has_2_isog", "has_3_isog", "has_5_isog", "has_7_isog", "log_conductor"],
   {"AN": (["rank_c", "root_number", "v2_sha", "log_odd_sha", "v2_moddeg", "log_odd_moddeg", "petersson_resid", "log_regulator", "log_omega"], ""), "LOC": (LOC_ALL, ""),
    "EQ": (["log_abs_disc", "log_j_height", "log_abs_c4", "c4_sign", "c6_sign"], ""), "AP": (AP_NOIMG, ""), "DIO": (DIO, "")}),
 T("class_size", "class_size", "cat", REP,
   ["torsion_structure", "log_conductor", "is_cm", "n_bad_primes", "log_j_height"],
   {"AN": (["rank_c", "root_number", "v2_sha", "log_odd_sha"] + MODDEG + ["log_regulator", "log_omega"], "U5"), "LOC": (LOC_ALL, ""),
    "EQ": (["log_abs_disc", "disc_sign", "log_abs_c4", "c4_sign", "c6_sign"], ""), "AP": (AP_NOIMG, ""), "DIO": (DIO, "")}),
 T("has5isog", "has_5_isog", "num", REP,
   ["torsion_structure", "log_conductor", "is_cm", "n_bad_primes", "log_j_height"],
   {"AN": (["rank_c", "root_number", "v2_sha", "log_odd_sha"] + MODDEG + ["log_regulator", "log_omega"], "U5"), "LOC": (LOC_ALL, ""),
    "EQ": (["log_abs_disc", "disc_sign", "log_abs_c4", "c4_sign", "c6_sign"], ""), "AP": (AP_NOIMG, ""), "DIO": (DIO, "")}),
 T("has7isog", "has_7_isog", "num", REP,
   ["torsion_structure", "log_conductor", "is_cm", "n_bad_primes", "log_j_height"],
   {"AN": (["rank_c", "root_number", "v2_sha", "log_odd_sha"] + MODDEG + ["log_regulator", "log_omega"], "U5"), "LOC": (LOC_ALL, ""),
    "EQ": (["log_abs_disc", "disc_sign", "log_abs_c4", "c4_sign", "c6_sign"], ""), "AP": (AP_NOIMG, ""), "DIO": (DIO, "")}),
 T("disc_sign", "disc_sign", "cat", ALL,
   ["torsion_structure", "log_conductor", "n_bad_primes", "v2_torsion"],
   {"AN": (["rank_c", "root_number", "v2_sha", "log_odd_sha"] + MODDEG + ["log_regulator"], "U8; H1 for v2_sha"), "LOC": (LOC_ALL, ""),
    "GAL": (["class_size", "max_isogeny_degree", "has_2_isog", "has_3_isog", "has_5_isog", "has_7_isog", "two_adic_log_index", "two_adic_level", "n_nonsurj_primes", "exceptional_image", "is_cm"], ""),
    "AP": (AP_ALL, ""), "DIO": (DIO, "U8")}),
 T("n_additive", "n_additive", "cat", ALL,
   ["log_conductor", "n_bad_primes", "log_abs_disc", "log_j_height"],
   {"AN": (AN_ALL, "U7"), "GAL": (GAL_ALL, ""), "DIO": (DIO, ""), "AP": (AP_ALL, "")}),
 T("disc_resid", "log_abs_disc", "num", ALL,
   ["log_conductor", "n_bad_primes"] + LOC_TYPES + ["torsion_structure"],
   {"GAL": (GAL_REST + ["has_2_isog", "has_3_isog", "is_cm"], ""), "AN": (["rank_c", "root_number", "v2_sha", "log_odd_sha"] + MODDEG + ["log_regulator"], "C8 for regulator"),
    "DIO": (DIO, "known T15"), "AP": (AP_ALL, "")}),
]

SEMISTABLE_REP = lambda d: d["is_rep"] & (d["n_additive"] == 0)
CONTROLS = [
 T("NEG_rootno", "root_number", "cat", SEMISTABLE_REP, ["n_split_mult", "log_conductor"],
   {"EQ": (EQ_ALL, "must not hit"), "GAL": (GAL_ALL, "must not hit"), "DIO": (DIO, "must not hit"),
    "AN": (["v2_sha", "log_odd_sha"] + MODDEG + ["log_omega"], "must not hit")}, control="negative"),
 T("NEG_a997", "a997", "num", lambda d: d["is_rep"] & ~d["is_cm"].astype(bool) & (d["N"] % 997 != 0), ["torsion_structure", "log_conductor"],
   {"EQ": (EQ_ALL, "must not hit"), "GAL": (GAL_REST + ["has_2_isog", "has_3_isog"], "must not hit"), "DIO": (DIO, "must not hit"),
    "LOC": (LOC_ALL, "must not hit"), "AN": (AN_ALL, "C7 at most small")}, control="negative"),
 T("NEG_shuffled_rank", "rank_shuffled", "cat", REP, ["log_conductor"],
   {"EQ": (EQ_ALL, "must not hit"), "LOC": (LOC_ALL, "must not hit"), "GAL": (GAL_ALL, "must not hit"), "AN": (AN_ALL, "must not hit"),
    "AP": (AP_ALL, "must not hit"), "DIO": (DIO, "must not hit")}, control="negative"),
 T("POS_watkins", "rank_c", "cat", OPT, ["log_conductor"], {"v2_moddeg": (["v2_moddeg"], "must hit (C1)")}, control="positive"),
 T("POS_lorenzini", "v5_tamagawa", "num", ALL, ["log_conductor"] + LOC_TYPES, {"torsion": (["torsion_structure"], "must local-hit at [5] (T13)")}, control="positive"),
 T("POS_torsion_cong", "has_2_torsion", "num", REP, ["log_conductor"], {"frac_even": (["frac_even"], "must hit (T5/T18)")}, control="positive"),
 T("POS_nagao", "rank_c", "cat", REP, ["log_conductor"], {"nagao": (["nagao_1000"], "must hit (C7)")}, control="positive"),
 T("EXP_H1", "v2_sha", "num", lambda d: d["rank"] == 0, ["log_conductor"], {"disc_sign": (["disc_sign"], "expected small (H1)")}, control="expected small"),
]
AP_RAW = None
for lo, hi in ((10000, 20000), (100000, 200000)):
    CONTROLS.append(T(f"POS_murmur_{lo}", "root_number", "cat", (lambda lo, hi: (lambda d: d["is_rep"] & (d["N"] >= lo) & (d["N"] < hi)))(lo, hi),
                      ["log_conductor"], {"ap_raw": ([], "must hit (C7 murmurations)")}, control="positive"))
    CONTROLS[-1]["ap_block"] = True

# ----------------------------------------------------------------------------- run
if any(s.get("ap_block") for s in CONTROLS):
    apdf = load_classes(); P = [int(c[1:]) for c in apdf.columns if c.startswith("a") and c[1:].isdigit()]
    AP_RAW = pd.DataFrame(apdf[[f"a{p}" for p in P]].to_numpy().astype(float) / np.sqrt(np.array(P))[None, :], columns=[f"a{p}" for p in P], index=apdf["class_label"])
    df["a997"] = df["class_label"].map(apdf.set_index("class_label")["a997"]).astype(float)

n_cand = sum(len(s["blocks"]) for s in SPECS)
log(f"pre-registered candidate tests: {n_cand} across {len(SPECS)} targets; controls: {sum(len(s['blocks']) for s in CONTROLS)}; Bonferroni alpha = {0.05 / n_cand:.2e}")
todo = CONTROLS + ([] if args.controls_only else SPECS)
if args.only: todo = [s for s in todo if s["name"] == args.only]
rows_out, locs_out, gbt_out = [], [], []
for spec in todo:
    for bname, (bcols, known) in spec["blocks"].items():
        t1 = time.time()
        try:
            res, locs = run_test(spec, bname, bcols, known, args.perms, args.sub)
        except Exception as e:
            log(f"ERROR {spec['name']}/{bname}: {e!r}"); continue
        rows_out.append(res); locs_out += locs
        log(f"{spec['name']:>18s} {bname:>8s} n={res['n_rows']:>8d} D_full={res['D_full']:+.4f} D_sub={res['D_sub']:+.4f} null_max={res['null_max']:+.4f} "
            f"p={res['p_perm']:.3f} local={res['local_max']:.3f}@{res['local_max_level']} p_loc={res['p_local']:.3f} "
            f"{'HIT' if res['hit'] else ''}{' LOCAL' if res['local_hit'] else ''} [{known}] {time.time()-t1:.0f}s")
        pd.DataFrame(rows_out).to_csv("results/step5_results.csv", index=False)
        pd.DataFrame(locs_out, columns=["test", "block", "level", "n_test", "local_gain"]).to_csv("results/step5_local.csv", index=False)
        if args.gbt and not spec.get("ap_block"):
            try:
                g = run_gbt(spec, bname, bcols, args.sub); gbt_out.append(g); log(f"      gbt: r2_K={g['gbt_r2_K']:.4f} r2_KX={g['gbt_r2_KX']:.4f} gain={g['gbt_gain']:+.4f}")
                pd.DataFrame(gbt_out).to_csv("results/step5_gbt.csv", index=False)
            except Exception as e:
                log(f"      gbt ERROR: {e!r}")
log(f"done in {time.time()-t0:.0f}s")
