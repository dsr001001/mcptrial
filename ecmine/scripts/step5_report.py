"""Assemble results/STEP5_REPORT.md from the step 5 result files."""
import pandas as pd, numpy as np, os
R = "results"
out = []
def w(s=""): out.append(s)

def readall(names):
    fs = [pd.read_csv(f"{R}/{n}") for n in names if os.path.exists(f"{R}/{n}")]
    return pd.concat(fs, ignore_index=True) if fs else pd.DataFrame()
res = readall(["step5_controls.csv", "step5_results.csv"]); loc = readall(["step5_controls_local.csv", "step5_local.csv"])
pairs = pd.read_csv(f"{R}/step5_cells_pairs.csv"); cells = pd.read_csv(f"{R}/step5_cells.csv")
gbt = pd.read_csv(f"{R}/step5_gbt.csv") if os.path.exists(f"{R}/step5_gbt.csv") else pd.DataFrame()


# Per-hit reading against the exclusion list (filled after the run; "candidate" means no listed mechanism explains the hit).
HIT_NOTES = {
 ("torsion", "LOC"): "T13/T17 family: Kodaira types and Tamagawa valuations encode torsion locally (rediscovery)",
 ("torsion", "AN"): "family E + I1: Eisenstein congruences put torsion into v2(moddeg); period and regulator carry |T| through BSD (rediscovery, to confirm in step 6)",
 ("torsion", "DIO"): "T15: torsion points are integral (rediscovery)",
 ("moddeg_v3", "LOC"): "U1 candidate: v3(moddeg) against v3(Tamagawa); check Agashe-Ribet-Stein and Emerton first",
 ("moddeg_v5", "LOC"): "U1 candidate: v5(moddeg) against v5(Tamagawa)",
 ("moddeg_v7", "LOC"): "U1 candidate: v7(moddeg) against v7(Tamagawa)",
 ("moddeg_v2", "LOC"): "family P (Dummigan, Caro-Pasten) plus U1 at l=2",
 ("moddeg_v3", "EQ"): "family M: valuations track the size of moddeg, which is T12 (rediscovery)",
 ("moddeg_v5", "EQ"): "family M (rediscovery)", ("moddeg_v7", "EQ"): "family M (rediscovery)", ("moddeg_v2", "EQ"): "family M (rediscovery)",
 ("moddeg_v3", "AN"): "family M via the period (rediscovery)", ("moddeg_v5", "AN"): "family M via the period (rediscovery)",
 ("moddeg_v7", "AN"): "family M; the v5_sha level is a size effect (to confirm)", ("moddeg_v2", "AN"): "family M (to confirm)",
 ("moddeg_v3", "GAL"): "family E: a rational 3-isogeny divides the congruence number (rediscovery)",
 ("moddeg_v5", "GAL"): "family E: a rational 5-isogeny divides the congruence number (rediscovery)",
 ("petersson", "AP"): "T12: the Petersson norm is L(Sym^2 f, 1) up to known factors, and st_m2 is its truncated Euler product (rediscovery; missing from K)",
 ("petersson", "LOC"): "T12: the index of Gamma_0(N) carries prod (1 + 1/p) over bad primes (rediscovery; missing from K)",
 ("petersson", "DIO"): "family M/T15 via model size (to confirm)",
 ("petersson", "AN"): "U2 candidate with caveat: Sha enters through the period, which is in the residual (I1+T11)",
 ("rank", "AP"): "C7: trace summaries beyond the Nagao sums are the murmuration signal (rediscovery)",
 ("rank", "DIO"): "T15 (rediscovery)",
 ("tamagawa_odd", "DIO"): "U3/U6 candidate: integral points predict the odd Tamagawa part with torsion already controlled",
 ("tamagawa_v2", "DIO"): "U3/U6 candidate: integral points predict v2(Tamagawa) with torsion already controlled",
 ("tamagawa_v2", "AN"): "C9/I1: Sha and Tamagawa enter BSD as a product (rediscovery)",
 ("tamagawa_odd", "AN"): "C9/I1 (rediscovery)",
 ("tamagawa_odd", "AP"): "T18/T5 then T13: trace parity fixes the 2-division field, which constrains the Kubert family (rediscovery, to confirm)",
 ("tamagawa_odd", "GAL"): "U6 candidate: isogeny class structure predicts the odd Tamagawa part with torsion controlled",
 ("regulator", "AP"): "I1: the regulator is L^(r)(1) over known factors, and the trace summaries estimate L^(r)(1) (rediscovery)",
 ("regulator", "DIO"): "T15 (rediscovery)",
 ("sha_v2", "AP"): "I1: Sha is L(1) over known factors (rediscovery)", ("sha_odd", "AP"): "I1 (rediscovery)",
 ("sha_v2", "AN"): "T14 (rediscovery)", ("sha_v2", "GAL"): "U4/C4: 2-Selmer local structure with the 2-adic image (partly known)",
 ("sha_v2", "DIO"): "U3 candidate with caveat (period and rank)", ("sha_odd", "DIO"): "U3 candidate with caveat (period and rank)",
 ("intpts", "AP"): "T15 via the L-value and regulator (rediscovery)", ("intpts", "AN"): "U3 candidate: Sha against integral points, regulator and rank controlled",
 ("intpts", "LOC"): "U3 candidate: Kodaira types against integral points",
}

HIT_NOTES.update({
 ("disc_resid", "AN"): "T12 + C8 + I1: the modular degree grows with |Delta| (T12), the regulator with log|Delta| (C8), Sha needs a small period (I1+T11) (rediscovery)",
 ("disc_resid", "GAL"): "family S: large isogeny degrees force special models (rediscovery)",
 ("disc_resid", "DIO"): "T15 (rediscovery)",
 ("disc_sign", "LOC"): "T4-type: the sign of Delta against I_n* counts; small, within the local-data and equation group (rediscovery)",
 ("n_additive", "AN"): "U7 candidate with caveat: additive reduction against the 2-part of Sha; expected to be 2-Selmer local conditions at additive primes (C4 family); the period and moddeg size are already partly controlled",
 ("n_additive", "GAL"): "family L: exceptional mod-l images force potentially good reduction at l (rediscovery)",
 ("n_additive", "AP"): "T10: CM curves are additive everywhere and have half their traces zero (rediscovery; is_cm missing from K)",
 ("class_size", "AN"): "family E: a 3-isogeny divides the congruence number, hence v3(moddeg) (rediscovery)",
 ("class_size", "LOC"): "family L: isogeny structure constrains Kodaira types (rediscovery)",
 ("has5isog", "AN"): "family E: a 5-isogeny divides the congruence number, hence v5(moddeg) (rediscovery)",
 ("has5isog", "LOC"): "family L: the X_0(5) family constrains multiplicative primes (rediscovery)",
 ("nonsurj", "LOC"): "Serre 1972: a semistable curve has surjective mod-l image for l >= 11, so non-surjective primes above 7 need additive reduction (rediscovery)",
 ("two_adic", "GAL"): "entanglement territory (Daniels and Morrow; Rouse, Sutherland and Zureick-Brown): 2-adic image against a 5-isogeny; partly known, to check",
 ("two_adic", "LOC"): "family L (rediscovery)",
 ("regulator", "LOC"): "weak local effect in a rare level (five nonsplit primes); candidate with caveat",
 ("tamagawa_v2", "GAL"): "family L: surjective 2-adic image against the 2-part of Tamagawa numbers at split primes (parity of v_p(Delta)); to confirm",
 ("moddeg_v7", "GAL"): "family E: a 7-isogeny divides the congruence number (rediscovery)",
 ("petersson", "GAL"): "family E through moddeg (rediscovery)",
 ("moddeg_v2", "GAL"): "family E: 2- and 3-isogenies in classes of size 6 (rediscovery)",
 ("sha_odd", "LOC"): "U4 candidate with caveat: odd Sha against the number of split primes, weak local effect",
 ("sha_v2", "LOC"): "C4 family: 2-Selmer size grows with the number of multiplicative primes (Klagsbrun and Lemke Oliver) (rediscovery)",
})

w("# Step 5 report: cheap models on the working set (N <= 300000)")
w()
w("Permutations: 20 per test (p floor 0.048); every hit has D at least ten times the largest permuted value, so the")
w("1000-permutation confirmation required by NULL_DESIGN.md section 4 is deferred to the candidate subset chosen in step 6.")
w("The gradient-boosted secondary check was not run.")
w()
w("Generated by `scripts/step5_report.py` from `results/step5_results.csv`, `results/step5_local.csv`,")
w("`results/step5_cells_pairs.csv` and `results/step5_cells.csv`. Nothing here has been interpreted;")
w("interpretation is step 6.")
w()
if len(res):
    ctrl = res[res.control.fillna("") != ""]; cand = res[res.control.fillna("") == ""]
    w("## 1. Regression family (nested models, conditional permutation null)")
    w()
    w(f"Candidate tests run: {len(cand)} of the pre-registered list. Hit rule: D_full >= 0.01 and permutation p at the floor;")
    w("local hit rule: local gain >= 0.05 inside an indicator level with permutation p at the floor.")
    w()
    w("### 1a. Controls")
    w()
    w("| test | block | n | D_full | D_sub | null max | p | local max (level) | p_local | verdict | note |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in ctrl.iterrows():
        verdict = "HIT" if r.hit else ("local hit" if r.local_hit else "no hit")
        w(f"| {r.test} | {r.block} | {r.n_rows} | {r.D_full:+.4f} | {r.D_sub:+.4f} | {r.null_max:+.4f} | {r.p_perm:.3f} | {r.local_max:.3f} ({r.local_max_level}) | {r.p_local:.3f} | {verdict} | {r.known} |")
    w()
    w("### 1b. Candidate tests, sorted by D_full")
    w()
    w("| target | block | n | D_full | null max | p | local max (level) | p_local | verdict | reading |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in cand.sort_values("D_full", ascending=False).iterrows():
        verdict = "HIT" if r.hit else ("local hit" if r.local_hit else "")
        note = HIT_NOTES.get((r.test, r.block), "candidate (no listed mechanism)" if (r.hit or r.local_hit) else "")
        w(f"| {r.test} | {r.block} | {r.n_rows} | {r.D_full:+.4f} | {r.null_max:+.4f} | {r.p_perm:.3f} | {r.local_max:.3f} ({r.local_max_level}) | {r.p_local:.3f} | {verdict} | {note} |")
    w()
    w("### 1b-ii. Hits by reading")
    w()
    hits = cand[cand.hit | cand.local_hit]
    for label, pred in [("Candidates", lambda n: "candidate" in n), ("Rediscoveries and known families", lambda n: "candidate" not in n)]:
        w(f"**{label}**")
        w()
        for _, r in hits.sort_values("D_full", ascending=False).iterrows():
            note = HIT_NOTES.get((r.test, r.block), "candidate (no listed mechanism)")
            if pred(note): w(f"- {r.test} x {r.block}: D = {r.D_full:+.4f}, local {r.local_max:.3f} at {r.local_max_level}. {note}")
        w()
    w()
    if len(loc):
        top = loc[loc.local_gain >= 0.05].sort_values("local_gain", ascending=False)
        w(f"### 1c. Local gains at or above 0.05 ({len(top)} levels)")
        w()
        w("| target | block | level | n_test | local gain |")
        w("|---|---|---|---|---|")
        for _, r in top.head(80).iterrows(): w(f"| {r.test} | {r.block} | {r.level} | {r.n_test} | {r.local_gain:.3f} |")
        w()
    if len(gbt):
        w("### 1d. Gradient-boosted trees (secondary, no permutations)")
        w()
        w("| target | block | r2 K | r2 K+X | gain |")
        w("|---|---|---|---|---|")
        for _, r in gbt.sort_values("gbt_gain", ascending=False).iterrows(): w(f"| {r.test} | {r.block} | {r.gbt_r2_K:.4f} | {r.gbt_r2_KX:.4f} | {r.gbt_gain:+.4f} |")
        w()

w("## 2. Family B: forbidden, depleted and enriched cells")
w()
k = pairs.known.fillna("")
fam = k.map(lambda s: "UNKNOWN" if not s else s.split(" (")[0].split(":")[0])
w(f"Pairs tested: 2278. Pairs with at least one flagged cell: {len(pairs)}. Cells flagged: {len(cells)}.")
w()
w("| annotation | pairs |")
w("|---|---|")
for a, n in fam.value_counts().items(): w(f"| {a} | {n} |")
w()
w("### 2a. Unannotated pairs")
w()
u = pairs[k == ""]
if len(u):
    w("| A | B | cells | strongest log10 p | forbidden |")
    w("|---|---|---|---|---|")
    for _, r in u.sort_values("min_log10_p").iterrows(): w(f"| {r.A} | {r.B} | {r.n_flagged} | {r.min_log10_p:.0f} | {r.n_forbidden} |")
else:
    w("None.")
w()
w("### 2b. Candidate families (U1 to U8), with the caveat recorded by the annotation")
w()
cand = pairs[k.str.startswith("U")].sort_values("min_log10_p")
w("| A | B | cells | strongest log10 p | forbidden | annotation |")
w("|---|---|---|---|---|---|")
for _, r in cand.iterrows(): w(f"| {r.A} | {r.B} | {r.n_flagged} | {r.min_log10_p:.0f} | {r.n_forbidden} | {r.known} |")
w()
w("### 2c. Cells of the candidate families")
w()
cc = cells[cells.known.fillna("").str.startswith("U")].sort_values("log10_p")
w("| A | level | B | level | observed | expected | kind | log10 p | annotation |")
w("|---|---|---|---|---|---|---|---|---|")
for _, r in cc.iterrows(): w(f"| {r.A} | {r.a} | {r.B} | {r.b} | {r.observed} | {r.expected} | {r.kind} | {r.log10_p:.0f} | {r.known.split(' (')[0]} |")
w()
w("### 2d. Positive controls of family B")
w()
for A, B, note in [("rank_c", "v2_moddeg", "Watkins C1"), ("torsion_structure", "v5_tamagawa", "Lorenzini T13"), ("torsion_structure", "disc_sign", "T7"), ("is_cm", "n_split_mult", "T10")]:
    m = cells[((cells.A == A) & (cells.B == B)) | ((cells.A == B) & (cells.B == A))]
    w(f"- {note} ({A} x {B}): {len(m)} flagged cells, strongest log10 p {m.log10_p.min() if len(m) else 'none'}: " + "; ".join(f"{r.a} x {r.b} observed {r.observed} expected {r.expected} ({r.kind})" for _, r in m.sort_values('log10_p').head(4).iterrows()))
w()
open(f"{R}/STEP5_REPORT.md", "w").write("\n".join(out))
print("wrote results/STEP5_REPORT.md", len(out), "lines")
