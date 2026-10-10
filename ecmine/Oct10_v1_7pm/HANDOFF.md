# HANDOFF: state of the project on 2026-10-10 and how to resume

Read this file first in a new session, then the README sections from "Referee responses (2026-10-10)" onwards.
Everything below is in the directory `ecmine` of the repository `dsr001001/mcptrial`, branch `ccr-f42a94ab-alw7cu`.

## 1. What the project is

An AI-assisted search for new relations in Cremona's tables of elliptic curves over Q (ecdata, commit
`25cec5ecfec8b9f016eb1631ac633194c2bed39f`, every curve of conductor below 500000). Steps 1 to 5 (frozen snapshot,
feature table, exclusion list of known relations, pre-registered null design, cheap models) produced one candidate
worth pursuing, U1: the l-adic valuation of the modular degree against local invariants at the bad primes. Steps 6 to 11
turned U1 into a note with conjectures, theorems and verification on four ranges (working set N <= 300000, hold-out
(300000, 400000], extension (400000, 500000] with 359,009 optimal-known classes, and 893 out-of-table curves of
conductor up to 3,994,600 with degrees from `ellmoddegree`). Two referee reports (an AI referee and a research note)
were received on 2026-10-10; items 2 and 3 (proof gaps, editorial) were settled, and item 1 (what are the exact local
factors) led to the Euler-factor rule, which now organises everything.

## 2. The mathematics as it stands (`paper/note.tex`, numbering of the 2026-10-10 evening version)

Notation: E the X_0(N)-optimal curve, f its newform, l odd with E[l] irreducible, deg phi_E the modular degree,
r_E the congruence number (deg | r_E with equal l-adic valuation when l^2 does not divide N, ARS Theorem 2.2).
T = T_l E, ad^0 T the trace-zero adjoint lattice, A_f = ad^0 V. Naive adjoint L-function (DFG): trivial factor at additive primes.

- Definition 1.1. e_l(E,q) = v_l(omitted factor of L(A_f,s) at s = 1 at q) = v_l(P_q(q^-2) / (1 - a_q^2 q^-2)) with
  L_q(Sym^2 f, s) = P_q(q^-s)^-1 (PARI `lfunsympow`, `lfuneuler`); t_l(E,q) = v_l(-v_q(j)) if v_q(j) < 0, else 0.
  Tame q >= 5: I_n: 0; I_n*: v_l(q^2-1); II, II*, IV, IV*: v_l(q - chi_{-3}(q)); III, III*: v_l(q - chi_{-4}(q));
  I_0*: v_l(q-1) + v_l((q+1)^2 - a_q(E')^2), E' the good twist (Lemma 2.5).
- THEOREM 1.2 (main; proved in Section 4.3). l not dividing 2N, rho-bar absolutely irreducible on Q(sqrt(l*)) (= l not in
  DFG's S_f): v_l(deg) >= length H^1_f(Q, ad^0 T (x) Q_l/Z_l) + sum_{q|N} v_l(Tam_q(ad^0 T)) + sum_{q|N} e_l(E,q), with
  equality when H^1(X_0(N), Z_l)_m is free over T_m (multiplicity one, Remark 4.4); hence v_l(deg) >= sum e + sum t,
  all types. Ingredients: Lemma 4.3 (DDT 4.17 inequality applied to H^1(X_0(N),Z_l)_m with DFG's pairing delta-hat_!,
  which is cup product composed with w_N, perfect, alternating, Hecke self-adjoint; DFG 1.5.3, 1.6.2, 1.7.3), Theorem 4.1
  = DFG Prop 1.4(c), Theorem 4.5 = DFG Thm 3.7, Theorem 4.6 = DFG Thm 2.7, Lemma 4.7 = DFG Lemma 2.1, Lemma 4.8 = FPR
  I.4.2.2 as quoted in DFG's proof of Thm 2.15 (+ DFG (57)), Lemma 4.9 (H^0(Q, ad^0 T(1) (x) Q_l/Z_l) = 0 iff rho-bar not
  isomorphic to its twist by chi_l; for l = 3 iff TW), Lemma 2.7 (Tamagawa factor at a multiplicative prime, full
  computation with the q^{-1}-twisted Frobenius action on H^1(I_q, M) = M/(t-1)M; sanity check against the Tamagawa
  number of E itself).
- Conjecture 1.3 (beyond the hypotheses): v_l(deg) >= sum_{q != l} e_l + sum_q t_l + [l=3] w_3(E) - h_l(E), with
  h_l(E) = length H^0(Q, ad^0 T(1) (x) Q_l/Z_l) (1 exactly for 3Ns at l = 3, else 0 in the tables) and w_3 the empirical
  weight at q = l = 3 (1 for II, IV, III*; 2 for II*, IV*; 0 for III, I_0*). Zero violations on all four ranges at
  l = 3, 5, 7, 11, 13 (Table 1). Equality at l = 3: 57.5% working set, 54.6% hold-out, 54.0% extension, 52.5% out of table.
- Proposition 1.4: the component-group (1.4) and twist-symmetrised (1.5) inequalities follow (type IV/IV* at 2 needs
  e_3(E,2) >= 1, true for all 92,812 such working-set curves).
- Theorem 1.5 (proved, Section 3; unchanged): l | deg for IV/IV* (q >= 5, l = 3), II/II* (q >= 5, l = 3), I_n* with l | n
  (q odd); l not dividing N; for l >= 5 only irreducibility is needed.
- Theorem 1.6 (proved, Section 4.4): without TW, for l not dividing 2N and E[l] irreducible, v_l(deg) >= v_l(r_{E_S}) +
  sum_{q in S} e_l(E,q), S = primes >= 5 of type I_0* or I_n* (twist monotonicity Theorem 4.11 + DFG Prop 1.4(c) +
  Lemma 4.2 + multiplicity one at the lower level).
- Proposition 1.7 (proved, Section 5): for non-CM E with E[3] irreducible, TW at 3 fails iff image = N_s(F_3) (3Ns);
  then rho-bar = rho-bar (x) chi_{-3}, h_3 >= 1, and the 3-division polynomial factors as two quadratics (Zywina's
  classification of mod 3 images is cited for the list GL_2, 3Nn, 3Ns).
- Remark 5.1: the heuristic identity with the correction -h_l(E) from DFG's Lemma 2.1 when TW fails (conditional on
  H^1_f(Q, B_f) = 0 and on v(eta^Sigma) = length H^1_Sigma, which are not proved for 3Ns).
- Section 6.4: Tamagawa factor test (step 12). Section 6.7: the 3Ns boundary (step 11, Table 4).
- Conjecture 1.8 (Eisenstein deficit) unchanged.

Remaining risks, stated in the paper as hypotheses or citations rather than as items for an outside check:
1. DFG Theorem 3.7 is applied to f with Sigma = all bad primes ("minimally ramified outside Sigma" holds trivially);
   its proof ends with "the theorem holds for a twist of f, hence for f itself", so the statement is used as stated.
2. The identification of DFG's pairing delta-hat_! on M(N,1)_! with cup product composed with w_N (DFG (13), (14)) and
   of M_{f,lambda} with the kernel of ker(phi_f) on H^1(X_0(N), Z_l) (DFG 1.6.2) is the basis of Lemma 4.3.
3. Lemma 2.7 (Tamagawa factor) is our computation; the data test of Section 6.4 (zero violations, equality rate
   rising exactly where predicted) supports it.
4. Bibliography entries added from memory: Zagier85 (Canad. Math. Bull. 28 (1985) 372-384), FPR94 (PSPM 55.1, 599-706),
   Hida81, Flach92, Tilouine97, Zywina15 (arXiv:1508.07660). DFG04 and ARS checked against the sources today.
5. w_3 at q = l = 3 and everything with l | N are empirical only; Remark 5.1 is heuristic for 3Ns.

## 3. Documents

- `paper/note.tex` + `paper/refs.bib` + `paper/tables/*.tex` -> `paper/note.pdf` (24 pages, evening version of 2026-10-10). Build:
  `cd paper && pdflatex note && bibtex note && pdflatex note && pdflatex note`. Author (Dharmaj Soni), email and ORCID are
  filled in (amsart `\email`, `\urladdr`); the AI-use statement (an unnumbered section at the end of the note, a short version at the end of the memorandum, and a sentence in the outreach message) was written on 2026-10-10 and approved in substance by the user. Tables: `counts.tex`, `isolated3.tex`, `oot.tex`, `twist.tex`
  (no longer input) from `scripts/make_tables.py`; `euler_full.tex` (step 9, q >= 5 rule, no longer input by the note but
  used by nothing else; keep), `euler_uniform.tex` and `ns3.tex` from `scripts/step11_tw_boundary.py` (the older `scripts/step10_make_table.py` wrote the
  III-free version), `euler_iso3.tex` (hand-copied from `results/step9_euler_l3.txt`).
- `paper/memo_local_factors.tex` -> `.pdf` (10 pages; `pdflatex` twice, bibliography inline). The memorandum answering
  referee item 1, with a postscript (evening of 2026-10-10) recording the resolution; its Section 6 and the q = 1 mod 12
  description are superseded, as the postscript says.
- `paper/outreach.md`: optional notification to Diamond/Flach and a short announcement (evening version).
- `README.md`: chronological log of all steps with commands; `STEP6_U1_REPORT.md` (working record, superseded, has update
  headers); `KNOWN_RELATIONS.md`, `NULL_DESIGN.md`, `results/STEP5_REPORT.md` (pre-registration and step 5 records, historical).
- `Oct10/`: morning snapshot of 2026-10-10 (superseded). `Oct10_v1_7pm/`: evening snapshot, with pdf, tex and plain-text
  (`.txt`, from pdftotext) versions of the note and the memorandum, tables, outreach, README, this file, scripts of steps 10-12.
- The referee reports (`review1_L_adic.pdf`, `author_note_twist_symmetrised_bound.pdf/.tex`, `review2_L_adic.pdf`,
  `independent_review_DFG_N2_modular_degree.pdf/.tex`) were uploaded by the user in the session and are NOT in the repository; ask the user for them if needed. Their items: (1) identify the local
  factors in the adjoint / congruence-ideal theory (done: the memorandum and the restructured note); (2) complete the proof
  of the qualitative theorem (done: Lemmas 3.1, 3.2, optimal-level theorem as DDT Theorem 3.15, l not dividing N added);
  (3) editorial: ARS citation and 99A1, Kim-Ota journal, Pasten, CNS24, abc sentence, q = 3 convention, Lemma 2.4 n = 0,
  Conjecture 1.7 bracket, calibration rates, control rows (done). AI-use statement: done (see section 3). Not done: Zenodo archive.

## 4. Data and pipeline (from a clean clone)

Environment: PARI/GP 2.15.4 (`gp`), python3 with pandas 3, numpy, pyarrow, cypari2; pdflatex with amsart, booktabs,
hyperref; bibtex. No `elldata` package is installed, so `ellinit("121a1")` does not work: take a-invariants from
`data/curves.parquet` (column `ainvs`) or from the ecdata files.

1. `sh scripts/download_ecdata.sh data/raw` (pinned commit; `data/` is gitignored, about 2 GB).
2. PARI features per 10000-conductor block: see the loop in `scripts/repro_run.sh` (`scripts/pari_features.gp`, outputs
   `data/pari/curves.*.tsv`, `ap.*.tsv`).
3. `python3 scripts/build_features.py` -> `data/curves.parquet` (one row per curve, all columns); `scripts/add_columns.py`
   adds step-5 columns.
4. Optimal-curve tables: the python block in `scripts/repro_run.sh` writes `results/optimal_{work,holdout,ext}.parquet`.
   Out-of-table curves: `scripts/out_of_table.gp` (generator; needs `-D threadsizemax=...` at gp startup for
   `ellmoddegree`), `scripts/oot_convert.py` -> `results/oot_table.parquet`.
5. Checker for the corollary forms: `python3 scripts/conj_check.py results/optimal_work.parquet --out results/check_work`
   (summary columns `viol_conj1` = inequality (1.2), `viol_conj2` = inequality (1.3), `viol_conj3` = Conjecture 1.7);
   for oot add `--nmax 4000000`. Tables: `python3 scripts/make_tables.py`.
6. Euler factors: `python3 scripts/step10_sym2_euler.py <work|holdout|ext|oot> <shard> <nshards>` -> `results/sym2_<set>_<shard>.parquet`
   (work was run as 3 shards in parallel; each shard well under an hour; `pari.allocatemem(600_000_000)` inside).
7. The rule: `python3 scripts/step10_rule.py <set>` -> `results/step10_rule_<set>_l{3,5,7,11,13}.parquet` (per curve:
   e, tam, wl, eIII, eIIIall, V, and the variants full / noIII / noIIIall / noql) and `results/step10_rule_<set>_summary.csv`;
   the note's rule is the `noIIIall` variant. Log of the last run: `results/step10_rule.log`.
   Then `python3 scripts/step10_make_table.py` -> `paper/tables/euler_uniform.tex`.
7b. Step 11: `python3 scripts/step11_tw_boundary.py` -> `results/step11_summary.csv`, `results/step11_ns3.csv`,
    `paper/tables/euler_uniform.tex`, `paper/tables/ns3.tex`, log `results/step11.log` (needs the step10 parquet files and
    `data/curves.parquet` for the galrep labels and a-invariants; ~5 minutes). Step 12: `python3 scripts/step12_tamagawa.py work`
    -> `results/step12_work.csv`, log `results/step12_work.log` (Tate parameters via `ellinit(E, O(q^k))`, member `tate`;
    ~10 minutes for the working set at l = 3 and 5).
8. Step 9 (Kodaira-type formulas, q >= 5 only, kept for the isolation table and the twist-degree check):
   `scripts/step9_euler.py` (`results/step9_euler_l{3,5,7}.txt`), `scripts/step9_fullrule.py`, `scripts/step9_additivity.py`,
   `scripts/step9_twist_degree.py` (`results/step9_twist_degree_work.csv`: 157,938 I_n* pairs and 292,580 potentially good
   pairs, zero violations of twist monotonicity of the degree).
9. One curve: `scripts/euler_rule.gp` (function `rule(E, l)`), examples in the note's appendix.
10. `*.parquet` under `results/` and `data/` are gitignored; summaries, logs and text outputs are committed. To regenerate
    the parquet files run steps 1-7 (a few hours in total).

Gotchas met: PARI thread stack must be set at gp startup (`-D threadsizemax=`), not inside a script; feed gp scripts via
stdin with a final `quit;` and braces around multi-line blocks (a file argument hung at the REPL); `ellmoddegree` for
conductors above a few thousand needs `-D parisizemax=400000000`; pandas 3 `groupby.apply` excludes grouping columns,
`itertuples` renames columns with special characters (use `to_dict("records")`); OT1 has no ogonek (write "K." for
Cesnavicius); never put Unicode in the .tex files (pdflatex, no inputenc setup).

## 5. Conventions for the repository and the session

- Work only on branch `ccr-f42a94ab-alw7cu`; push with `git push -u origin ccr-f42a94ab-alw7cu`; never open a pull request
  unless asked. Commit as `dsr001001 <dsr001@outlook.in>` (use `-c user.name=... -c user.email=...`).
- Every commit message ends with the two trailer lines `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_011kYUdULqcd2A9SQLVwp2R2` (as instructed in the session); no model identifiers
  anywhere in repository content (papers, scripts, README).
- The user's strict preference: never use an em dash or long dash in anything written (use colon, semicolon or comma).
- Do not incorporate referee suggestions into the papers before discussing them with the user; assess first.
- Keep `results/*.parquet` and `data/` out of git; keep compiled PDFs in git (`paper/*.pdf`) so the user can read them.
- Dates in the documents and README are 2026-10-10 for everything in this phase.

## 6. Where things stand and what could come next

Done today (evening): two more reports assessed; the 3Ns claim verified; DFG's published text read and the rule proved
under TW as Theorem 1.2 (24-page note); Lemma 2.7 (Tamagawa factor) proved and tested on the data (step 12); the
correction term h_l identified and verified (step 11); memorandum postscript; outreach rewritten; `Oct10_v1_7pm/` snapshot.

Open, for the user to decide:
1. Whether to submit the note to a journal directly (the user's stated inclination) or first notify Diamond/Flach with
   the outreach text A; the note does not assume any outside review.
2. Zenodo archive of the result files (referee item, not done).
3. Possible further computations: (a) the type III anomaly at larger scale (configurations with q >= 5 only were
   analysed in `results/step9_fullrule_*`; the q = 2 cases have v_2(N) = 8 and factor 1 + 2X); (b) the weights at q = l = 3
   against the local condition at l in DFG; (c) a direct computation of the adjoint Selmer groups for a handful of
   curves, which would test Conjecture 5.1 and the dual correction (needs a tool; not attempted).
4. Possible further writing: a short version of the note for a journal, with Sections 3 and 6 compressed.

To resume: `git log --oneline | head`, read this file, `ls results/*.csv results/*.log`, and if the parquet files are
missing rebuild them (section 4). The note and the memorandum compile with the commands in section 3.
