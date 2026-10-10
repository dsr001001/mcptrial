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

## 2. The mathematics as it stands (all in `paper/note.tex`, numbering of the 2026-10-10 version)

Notation: E the X_0(N)-optimal curve, f its newform, l odd with E[l] irreducible, deg phi_E the modular degree,
r_E the congruence number (deg phi_E | r_E with equal l-adic valuation when l^2 does not divide N, Agashe-Ribet-Stein
Theorem 2.2). A_f = ad^0 V_l E. Naive adjoint L-function (Diamond-Flach-Guo): trivial Euler factor at additive primes.

- Definition 1.1. e_l(E,q) = v_l(omitted factor of L(A_f, s) at s = 1 at q) = v_l(P_q(q^-2) / (1 - a_q^2 q^-2)), with
  L_q(Sym^2 f, s) = P_q(q^-s)^-1 (PARI: `lfunsympow(E,2)`, `lfuneuler`); t_l(E,q) = v_l(-v_q(j)) if v_q(j) < 0, else 0.
  Tame q >= 5: I_n: 0; I_n*: v_l(q^2-1); II, II*, IV, IV*: v_l(q - chi_{-3}(q)); III, III*: v_l(q - chi_{-4}(q));
  I_0*: v_l(q-1) + v_l((q+1)^2 - a_q(E')^2), E' the good twist.
- Conjecture 1.2 (Euler-factor rule): v_l(deg phi_E) >= sum_{q != l, type != III} e_l(E,q) + sum_q t_l(E,q) + [l=3] w_3(E),
  w_3 at q = l = 3 potentially good: 1 for II, IV, III*, 2 for II*, IV*, 0 for III, I_0* (empirical).
  Verified: zero violations on all four ranges at l = 3, 5, 7, 11, 13 (Table 1 of the note; `results/step10_rule_*_summary.csv`).
  Equality rate at l = 3 on the working set: 56.8% (861,074 curves with positive bound). With the III terms added: 364 / 68 / 43 / 0
  exceptions, all at l = 3, shortfall exactly 1, one shape (Section 6.6).
- Proposition 1.3: the rule implies the two earlier conjectures (component groups (1.2); twist-symmetrised (1.3)), except
  for type IV/IV* at 2 where it needs e_3(E,2) >= 1 (true on the data).
- Theorem 1.4 (Kim-Ota + ARS): multiplicative part for l >= 5, l not dividing N, q not = +-1 mod l.
- Theorem 1.5 (proved, Section 3): l | deg for IV/IV* (q >= 5, l = 3), II/II* (q >= 5, l = 3), I_n* with l | n (q odd); l not dividing N.
- Theorem 1.6 (proved, Section 4): for every odd l not dividing 2N with E[l] irreducible, v_l(deg) >= sum of the Euler terms of
  the I_0* and I_n* primes q >= 5 (Theorem 4.4: DFG Proposition 1.4(c) + DDT Lemma 4.17 inequality (Lemma 4.2, proved in
  full) + multiplicity one at the lower level + twist monotonicity Theorem 4.3); for l >= 5 with (TW) add the Tamagawa
  terms of the primes q not = +-1 mod l (Theorem 4.5 from Kim-Ota; Corollary 4.6).
- Section 5: for the potentially good types the rule is equivalent (under DFG Theorem 0.2 and a Greenberg-Wiles
  comparison, hypotheses stated) to Conjecture 5.1: length of the Bloch-Kato adjoint Selmer group plus the local
  H^0(Q_p, ad rho(1) (x) K/O) minus a dual Selmer correction >= the local terms. The type III anomaly is the case where the
  dual correction is 1. THIS IS THE QUESTION TO PUT TO KIM/OTA (or Diamond/Flach); the user is deciding whether to approach
  them directly or via a scientist friend.
- Conjecture 1.7 (Eisenstein deficit) unchanged.

Points an expert must check (listed in the note's Remark 4.7 and here):
1. That DFG's naive Sigma-finite congruence ideal (their section 1.7.3) is the square root of the Gram determinant of the
   w_N-twisted intersection pairing on the rank-two lattice M_{g^S, lambda}, up to units (Step 2 of Theorem 4.4).
2. Ribet-Stein Theorem 3.5 is stated for J_1(N); for J_0(N) the references in its proof (Mazur, Ribet 1990) give it; the
   only exception needs Serre weight l, which does not arise in weight 2. Gorenstein => free of rank 2: Tilouine 1997.
3. Kim-Ota Theorem 1.1 applied with N^- = any square-free set of Steinberg primes not = +-1 mod l (vacuous condition (6)),
   and their congruence ideal being the full-space one (checked in their paper on 2026-10-10 before compaction).
4. The four bibliography entries added from memory on 2026-10-10: DFG04 (Ann. Sci. ENS 37 (2004) 663-727), Flach92
   (Invent. Math. 109 (1992) 307-327), Hida81 (Invent. Math. 63 (1981) 225-261), Tilouine97 (Modular forms and FLT,
   Springer 1997, 327-342). Check page numbers.
5. The weights w_3 at q = l = 3 and everything with l | N are empirical only.

## 3. Documents

- `paper/note.tex` + `paper/refs.bib` + `paper/tables/*.tex` -> `paper/note.pdf` (20 pages). Build:
  `cd paper && pdflatex note && bibtex note && pdflatex note && pdflatex note`. Author (Dharmaj Soni), email and ORCID are
  filled in (amsart `\email`, `\urladdr`); the acknowledgement in double brackets (the AI-use statement) is still for the user. Tables: `counts.tex`, `isolated3.tex`, `oot.tex`, `twist.tex`
  (no longer input) from `scripts/make_tables.py`; `euler_full.tex` (step 9, q >= 5 rule, no longer input by the note but
  used by nothing else; keep), `euler_uniform.tex` from `scripts/step10_make_table.py`, `euler_iso3.tex` (hand-copied from
  `results/step9_euler_l3.txt`).
- `paper/memo_local_factors.tex` -> `.pdf` (9 pages; `pdflatex` twice, bibliography inline). The memorandum answering
  referee item 1; its last section records the changes made to the note; its proofs are the same as the note's Section 4.
- `paper/outreach.md`: message to an expert and a MathOverflow question, both rewritten for the rule.
- `README.md`: chronological log of all steps with commands; `STEP6_U1_REPORT.md` (working record, superseded, has update
  headers); `KNOWN_RELATIONS.md`, `NULL_DESIGN.md`, `results/STEP5_REPORT.md` (pre-registration and step 5 records, historical).
- `Oct10/`: copy of all of the above documents as of 2026-10-10 (note, memo, bib, tables, outreach, README, this file).
- The two referee reports (`review1_L_adic.pdf`, `author_note_twist_symmetrised_bound.pdf/.tex`) were uploaded by the
  user in the session and are NOT in the repository; ask the user for them if needed. Their items: (1) identify the local
  factors in the adjoint / congruence-ideal theory (done: the memorandum and the restructured note); (2) complete the proof
  of the qualitative theorem (done: Lemmas 3.1, 3.2, optimal-level theorem as DDT Theorem 3.15, l not dividing N added);
  (3) editorial: ARS citation and 99A1, Kim-Ota journal, Pasten, CNS24, abc sentence, q = 3 convention, Lemma 2.4 n = 0,
  Conjecture 1.7 bracket, calibration rates, control rows (done). Not done: Zenodo archive, AI-use statement wording.

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

Done today: memorandum v2 with proofs re-based on DFG; step 10 (PARI Euler factors at all bad primes, uniform rule, zero
violations); note restructured around the rule (20 pages, compiles clean, bibliography resolved); outreach texts
rewritten; README, this handoff, `Oct10/` snapshot.

Open, for the user to decide:
1. Whether and how to approach Kim/Ota (or Diamond/Flach) with Section 5 of the note (Conjecture 5.1) and the memorandum.
   The outreach text A is drafted for that.
2. Zenodo archive of the result files and the AI-use statement wording (referee item, not done).
3. Possible further computations: (a) the type III anomaly at larger scale (configurations with q >= 5 only were
   analysed in `results/step9_fullrule_*`; the q = 2 cases have v_2(N) = 8 and factor 1 + 2X); (b) the weights at q = l = 3
   against the local condition at l in DFG; (c) a direct computation of the adjoint Selmer groups for a handful of
   curves, which would test Conjecture 5.1 and the dual correction (needs a tool; not attempted).
4. Possible further writing: a short version of the note for a journal, with Sections 3 and 6 compressed.

To resume: `git log --oneline | head`, read this file, `ls results/*.csv results/*.log`, and if the parquet files are
missing rebuild them (section 4). The note and the memorandum compile with the commands in section 3.
