# Replication package

**Giant Killing in Football: An Information-Theoretic Analysis of Underdog Upsets Using
Causal Emergence**

Everything in the manuscript is produced by **one script**, `src/pipeline.py`. This
document tells you how to run it, what each stage costs, and how to check you got the
same answer.

---

## 0. Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r src/replication/requirements.txt

cd src
python src/pipeline.py --stage all --limit 20 --quick    # ~14 min, end-to-end smoke test
python src/pipeline.py --stage all                       # the real thing (see §3)
```

`--stage all --limit 20 --quick` exercises every stage on a stratified 10-upset /
10-control subset with reduced permutation counts. It is the fastest way to confirm the
environment works end to end. The numbers it prints are **not** the published ones; drop
both flags for those.

Do not go much below `--limit 20`: the `stats` stage needs at least 5 matches of each
outcome to fit the mixed models, and exits with a clear message if given fewer. The other
stages work on any subset.

---

## 1. What is in the package

The package is tiered because the full intermediate data is 1.2 GB.

| Tier | Contents | Size | Rebuilds |
|---|---|---|---|
| **A — analysis** | code + `data/stat_analysis/` (panel, tables, figures) | ~25 MB | every figure and table in the paper |
| **B — emergence** | `data/caus_emg/`, `data/ksg/`, `data/preprocessed/` | ~1.2 GB | the analysis panel, from Ψ upward |
| **C — source** | StatsBomb 360, FBref, Transfermarkt, Understat | — | *not redistributed*; fetched from public sources by the pipeline |

Tier A is the package you want unless you are auditing the estimator. Tier B is archived
separately (see `data/stat_analysis/MANIFEST.csv`). Tier C is not ours to redistribute;
the pipeline pulls it from public endpoints.

```
giant-killing/
├── statsbomb-360/data/         # Tier C: 360 freeze frames + match metadata
├── open-data/data/events/      # Tier C: StatsBomb event feed
└── src/
    ├── pipeline.py             # ← THE SCRIPT: Sprints 2-6, all stages
    ├── make_figures.py         # Sprint 6 manuscript figures (called by pipeline.py)
    ├── identify_upsets.r       # Sprint 1: upset scoring        → data/upsets.rds
    ├── extract_upsets.r        # Sprint 1: 360-enabled fixtures
    ├── upset_list.xlsx         # Sprint 1 output: 30 upsets
    ├── control_lists.xlsx      # Sprint 1 output: 106 controls
    ├── data/
    │   ├── preprocessed/       # stage: preprocess
    │   ├── ksg/                # stage: ksg
    │   ├── caus_emg/           # stage: emergence
    │   └── stat_analysis/      # stage: stats  (panel, 25 tables, 8 figures)
    ├── manuscript/             # manuscript.tex, refs.bib, figures/
    ├── replication/            # this file, requirements.txt, checksums
```

### The notebooks

`pipeline.py` replaces five notebooks, which are retained for reference only:

| Notebook | Sprint | Replaced by |
|---|---|---|
| `preprocessing.ipynb` | 2 | `--stage preprocess` |
| `ksg_impl.ipynb` | 3 | `--stage validate` + `--stage ksg` |
| `causal_emergence.ipynb` | 4 | `--stage emergence` |
| `stat_analysis.ipynb` | 5 | `--stage stats` |
| `caus_sample.ipynb` | 4 | superseded by the specification curve in `--stage stats` |

Running the notebooks is no longer necessary and is not supported. See §6 for what
changed in consolidation and why the outputs are unaffected.

---

## 2. Stages

```
python pipeline.py --stage <name> [<name> ...]
```

| Stage | Reads | Writes | Cost |
|---|---|---|---|
| `validate` | — | `stat_analysis/validation_report.csv` | 30 s |
| `preprocess` | Tier C | `data/preprocessed/` | ~40 min |
| `ksg` | Tier C + match lists | `data/ksg/` | ~15 min |
| `emergence` | `data/ksg/` + Tier C | `data/caus_emg/` | **~5 days** (4,898 core-min) |
| `stats` | `data/caus_emg/` | `data/stat_analysis/` (25 tables, 8 figures) | ~12 min |
| `figures` | `data/stat_analysis/` | `manuscript/figures/` (6 figures) | 10 s |
| `all` | | | `validate ksg emergence stats figures` |

`all` deliberately skips `preprocess`: nothing downstream reads `data/preprocessed/`.
The critical path is `ksg → emergence → stats → figures`; each of those stages rebuilds
the tracking it needs directly from the 360 files. Run `preprocess` only if you want the
tidy per-match tracking tables for inspection.

### Options

| Flag | Effect |
|---|---|
| `--root PATH` | project root containing `src/`, `statsbomb-360/`, `open-data/` (default: parent of the script) |
| `--limit N` | process at most N matches, **stratified** half upsets / half controls |
| `--overwrite` | recompute matches that already have output (default: resume) |
| `--jobs N` | parallel workers for `emergence` (default: cores − 1) |
| `--quick` | reduced permutation counts — smoke tests only |
| `--clock {legacy,corrected}` | match-clock formula. Default `legacy` reproduces the published corpus but contains a known defect — **see §7** |

`emergence` is **resumable**. It skips matches already written to `data/caus_emg/`, so an
interrupted run picks up where it stopped. This is the default; pass `--overwrite` to
force recomputation.

`--quick` changes: gap-trend permutations 20,000 → 500; RF null 200 → 10; placebo
shuffles 25 → 2; sensitivity grid 6 configs → 2; sensitivity controls 30 → 8. Every
other number is unaffected.

---

## 3. The three reproduction paths

### Path A — figures only (2 minutes, Tier A)

```bash
cd src
python pipeline.py --stage figures
```

Writes six PDF/PNG pairs to `manuscript/figures/`. Every number annotated on a figure is
read from `data/stat_analysis/tables/*.csv` or aggregated from `analysis_panel.parquet`;
nothing is re-estimated. Two figure elements are *derived* rather than read, and are
labelled as such in `make_figures.py`:

- **Fig. 5a** draws the RQ5 permutation null as a normal approximation from the stored
  mean and sd in `rq5_performance.csv`. The 200 individual draws were not persisted.
- **Fig. 6c** is the analytic non-central-*t* power curve. It reproduces `sens_power.csv`
  exactly (MDE 0.579 at *n* = 30 vs 106).

Then build the PDF:

```bash
cd manuscript
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

### Path B — statistics (~12 minutes, Tier B)

```bash
cd src
python pipeline.py --stage stats figures
```

Rebuilds `data/stat_analysis/` in full: the analysis panel, all 25 result tables, the
clock-void diagnostic, the sensitivity battery, the 8 Sprint-5 diagnostic figures, and
then the 6 manuscript figures. The sensitivity grid reuses cached
`sensitivity/psi_W*_k*.parquet` files if present; delete them to force recomputation
(adds ~25 min).

### Path C — everything from source (~5 days, Tier C)

```bash
# Sprint 1 — upset and control lists  (R >= 4.3; produces the two .xlsx files)
Rscript identify_upsets.r --season 2022 --threshold 0.65 --out data/upsets.rds
Rscript extract_upsets.r  --upsets data/upsets.rds

# Sprints 2-6
python pipeline.py --stage all
```

`emergence` is the expensive stage: 4,898 core-minutes for the production run
(136 matches × 2 teams × 6 macro features × 3 orders × ~116,700 windows). On 8 cores
that is roughly 10 hours of wall clock; the "~5 days" figure is single-core.

R dependencies for Sprint 1:

```r
install.packages(c(
  "worldfootballR", "dplyr", "tidyr", "purrr", "stringr", "lubridate",
  "readr", "rvest", "jsonlite", "optparse", "fs", "zoo", "igraph",
  "RANN", "FNN", "MASS", "Matrix", "RcppHungarian", "remotes"
))
remotes::install_github("statsbomb/StatsBombR")
```

FBref and Transfermarkt are rate-limited; `worldfootballR` inserts a ~3 s delay
internally. Expect Sprint 1 to take a few hours per league-season.

---

## 4. Determinism

`pipeline.py` fixes every seed it controls. Three components are stochastic; their
settings and sensitivity are:

| Component | Draws | Seed | Re-run spread |
|---|---|---|---|
| RQ3b gap-trend permutation | 20,000 | 0 | ±0.003 on *p* |
| RQ5 classifier permutation null | 200 | 0 | ±0.03 on *p* |
| Placebo calibration | 25 | 7 | ±0.02 on the empirical *p* |

None of these changes a conclusion.

**One genuine cross-version difference.** `RandomForestClassifier` is not bit-reproducible
across scikit-learn versions even at a fixed `random_state`: tree-split tie-breaking and
float summation order change. On this data that moves the temporal-blocking AUCs by up to
0.007 (e.g. 0.7054 → 0.7074 at the 0.6 split). The manuscript already reports that spread
as sampling noise across split points, so nothing downstream depends on it. Everything
else — every mutual-information estimate, every Ψ, every mixed model, the specification
curve — is bit-identical across versions.

---

## 5. Verifying you got the same answer

### 5.1 Automated checks

```bash
python pipeline.py --stage validate
```

Seven estimator checks, written to `data/stat_analysis/validation_report.csv`. Exits
non-zero if any fails.

| Check | Expected |
|---|---|
| KSG vs closed-form Gaussian MI | max \|bias\| < 0.01 nats over 20 replicates |
| Kozachenko–Leonenko entropy | max \|bias\| < 0.1 nats, growing with dimension |
| brute force == k-d tree | \|diff\| ~ 1e-15 |
| tree == matrix == batched | \|diff\| = 0 |
| piecewise-constant input | finite (not `inf`) |
| vectorised Onnela == networkx | \|diff\| ~ 2e-16 |
| convergence at large N | \|error\| < 0.05 nats at N ≥ 2000 |

Two notes on the first two rows, because both were stated too tightly in earlier drafts:

- **KSG is tested on the mean over 20 replicates, not a single draw.** At N = 4000 a
  single estimate has sd ≈ 0.013 nats, so a one-draw tolerance of 0.01 fails on sampling
  noise about half the time and says nothing about the estimator. The bias is 0.0007 to
  0.0040 nats — the estimator is unbiased to well within its own noise.
- **The Kozachenko–Leonenko entropy has a real, systematic negative bias that grows with
  dimension**: −0.005 (d=1), −0.003 (d=2), −0.019 (d=3), −0.054 (d=5), against a
  replicate sd of ~0.016. This is the standard curse-of-dimensionality behaviour of kNN
  entropy estimators, not noise. It is recorded per dimension in
  `validation_kl_entropy_bias.csv` rather than asserted away. Consequence for this
  project: *H*(V_fav) runs about −5 nats, so a −0.05 nat bias is ~1% and does not affect
  the RQ4 conclusion, which is null and is reported through `dc_ic` rather than the raw
  ratio.

### 5.2 File checksums

`checksums.sha256` covers every file in Tier A:

```bash
cd /path/to/project
sha256sum -c src/replication/checksums.sha256      # Linux
shasum -a 256 -c src/replication/checksums.sha256  # macOS
```

Figure PDFs will **not** match byte-for-byte across matplotlib versions (font subsetting
and the embedded creation date differ). The CSV tables and the parquet panel will. If the
tables match, the paper reproduces.

### 5.3 Headline numbers

In `data/stat_analysis/tables/`:

| Claim | File | Value |
|---|---|---|
| RQ1 null | `summary_all_rqs.csv` | 0 of 54 tests significant, min *p* = 0.081 |
| RQ2 interaction null | `rq2_mixed_anova.csv` | `C(outcome):C(order)` *p* = 0.995 |
| RQ3b effect | `rq3_gap_trend_permutation.csv` | *g* = −0.489, perm *p* = 0.024, 20,000 draws |
| RQ3b FDR replication | `rq3_gap_trend.csv` | 6 of 18 cells with `p_fdr` < 0.05 |
| RQ4 null | `rq4_disruption.csv` | min *p* = 0.222 |
| RQ5 chance | `rq5_performance.csv` | ROC-AUC 0.547, perm *p* = 0.255 |
| Specification curve | `sens_specification_curve.csv` | 323 of 324 cells with *g* < 0 |
| Power | `sens_power.csv` | MDE 0.579 at 30 vs 106 |
| Sprint-3 reconciliation | `caus_emg/computation_log.csv` | `max_recon_delta` = 0.0 nats |

Two manuscript claims are computed from the panel rather than a stored table:

```python
import pandas as pd, numpy as np
p = pd.read_parquet("src/data/stat_analysis/analysis_panel.parquet")

# 1. Psi is essentially the negated micro sum  (manuscript §4.1, Fig. 1b)
v = p[p.macro_feature == "V_all"].dropna(subset=["I_macro", "psi_l1"])
print(np.corrcoef(v.psi_l1, -(v.I_macro - v.psi_l1))[0, 1])          # 0.9955

# 2. The estimator ceiling  (manuscript §4.1, Fig. 1c)
lin = p[p.macro_feature.isin(["V_com", "V_dist", "V_fast", "V_all"])]
per_term = (lin.I_macro - lin.psi_l1) / 10.0
print(per_term.max(), (per_term > per_term.max() - 0.01).mean())     # 2.5652, 0.235

# 3. Headroom for positive emergence, by feature  (manuscript Table 3)
print(p.groupby("macro_feature").psi_l1.apply(lambda s: (s > 0).mean() * 100).round(3))
```

---

## 6. What changed when the notebooks were consolidated

The five notebooks became one script. Three substantive changes, all verified:

**1. One definition per estimator.** The notebooks carried duplicate copies of `ksg_mi`,
the clustering coefficient, `_norm`, `COMP_MAP` and the StatsBomb readers. Those copies
were identical or near-identical and are now merged. Where two versions genuinely
differed, the difference is an explicit argument rather than a fork — see
`build_bundle(common_grid=...)`: Sprint 3 grids each team from its own first frame,
Sprint 4 puts both teams on one absolute-second grid (required for the cross-team EAI and
DC terms). `_grid_offset` converts between them, and the Sprint-3 reconciliation column
verifies the correspondence on every 25th window.

**2. The AST import is gone.** `stat_analysis.ipynb` §7.5a imported Sprint 4's estimators
by AST-filtered execution of `causal_emergence.ipynb` — parsing the notebook JSON,
selecting cells that defined names in a hard-coded `NEEDED` set, and `exec`-ing them. That
machinery existed only because the code lived in a notebook. The functions are now in one
module, so there is nothing to drift and nothing to filter.

**3. The per-frame networkx clustering coefficient is gone.** Sprints 2–3 built a
`networkx` graph per frame; Sprint 4 replaced it with a vectorised `tr(W³)` form. The
script uses the vectorised version throughout, and `--stage validate` asserts the two
agree (they match to 2.2e-16). `networkx` is now needed only for that check.

### Verified equivalence

The consolidated script was run against the notebook outputs on the same inputs:

| Output | Result |
|---|---|
| `data/ksg/*_ksg.parquet` (4 matches, 6,810 rows) | **bit-identical** — `I_macro`, `I_micro_sum`, `psi` all max \|diff\| = 0 |
| `data/caus_emg/*_caus.parquet` (4 matches, 40,860 rows, 63 columns) | **bit-identical** — every shared numeric column, max \|diff\| = 0 |
| 17 of the 25 result tables (full corpus, full settings) | **bit-identical** |
| `rq2_rm_anova_gg.csv`, `sens_window_and_k.csv` | max \|diff\| 5e-15 / 6e-15 — floating-point associativity |
| the 6 `rq5_*` tables + `summary_all_rqs.csv` | differ ≤ 0.08 on RF-derived quantities — scikit-learn version, see §4 |
| `diag_clock_void.csv` | new; the diagnostic described in §7 |

Headline values reproduced exactly at full settings: RQ3b observed slope difference
−0.4891 with permutation *p* = 0.0243 (20,000 draws), 9/18 raw and 6/18 FDR-surviving
cells, specification curve 71 of 324 significant with 323 of 324 negative, placebo 21.9%
vs 3.6% (*p*<sub>emp</sub> = 0.080), RQ1 0 of 54 with min *p* = 0.0813, MDE 0.579.

### One bug fixed in transit

`rq2_order_of_emergence` called `.astype(float)` on the statsmodels Wald table. Under
`scalar=False` statsmodels returns each statistic as a `(1,1)` array; older pandas
coerced those silently, current pandas raises `ValueError: setting an array element with
a sequence`. The script now unwraps them explicitly, so RQ2 runs on any pandas version.
The values are unchanged — `rq2_mixed_anova.csv` is bit-identical to the notebook's.

---

## 7. The match-clock defect — read this before quoting any number

The absolute match clock is computed as

```
t = (period - 1) * 45 * 60 + minute * 60 + second       # "legacy"
```

but StatsBomb's `minute` field **already runs continuously across the match** — period 2
spans minutes 45–92, not 0–47. The legacy formula therefore double-counts the first half
and displaces every second-half event 45 minutes into the future, opening a median
**2,516 s (≈42 min) void** with no observations. The 1 Hz reconstruction interpolates
linearly across that void for any player observed in both halves, and the resulting
windows report `n_eff = 59` — a full complement of valid frames — so none of the six
Sprint-4 diagnostics detects them.

| | inside the void | outside |
|---|---:|---:|
| share of analysis windows | **29.1 %** | 70.9 % |
| mean `n_eff` | 59.0 | 59.0 |
| mean Ψ⁽¹⁾ | −17.0 | −8.4 |
| per-player terms at the estimator ceiling | **71.7 %** | 3.8 % |

**88.6 % of all estimator saturation in the corpus lies inside the artefact.** Perfectly
smooth interpolated motion makes every player maximally informative about the centre of
mass, which drives the micro sum to the KSG per-term ceiling (2.5652 nats) and Ψ to its
floor (−23.1 nats).

### What this does and does not affect

| | status |
|---|---|
| Definitional redundancy (Ψ ≈ −Σᵢ I, *r* = 0.9996) | **unaffected** — arithmetic, true in every window |
| Ψ > 0 headroom by feature (Table 3) | **unaffected** — the linear/non-linear split holds either way |
| Estimator-saturation claim | **substantially an artefact** — 3.8 % outside the void, not 24 % |
| RQ3b gap-trend headline | **survives** — *g* = −0.458 (*p* = 0.009) excluding void windows, vs −0.468 (*p* = 0.005) published |
| RQ1/RQ2/RQ4/RQ5 nulls | **likely robust** — the artefact is near-identical across groups and roles, so it differences out. Not re-verified. |

### Running without it

```bash
python pipeline.py --stage ksg emergence stats figures --clock corrected --overwrite
```

`--clock legacy` is the **default**, because it is what produced the published corpus.
`--clock corrected` uses `t = minute*60 + second` and changes every downstream number, so
it needs a full recomputation (`--overwrite`, ~5 days of compute) — you cannot mix the two
in one `data/caus_emg/`.

Every `--stage stats` run writes `tables/diag_clock_void.csv` quantifying the void under
whichever clock is active, so the defect cannot silently return.

---

## 8. Other gotchas

- **`--stage emergence` resumes by default.** It skips matches already in
  `data/caus_emg/`. If you change an estimator parameter — including `--clock` — you must
  pass `--overwrite`, or you will silently mix old and new output.
- **`--limit` is stratified.** It takes half upsets and half controls, because the upset
  list is read first and a plain head() would give a single-class corpus. The `stats`
  stage needs at least 5 of each and exits with a clear message otherwise.
- **Window overlap.** Ψ is emitted every 10 s from a 60 s window, so consecutive rows
  share 50 of 60 seconds. All models use non-overlapping windows only
  (`window_start % 60 == 0`), retaining 231,492 of 1,384,566 rows. Analysing the full set
  gives spuriously small standard errors.
- **Standardisation is not optional.** KSG uses the Chebyshev norm, so unstandardised
  macro features let the largest-variance dimension dominate. Without per-window
  *z*-scoring the 5-dimensional `V_all` returns numerically identical Ψ to the
  3-dimensional `V_fast` — the clustering dimensions contribute nothing. See manuscript
  §3.5(ii).
- **`psi_l3` is the rescaled (Horvitz–Thompson) sum** over proximity-selected triples;
  `psi_l3_raw` is the unscaled sampled sum. The manuscript uses `psi_l3`. The rescaling
  assumes the retained triples are representative, which proximity selection deliberately
  violates; `--stage emergence` measures that bias against the exhaustive sum and writes
  it to `data/caus_emg/l3_subsampling_bias.csv` (median relative error ≈ 0%, range
  −0.6% to +0.7% on the audited match).
- **`data/caus_sample/` is empty.** That notebook was a 20-match aggregation-sensitivity
  demonstration, fully superseded by the 324-cell specification curve in `--stage stats`.
- **`torch` is not required.** It appears in two exploratory Sprint-1/2 notebooks and in
  a GPU hook in `ksg_impl.ipynb` that always fell back to CPU. No manuscript result uses
  it, and it is not in `requirements.txt`.

---

## 9. Data sources and terms

| Source | Used for | Access |
|---|---|---|
| StatsBomb Open Data | 360 freeze frames, events, goal times | <https://github.com/statsbomb/open-data> (CC BY-NC-SA 4.0) |
| FBref (via `worldfootballR`) | fixtures, results, basic xG | public, rate-limited |
| Transfermarkt (via `worldfootballR`) | squad market values | public, rate-limited |
| Understat (via `worldfootballR`) | expected goals | public |

All sources are publicly available and used in accordance with their terms. Raw provider
data are **not** redistributed in this package; the pipeline reproduces them. `--stage
ksg` will fetch and cache missing event files from the StatsBomb open-data repository. No
personally identifiable information is used; tracking data are aggregated to team-level
features before any analysis.

## 10. Citation

If you use this code or these results, please cite the manuscript
(`manuscript/manuscript.tex`) and the three works it builds on: Rosas et al. (2020) for
the emergence framework, Cheng et al. (2025) for the football instantiation, and Penn et
al. (2023) for the tracking reconstruction.
