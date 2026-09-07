# Giant Killing

**An information-theoretic analysis of underdog upsets in football using causal emergence.**

This repo asks whether giant killings (large underdog wins) show a measurable surge in
team-level coordination — beyond what individual players already predict about the near
future. The short answer from the published analysis: **no average “togetherness surge,”**
with one exploratory signal that underdogs in upsets may *consolidate* structure over the
match. For a coach-facing summary, see [`src/coaching_guide.md`](src/coaching_guide.md).

Everything that produces the manuscript tables and figures runs through one script:
[`src/pipeline.py`](src/pipeline.py). Step-by-step replication, costs, and checksums live
in [`src/replication/REPRODUCE.md`](src/replication/REPRODUCE.md).

---

## What’s in here

```
giant-killing/
├── statsbomb-360/          # Tier C: StatsBomb 360 freeze frames + match metadata
├── open-data/              # Tier C: StatsBomb event feed (goals, clock, etc.)
└── src/
    ├── pipeline.py         # End-to-end pipeline (Sprints 2–6)
    ├── make_figures.py     # Manuscript figures (called by --stage figures)
    ├── identify_upsets.r   # Sprint 1: score upsets
    ├── extract_upsets.r    # Sprint 1: 360-enabled fixtures
    ├── coaching_guide.md   # One-page summary for coaches/analysts
    ├── manuscript/         # Paper + figures
    ├── replication/        # REPRODUCE.md, requirements, checksums
    └── data/
        ├── game_list/      # Upset / control match lists
        ├── ksg/            # Baseline mutual information windows
        ├── caus_emg/       # Causal-emergence (Ψ) outputs
        └── stat_analysis/  # Analysis panel, tables, figures
```

Older Jupyter notebooks under `src/` are **reference only**; `pipeline.py` replaces them.

---

## Pipeline stages

| Stage | What it does |
|---|---|
| `validate` | Estimator correctness checks |
| `ksg` | Per-window KSG mutual information from 360 tracking |
| `emergence` | Causal emergence Ψ at interaction orders 1–3 |
| `stats` | RQ1–RQ5 mixed models, sensitivity battery, tables |
| `figures` | Manuscript figures |
| `all` | `validate → ksg → emergence → stats → figures` |

`--stage all` skips `preprocess` (nothing downstream needs it). `emergence` is the expensive
stage (~days for the full 136-match corpus) and is resumable by default.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r src/replication/requirements.txt

# Smoke test (~15 min): stratified 10 upsets / 10 controls, reduced permutations
python src/pipeline.py --stage all --limit 20 --quick

# Full published path (needs Tier C data; see REPRODUCE.md)
python src/pipeline.py --stage all
```

`--quick` numbers are **indicative only**. Drop `--limit` / `--quick` for the published corpus.

### Data you must supply (Tier C)

Not redistributed here:

1. **`statsbomb-360/data/three-sixty/`** — 360 freeze-frame JSONs (and match metadata under `matches/`)
2. **`open-data/data/events/`** — StatsBomb event JSONs ([statsbomb/open-data](https://github.com/statsbomb/open-data))

Without local events, KSG can still fetch some event files on demand, but the **stats**
stage needs events for goal timelines / game-state covariates (`gd_c`). An empty events
directory leaves `goal_diff` all zeros and can crash MixedLM with a singular matrix when
fitting the game-state-adjusted RQ1 models.

Match lists (30 upsets / 106 controls) come from Sprint 1 R scripts, or use the checked-in
lists under `src/data/game_list/`.

---

## Research design (short)

- **Sample:** 136 matches with StatsBomb 360 coverage — 30 giant killings vs 106 matched
  “expected result” controls between similarly mismatched sides.
- **Measure:** causal emergence Ψ — how much collective team shape predicts the near future
  beyond the sum of individual players (macro features: centre of mass, compactness,
  clustering, velocity similarity, etc.).
- **Main findings (see coaching guide / manuscript):** no planned upset effect on average Ψ;
  exploratory consolidation-over-time signal; spatial configuration carries more signal
  than velocity synchrony.

Default `--clock legacy` reproduces the published corpus and includes a known half-time
clock defect; see REPRODUCE.md §7 for `--clock corrected`.

---

## Further reading

| Doc | Audience |
|---|---|
| [`src/coaching_guide.md`](src/coaching_guide.md) | Coaches / analysts — what the evidence means |
| [`src/replication/REPRODUCE.md`](src/replication/REPRODUCE.md) | Replicators — how to run and verify |
| [`src/manuscript/`](src/manuscript/) | Full paper |

## License / data

StatsBomb open data is CC BY-NC-SA 4.0. Tracking/event source data are not redistributed in
this package; the pipeline rebuilds derived artefacts from public sources.
