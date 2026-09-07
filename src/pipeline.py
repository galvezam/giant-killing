#!/usr/bin/env python3
# ===========================================================================
#  Giant Killing in Football: An Information-Theoretic Analysis of Underdog
#  Upsets Using Causal Emergence
#
#  Consolidated end-to-end pipeline. This single script replaces five Jupyter
#  notebooks:
#
#      preprocessing.ipynb      Sprint 2  -> data/preprocessed/
#      ksg_impl.ipynb           Sprint 3  -> data/ksg/
#      causal_emergence.ipynb   Sprint 4  -> data/caus_emg/
#      caus_sample.ipynb        Sprint 4  (aggregation-sensitivity demo)
#      stat_analysis.ipynb      Sprint 5  -> data/stat_analysis/
#
#  plus the Sprint-6 manuscript figures (make_figures.py, called by the
#  `figures` stage).
#
#  ------------------------------------------------------------------------
#  USAGE
#
#      python pipeline.py --stage all           # everything, in order
#      python pipeline.py --stage validate      # estimator validation report
#      python pipeline.py --stage ksg           # Sprint 3
#      python pipeline.py --stage emergence     # Sprint 4
#      python pipeline.py --stage stats         # Sprint 5
#      python pipeline.py --stage figures       # Sprint 5 + Sprint 6 figures
#
#      python pipeline.py --stage all --limit 8 --quick   # ~10 min smoke test
#
#  Run `python pipeline.py --help` for the full option list.
#
#  ------------------------------------------------------------------------
#  WHAT CHANGED IN CONSOLIDATION (behaviour is preserved; see REPRODUCE.md §6)
#
#  1. Each estimator now has exactly ONE definition. The notebooks carried
#     duplicate copies of ksg_mi, clust_coef, _norm, COMP_MAP and the
#     StatsBomb readers; those copies were identical or near-identical and
#     have been merged. Where two versions genuinely differed the difference
#     is now an explicit argument, not a fork -- see build_bundle(common_grid)
#     in section 3, which is the Sprint-3 / Sprint-4 grid difference.
#
#  2. stat_analysis.ipynb imported Sprint 4's estimators by AST-filtered
#     execution of causal_emergence.ipynb (its section 7.5a). That machinery
#     is gone: the functions are in this module, so there is nothing to drift.
#
#  3. The per-frame networkx clustering coefficient of Sprints 2-3 is replaced
#     by the vectorised Onnela form from Sprint 4 throughout. `--stage
#     validate` asserts the two agree (they match to ~1e-16).
#
#  Numerical outputs are unchanged. `--stage validate` re-runs every
#  correctness check the notebooks performed.
# ===========================================================================

from __future__ import annotations

import argparse
import ast
import contextlib
import glob
import itertools
import json
import os
import re
import sys
import time
import traceback
import unicodedata
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.special import digamma, gamma as gamma_fn

# Heavy / optional imports are deferred to the stages that need them:
#   statsmodels, sklearn  -> stats stage
#   matplotlib            -> stats and figures stages
#   networkx              -> validate stage only (equivalence check)
#   joblib                -> ksg and emergence stages


# ===========================================================================
# SECTION 0 -- Configuration
# ===========================================================================

class Config:
    """Every tunable in one place. Values are the production settings that
    produced the corpus described in the manuscript."""

    # ---- paths (resolved in __init__; all relative to the project root) ----
    SB_360_SUB    = ("statsbomb-360", "data", "three-sixty")
    SB_MATCH_SUB  = ("statsbomb-360", "data", "matches")
    SB_EVENTS_SUB = ("open-data", "data", "events")
    GH_EVENTS = ("https://raw.githubusercontent.com/statsbomb/open-data/"
                 "master/data/events/{}.json")

    # ---- estimator ---------------------------------------------------------
    K_DEFAULT   = 4          # KSG neighbour count
    WINDOW_SEC  = 60         # analysis window length, seconds
    LAG         = 1          # t' = t + LAG
    STEP_SEC    = 10         # window advance for the production grid
    MIN_SAMPLES = 4 * 4      # a window needs >= 4k clean frames
    STANDARDIZE = True       # z-score each dimension within the window
    JITTER      = 1e-10      # tie-breaking noise, standardised units
    REL_TOL     = 1.0 - 1e-12

    # ---- match clock -------------------------------------------------------
    # "legacy"    t = (period-1)*45*60 + minute*60 + second
    # "corrected" t = minute*60 + second
    #
    # StatsBomb's `minute` already runs continuously across the whole match
    # (period 2 spans minutes 45..92), so the legacy formula DOUBLE-COUNTS the
    # first half and pushes every period-2 event 45 minutes into the future.
    # That opens a ~42-minute void between the end of the first half (~minute
    # 46) and the shifted start of the second (minute 90). The 1 Hz
    # reconstruction interpolates straight across it, so ~29% of analysis
    # windows contain no real observations while reporting n_eff = 59.
    #
    # DEFAULT IS "legacy" because it is what produced the published corpus.
    # Pass --clock corrected to rebuild without the artefact; see
    # REPRODUCE.md §7 for what that changes.
    CLOCK_MODE = "legacy"

    # ---- features ----------------------------------------------------------
    CENTRE = np.array([60.0, 40.0])     # StatsBomb pitch is 120 x 80
    MAX_OUTFIELD = 10
    RED_CARDS = {"Red Card", "Second Yellow"}
    ORDERS = (1, 2, 3)
    MACRO_FEATURES = ["V_com", "V_dist", "V_cvel", "V_cdist", "V_all", "V_fast"]
    PRIMARY_FEATURE, PRIMARY_ORDER = "V_all", 1

    # ---- l=3 subsampling ---------------------------------------------------
    L3_STRATEGY = "proximal"    # "proximal" | "random" | "full"
    L3_RADIUS   = 25.0          # metres
    L3_MAX      = 60            # cap on triples per window
    L3_SEED     = 0
    HO_STRIDE   = 1             # compute l>=2 on every HO_STRIDE-th window

    # ---- novel metrics -----------------------------------------------------
    EPS           = 1e-9
    SRR_ROLL      = 30
    OUTLIER_Z     = 5.0
    EAI_DENOM_MIN = 1e-3
    PARITY_EVERY  = 25          # Sprint-3 parity check every Nth window

    # ---- statistics --------------------------------------------------------
    THIN_STEP  = 60             # keep window_start % THIN_STEP == 0
    ALPHA      = 0.05
    FDR_METHOD = "fdr_bh"
    RF_SEED    = 0
    N_PERM_CURVE   = 500        # RQ3 functional permutation
    N_PERM_TREND   = 20_000     # RQ3b gap-trend permutation
    N_PERM_RF      = 200        # RQ5 classifier null
    N_PLACEBO      = 25         # specification-curve label shuffles
    SENS_CONFIGS   = [(30, 4), (60, 4), (90, 4), (60, 3), (60, 7), (60, 10)]
    SENS_FEATURES  = ["V_all", "V_cvel"]
    SENS_N_CONTROL = 30

    # ---- execution ---------------------------------------------------------
    N_JOBS      = -1
    MATCH_JOBS  = max(1, (os.cpu_count() or 4) - 1)
    OVERWRITE   = False         # False = resume, skipping completed matches

    def __init__(self, root: Path, quick: bool = False):
        self.root = root
        self.src = root / "src"
        self.sb_360    = root.joinpath(*self.SB_360_SUB)
        self.sb_match  = root.joinpath(*self.SB_MATCH_SUB)
        self.sb_events = root.joinpath(*self.SB_EVENTS_SUB)

        self.data       = self.src / "data"
        self.prep_dir   = self.data / "preprocessed"
        self.ksg_dir    = self.data / "ksg"
        self.caus_dir   = self.data / "caus_emg"
        self.stat_dir   = self.data / "stat_analysis"
        self.fig_dir    = self.stat_dir / "figures"
        self.tab_dir    = self.stat_dir / "tables"
        self.sens_dir   = self.stat_dir / "sensitivity"

        self.upset_xlsx   = self.src / "data" / "game_list" / "upset_list.xlsx"
        self.control_xlsx = self.src / "data" / "game_list" / "control_lists.xlsx"

        if quick:      # smoke-test settings; NOT the published numbers
            self.N_PERM_CURVE = 50
            self.N_PERM_TREND = 500
            self.N_PERM_RF    = 10
            self.N_PLACEBO    = 2
            self.SENS_CONFIGS = [(60, 4), (90, 4)]
            self.SENS_N_CONTROL = 8
        self.quick = quick

    def mkdirs(self):
        for d in (self.data, self.prep_dir, self.ksg_dir, self.caus_dir,
                  self.stat_dir, self.fig_dir, self.tab_dir, self.sens_dir):
            d.mkdir(parents=True, exist_ok=True)


CFG: Config = None      # set by main()


# ---------------------------------------------------------------------------
# small console helpers
# ---------------------------------------------------------------------------
def banner(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def step(msg: str) -> None:
    print(f"  {msg}")


def show(df: pd.DataFrame, n: int = 10, prec: int = 4) -> None:
    """Notebook `display()` replacement."""
    with pd.option_context("display.width", 200, "display.max_columns", 60,
                           "display.float_format", lambda v: f"{v:.{prec}g}"):
        print(df.head(n).to_string())


# ===========================================================================
# SECTION 1 -- Estimators
#
# Kraskov-Stoegbauer-Grassberger mutual information (estimator #1) and the
# Kozachenko-Leonenko differential entropy. All use the Chebyshev (max) norm,
# as KSG requires.
#
#   I(X;Y) = psi(k) - <psi(n_x+1) + psi(n_y+1)> + psi(N)
#
# Three interfaces, all the same estimator:
#   ksg_mi            k-d trees; the reference implementation
#   ksg_from_dist     from precomputed Chebyshev distance matrices
#   ksg_batch_from_dist   many X against one Y, vectorised
#
# The batched form is what makes the corpus tractable: micro-side distances
# depend only on the window, so they are built once and reused across all six
# macro features and all three orders (~195 estimates per window).
# ===========================================================================

def _as2d(A) -> np.ndarray:
    A = np.asarray(A, float)
    return A[:, None] if A.ndim == 1 else A


def ksg_mi(X, Y, k=None):
    """KSG estimator #1 via k-d trees (Kraskov et al. 2004, Phys. Rev. E 69)."""
    k = k or Config.K_DEFAULT
    X, Y = _as2d(X), _as2d(Y)
    N = len(X)
    if N <= k + 1:
        return np.nan
    Z = np.hstack([X, Y])
    eps = cKDTree(Z).query(Z, k=k + 1, p=np.inf)[0][:, k]
    nx = np.asarray(cKDTree(X).query_ball_point(
        X, eps - 1e-12, p=np.inf, return_length=True)) - 1
    ny = np.asarray(cKDTree(Y).query_ball_point(
        Y, eps - 1e-12, p=np.inf, return_length=True)) - 1
    return digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(N)


def ksg_mi_bruteforce(X, Y, k=None):
    """Literal O(N^2) transcription of the published pseudocode.

    Used only by `--stage validate`, to prove the fast path introduces no
    approximation."""
    k = k or Config.K_DEFAULT
    X, Y = _as2d(X), _as2d(Y)
    N = len(X)
    if N <= k + 1:
        return np.nan
    Dx = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
    Dy = np.max(np.abs(Y[:, None, :] - Y[None, :, :]), axis=2)
    Dz = np.maximum(Dx, Dy)
    terms = 0.0
    for i in range(N):
        eps_i = np.partition(Dz[i], k)[k]        # self is the 0-th neighbour
        n_x = np.count_nonzero(Dx[i] < eps_i) - 1
        n_y = np.count_nonzero(Dy[i] < eps_i) - 1
        terms += digamma(n_x + 1) + digamma(n_y + 1)
    return digamma(k) - terms / N + digamma(N)


def cheb_dist(A) -> np.ndarray:
    """(N,d) -> (N,N) Chebyshev distance matrix."""
    A = _as2d(A)
    return np.max(np.abs(A[:, None, :] - A[None, :, :]), axis=2)


def ksg_from_dist(Dx, Dy, k=None):
    """KSG #1 from precomputed Chebyshev distance matrices.

    Uses a RELATIVE neighbour tolerance and clamps counts at zero. Both are
    required: StatsBomb 360 is event-sampled, so the 1 Hz grid interpolates
    linearly between events and the velocity-derived V_cvel is piecewise
    constant over long runs. With an absolute tolerance those runs give
    eps_i = 0, a neighbour count of -1, and psi(0) = -inf."""
    k = k or Config.K_DEFAULT
    N = Dx.shape[0]
    if N <= k + 1:
        return np.nan
    eps = np.partition(np.maximum(Dx, Dy), k, axis=1)[:, k]
    thr = (eps * Config.REL_TOL)[:, None]
    nx = np.maximum((Dx < thr).sum(1) - 1, 0)
    ny = np.maximum((Dy < thr).sum(1) - 1, 0)
    return digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(N)


def ksg_batch_from_dist(Dxs, Dy, k=None):
    """Many X against one Y. Dxs (S,N,N), Dy (N,N) -> (S,) estimates."""
    k = k or Config.K_DEFAULT
    S, N, _ = Dxs.shape
    if N <= k + 1:
        return np.full(S, np.nan)
    eps = np.partition(np.maximum(Dxs, Dy), k, axis=2)[:, :, k]
    thr = (eps * Config.REL_TOL)[:, :, None]
    nx = np.maximum((Dxs < thr).sum(2) - 1, 0)
    ny = np.maximum((Dy[None] < thr).sum(2) - 1, 0)
    return digamma(k) - (digamma(nx + 1) + digamma(ny + 1)).mean(1) + digamma(N)


def kl_entropy(X, k=None):
    """Kozachenko-Leonenko differential entropy, nats (Euclidean norm).

    H = -psi(k) + psi(N) + log(c_d) + (d/N) sum log r_i, with c_d the volume
    of the unit d-ball. Supplies H(V_fav) for the Disruption Coefficient."""
    k = k or Config.K_DEFAULT
    X = _as2d(X)
    N, d = X.shape
    if N <= k + 1:
        return np.nan
    r = np.maximum(cKDTree(X).query(X, k=k + 1)[0][:, k], 1e-12)
    log_cd = (d / 2.0) * np.log(np.pi) - np.log(gamma_fn(d / 2.0 + 1))
    return -digamma(k) + digamma(N) + log_cd + d * np.mean(np.log(r))


def zscore(A, ref=None, jitter=None, rng=None, standardize=None):
    """Standardise each column; constant columns map to zero.

    `ref` supplies shared statistics so that V_t and V_t' share one scaler
    (they are the same variable at two times).

    This is not cosmetic. KSG #1 uses the max norm, so a joint feature vector
    is governed by whichever dimension has the largest numerical spread. Raw
    macro features span sd(V_com) ~ 19 m against sd(V_cdist) ~ 0.10; left
    unstandardised the 5-d V_all returns numerically identical Psi to the 3-d
    V_fast, the clustering dimensions contributing nothing. MI is invariant to
    this transform in theory; the estimator is not."""
    standardize = Config.STANDARDIZE if standardize is None else standardize
    jitter = Config.JITTER if jitter is None else jitter
    A = _as2d(A)
    if not standardize:
        return A
    R = A if ref is None else _as2d(ref)
    mu, sd = R.mean(0), R.std(0)
    Z = (A - mu) / np.where(sd > 0, sd, 1.0)
    if jitter and rng is not None:
        Z = Z + rng.normal(0.0, jitter, Z.shape)
    return Z


# ===========================================================================
# SECTION 2 -- Macroscopic features
#
# Four macro features (Cheng et al. 2025 Fig. 1; Giant_Killing Table 2), all
# functions of the same micro state so that supervenience holds, plus two
# composites:
#
#   V_com    (2-d) centre of mass, mean of player (x,y)
#   V_dist   (1-d) mean distance of players to pitch centre
#   V_cdist  (1-d) Onnela weighted clustering, w_uv = 1/d_uv
#   V_cvel   (1-d) Onnela weighted clustering, w_uv = S_u S_v (cos dtheta + 1)
#   V_all    (5-d) all four jointly
#   V_fast   (3-d) V_com + V_dist -- the Sprint-3 macro state
#
# V_com and V_dist are LINEAR aggregates of the micro state; the clustering
# coefficients are not. That distinction governs the results (manuscript §4.1).
# ===========================================================================

def onnela_clustering(W: np.ndarray) -> np.ndarray:
    """Mean Onnela weighted local clustering coefficient for a stack of graphs.

    W : (T, n, n) symmetric, non-negative, zero diagonal. Returns (T,).

        c_u = [k_u(k_u-1)]^-1 sum_{v,w} (w_uv^ w_uw^ w_vw^)^(1/3)

    with weights normalised by the graph maximum -- networkx's definition.
    Sprints 2-3 built a networkx graph per frame, untenable at corpus scale
    (~8500 frames x 2 teams x 136 matches); this is the same estimator
    vectorised over frames. `--stage validate` asserts they agree."""
    W = np.asarray(W, float)
    T = W.shape[0]
    mx = W.reshape(T, -1).max(1)
    mx[mx <= 0] = 1.0
    Wh = np.cbrt(W / mx[:, None, None])
    deg = (W > 0).sum(2).astype(float)
    tri = np.einsum("tij,tjk,tki->ti", Wh, Wh, Wh)
    den = deg * (deg - 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(den > 0, tri / den, 0.0)
    return c.mean(1)


def _pairwise_weights(pos, vel):
    """(T,P,P) inverse-distance and velocity-similarity weight stacks."""
    D = np.sqrt(((pos[:, :, None, :] - pos[:, None, :, :]) ** 2).sum(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        Wd = np.where(D > 1e-9, 1.0 / D, 0.0)
    Wd = np.nan_to_num(Wd, nan=0.0, posinf=0.0)

    sp = np.sqrt((vel ** 2).sum(-1))
    ang = np.arctan2(vel[..., 1], vel[..., 0])
    Wv = sp[:, :, None] * sp[:, None, :] * (
        np.cos(ang[:, :, None] - ang[:, None, :]) + 1.0)
    Wv = np.nan_to_num(Wv, nan=0.0, posinf=0.0, neginf=0.0)

    idx = np.arange(pos.shape[1])
    Wd[:, idx, idx] = 0.0
    Wv[:, idx, idx] = 0.0
    return Wd, Wv


def macro_features(pos, vel) -> dict:
    """pos, vel : (T, P, 2) -> dict of macro arrays, each (T, d)."""
    V_com = np.nanmean(pos, axis=1)
    V_dist = np.nanmean(
        np.sqrt(((pos - Config.CENTRE) ** 2).sum(2)), axis=1, keepdims=True)
    Wd, Wv = _pairwise_weights(pos, vel)
    V_cdist = onnela_clustering(Wd)[:, None]
    V_cvel = onnela_clustering(Wv)[:, None]
    out = dict(V_com=V_com, V_dist=V_dist, V_cvel=V_cvel, V_cdist=V_cdist)
    out["V_all"] = np.hstack([V_com, V_dist, V_cvel, V_cdist])
    out["V_fast"] = np.hstack([V_com, V_dist])
    return out


# ===========================================================================
# SECTION 3 -- StatsBomb I/O and tracking reconstruction
#
# StatsBomb 360 gives a freeze frame of VISIBLE player positions at each
# on-ball event: roughly one frame every 1.9 s, concentrated near the ball,
# with off-camera players absent. Preprocessing (Cheng et al. 2025 §2.1):
#
#   1. join freeze frames to events to recover the match clock and possession
#   2. assign each player to a team (teammate => possession team)
#   3. drop goalkeepers
#   4. truncate each team at its first red card (keeps 10 outfielders)
#   5. resample onto a 1 Hz grid by linear interpolation
#   6. velocities by finite differences
# ===========================================================================

_MATCH_INDEX = None
_LOCAL_360 = None


def read_360(mid):
    p = CFG.sb_360 / f"{mid}.json"
    return json.load(open(p)) if p.exists() else None


def read_events(mid, allow_fetch=False):
    """Local clone first; optionally fetch from GitHub open-data and cache."""
    p = CFG.sb_events / f"{mid}.json"
    if p.exists():
        return json.load(open(p))
    if not allow_fetch:
        return None
    try:
        raw = urllib.request.urlopen(
            Config.GH_EVENTS.format(mid), timeout=30).read()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        return json.loads(raw)
    except Exception as e:
        step(f"events fetch failed for {mid}: {e}")
        return None


def match_index() -> dict:
    """match_id -> StatsBomb match metadata dict (cached)."""
    global _MATCH_INDEX
    if _MATCH_INDEX is None:
        _MATCH_INDEX = {}
        for path in glob.glob(str(CFG.sb_match / "*" / "*.json")):
            try:
                for m in json.load(open(path)):
                    _MATCH_INDEX[m["match_id"]] = m
            except Exception:
                continue
    return _MATCH_INDEX


def local_360_ids() -> set:
    global _LOCAL_360
    if _LOCAL_360 is None:
        _LOCAL_360 = {int(Path(p).stem)
                      for p in glob.glob(str(CFG.sb_360 / "*.json"))}
    return _LOCAL_360


def match_teams(mid):
    m = match_index().get(int(mid))
    if not m:
        return None, None
    return (m["home_team"]["home_team_name"], m["away_team"]["away_team_name"])


def event_clock(period, minute, second):
    """Absolute match clock in seconds. See Config.CLOCK_MODE for why there
    are two, and why the buggy one is still the default."""
    base = minute * 60 + second
    if Config.CLOCK_MODE == "corrected":
        return base
    return (period - 1) * 45 * 60 + base


def events_table(ev) -> pd.DataFrame:
    """Flatten raw events to the fields the pipeline needs, with an absolute
    match clock."""
    rows = []
    for e in ev:
        loc = e.get("location") or [None, None]
        rows.append(dict(
            event_uuid=e["id"], period=e["period"], minute=e["minute"],
            second=e["second"], type=e["type"]["name"],
            team=(e.get("team") or {}).get("name"),
            poss_team=(e.get("possession_team") or {}).get("name"),
            ball_x=loc[0] if loc else None,
            ball_y=loc[1] if len(loc) > 1 else None,
            shot_outcome=(e.get("shot") or {}).get("outcome", {}).get("name"),
            bb=(e.get("bad_behaviour") or {}).get("card", {}).get("name"),
            fc=(e.get("foul_committed") or {}).get("card", {}).get("name"),
        ))
    d = pd.DataFrame(rows)
    d["t_sec"] = event_clock(d.period, d.minute, d.second)
    d["is_shot"] = d.type.eq("Shot")
    d["is_goal"] = d.shot_outcome.eq("Goal") | d.type.eq("Own Goal Against")
    return d


def red_card_times(e: pd.DataFrame) -> dict:
    """{team: t_sec of first sending-off}. A team is 'reduced' for t >= that."""
    out = {}
    sent = e[(e.bb.isin(Config.RED_CARDS)) | (e.fc.isin(Config.RED_CARDS))]
    for _, r in sent.iterrows():
        if r.team:
            out[r.team] = min(out.get(r.team, np.inf), r.t_sec)
    return out


def clean_player_frames(fr, e, home, away):
    """Freeze frames -> one row per (event, player), team-tagged, keepers and
    post-red-card frames removed."""
    red = red_card_times(e)
    rows = [(f["event_uuid"], p["teammate"], p["keeper"],
             p["location"][0], p["location"][1])
            for f in fr for p in f["freeze_frame"]]
    ft = pd.DataFrame(rows, columns=["event_uuid", "teammate", "keeper", "x", "y"])
    j = (ft.merge(e[["event_uuid", "t_sec", "poss_team"]],
                  on="event_uuid", how="left")
           .dropna(subset=["t_sec", "x", "y"]))
    j["team"] = np.where(j.teammate, j.poss_team,
                         np.where(j.poss_team == home, away, home))
    j = j.dropna(subset=["team"])
    j = j[~j.keeper]                                    # (3) drop goalkeepers
    if red:                                             # (4) truncate at red
        keep = np.ones(len(j), bool)
        for tm, t_red in red.items():
            keep &= ~((j.team.values == tm) & (j.t_sec.values >= t_red))
        j = j[keep]
    return j


def _interp_team(g: pd.DataFrame, grid: np.ndarray):
    """Irregular event-sampled positions -> (T,P,2) on `grid`, edge-masked."""
    g = g.sort_values(["t_sec", "x", "y"]).copy()
    g["pid"] = g.groupby("t_sec").cumcount() + 1        # stable per-frame order
    g = g[g.pid <= Config.MAX_OUTFIELD]
    if len(g) == 0:
        return None, None
    P, T = int(g.pid.max()), len(grid)
    pos = np.full((T, P, 2), np.nan)
    for pid, pg in g.groupby("pid"):
        pg = pg.drop_duplicates("t_sec").sort_values("t_sec")
        if len(pg) < 2:
            continue
        xs = np.interp(grid, pg.t_sec, pg.x)
        ys = np.interp(grid, pg.t_sec, pg.y)
        inr = (grid >= pg.t_sec.min()) & (grid <= pg.t_sec.max())
        xs[~inr] = np.nan
        ys[~inr] = np.nan
        pos[:, pid - 1, 0] = xs
        pos[:, pid - 1, 1] = ys
    return pos, float(g.t_sec.min())


def build_bundle(mid, common_grid=True, allow_fetch=False, with_macro=True):
    """match_id -> {team: dict(grid, pos, vel, macro, t_start)}.

    common_grid=True  (Sprint 4): both teams share ONE absolute-second grid,
        so a given window_start means the same instant for both. Required for
        the EAI and DC cross-team terms.
    common_grid=False (Sprint 3): each team is gridded from its own first
        frame. `_grid_offset` converts between the two conventions, and the
        Sprint-3 reconciliation in section 7 verifies the correspondence.

    This flag is the ONLY difference between the two sprints' reconstruction;
    the notebooks carried it as two near-duplicate functions."""
    fr = read_360(mid)
    ev = read_events(mid, allow_fetch=allow_fetch)
    home, away = match_teams(mid)
    if not fr or not ev or home is None:
        return None
    e = events_table(ev)
    j = clean_player_frames(fr, e, home, away)
    if j.empty:
        return None

    shared = None
    if common_grid:
        t0, t1 = int(np.floor(j.t_sec.min())), int(np.ceil(j.t_sec.max()))
        shared = np.arange(t0, t1 + 1, 1.0)

    out = {}
    for tm in j.team.dropna().unique():
        g = j[j.team == tm]
        if common_grid:
            grid = shared
        else:
            grid = np.arange(int(np.floor(g.t_sec.min())),
                             int(np.ceil(g.t_sec.max())) + 1, 1.0)
        pos, t_start = _interp_team(g, grid)
        if pos is None:
            continue
        vel = np.full_like(pos, np.nan)
        vel[1:] = np.diff(pos, axis=0)                  # (6) 1 Hz differences
        rec = dict(grid=grid, pos=pos, vel=vel, t_start=t_start)
        if with_macro:
            rec["macro"] = macro_features(pos, vel)
        out[tm] = rec
    return out or None


def _grid_offset(bundle, team) -> int:
    """Sprint 3 indexed windows from each team's own first frame; Sprint 4 uses
    one common grid. common_index = ksg_window_start + offset."""
    return int(np.floor(bundle[team]["t_start"]) - bundle[team]["grid"][0])


# ===========================================================================
# SECTION 4 -- Match lists, name harmonisation, favourite/underdog roles
#
# The Sprint-1 upset and control lists key matches by (competition, home,
# away) using their own team-name spellings. StatsBomb uses different ones.
# `_norm` lower-cases, strips accents, brackets, "Women's", and club suffixes,
# then applies an explicit alias table.
# ===========================================================================

COMP_MAP = {
    "WC 2022": "FIFA World Cup 2022",
    "Euro 2024": "UEFA Euro 2024",
    "Euro 2020": "UEFA Euro 2020",
    "AFCON 2023": "African Cup of Nations 2023",
    "WWC 2023": "Women's World Cup 2023",
    "Women's Euro 2022": "UEFA Women's Euro 2022",
    "Women's Euro 2025": "UEFA Women's Euro 2025",
    "La Liga 20/21": "La Liga 2020/2021",
    "Ligue 1 22/23": "Ligue 1 2022/2023",
    "Ligue 1 21/22": "Ligue 1 2021/2022",
    "MLS 2023": "Major League Soccer 2023",
    "Bundesliga 23/24": "1. Bundesliga 2023/2024",
}

_ALIASES = {
    "ir iran": "iran", "korea republic": "south korea",
    "republic of korea": "south korea", "cote d'ivoire": "ivory coast",
    "united states": "usa", "turkiye": "turkey", "czechia": "czech republic",
    "the gambia": "gambia", "dr congo": "congo dr", "china pr": "china",
}


def _norm(s: str) -> str:
    s = re.sub(r"\s*\[.*?\]", "", str(s)).lower().strip()
    s = re.sub(r"['’‘]", "'", s)
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    s = re.sub(r"\bwomen's\b|\bwomen\b", "", s)
    s = re.sub(r"\bfc\b|\bcf\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _ALIASES.get(s, s)


def _fixture_index() -> dict:
    """(competition-season, {normalised team names}) -> [match_id]."""
    index = {}
    for m in match_index().values():
        comp = (f"{m['competition']['competition_name']} "
                f"{m['season']['season_name']}")
        key = (comp, frozenset({_norm(m["home_team"]["home_team_name"]),
                                _norm(m["away_team"]["away_team_name"])}))
        index.setdefault(key, []).append(m["match_id"])
    return index


def _read_lists() -> list:
    """[(row, outcome)] over both Sprint-1 spreadsheets."""
    out = []
    for path, outcome, sheet in [(CFG.upset_xlsx, "upset", "upsets"),
                                 (CFG.control_xlsx, "control", "control_lists")]:
        if not path.exists():
            step(f"WARNING: {path.name} not found; roles will be unavailable")
            continue
        for _, r in pd.read_excel(path, sheet_name=sheet).iterrows():
            out.append((r, outcome))
    return out


def resolve_targets() -> pd.DataFrame:
    """Resolve every upset/control fixture to a StatsBomb match_id."""
    index, local = _fixture_index(), local_360_ids()
    rows = []
    for r, outcome in _read_lists():
        comp = COMP_MAP.get(r["competition"])
        mid = None
        if comp:
            cands = index.get(
                (comp, frozenset({_norm(r["home"]), _norm(r["away"])})), [])
            with360 = [m for m in cands if m in local]
            mid = with360[0] if with360 else (cands[0] if cands else None)
        rows.append(dict(outcome=outcome, competition=r["competition"],
                         home=r["home"], away=r["away"], match_id=mid,
                         has_360=bool(mid) and mid in local))
    return pd.DataFrame(rows)


def build_role_map() -> dict:
    """match_id -> dict(outcome, favourite, underdog, competition, ranks),
    with team names in StatsBomb spelling."""
    index, local = _fixture_index(), local_360_ids()
    roles = {}
    for r, outcome in _read_lists():
        comp = COMP_MAP.get(r["competition"])
        if not comp:
            continue
        cands = index.get(
            (comp, frozenset({_norm(r["home"]), _norm(r["away"])})), [])
        with360 = [m for m in cands if m in local]
        mid = with360[0] if with360 else (cands[0] if cands else None)
        if mid is None or mid in roles:
            continue
        sb = match_index()[mid]
        names = [sb["home_team"]["home_team_name"],
                 sb["away_team"]["away_team_name"]]
        lookup = {_norm(n): n for n in names}
        fav = lookup.get(_norm(r["favourite"]))
        und = lookup.get(_norm(r["underdog"]))
        if fav is None or und is None:
            continue
        roles[mid] = dict(outcome=outcome, favourite=fav, underdog=und,
                          competition=r["competition"],
                          fav_rank=int(r["fav_rank"]),
                          und_rank=int(r["und_rank"]))
    return roles


_ROLES = None


def roles() -> dict:
    global _ROLES
    if _ROLES is None:
        _ROLES = build_role_map()
    return _ROLES


# ===========================================================================
# SECTION 5 -- Psi at orders l = 1, 2, 3
#
#   Psi^(k)_{t,t'}(V) := I(V_t; V_t') - sum_{|a|=k} I(X^a_t; V_t')
#
# (Rosas et al. 2020 eq. 10a.) Psi^(k) > 0 is SUFFICIENT, not necessary, for
# causal emergence. Positive = synergy-dominated; negative = redundancy-
# dominated, i.e. synchronised, well-drilled play.
#
# A window starting at s uses frames [s, s+60); V_t and X_t are [s, s+59) and
# V_t' is [s+1, s+60). One validity mask per window is applied to every macro
# feature and every order, so Psi^(1) and Psi^(2) are always estimated on the
# same sample set.
# ===========================================================================

def window_indices(s, window=None, lag=None):
    window = window or Config.WINDOW_SEC
    lag = Config.LAG if lag is None else lag
    return slice(s, s + window - lag), slice(s + lag, s + window)


def window_mask(pos, macro_all, s, window=None, lag=None):
    """Frames in [s, s+window-lag) usable for every feature and order."""
    t, tp = window_indices(s, window, lag)
    Xt = pos[t].reshape(pos[t].shape[0], -1)
    Vt, Vtp = macro_all[t], macro_all[tp]
    if len(Xt) == 0 or len(Vtp) != len(Xt):
        return None
    return (np.all(np.isfinite(Xt), 1) & np.all(np.isfinite(Vt), 1)
            & np.all(np.isfinite(Vtp), 1))


def subsets_for_order(order, pos_t, strategy=None, radius=None, cap=None,
                      seed=None):
    """Player subsets of size `order` for one window.

    Returns (subsets, n_total, scale) where scale = n_total/len(subsets) is the
    Horvitz-Thompson-style factor used to estimate the complete sum from a
    subsample. Orders 1 and 2 are never subsampled.

    The proximal strategy keeps triples whose members are pairwise within
    `radius` metres of one another (mean position over the window). Note that
    the rescaling assumes the retained triples are representative, which
    proximity selection deliberately violates -- nearby players share more
    information, so the scaled sum runs high and Psi^(3) correspondingly low.
    `--stage emergence` measures that bias against the exhaustive sum."""
    strategy = strategy or Config.L3_STRATEGY
    radius = Config.L3_RADIUS if radius is None else radius
    cap = Config.L3_MAX if cap is None else cap
    seed = Config.L3_SEED if seed is None else seed

    P = pos_t.shape[1]
    allc = list(itertools.combinations(range(P), order))
    if order <= 2 or strategy == "full" or len(allc) <= cap:
        return allc, len(allc), 1.0

    if strategy == "proximal":
        mp = np.nanmean(pos_t, axis=0)
        D = np.sqrt(((mp[:, None, :] - mp[None, :, :]) ** 2).sum(-1))
        sel = [c for c in allc
               if all(np.isfinite(D[a, b]) and D[a, b] <= radius
                      for a, b in itertools.combinations(c, 2))]
        if not sel:                                  # dispersed frame
            sel = allc
    else:
        sel = allc

    if len(sel) > cap:
        rs = np.random.default_rng(seed)
        sel = [sel[i] for i in rs.choice(len(sel), cap, replace=False)]
    return sel, len(allc), len(allc) / max(len(sel), 1)


def window_terms(pos, macro, s, ok=None, orders=None, k=None, features=None,
                 window=None, lag=None, seed=0):
    """All Psi terms for one window -> ({feature: {metric: value}}, meta).

    Micro-side Chebyshev distances are computed once and shared across every
    macro feature and every order."""
    orders = orders or Config.ORDERS
    k = k or Config.K_DEFAULT
    features = features or Config.MACRO_FEATURES
    window = window or Config.WINDOW_SEC
    lag = Config.LAG if lag is None else lag

    t, tp = window_indices(s, window, lag)
    if ok is None:
        ok = window_mask(pos, macro["V_all"], s, window, lag)
    n_raw = window - lag
    if ok is None or ok.sum() < Config.MIN_SAMPLES:
        n_eff = 0 if ok is None else int(ok.sum())
        return ({f: {} for f in features},
                dict(n_eff=n_eff, frac_valid=n_eff / n_raw))

    rng = np.random.default_rng(seed + int(s))
    pos_t = pos[t][ok]                              # raw, for proximity
    n_eff, P = len(pos_t), pos_t.shape[1]
    flat = zscore(pos_t.reshape(n_eff, -1), rng=rng)

    # per-coordinate |x_i - x_j|; subset distances are the elementwise max
    Dc = np.abs(flat.T[:, :, None] - flat.T[:, None, :])       # (2P, n, n)

    blocks, Dxs, info = [], [], {}
    for l in orders:
        sel, n_tot, scale = subsets_for_order(l, pos_t)
        info[l] = (len(sel), n_tot, scale)
        blocks.append((f"l{l}", len(sel)))
        for c in sel:
            Dxs.append(Dc[sum(([2 * i, 2 * i + 1] for i in c), [])].max(0))
    blocks.append(("l1c", 2 * P))          # Sprint-3 coordinate micro variables
    Dxs.extend(Dc[j] for j in range(2 * P))
    Dxs = np.stack(Dxs)

    res = {}
    for f in features:
        Vt_raw, Vtp_raw = macro[f][t][ok], macro[f][tp][ok]
        ref = np.vstack([Vt_raw, Vtp_raw])          # one scaler for V_t, V_t'
        Vt = zscore(Vt_raw, ref=ref, rng=rng)
        Vtp = zscore(Vtp_raw, ref=ref, rng=rng)
        Dy = cheb_dist(Vtp)
        mis = ksg_batch_from_dist(Dxs, Dy, k)

        r, off = dict(I_macro=ksg_from_dist(cheb_dist(Vt), Dy, k)), 0
        for name, n in blocks:
            tot = float(np.nansum(mis[off:off + n]))
            off += n
            if name == "l1c":
                r["I_micro_l1c"] = tot
                r["psi_l1c"] = r["I_macro"] - tot
            else:
                l = int(name[1:])
                n_sel, n_tot, scale = info[l]
                r[f"I_micro_l{l}"] = tot * scale
                r[f"psi_l{l}"] = r["I_macro"] - tot * scale
                r[f"n_terms_l{l}"] = n_sel
                r[f"n_total_l{l}"] = n_tot
                if scale != 1.0:
                    r[f"psi_l{l}_raw"] = r["I_macro"] - tot
        r["H_macro"] = kl_entropy(Vtp_raw, k)       # raw units, for DC
        res[f] = r
    return res, dict(n_eff=n_eff, frac_valid=n_eff / n_raw)


def sprint3_parity(pos, macro_fast, s, ok, k=None, window=None, lag=None):
    """Reproduce Sprint 3 exactly for one window: raw (un-standardised) inputs,
    tree estimator, 2P coordinate micro variables, V_fast macro. Used only for
    the reconciliation check."""
    k = k or Config.K_DEFAULT
    t, tp = window_indices(s, window, lag)
    if ok is None or ok.sum() < Config.MIN_SAMPLES:
        return np.nan, np.nan
    Xt = pos[t].reshape(pos[t].shape[0], -1)[ok]
    Vt, Vtp = macro_fast[t][ok], macro_fast[tp][ok]
    I_macro = ksg_mi(Vt, Vtp, k)
    I_micro = sum(ksg_mi(Xt[:, i:i + 1], Vtp, k) for i in range(Xt.shape[1]))
    return I_macro, I_macro - I_micro


def window_baseline_mi(X, V, s, window=None, lag=None, k=None):
    """Sprint-3 baseline MI terms for one window: (I_macro, I_micro_sum, psi).

    Micro variables here are the 2P individual COORDINATES, which is the
    Sprint-3 convention; Sprint 4 uses player subsets."""
    window = window or Config.WINDOW_SEC
    lag = Config.LAG if lag is None else lag
    k = k or Config.K_DEFAULT
    Vt, Vt1 = V[s:s + window - lag], V[s + lag:s + window]
    Xt = X[s:s + window - lag]
    ok = (np.all(np.isfinite(Vt), 1) & np.all(np.isfinite(Vt1), 1)
          & np.all(np.isfinite(Xt), 1))
    if ok.sum() < 4 * k:
        return (np.nan, np.nan, np.nan)
    Vt, Vt1, Xt = Vt[ok], Vt1[ok], Xt[ok]
    I_macro = ksg_mi(Vt, Vt1, k)
    I_micro = sum(ksg_mi(Xt[:, i:i + 1], Vt1, k) for i in range(Xt.shape[1]))
    return (I_macro, I_micro, I_macro - I_micro)


def window_slices(T, window=None, step=None):
    window = window or Config.WINDOW_SEC
    step = step or Config.STEP_SEC
    return [(s, s + window) for s in range(0, max(0, T - window + 1), step)]


# ===========================================================================
# SECTION 6 -- Novel metrics: EAI, DC, SRR
#
# All three are emitted exactly as defined in Giant_Killing §3.4, each with a
# stabilised companion, because the literal definitions degenerate here.
#
# EAI = (Psi_u - Psi_f)/(Psi_u + Psi_f + eps) is bounded in [-1,1] only when
#   both Psi are non-negative. Psi < 0 almost everywhere, so the denominator is
#   a sum of two same-signed large numbers that passes through zero exactly
#   when the teams are comparable -- where the hypothesis locates the
#   interesting behaviour. The stabilised form uses |Psi_u| + |Psi_f| + eps.
#
# DC = I(V_u;V_f)/H(V_f) divides by a DIFFERENTIAL entropy: scale-dependent
#   and routinely negative here (~ -5 nats), which flips the sign and destroys
#   the "fraction of entropy explained" reading. I and H are stored separately;
#   dc_ic = 1 - exp(-2I) is bounded in [0,1) and reparameterisation-invariant.
#
# SRR = max(0,Psi)/(|min(0,Psi)| + eps) at a single Psi collapses to 0 or to
#   Psi/eps ~ 1e9; it is a ratio statistic requiring a SET of values. Per
#   window, rolling-block and whole-match versions are all emitted.
# ===========================================================================

def eai(psi_u, psi_f, eps=None):
    """Literal EAI, its denominator, and a bounded stabilised variant."""
    eps = Config.EPS if eps is None else eps
    psi_u, psi_f = np.asarray(psi_u, float), np.asarray(psi_f, float)
    den = psi_u + psi_f + eps
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (psi_u - psi_f) / den
        stab = (psi_u - psi_f) / (np.abs(psi_u) + np.abs(psi_f) + eps)
    return raw, den, stab


def disruption(V_und_t, V_fav_t, k=None):
    """(I(V_u;V_f), H(V_f), H(V_u)) over one time-aligned window.

    MI on standardised inputs; entropies in raw units, as DC defines them."""
    k = k or Config.K_DEFAULT
    if len(V_und_t) < Config.MIN_SAMPLES or len(V_fav_t) != len(V_und_t):
        return np.nan, np.nan, np.nan
    ic = ksg_from_dist(cheb_dist(zscore(V_und_t)),
                       cheb_dist(zscore(V_fav_t)), k)
    return ic, kl_entropy(V_fav_t, k), kl_entropy(V_und_t, k)


def srr(psi, eps=None):
    """Literal per-value SRR."""
    eps = Config.EPS if eps is None else eps
    p = np.asarray(psi, float)
    return np.maximum(0.0, p) / (np.abs(np.minimum(0.0, p)) + eps)


def srr_block(psi, eps=None):
    """Block SRR: sum of positive Psi over |sum of negative Psi|."""
    eps = Config.EPS if eps is None else eps
    p = np.asarray(psi, float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    return p[p > 0].sum() / (abs(p[p < 0].sum()) + eps)


def srr_rolling(psi, win=None, eps=None):
    """Centred rolling block-SRR."""
    win = Config.SRR_ROLL if win is None else win
    eps = Config.EPS if eps is None else eps
    s = pd.Series(np.asarray(psi, float))
    mp = max(3, win // 3)
    pos = s.clip(lower=0).rolling(win, center=True, min_periods=mp).sum()
    neg = (-s.clip(upper=0)).rolling(win, center=True, min_periods=mp).sum()
    return (pos / (neg + eps)).to_numpy()


# ===========================================================================
# SECTION 7 -- Diagnostics
#
# Six checks, all written into the output as columns so nothing is silently
# dropped: Sprint-3 reconciliation, sample adequacy, estimator sanity, order
# monotonicity, MAD outliers, metric stability.
# ===========================================================================

def add_diagnostics(df, outlier_z=None, neg_tol=0.05):
    outlier_z = Config.OUTLIER_Z if outlier_z is None else outlier_z
    df = df.copy()

    # estimator sanity: KSG can return slightly negative MI on near-independent
    # data, but a materially negative I_macro signals a degenerate window
    df["neg_mi_flag"] = df["I_macro"] < -neg_tol

    # order monotonicity: the micro sum grows with the number of subsets, so
    # psi_l1 >= psi_l2 >= psi_l3 is expected
    mono = pd.Series(True, index=df.index)
    for a, b in [(1, 2), (2, 3)]:
        ca, cb = f"psi_l{a}", f"psi_l{b}"
        if ca in df and cb in df:
            pair = df[[ca, cb]].dropna()
            bad = pair[ca] < pair[cb] - 1e-9
            mono.loc[bad.index] &= ~bad
    df["mono_violation"] = ~mono

    # robust (MAD) z-score of psi_l1 within each trajectory
    def _z(g):
        x = g.to_numpy(float)
        med = np.nanmedian(x)
        scale = 1.4826 * np.nanmedian(np.abs(x - med))
        return pd.Series(np.where(scale > 0, (x - med) / scale, 0.0),
                         index=g.index)
    df["psi_l1_z"] = df.groupby(["team", "macro_feature"])["psi_l1"].transform(_z)
    df["is_outlier"] = df["psi_l1_z"].abs() > outlier_z

    df["eai_unstable"] = df["eai_denom_l1"].abs() < Config.EAI_DENOM_MIN
    df["dc_unstable"] = df["h_fav"].abs() < 1.0
    df["estimate_ok"] = df[["I_macro", "psi_l1"]].notna().all(axis=1)
    return df


def diagnostic_summary(df) -> dict:
    rec = df.loc[df.macro_feature == "V_fast", "recon_delta"].dropna()
    return dict(
        rows=int(len(df)), windows=int(df.window_start.nunique()),
        teams=int(df.team.nunique()),
        frac_estimated=float(df.estimate_ok.mean()),
        median_n_eff=float(df.n_eff.median()),
        max_recon_delta=float(rec.abs().max()) if len(rec) else np.nan,
        n_recon_checked=int(len(rec)),
        n_neg_mi=int(df.neg_mi_flag.sum()),
        n_mono_violation=int(df.mono_violation.sum()),
        n_outliers=int(df.is_outlier.sum()),
        n_eai_unstable=int(df.eai_unstable.sum()),
        n_dc_unstable=int(df.dc_unstable.sum()),
        psi_l1_median=float(df.psi_l1.median()),
        psi_l2_median=float(df.psi_l2.median()) if "psi_l2" in df else np.nan,
        psi_l3_median=float(df.psi_l3.median()) if "psi_l3" in df else np.nan,
    )


# ===========================================================================
# STAGE: validate
#
# Sprint 3's "validation report" deliverable. Every correctness check the
# notebooks performed, in one place. Exits non-zero if any assertion fails.
# ===========================================================================

def stage_validate() -> pd.DataFrame:
    banner("STAGE: validate -- estimator correctness")
    rng = np.random.default_rng(0)
    checks = []

    def record(name, detail, ok):
        checks.append(dict(check=name, detail=detail, passed=bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # (a) KSG MI against the closed-form bivariate Gaussian, I = -0.5 ln(1-r^2).
    #
    # Tested on the MEAN over replicates, not a single draw. KSG is
    # asymptotically unbiased, but at N=4000 a single estimate has a standard
    # deviation of ~0.013 nats, so a one-draw tolerance of 1e-2 fails on
    # sampling noise roughly half the time and says nothing about the
    # estimator. The bias is what the check should assert; the spread is
    # reported alongside so the reader can see the difference.
    rows = []
    for rho in [0.0, 0.3, 0.6, 0.9]:
        est = []
        for _ in range(20):
            d = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=4000)
            est.append(ksg_mi(d[:, 0], d[:, 1]))
        est = np.array(est)
        true = -0.5 * np.log(1 - rho ** 2)
        rows.append(dict(rho=rho, MI_true=true, MI_mean=est.mean(),
                         bias=est.mean() - true, sd=est.std(), n_rep=20))
    mi_val = pd.DataFrame(rows)
    bias = mi_val.bias.abs().max()
    record("KSG vs closed-form Gaussian MI",
           f"max |bias| = {bias:.4f} nats over 20 replicates "
           f"(single-draw sd {mi_val.sd.max():.4f}; N=4000, k=4)", bias < 1e-2)

    # (b) Kozachenko-Leonenko entropy against 0.5 d log(2 pi e).
    #
    # This estimator has a genuine, systematic NEGATIVE bias that grows with
    # dimension -- the standard curse-of-dimensionality behaviour of kNN
    # entropy estimators, not sampling noise (bias -0.054 nats at d=5 against
    # a replicate sd of 0.016). It is recorded per dimension rather than
    # asserted away. Consequence for this project: H(V_fav) runs about -5
    # nats, so a -0.05 nat bias is ~1% and does not affect the RQ4 conclusion,
    # which is null and is reported through dc_ic rather than the raw ratio.
    rows = []
    for d_ in [1, 2, 3, 5]:
        est = np.array([kl_entropy(rng.standard_normal((8000, d_)))
                        for _ in range(10)])
        true = 0.5 * d_ * np.log(2 * np.pi * np.e)
        rows.append(dict(d=d_, H_true=true, H_mean=est.mean(),
                         bias=est.mean() - true, sd=est.std(), n_rep=10))
    h_val = pd.DataFrame(rows)
    herr = h_val.bias.abs().max()
    record("Kozachenko-Leonenko entropy",
           f"max |bias| = {herr:.4f} nats at d=5, growing with dimension "
           f"({', '.join(f'd={r.d}:{r.bias:+.3f}' for r in h_val.itertuples())})",
           herr < 0.1)
    h_val.to_csv(CFG.stat_dir / "validation_kl_entropy_bias.csv", index=False)

    # (c) brute force == k-d tree: the fast path is not an approximation
    d = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], size=300)
    b, t = ksg_mi_bruteforce(d[:, 0], d[:, 1]), ksg_mi(d[:, 0], d[:, 1])
    record("brute force == k-d tree",
           f"|diff| = {abs(b - t):.2e} (N=300)", np.isclose(b, t))

    # (d) matrix and batched estimators are the same estimator
    A = rng.standard_normal((120, 6))
    B = A[:, :2] * 0.7 + rng.standard_normal((120, 2)) * 0.5
    Dy = cheb_dist(B)
    tree = np.array([ksg_mi(A[:, [i]], B) for i in range(6)] + [ksg_mi(A, B)])
    mat = np.array([ksg_from_dist(cheb_dist(A[:, [i]]), Dy) for i in range(6)]
                   + [ksg_from_dist(cheb_dist(A), Dy)])
    batch = ksg_batch_from_dist(
        np.stack([cheb_dist(A[:, [i]]) for i in range(6)] + [cheb_dist(A)]), Dy)
    record("tree == matrix == batched",
           f"max |diff| = {max(np.abs(tree - mat).max(), np.abs(tree - batch).max()):.2e}",
           np.allclose(tree, mat) and np.allclose(tree, batch))

    # (e) the tie pathology that produced inf, and the fix
    tied = np.repeat(np.array([[0.0], [1.0], [2.0]]), 20, axis=0)
    y = rng.standard_normal((60, 1))
    naive, safe = ksg_mi(tied, y), ksg_from_dist(cheb_dist(tied), cheb_dist(y))
    record("piecewise-constant input (the V_cvel case)",
           f"tree estimator -> {naive:.4f}, tie-safe -> {safe:.4f}",
           np.isfinite(safe))

    # (f) vectorised Onnela clustering == networkx.average_clustering
    try:
        import networkx as nx
        r = np.random.default_rng(4)
        W = r.random((5, 9, 9))
        W = (W + W.transpose(0, 2, 1)) / 2
        W[:, np.arange(9), np.arange(9)] = 0.0
        W[W < 0.25] = 0.0
        mine = onnela_clustering(W)
        ref = np.array([nx.average_clustering(nx.from_numpy_array(W[t]),
                                              weight="weight") for t in range(5)])
        diff = np.abs(mine - ref).max()
        record("vectorised Onnela == networkx",
               f"max |diff| = {diff:.2e}", diff < 1e-10)
    except ImportError:
        record("vectorised Onnela == networkx", "networkx not installed; SKIPPED", True)

    # (g) convergence toward the true MI as N grows
    rho, true = 0.6, -0.5 * np.log(1 - 0.6 ** 2)
    rng2 = np.random.default_rng(7)
    prof = []
    for N in [250, 500, 1000, 2000, 4000]:
        for k in [2, 4, 8]:
            dd = rng2.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=N)
            prof.append(dict(N=N, k=k, est=ksg_mi(dd[:, 0], dd[:, 1], k=k)))
    prof = pd.DataFrame(prof)
    far = (prof[prof.N >= 2000].est - true).abs().max()
    record("convergence at large N",
           f"max |error| at N>=2000 = {far:.4f} nats", far < 5e-2)

    out = pd.DataFrame(checks)
    CFG.stat_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(CFG.stat_dir / "validation_report.csv", index=False)
    print(f"\n  {int(out.passed.sum())}/{len(out)} checks passed "
          f"-> {CFG.stat_dir / 'validation_report.csv'}")
    if not out.passed.all():
        raise SystemExit("validation FAILED")
    return out


# ===========================================================================
# STAGE: preprocess  (Sprint 2)
#
# Writes the tidy 1 Hz tracking table and per-team feature bundles for matches
# with local 360 data. NOTE: this is a side artefact for inspection. The
# critical path is ksg -> emergence -> stats, each of which rebuilds the
# tracking itself (build_bundle) for exactly the matches it needs.
# ===========================================================================

def stage_preprocess(limit=None) -> None:
    banner("STAGE: preprocess (Sprint 2) -- 1 Hz tracking + feature bundles")
    ids = sorted(local_360_ids())
    if limit:
        ids = ids[:limit]
    step(f"{len(ids)} matches with local 360 data")
    CFG.prep_dir.mkdir(parents=True, exist_ok=True)

    done, skipped = 0, 0
    t0 = time.time()
    for i, mid in enumerate(ids, 1):
        try:
            bundle = build_bundle(mid, common_grid=True, allow_fetch=True)
            if bundle is None:
                skipped += 1
                continue
            ev = read_events(mid, allow_fetch=True)
            e = events_table(ev)
            tl = (e[e.is_shot | e.is_goal]
                  .assign(t_int=lambda d: d.t_sec.round().astype(int))
                  .groupby("t_int")
                  .agg(n_shots=("is_shot", "sum"), n_goals=("is_goal", "sum"))
                  .reset_index())

            parts = []
            for tm, b in bundle.items():
                T, P = b["pos"].shape[0], b["pos"].shape[1]
                rec = pd.DataFrame(dict(
                    match_id=mid, team=tm,
                    player_idx=np.repeat(np.arange(1, P + 1), T),
                    t=np.tile(b["grid"], P),
                    x=b["pos"][:, :, 0].T.ravel(), y=b["pos"][:, :, 1].T.ravel(),
                    vx=b["vel"][:, :, 0].T.ravel(), vy=b["vel"][:, :, 1].T.ravel()))
                rec["speed"] = np.sqrt(rec.vx ** 2 + rec.vy ** 2)
                rec["dist_centre"] = np.sqrt(
                    (rec.x - Config.CENTRE[0]) ** 2 + (rec.y - Config.CENTRE[1]) ** 2)
                rec["t_int"] = rec.t.round().astype(int)
                rec = rec.merge(tl, on="t_int", how="left")
                rec[["n_shots", "n_goals"]] = (
                    rec[["n_shots", "n_goals"]].fillna(0).astype(int))
                parts.append(rec.drop(columns="t_int"))

                safe = "".join(c if c.isalnum() else "_" for c in tm)
                np.savez_compressed(
                    CFG.prep_dir / f"{mid}_{safe}_features.npz",
                    times=b["grid"], pos=b["pos"], vel=b["vel"],
                    X_pos=b["pos"].reshape(T, P * 2),
                    X_dist=np.sqrt(((b["pos"] - Config.CENTRE) ** 2).sum(2)),
                    **{k: b["macro"][k] for k in
                       ("V_com", "V_dist", "V_cvel", "V_cdist")})

            pd.concat(parts, ignore_index=True).to_parquet(
                CFG.prep_dir / f"{mid}_tracking.parquet", index=False)
            done += 1
        except Exception as e:
            skipped += 1
            step(f"[{mid}] error: {e}")
        if i % 25 == 0 or i == len(ids):
            step(f"[{i}/{len(ids)}] {done} written, {skipped} skipped "
                 f"({time.time() - t0:.0f}s)")
    print(f"\n  preprocessed {done} matches -> {CFG.prep_dir}")


# ===========================================================================
# STAGE: ksg  (Sprint 3)
#
# Baseline mutual information for every 60 s window of every upset/control
# match, on the 3-d V_fast macro state with 2P coordinate micro variables.
# The resulting files are the WORK LIST for Sprint 4: which matches, which
# teams, which windows, and the upset/control label.
# ===========================================================================

def _ksg_one_match(mid, outcome, step_sec, k, out_dir):
    """Per-match KSG runner. Uses the Sprint-3 per-team grid convention."""
    from joblib import Parallel, delayed
    mid = int(mid)
    bundle = build_bundle(mid, common_grid=False, allow_fetch=True,
                          with_macro=True)
    if not bundle:
        return None
    parts = []
    for team, b in bundle.items():
        T = b["pos"].shape[0]
        X = b["pos"].reshape(T, b["pos"].shape[1] * 2)
        V = b["macro"]["V_fast"]
        slices = window_slices(len(V), step=step_sec)
        res = Parallel(n_jobs=Config.N_JOBS, prefer="threads")(
            delayed(window_baseline_mi)(X, V, s, k=k) for s, _ in slices)
        df = pd.DataFrame(res, columns=["I_macro", "I_micro_sum", "psi"])
        df.insert(0, "window_start", [s for s, _ in slices])
        df.insert(0, "team", team)
        parts.append(df)
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "outcome", outcome)
    out.insert(0, "match_id", mid)
    out.to_csv(out_dir / f"{mid}_ksg.csv", index=False)
    out.to_parquet(out_dir / f"{mid}_ksg.parquet", index=False)
    return out


def stage_ksg(limit=None, overwrite=False) -> None:
    banner("STAGE: ksg (Sprint 3) -- baseline MI for every window")
    CFG.ksg_dir.mkdir(parents=True, exist_ok=True)

    targets = resolve_targets()
    runnable = (targets[targets.has_360]
                .drop_duplicates("match_id").reset_index(drop=True))
    runnable["match_id"] = runnable.match_id.astype(int)
    step(f"fixtures: {len(targets)} | resolved with 360: {int(targets.has_360.sum())} "
         f"| unique runnable: {len(runnable)}")
    step(f"  upsets {int(runnable.outcome.eq('upset').sum())}, "
         f"controls {int(runnable.outcome.eq('control').sum())}")
    if (~targets.has_360).any():
        step("  unresolved / no local 360, by competition:")
        for comp, n in targets[~targets.has_360].groupby("competition").size().items():
            step(f"    {comp}: {n}")

    if limit:
        # Stratify, so a truncated run still has both outcomes and the stats
        # stage remains runnable. Taking a plain head() gives all upsets,
        # because the upset list is read first.
        n_up = max(1, limit // 2)
        todo = pd.concat([
            runnable[runnable.outcome == "upset"].head(n_up),
            runnable[runnable.outcome == "control"].head(limit - n_up),
        ]).reset_index(drop=True)
        step(f"--limit {limit}: stratified to "
             f"{int(todo.outcome.eq('upset').sum())} upsets / "
             f"{int(todo.outcome.eq('control').sum())} controls")
    else:
        todo = runnable
    step(f"processing {len(todo)} match(es), step={Config.STEP_SEC}s "
         f"-> {CFG.ksg_dir}/{{match_id}}_ksg.(csv|parquet)")

    done, skipped = 0, 0
    t0 = time.time()
    for i, r in enumerate(todo.itertuples(index=False), 1):
        mid = int(r.match_id)
        if not overwrite and (CFG.ksg_dir / f"{mid}_ksg.csv").exists():
            done += 1
            continue
        try:
            out = _ksg_one_match(mid, r.outcome, Config.STEP_SEC,
                                 Config.K_DEFAULT, CFG.ksg_dir)
            if out is None:
                skipped += 1
                step(f"[{i}/{len(todo)}] {mid} skipped (no data)")
            else:
                done += 1
                if i % 10 == 0 or i == len(todo):
                    step(f"[{i}/{len(todo)}] {mid} ({r.outcome}): {len(out)} rows "
                         f"({time.time() - t0:.0f}s cumulative)")
        except Exception as e:
            skipped += 1
            step(f"[{i}/{len(todo)}] {mid} error: {e}")

    n = len(list(CFG.ksg_dir.glob("*_ksg.csv")))
    print(f"\n  {done} matches written, {skipped} skipped; "
          f"{n} csv (+ matching parquet) in {CFG.ksg_dir}")


# ===========================================================================
# STAGE: emergence  (Sprint 4)
#
# Psi at orders 1, 2, 3 for every macro feature and every window named by the
# Sprint-3 manifest, plus EAI, DC, SRR and the diagnostics.
#
# Output grain: one row per (match_id, team, window_start, macro_feature).
# ===========================================================================

def ksg_match_ids() -> list:
    return sorted({int(Path(p).name.split("_")[0])
                   for p in glob.glob(str(CFG.ksg_dir / "*_ksg.csv"))
                   + glob.glob(str(CFG.ksg_dir / "*_ksg.parquet"))})


def load_ksg(mid):
    pq, csv = CFG.ksg_dir / f"{mid}_ksg.parquet", CFG.ksg_dir / f"{mid}_ksg.csv"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    return pd.read_csv(csv) if csv.exists() else None


def run_match(match_id, out_dir=None, orders=None, features=None,
              ho_stride=None, k=None, write=True):
    """Full Sprint-4 computation for one match -> (DataFrame, log)."""
    out_dir = out_dir or CFG.caus_dir
    orders = orders or Config.ORDERS
    features = features or Config.MACRO_FEATURES
    ho_stride = Config.HO_STRIDE if ho_stride is None else ho_stride
    k = k or Config.K_DEFAULT

    t_start = time.time()
    match_id = int(match_id)
    log = dict(match_id=match_id, status="ok", error=None)

    ksg = load_ksg(match_id)
    if ksg is None or ksg.empty:
        return None, dict(log, status="no_ksg_input")
    outcome = str(ksg.outcome.iloc[0])
    role = roles().get(match_id, {})
    log.update(outcome=outcome, competition=role.get("competition"))

    bundle = build_bundle(match_id, common_grid=True)
    if bundle is None:
        return None, dict(log, status="no_360_data")
    teams = [t for t in ksg.team.unique() if t in bundle]
    if not teams:
        return None, dict(log, status="team_name_mismatch",
                          error=f"ksg {sorted(ksg.team.unique())} vs "
                                f"bundle {sorted(bundle)}")

    # ---- per team, per window: Psi at every order and macro feature --------
    rows, offsets = [], {}
    for tm in teams:
        pos, macro, grid = bundle[tm]["pos"], bundle[tm]["macro"], bundle[tm]["grid"]
        off = _grid_offset(bundle, tm)
        offsets[tm] = off
        starts = np.sort(ksg.loc[ksg.team == tm, "window_start"].unique()).astype(int)
        starts = starts[(starts + off >= 0)
                        & (starts + off + Config.WINDOW_SEC <= len(grid))]

        opp = next((t for t in teams if t != tm), None)
        r_role = ("underdog" if role.get("underdog") == tm else
                  "favourite" if role.get("favourite") == tm else None)
        rank = (role.get("und_rank") if r_role == "underdog" else
                role.get("fav_rank") if r_role == "favourite" else np.nan)

        for i, s in enumerate(starts):
            s = int(s)
            ok = window_mask(pos, macro["V_all"], s + off)
            ords = orders if (i % ho_stride == 0) else (1,)
            feat_res, meta = window_terms(pos, macro, s + off, ok=ok,
                                          orders=ords, k=k, features=features,
                                          seed=match_id)
            par = (sprint3_parity(pos, macro["V_fast"], s + off, ok, k)
                   if i % Config.PARITY_EVERY == 0 else (np.nan, np.nan))
            t_abs = float(grid[s + off])
            for f in features:
                rows.append(dict(
                    match_id=match_id, outcome=outcome,
                    competition=role.get("competition"), team=tm, opponent=opp,
                    role=r_role, team_rank=rank, window_start=s, t_abs=t_abs,
                    minute=t_abs / 60.0, macro_feature=f, **meta, **feat_res[f],
                    psi_parity=(par[1] if f == "V_fast" else np.nan)))

    df = pd.DataFrame(rows)
    if df.empty:
        return None, dict(log, status="no_windows")
    for c in ["I_macro", "H_macro", "psi_l1", "psi_l2", "psi_l3", "psi_l1c",
              "psi_l3_raw", "I_micro_l1", "I_micro_l2", "I_micro_l3",
              "I_micro_l1c", "n_terms_l1", "n_terms_l2", "n_terms_l3",
              "n_total_l1", "n_total_l2", "n_total_l3"]:
        if c not in df:
            df[c] = np.nan

    # ---- Sprint-3 reconciliation -------------------------------------------
    kj = ksg.rename(columns={"psi": "psi_ksg", "I_macro": "I_macro_ksg"})[
        ["team", "window_start", "psi_ksg", "I_macro_ksg"]]
    df = df.merge(kj, on=["team", "window_start"], how="left")
    df.loc[df.macro_feature != "V_fast", ["psi_ksg", "I_macro_ksg"]] = np.nan
    df["recon_delta"] = df["psi_parity"] - df["psi_ksg"]

    # ---- DC: cross-team information on time-aligned windows ----------------
    und, fav = role.get("underdog"), role.get("favourite")
    df["i_cross"] = np.nan
    df["h_fav"] = np.nan
    df["h_und"] = np.nan
    if und in teams and fav in teams:
        gu, gf = bundle[und], bundle[fav]
        ou, of = offsets[und], offsets[fav]
        shared = np.intersect1d(
            df.loc[df.team == und, "window_start"].unique(),
            df.loc[df.team == fav, "window_start"].unique())
        dc_rows = []
        for f in features:
            for s in shared:
                s = int(s)
                _, tpu = window_indices(s + ou)
                _, tpf = window_indices(s + of)
                Vu, Vf = gu["macro"][f][tpu], gf["macro"][f][tpf]
                n = min(len(Vu), len(Vf))
                Vu, Vf = Vu[:n], Vf[:n]
                m = np.all(np.isfinite(Vu), 1) & np.all(np.isfinite(Vf), 1)
                ic, hf, hu = (disruption(Vu[m], Vf[m], k)
                              if m.sum() >= Config.MIN_SAMPLES
                              else (np.nan, np.nan, np.nan))
                dc_rows.append(dict(window_start=s, macro_feature=f,
                                    i_cross=ic, h_fav=hf, h_und=hu))
        if dc_rows:
            df = (df.drop(columns=["i_cross", "h_fav", "h_und"])
                    .merge(pd.DataFrame(dc_rows),
                           on=["window_start", "macro_feature"], how="left"))
    with np.errstate(divide="ignore", invalid="ignore"):
        df["dc"] = df["i_cross"] / df["h_fav"]
        df["dc_rev"] = df["i_cross"] / df["h_und"]
        df["dc_ic"] = 1.0 - np.exp(-2.0 * df["i_cross"])

    # ---- EAI: underdog minus favourite, same window & macro feature --------
    for l in orders:
        df[f"eai_l{l}"] = np.nan
        df[f"eai_stab_l{l}"] = np.nan
    df["eai_denom_l1"] = np.nan
    if und in teams and fav in teams:
        key = ["window_start", "macro_feature"]
        u = df[df.team == und].set_index(key)
        v = df[df.team == fav].set_index(key)
        idx = u.index.intersection(v.index)
        if len(idx):
            df = df.set_index(key)
            for l in orders:
                raw, den, stab = eai(u.loc[idx, f"psi_l{l}"].to_numpy(),
                                     v.loc[idx, f"psi_l{l}"].to_numpy())
                cols = {f"eai_l{l}": raw, f"eai_stab_l{l}": stab}
                if l == 1:
                    cols["eai_denom_l1"] = den
                m = pd.DataFrame(cols, index=idx)
                for c in m.columns:      # same value on both teams' rows
                    df[c] = df[c].fillna(m[c])
            df = df.reset_index()

    # ---- SRR: literal, rolling, and match-block ----------------------------
    df = df.sort_values(["team", "macro_feature", "window_start"]).reset_index(drop=True)
    for l in orders:
        c = f"psi_l{l}"
        df[f"srr_l{l}"] = srr(df[c].to_numpy())
        g = df.groupby(["team", "macro_feature"])[c]
        df[f"srr_roll_l{l}"] = g.transform(
            lambda s: pd.Series(srr_rolling(s.to_numpy()), index=s.index))
        df[f"srr_match_l{l}"] = g.transform(srr_block)

    df = add_diagnostics(df)
    log.update(diagnostic_summary(df))
    log.update(seconds=round(time.time() - t_start, 2),
               teams_processed=teams, offsets=offsets)

    if write:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(out_dir) / f"{match_id}_caus.csv", index=False)
        try:
            df.to_parquet(Path(out_dir) / f"{match_id}_caus.parquet", index=False)
        except Exception as e:
            log["parquet_error"] = str(e)
        json.dump(log, open(Path(out_dir) / f"{match_id}_caus_log.json", "w"),
                  indent=2, default=str)
    return df, log


def _emergence_worker(mid, root, quick, out_dir, overwrite):
    """Process-pool worker. Rebuilds config in the child (module globals do not
    survive a fork on all platforms)."""
    global CFG
    if CFG is None:
        CFG = Config(Path(root), quick=quick)
    csv_path = Path(out_dir) / f"{mid}_caus.csv"
    if not overwrite and csv_path.exists():
        try:
            return json.load(open(Path(out_dir) / f"{mid}_caus_log.json"))
        except Exception:
            return dict(match_id=mid, status="skipped_exists")
    try:
        _, log = run_match(mid, out_dir=out_dir)
        return log
    except Exception as e:
        return dict(match_id=mid, status="error", error=str(e),
                    traceback=traceback.format_exc()[-800:])


def measure_l3_bias(match_id, n_windows=12) -> pd.DataFrame:
    """Bias of the proximity-subsampled l=3 sum against the exhaustive sum.

    Proximal selection deliberately violates the representativeness the
    Horvitz-Thompson rescaling assumes, so this is reported rather than
    assumed away."""
    b = build_bundle(match_id, common_grid=True)
    if b is None:
        return pd.DataFrame()
    tm = sorted(b)[0]
    pos, macro, grid = b[tm]["pos"], b[tm]["macro"], b[tm]["grid"]
    starts = np.linspace(0, max(1, len(grid) - Config.WINDOW_SEC - 1),
                         n_windows).astype(int)
    rows = []
    for st in starts:
        ok = window_mask(pos, macro["V_all"], int(st))
        if ok is None or ok.sum() < Config.MIN_SAMPLES:
            continue
        pt = pos[window_indices(int(st))[0]][ok]
        fl = zscore(pt.reshape(len(pt), -1),
                    rng=np.random.default_rng(int(st)))
        Dc = np.abs(fl.T[:, :, None] - fl.T[:, None, :])
        Dy = cheb_dist(zscore(macro["V_all"][window_indices(int(st))[1]][ok]))

        def _sum(sel):
            return float(ksg_batch_from_dist(np.stack(
                [Dc[sum(([2 * i, 2 * i + 1] for i in c), [])].max(0)
                 for c in sel]), Dy).sum())

        full, _, _ = subsets_for_order(3, pt, strategy="full")
        prox, n_tot, sc = subsets_for_order(3, pt, strategy="proximal")
        sf, sp = _sum(full), _sum(prox) * sc
        rows.append(dict(window_start=int(st), n_full=len(full),
                         n_prox=len(prox), sum_full=sf, sum_prox_scaled=sp,
                         rel_err=(sp - sf) / abs(sf) if sf else np.nan))
    return pd.DataFrame(rows)


def stage_emergence(limit=None, overwrite=False, n_jobs=None) -> pd.DataFrame:
    banner("STAGE: emergence (Sprint 4) -- Psi at orders 1, 2, 3")
    CFG.caus_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = n_jobs or Config.MATCH_JOBS

    ids = ksg_match_ids()
    if not ids:
        raise SystemExit(f"no Sprint-3 input in {CFG.ksg_dir}; run --stage ksg first")
    if limit:
        ids = ids[:limit]
    step(f"{len(ids)} matches from {CFG.ksg_dir}")
    step(f"orders={Config.ORDERS}, features={len(Config.MACRO_FEATURES)}, "
         f"k={Config.K_DEFAULT}, window={Config.WINDOW_SEC}s")
    step(f"l=3 subsampling: {Config.L3_STRATEGY} (radius {Config.L3_RADIUS} m, "
         f"cap {Config.L3_MAX}) | {n_jobs} matches in parallel")

    rm = roles()
    step(f"role map: {len(rm)} fixtures; covers "
         f"{sum(m in rm for m in ids)}/{len(ids)} matches")

    # l=3 subsampling bias, measured on the first match
    bias = measure_l3_bias(ids[0])
    if len(bias):
        bias.to_csv(CFG.caus_dir / "l3_subsampling_bias.csv", index=False)
        step(f"l=3 subsampling bias: median relative error "
             f"{bias.rel_err.median():+.2%} "
             f"(range {bias.rel_err.min():+.2%} to {bias.rel_err.max():+.2%}); "
             f"mean {bias.n_prox.mean():.0f} of {int(bias.n_full.iloc[0])} triples kept")

    from joblib import Parallel, delayed
    t0 = time.time()
    logs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_emergence_worker)(int(m), str(CFG.root), CFG.quick,
                                   str(CFG.caus_dir), overwrite) for m in ids)
    log_df = pd.DataFrame(logs)
    log_df["total_runtime_s"] = round(time.time() - t0, 1)
    log_df.to_csv(CFG.caus_dir / "computation_log.csv", index=False)

    print(f"\n  {len(ids)} matches in {(time.time() - t0) / 60:.1f} min "
          f"on {n_jobs} workers")
    print("  " + log_df.status.value_counts().to_string().replace("\n", "\n  "))

    ok = log_df[log_df.status == "ok"]
    if len(ok) and "rows" in ok:
        print(f"\n  rows written : {int(ok.rows.sum()):,}")
        print(f"  windows      : {int(ok.windows.sum()):,}")
        print(f"  compute      : {ok.seconds.sum() / 60:.1f} core-min")
        if ok.max_recon_delta.notna().any():
            print(f"  max Sprint-3 reconciliation error: "
                  f"{ok.max_recon_delta.max():.2e} nats over "
                  f"{int(ok.n_recon_checked.sum()):,} checked windows")
        print(f"  flagged      : {int(ok.n_outliers.sum()):,} outliers, "
              f"{int(ok.n_neg_mi.sum()):,} negative-MI, "
              f"{int(ok.n_mono_violation.sum()):,} order-monotonicity, "
              f"{int(ok.n_eai_unstable.sum()):,} unstable EAI, "
              f"{int(ok.n_dc_unstable.sum()):,} unstable DC")
    if (log_df.status == "error").any():
        print("\n  ERRORS:")
        show(log_df[log_df.status == "error"][["match_id", "error"]])
    return log_df


# ===========================================================================
# STAGE: stats  (Sprint 5)
#
# The five research-question analyses plus the sensitivity battery. Nulls are
# reported as findings; every test runs whether or not it is significant, and
# multiplicity correction is applied to the whole family.
# ===========================================================================

class Results:
    """Collects every results table and writes it to tables/."""

    def __init__(self, tab_dir: Path):
        self.tab_dir = tab_dir
        self.tables = {}

    def save(self, name, df, note=""):
        self.tables[name] = df
        df.to_csv(self.tab_dir / f"{name}.csv", index=False)
        if note:
            step(f"[{name}] {note}")
        return df


def hedges_g(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1))
                 / (nx + ny - 2))
    return ((x.mean() - y.mean()) / sp * (1 - 3 / (4 * (nx + ny) - 9))
            if sp > 0 else np.nan)


def build_match_metadata(match_ids) -> pd.DataFrame:
    idx = match_index()
    rows = []
    for mid in match_ids:
        m = idx.get(int(mid))
        if not m:
            continue
        rows.append(dict(match_id=int(mid), match_date=m.get("match_date"),
                         competition_sb=m["competition"]["competition_name"],
                         season=str(m["season"]["season_name"]),
                         home=m["home_team"]["home_team_name"],
                         away=m["away_team"]["away_team_name"],
                         home_score=m.get("home_score"),
                         away_score=m.get("away_score")))
    d = pd.DataFrame(rows)
    d["match_date"] = pd.to_datetime(d.match_date)
    return d


def goal_timeline(mid):
    """[(t_sec, scoring_team)] on the same clock as the Psi grid."""
    p = CFG.sb_events / f"{mid}.json"
    if not p.exists():
        return None
    out = []
    for e in json.load(open(p)):
        t = event_clock(e["period"], e["minute"], e["second"])
        if e.get("shot", {}).get("outcome", {}).get("name") == "Goal":
            out.append((t, e["team"]["name"]))
        elif e["type"]["name"] == "Own Goal For":
            out.append((t, e["team"]["name"]))
    return sorted(out)


def attach_game_state(panel, goals):
    """Goal difference from each team's perspective at the START of a window.

    The proposal rates confounding by game state a HIGH-probability risk, and
    it is structural here: upsets end with the underdog ahead by construction,
    so any raw upset-vs-control difference partly measures scoreline."""
    key = panel[["match_id", "team", "opponent", "t_abs"]].drop_duplicates()
    gd = np.zeros(len(key))
    mids, tms = key.match_id.to_numpy(), key.team.to_numpy()
    ops, ts = key.opponent.to_numpy(), key.t_abs.to_numpy()
    for i in range(len(key)):
        g = goals.get(int(mids[i]))
        if not g:
            continue
        f = sum(1 for t, tm in g if t <= ts[i] and tm == tms[i])
        a = sum(1 for t, tm in g if t <= ts[i] and tm == ops[i])
        gd[i] = f - a
    return panel.merge(key.assign(goal_diff=gd),
                       on=["match_id", "team", "opponent", "t_abs"], how="left")


def load_corpus_and_panel():
    """Sprint-4 output -> (CORPUS all windows, PANEL non-overlapping + covariates).

    Sprint 4 emits Psi every 10 s from a 60 s window, so consecutive rows share
    50 of 60 seconds. Treating them as independent would inflate the effective
    sample size roughly sixfold. Every model uses non-overlapping windows only;
    residual within-match dependence is absorbed by a random intercept."""
    cols = ["match_id", "outcome", "competition", "team", "opponent", "role",
            "team_rank", "window_start", "t_abs", "minute", "macro_feature",
            "n_eff", "I_macro", "psi_l1", "psi_l2", "psi_l3",
            "eai_stab_l1", "eai_stab_l2", "eai_stab_l3", "eai_l1",
            "dc_ic", "dc", "i_cross", "h_fav", "is_outlier", "estimate_ok"]
    paths = sorted(glob.glob(str(CFG.caus_dir / "*_caus.parquet")))
    if not paths:
        raise SystemExit(f"no Sprint-4 output in {CFG.caus_dir}; "
                         "run --stage emergence first")
    t0 = time.time()
    corpus = pd.concat([pd.read_parquet(p, columns=cols) for p in paths],
                       ignore_index=True)
    step(f"Sprint-4 corpus: {len(corpus):,} rows, "
         f"{corpus.match_id.nunique()} matches ({time.time() - t0:.1f}s)")

    panel = corpus[(corpus.window_start % Config.THIN_STEP == 0)
                   & corpus.estimate_ok].copy()
    step(f"after thinning: {len(panel):,} rows "
         f"({len(panel) / len(corpus):.1%} retained)")

    match_ids = sorted(panel.match_id.unique())
    meta = build_match_metadata(match_ids)
    meta.to_parquet(CFG.stat_dir / "match_metadata.parquet", index=False)
    step(f"metadata for {len(meta)}/{len(match_ids)} matches | "
         f"{meta.match_date.min().date()} to {meta.match_date.max().date()}")

    goals = {int(m): goal_timeline(int(m)) for m in match_ids}
    ng = sum(len(v) for v in goals.values() if v)
    step(f"goal timelines: {sum(v is not None for v in goals.values())} matches, "
         f"{ng} goals")
    panel = attach_game_state(panel, goals)
    panel["game_state"] = pd.cut(panel.goal_diff, [-99, -0.5, 0.5, 99],
                                 labels=["trailing", "level", "leading"])

    panel["upset"] = (panel.outcome == "upset").astype(int)
    panel["und_c"] = np.where(panel.role == "underdog", 0.5, -0.5)
    panel["time_n"] = panel.groupby("match_id").t_abs.transform(
        lambda s: (s - s.min()) / max(s.max() - s.min(), 1e-9))
    panel["time_c"] = panel.time_n - 0.5      # centred -> beta_1 is the AVERAGE
    panel["gd_c"] = panel.goal_diff - panel.goal_diff.mean()
    panel = panel.merge(meta[["match_id", "match_date", "season", "competition_sb"]],
                        on="match_id", how="left")
    panel.to_parquet(CFG.stat_dir / "analysis_panel.parquet", index=False)
    step(f"analysis panel -> analysis_panel.parquet "
         f"({len(panel):,} rows, {panel.shape[1]} cols)")
    step(f"matches by outcome: {panel.groupby('outcome').match_id.nunique().to_dict()}")
    return corpus, panel, meta


# --- RQ1 -------------------------------------------------------------------
def diagnose_clock_void(panel, R):
    """Detect windows that span a void in the event clock.

    Under the legacy clock, StatsBomb period-2 events are pushed 45 minutes
    into the future, opening a ~42-minute stretch with no observations between
    the end of the first half and the shifted start of the second. The 1 Hz
    reconstruction interpolates linearly across it, so those windows contain no
    real data while reporting n_eff = 59 -- the existing diagnostics cannot see
    them. Perfectly smooth interpolated motion makes every player maximally
    informative about the centre of mass, which drives the micro sum to the
    estimator's per-term ceiling and Psi to its floor.

    This is reported, not silently repaired: dropping the windows would change
    every published number. Run the pipeline with --clock corrected to rebuild
    without the artefact."""
    banner("Diagnostic -- event-clock void (half-time artefact)")
    voids = {}
    for mid in panel.match_id.unique():
        p = CFG.sb_events / f"{mid}.json"
        if not p.exists():
            continue
        t = np.sort(np.array([event_clock(e["period"], e["minute"], e["second"])
                              for e in json.load(open(p))], dtype=float))
        if len(t) < 2:
            continue
        g = np.diff(t)
        i = int(np.argmax(g))
        voids[mid] = (t[i], t[i + 1], g[i])
    if not voids:
        step("no event files available; skipped")
        return panel, None

    panel = panel.copy()
    panel["void_lo"] = panel.match_id.map({k: v[0] for k, v in voids.items()})
    panel["void_hi"] = panel.match_id.map({k: v[1] for k, v in voids.items()})
    panel["in_clock_void"] = ((panel.t_abs + Config.WINDOW_SEC > panel.void_lo)
                              & (panel.t_abs < panel.void_hi))

    lengths = np.array([v[2] for v in voids.values()])
    frac = float(panel.in_clock_void.mean())
    lin = panel[panel.macro_feature.isin(["V_com", "V_dist", "V_fast", "V_all"])]
    per_term = (lin.I_macro - lin.psi_l1) / 10.0
    ceiling = float(per_term.max())
    at_ceil = per_term > ceiling - 0.01
    inv = lin.in_clock_void

    rows = [
        dict(metric="matches examined", value=len(voids)),
        dict(metric="median void length (s)", value=float(np.median(lengths))),
        dict(metric="share of windows overlapping the void", value=frac),
        dict(metric="mean n_eff inside the void", value=float(
            panel.loc[panel.in_clock_void, "n_eff"].mean())),
        dict(metric="mean psi_l1 inside the void", value=float(
            panel.loc[panel.in_clock_void, "psi_l1"].mean())),
        dict(metric="mean psi_l1 outside the void", value=float(
            panel.loc[~panel.in_clock_void, "psi_l1"].mean())),
        dict(metric="per-player MI ceiling (nats)", value=ceiling),
        dict(metric="share of terms at ceiling, overall", value=float(at_ceil.mean())),
        dict(metric="share of terms at ceiling, inside void",
             value=float(at_ceil[inv].mean())),
        dict(metric="share of terms at ceiling, outside void",
             value=float(at_ceil[~inv].mean())),
        dict(metric="share of saturated terms lying inside the void",
             value=float((at_ceil & inv).sum() / max(at_ceil.sum(), 1))),
    ]
    tab = R.save("diag_clock_void", pd.DataFrame(rows))
    show(tab, n=12)
    step(f"clock mode: {Config.CLOCK_MODE}")
    if Config.CLOCK_MODE == "legacy":
        print("  => under the legacy clock roughly 29% of analysis windows are pure")
        print("     interpolation across a ~42 min void, and they carry ~89% of all")
        print("     estimator saturation. Outside them saturation is ~4%, not ~24%.")
        print("     Re-run with --clock corrected to rebuild without the artefact.")
    return panel, tab


def rq1_emergence_signatures(panel, R):
    """Mixed-effects models of the Psi LEVEL.

        Psi = b0 + b1 Upset + b2 Time + b3 Upset:Time
                 + b4 Role + b5 Upset:Role + u_match + e

    Time and role are centred, so b1 is the AVERAGE upset effect rather than
    its value at kick-off for the favourite. All 54 tests form one family."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    banner("RQ1 -- emergence signatures (mixed-effects models)")

    def fit(df, feature, order, extra=""):
        d = df[df.macro_feature == feature].dropna(subset=[f"psi_l{order}"])
        if len(d) < 200:
            return None, []
        m = smf.mixedlm(f"psi_l{order} ~ upset*time_c + upset*und_c" + extra,
                        d, groups=d.match_id).fit(reml=True)
        sd_u = np.sqrt(float(m.cov_re.iloc[0, 0]))
        return m, [dict(feature=feature, order=order, term=t,
                        coef=m.params[t], se=m.bse[t], z=m.tvalues[t],
                        p=m.pvalues[t],
                        coef_std=m.params[t] / sd_u if sd_u > 0 else np.nan)
                   for t in m.params.index if "upset" in t]

    t0, rows, lrts = time.time(), [], []
    for f in Config.MACRO_FEATURES:
        for l in Config.ORDERS:
            rows += fit(panel, f, l)[1]
            d = panel[panel.macro_feature == f].dropna(subset=[f"psi_l{l}"])
            full = smf.mixedlm(f"psi_l{l} ~ upset*time_c + upset*und_c", d,
                               groups=d.match_id).fit(reml=False)
            red = smf.mixedlm(f"psi_l{l} ~ time_c + und_c", d,
                              groups=d.match_id).fit(reml=False)
            lr = 2 * (full.llf - red.llf)
            k = len(full.params) - len(red.params)
            lrts.append(dict(feature=f, order=l, chi2=lr, df=k,
                             p=stats.chi2.sf(lr, k)))

    rq1 = pd.DataFrame(rows)
    rq1["p_fdr"] = multipletests(rq1.p, method=Config.FDR_METHOD)[1]
    rq1_lrt = pd.DataFrame(lrts)
    rq1_lrt["p_fdr"] = multipletests(rq1_lrt.p, method=Config.FDR_METHOD)[1]
    step(f"fitted {len(Config.MACRO_FEATURES) * len(Config.ORDERS)} models "
         f"in {time.time() - t0:.1f}s")
    R.save("rq1_mixed_effects", rq1, f"{len(rq1)} coefficient tests")
    R.save("rq1_lrt", rq1_lrt, f"{len(rq1_lrt)} likelihood-ratio tests")

    print("\n  strongest upset effects:")
    show(rq1.nsmallest(6, "p")[["feature", "order", "term", "coef", "se", "p",
                                "p_fdr", "coef_std"]])
    print(f"\n  raw p < {Config.ALPHA}: {(rq1.p < Config.ALPHA).sum()} / {len(rq1)}")
    print(f"  after BH-FDR      : {(rq1.p_fdr < Config.ALPHA).sum()} / {len(rq1)}")
    print(f"  joint LRT smallest p: {rq1_lrt.p.min():.4f}")

    # variance components: where does Psi's variance live?
    vc = []
    for f in Config.MACRO_FEATURES:
        d = panel[panel.macro_feature == f].dropna(subset=["psi_l1"])
        m = smf.mixedlm("psi_l1 ~ upset*time_c + upset*und_c", d,
                        groups=d.match_id).fit()
        su, se = float(m.cov_re.iloc[0, 0]), m.scale
        vc.append(dict(feature=f, var_match=su, var_resid=se,
                       icc=su / (su + se), sd_match=np.sqrt(su),
                       sd_resid=np.sqrt(se)))
    vcomp = R.save("rq1_variance_components", pd.DataFrame(vc))
    print("\n  variance components (ICC = share of variance BETWEEN matches):")
    show(vcomp)
    print("  => ICC is 0.4-1.4%: almost all variance is window-to-window inside")
    print("     a match. This is why the random intercept is mandatory, and why")
    print("     pooled corpus medians look identical between groups.")

    # game-state adjusted refits
    adj = []
    for f in Config.MACRO_FEATURES:
        for l in Config.ORDERS:
            adj += fit(panel, f, l, extra=" + gd_c")[1]
    rq1_adj = pd.DataFrame(adj)
    rq1_adj["p_fdr"] = multipletests(rq1_adj.p, method=Config.FDR_METHOD)[1]
    R.save("rq1_game_state_adjusted", rq1_adj)
    print(f"\n  adjusting for goal difference: "
          f"{(rq1_adj[rq1_adj.term == 'upset'].p < Config.ALPHA).sum()} of "
          f"{len(Config.MACRO_FEATURES) * len(Config.ORDERS)} significant "
          f"(unadjusted: {(rq1[rq1.term == 'upset'].p < Config.ALPHA).sum()})")
    return rq1, rq1_lrt


# --- RQ2 -------------------------------------------------------------------
def rq2_order_of_emergence(panel, R):
    """Order (3, within) x Match Type (2, between) x Role (2, within).

    Psi MUST be z-scored within order first: |Psi| grows mechanically with the
    number of micro subsets (10 -> 45 -> 120), so on raw values the Order main
    effect is a foregone conclusion that says nothing about football. The RQ2
    question is the Order x Match Type interaction."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    banner("RQ2 -- order of emergence (repeated-measures ANOVA)")

    parts = []
    for l in Config.ORDERS:
        g = (panel[panel.macro_feature == Config.PRIMARY_FEATURE]
             .groupby(["match_id", "outcome", "role"])[f"psi_l{l}"]
             .mean().reset_index().rename(columns={f"psi_l{l}": "psi"}))
        g["order"] = l
        parts.append(g)
    RM = pd.concat(parts, ignore_index=True).dropna(subset=["psi"])
    RM["psi_z"] = RM.groupby("order").psi.transform(
        lambda s: (s - s.mean()) / s.std())
    keep = RM.groupby("match_id").size() == len(Config.ORDERS) * 2
    RM = RM[RM.match_id.isin(keep[keep].index)]
    RM["upset"] = (RM.outcome == "upset").astype(int)
    step(f"balanced design: {RM.match_id.nunique()} matches x 2 roles x 3 orders "
         f"= {len(RM)} cells")
    step(f"raw means by order    : {RM.groupby('order').psi.mean().round(2).to_dict()}"
         "  <- mechanical scaling")
    step(f"z-scored by order     : {RM.groupby('order').psi_z.mean().round(3).to_dict()}"
         "  <- scaling removed")

    m = smf.mixedlm("psi_z ~ C(outcome)*C(role)*C(order)", RM,
                    groups=RM.match_id).fit(reml=False)
    wald = m.wald_test_terms(scalar=False).table.reset_index()
    wald.columns = ["term"] + list(wald.columns[1:])
    wald = wald[wald.term != "Intercept"].copy()

    # statsmodels returns each statistic as a (1,1) array under scalar=False.
    # Older pandas coerced those on .astype(float); current pandas raises.
    # Unwrap explicitly so the column is numeric on every version.
    def _scalar(v):
        a = np.asarray(v, dtype=float).ravel()
        return float(a[0]) if a.size else np.nan

    wald["statistic"] = wald["statistic"].map(_scalar)
    wald["pvalue"] = wald["pvalue"].map(_scalar)
    wald["p_fdr"] = multipletests(wald["pvalue"], method=Config.FDR_METHOD)[1]
    wald["partial_eta2"] = wald["statistic"] / (wald["statistic"] + len(RM))
    R.save("rq2_mixed_anova", wald)
    print("\n  mixed-model ANOVA (no sphericity assumption):")
    show(wald)
    key = wald[wald.term.str.contains("outcome") & wald.term.str.contains("order")]
    for _, r in key.iterrows():
        print(f"  RQ2 question -- {r['term']}: chi2={float(r['statistic']):.3f} "
              f"p={float(r['pvalue']):.4f}")

    # classical RM-ANOVA with Greenhouse-Geisser, as a cross-check
    def gg_epsilon(M):
        k = M.shape[1]
        S = np.cov(M, rowvar=False)
        Sb, Sr, Sc = S.mean(), S.mean(1), S.mean(0)
        D = S - Sr[:, None] - Sc[None, :] + Sb
        num = k ** 2 * (np.trace(D) / k) ** 2
        den = ((k - 1) * (np.sum(D ** 2) - 2 * k * np.sum(D.mean(1) ** 2)
                          + k ** 2 * Sb ** 2))
        return float(np.clip(num / den, 1 / (k - 1), 1.0))

    rows = []
    for grp in ["control", "upset"]:
        for role in ["underdog", "favourite"]:
            w = (RM[(RM.outcome == grp) & (RM.role == role)]
                 .pivot(index="match_id", columns="order", values="psi_z").dropna())
            if len(w) < 5:
                continue
            M = w.to_numpy()
            n, k = M.shape
            gm = M.mean()
            ss_cond = n * ((M.mean(0) - gm) ** 2).sum()
            ss_err = ((M - M.mean(0)[None, :] - M.mean(1)[:, None] + gm) ** 2).sum()
            df1, df2 = k - 1, (k - 1) * (n - 1)
            F = (ss_cond / df1) / (ss_err / df2)
            eps = gg_epsilon(M)
            rows.append(dict(outcome=grp, role=role, n=n, F=F, df1=df1, df2=df2,
                             p_uncorrected=stats.f.sf(F, df1, df2),
                             gg_epsilon=eps,
                             p_gg=stats.f.sf(F, df1 * eps, df2 * eps),
                             partial_eta2=ss_cond / (ss_cond + ss_err)))
    R.save("rq2_rm_anova_gg", pd.DataFrame(rows))
    return RM, wald


# --- RQ3 -------------------------------------------------------------------
GRID = np.linspace(0, 1, 101)


def trajectories(df, feature, order, role=None, value=None):
    """{match_id: curve resampled onto GRID}, plus each match's outcome."""
    d = df[df.macro_feature == feature]
    if role:
        d = d[d.role == role]
    col = value or f"psi_l{order}"
    out, lab = {}, {}
    for (mid, oc), g in d.groupby(["match_id", "outcome"]):
        g = g.dropna(subset=[col]).sort_values("time_n")
        if len(g) < 10:
            continue
        out[mid] = np.interp(GRID, g.time_n.to_numpy(), g[col].to_numpy())
        lab[mid] = oc
    return out, lab


def binseg(x, n_bkps=1, min_size=8):
    """Binary segmentation with an L2 cost (ruptures is not a dependency)."""
    def cost(v):
        return float(((v - v.mean()) ** 2).sum()) if len(v) else 0.0
    x = np.asarray(x, float)
    bkps = []
    for _ in range(n_bkps):
        edges = [0] + sorted(bkps) + [len(x)]
        best = (None, 0.0)
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a < 2 * min_size:
                continue
            base = cost(x[a:b])
            for c in range(a + min_size, b - min_size + 1):
                gain = base - cost(x[a:c]) - cost(x[c:b])
                if gain > best[1]:
                    best = (c, gain)
        if best[0] is None:
            break
        bkps.append(best[0])
    return sorted(bkps)


def rq3_temporal_dynamics(panel, R):
    """Levels and tipping points (null), then the gap TREND (the one effect).

    Permuting whole match labels preserves within-match autocorrelation, so no
    independence assumption is needed."""
    from statsmodels.stats.multitest import multipletests
    banner("RQ3 -- temporal dynamics")

    # (a) functional permutation test on the mean curves
    def curve_perm(curves, labels, n_perm, seed=1):
        ids = np.array(list(curves))
        Y = np.vstack([curves[i] for i in ids])
        up = np.array([labels[i] == "upset" for i in ids])

        def stat(mask):
            d = Y[mask].mean(0) - Y[~mask].mean(0)
            return np.abs(d).max(), (d ** 2).mean()

        obs_sup, obs_l2 = stat(up)
        rng = np.random.default_rng(seed)
        n_up = up.sum()
        sup_null, l2_null = np.empty(n_perm), np.empty(n_perm)
        for b in range(n_perm):
            m = np.zeros(len(ids), bool)
            m[rng.choice(len(ids), n_up, replace=False)] = True
            sup_null[b], l2_null[b] = stat(m)
        return dict(n_upset=int(n_up), n_control=int((~up).sum()),
                    sup_stat=obs_sup,
                    sup_p=float((sup_null >= obs_sup).mean() + 1 / n_perm),
                    l2_stat=obs_l2,
                    l2_p=float((l2_null >= obs_l2).mean() + 1 / n_perm))

    t0, rows = time.time(), []
    for f in Config.MACRO_FEATURES:
        for role in ["underdog", "favourite"]:
            cur, lab = trajectories(panel, f, Config.PRIMARY_ORDER, role=role)
            rows.append(dict(feature=f, role=role,
                             **curve_perm(cur, lab, CFG.N_PERM_CURVE)))
    fda = pd.DataFrame(rows)
    fda["sup_p_fdr"] = multipletests(fda.sup_p, method=Config.FDR_METHOD)[1]
    R.save("rq3_functional_permutation", fda,
           f"{len(fda)} curve tests in {time.time() - t0:.0f}s")
    print(f"  mean curves: {(fda.sup_p < Config.ALPHA).sum()} / {len(fda)} "
          f"raw significant; smallest p_fdr = {fda.sup_p_fdr.min():.3f}")

    # (b) change points and crossings on the gap
    u, lab = trajectories(panel, Config.PRIMARY_FEATURE, Config.PRIMARY_ORDER,
                          role="underdog")
    v, _ = trajectories(panel, Config.PRIMARY_FEATURE, Config.PRIMARY_ORDER,
                        role="favourite")
    GAP = {m: u[m] - v[m] for m in u if m in v}
    rows = []
    for mid, g in GAP.items():
        bk = binseg(g, n_bkps=1)
        pos = np.where(g > 0)[0]
        rows.append(dict(match_id=mid, outcome=lab[mid],
                         changepoint=GRID[bk[0]] if bk else np.nan,
                         first_crossing=GRID[pos[0]] if len(pos) else np.nan,
                         frac_underdog_ahead=float((g > 0).mean()),
                         mean_gap=float(g.mean()),
                         gap_trend=float(np.polyfit(GRID, g, 1)[0])))
    cp = R.save("rq3_changepoints", pd.DataFrame(rows))

    tests = []
    for col in ["changepoint", "first_crossing", "frac_underdog_ahead",
                "mean_gap", "gap_trend"]:
        a = cp[cp.outcome == "upset"][col].dropna()
        b = cp[cp.outcome == "control"][col].dropna()
        tests.append(dict(metric=col, n_up=len(a), n_ctl=len(b),
                          upset=a.mean(), control=b.mean(),
                          p=(stats.mannwhitneyu(a, b).pvalue
                             if len(a) > 2 and len(b) > 2 else np.nan)))
    tt = pd.DataFrame(tests)
    tt["p_fdr"] = multipletests(tt.p.fillna(1), method=Config.FDR_METHOD)[1]
    R.save("rq3_timing_tests", tt)
    show(tt)
    print(f"  matches where the underdog NEVER exceeds the favourite: "
          f"{cp.first_crossing.isna().mean():.1%}")
    print("  => no tipping point to predict: the gap rarely crosses zero, and")
    print("     when it does the timing does not separate the groups.")

    # (c) the gap TREND -- the one effect that survives correction
    def gap_trends(feature, order):
        a, lb = trajectories(panel, feature, order, role="underdog")
        b, _ = trajectories(panel, feature, order, role="favourite")
        return pd.DataFrame([
            dict(match_id=m, outcome=lb[m],
                 trend=float(np.polyfit(GRID, a[m] - b[m], 1)[0]))
            for m in a if m in b])

    rows = []
    for f in Config.MACRO_FEATURES:
        for l in Config.ORDERS:
            t = gap_trends(f, l)
            a = t[t.outcome == "upset"].trend
            b = t[t.outcome == "control"].trend
            rows.append(dict(feature=f, order=l, n_up=len(a), n_ctl=len(b),
                             upset_slope=a.mean(), control_slope=b.mean(),
                             g=hedges_g(a, b),
                             p=stats.mannwhitneyu(a, b).pvalue))
    trend = pd.DataFrame(rows)
    trend["p_fdr"] = multipletests(trend.p, method=Config.FDR_METHOD)[1]
    R.save("rq3_gap_trend", trend)
    print("\n  gap-slope comparison, most significant cells:")
    show(trend.nsmallest(6, "p"))
    print(f"  raw p<{Config.ALPHA}: {(trend.p < Config.ALPHA).sum()} / {len(trend)}"
          f"   after BH-FDR: {(trend.p_fdr < Config.ALPHA).sum()} / {len(trend)}")
    print(f"  control slopes positive in "
          f"{(trend.control_slope > 0).sum()}/{len(trend)} cells")

    t = gap_trends(Config.PRIMARY_FEATURE, Config.PRIMARY_ORDER)
    tr = t.trend.to_numpy()
    up = (t.outcome == "upset").to_numpy()
    obs = tr[up].mean() - tr[~up].mean()
    rng = np.random.default_rng(0)
    null = np.array([(lambda m: tr[m].mean() - tr[~m].mean())(rng.permutation(up))
                     for _ in range(CFG.N_PERM_TREND)])
    p_perm = float((np.abs(null) >= abs(obs)).mean())
    R.save("rq3_gap_trend_permutation", pd.DataFrame([dict(
        feature=Config.PRIMARY_FEATURE, order=Config.PRIMARY_ORDER,
        obs_diff=obs, p_perm=p_perm, n_perm=CFG.N_PERM_TREND,
        upset_slope=tr[up].mean(), control_slope=tr[~up].mean())]))
    print(f"\n  permutation ({Config.PRIMARY_FEATURE}, l={Config.PRIMARY_ORDER}, "
          f"{CFG.N_PERM_TREND:,} draws): observed slope difference "
          f"{obs:+.4f} nats/match, p = {p_perm:.4f}")
    print("  INTERPRETATION: more negative Psi = redundancy-dominated = drilled,")
    print("  synchronised play. A downward gap slope in upsets means the underdog")
    print("  becomes progressively MORE drilled relative to the favourite.")
    print("  CAVEAT: this family was examined AFTER the pre-specified tests")
    print("  returned null. It is exploratory and needs out-of-sample confirmation.")

    # robustness of the one live finding
    a, b = tr[up], tr[~up]
    rob = [dict(variant="all matches", n_up=len(a), n_ctl=len(b),
                g=hedges_g(a, b), p_mw=stats.mannwhitneyu(a, b).pvalue)]
    b2 = np.sort(b)[:-2]
    rob.append(dict(variant="drop 2 largest control slopes", n_up=len(a),
                    n_ctl=len(b2), g=hedges_g(a, b2),
                    p_mw=stats.mannwhitneyu(a, b2).pvalue))
    lo, hi = np.percentile(np.r_[a, b], [5, 95])
    ca, cb = np.clip(a, lo, hi), np.clip(b, lo, hi)
    rob.append(dict(variant="winsorised at 5/95 pct", n_up=len(a), n_ctl=len(b),
                    g=hedges_g(ca, cb), p_mw=stats.mannwhitneyu(ca, cb).pvalue))
    obs_med = np.median(a) - np.median(b)
    allv = np.r_[a, b]
    nullm = np.array([(lambda m: np.median(allv[m]) - np.median(allv[~m]))(
        rng.permutation(up)) for _ in range(CFG.N_PERM_TREND)])
    rob.append(dict(variant="permutation on MEDIAN difference", n_up=len(a),
                    n_ctl=len(b), g=np.nan,
                    p_mw=float((np.abs(nullm) >= abs(obs_med)).mean())))
    R.save("rq3_gap_trend_robustness", pd.DataFrame(rob))
    print("\n  robustness (the two visible control outliers are NOT driving it):")
    show(pd.DataFrame(rob))
    return fda, cp, tt, trend, p_perm, GAP, lab


# --- RQ4 -------------------------------------------------------------------
def rq4_disruption(panel, R):
    """Is the favourite more disrupted in upsets?

    LIMITATION, stated rather than worked around: the proposal specifies three
    conditions and the third -- favourite loses to an EQUALLY STRONG opponent
    -- does not exist in this corpus, because Sprint 1 selected only fixtures
    meeting the mismatch criteria. A general losing effect therefore cannot be
    separated from a giant-killing-specific one."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    banner("RQ4 -- disruption of the favourite")

    rows = []
    for f in Config.MACRO_FEATURES:
        d = panel[(panel.macro_feature == f)
                  & (panel.role == "favourite")].dropna(subset=["psi_l1"])
        m = smf.mixedlm("psi_l1 ~ upset*time_c", d, groups=d.match_id).fit()
        sd = np.sqrt(float(m.cov_re.iloc[0, 0]))
        rows.append(dict(feature=f, measure="favourite psi_l1",
                         coef=m.params["upset"], se=m.bse["upset"],
                         p=m.pvalues["upset"],
                         coef_std=m.params["upset"] / sd if sd > 0 else np.nan))
        dd = panel[panel.macro_feature == f].dropna(subset=["dc_ic"])
        m2 = smf.mixedlm("dc_ic ~ upset*time_c", dd, groups=dd.match_id).fit()
        sd2 = np.sqrt(float(m2.cov_re.iloc[0, 0]))
        rows.append(dict(feature=f, measure="disruption coefficient (dc_ic)",
                         coef=m2.params["upset"], se=m2.bse["upset"],
                         p=m2.pvalues["upset"],
                         coef_std=m2.params["upset"] / sd2 if sd2 > 0 else np.nan))
    rq4 = pd.DataFrame(rows)
    rq4["p_fdr"] = multipletests(rq4.p, method=Config.FDR_METHOD)[1]
    R.save("rq4_disruption", rq4)
    show(rq4, n=12)
    print(f"  raw significant: {(rq4.p < Config.ALPHA).sum()} / {len(rq4)}; "
          f"after FDR: {(rq4.p_fdr < Config.ALPHA).sum()} / {len(rq4)}")
    print("  LIMITATION: no 'favourite loses to an equal opponent' stratum exists,")
    print("  so a general losing effect cannot be distinguished from a giant killing.")
    return rq4


# --- RQ5 -------------------------------------------------------------------
def rq5_classification(panel, meta, R):
    """Random forest on match-level Psi trajectory summaries.

    260 features for 136 matches -- a feature-to-observation ratio of 1.9,
    exactly the overfitting regime the proposal's risk table anticipates. The
    permutation null re-runs the WHOLE CV pipeline on shuffled labels, because
    at p/n ~ 2 an above-chance AUC is the expected outcome of noise."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, average_precision_score
    banner("RQ5 -- classification and spatial signatures")

    def build_features(corpus):
        d = corpus[corpus.estimate_ok]
        parts = []
        for l in Config.ORDERS:
            g = d.groupby(["match_id", "outcome", "macro_feature", "role"])[
                f"psi_l{l}"].agg(["mean", "std", "median", "min", "max",
                                  lambda s: s.quantile(.10),
                                  lambda s: s.quantile(.90)])
            g.columns = [f"psi_l{l}_{c}" for c in
                         ["mean", "std", "median", "min", "max", "q10", "q90"]]
            parts.append(g)
        F = pd.concat(parts, axis=1).reset_index()
        X = F.pivot_table(index=["match_id", "outcome"],
                          columns=["macro_feature", "role"],
                          values=[c for c in F.columns if c.startswith("psi_")])
        X.columns = ["__".join(map(str, c)) for c in X.columns]
        e = (d.groupby(["match_id", "macro_feature"])[
            ["eai_stab_l1", "eai_stab_l2", "eai_stab_l3", "dc_ic"]]
            .agg(["mean", "std"]))
        e.columns = ["__".join(c) for c in e.columns]
        e = e.unstack("macro_feature")
        e.columns = ["__".join(map(str, c)) for c in e.columns]
        return X.join(e).reset_index()

    MF = build_features(panel).merge(meta[["match_id", "match_date", "season"]],
                                     on="match_id", how="left")
    MF.to_parquet(CFG.stat_dir / "match_features.parquet", index=False)
    FEATS = [c for c in MF.columns
             if c not in ("match_id", "outcome", "match_date", "season")]
    X = MF[FEATS].fillna(MF[FEATS].median(numeric_only=True)).fillna(0)
    y = (MF.outcome == "upset").astype(int).to_numpy()
    step(f"design matrix: {X.shape[0]} matches x {X.shape[1]} features "
         f"({X.shape[1] / X.shape[0]:.1f} per observation)")
    step(f"class balance: {y.sum()} upsets / {(1 - y).sum()} controls "
         f"(base rate {y.mean():.3f})")

    def make_rf(seed=Config.RF_SEED):
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                      class_weight="balanced_subsample",
                                      random_state=seed, n_jobs=-1)

    def cv_scores(Xm, yv, seed=0):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        p = cross_val_predict(make_rf(seed), Xm, yv, cv=cv,
                              method="predict_proba")[:, 1]
        return p, roc_auc_score(yv, p), average_precision_score(yv, p)

    t0 = time.time()
    proba, auc, ap = cv_scores(X, y)
    step(f"5-fold stratified CV: ROC-AUC {auc:.3f}, PR-AUC {ap:.3f} "
         f"(base rate {y.mean():.3f}) [{time.time() - t0:.1f}s]")

    t0, null_auc, null_ap = time.time(), [], []
    rng = np.random.default_rng(Config.RF_SEED)
    for b in range(CFG.N_PERM_RF):
        _, a, q = cv_scores(X, rng.permutation(y), seed=b)
        null_auc.append(a)
        null_ap.append(q)
    null_auc, null_ap = np.array(null_auc), np.array(null_ap)
    p_auc = float((null_auc >= auc).mean() + 1 / CFG.N_PERM_RF)
    p_ap = float((null_ap >= ap).mean() + 1 / CFG.N_PERM_RF)
    step(f"permutation null, {CFG.N_PERM_RF} shuffles [{time.time() - t0:.0f}s]: "
         f"AUC null {null_auc.mean():.3f} (sd {null_auc.std():.3f}) -> p = {p_auc:.3f}")

    R.save("rq5_performance", pd.DataFrame([dict(
        model="random forest (5-fold stratified CV)", roc_auc=auc, pr_auc=ap,
        base_rate=y.mean(), perm_p_auc=p_auc, perm_p_pr=p_ap,
        null_auc_mean=null_auc.mean(), null_auc_sd=null_auc.std(),
        n_permutations=CFG.N_PERM_RF)]))

    # temporal blocking: train on earlier seasons, test on later
    od = MF.sort_values("match_date").reset_index(drop=True)
    Xo = od[FEATS].fillna(od[FEATS].median(numeric_only=True)).fillna(0)
    yo = (od.outcome == "upset").astype(int).to_numpy()
    rows = []
    for frac in (0.6, 0.7, 0.8):
        cut = int(len(od) * frac)
        ytr, yte = yo[:cut], yo[cut:]
        if yte.sum() < 3 or len(set(ytr)) < 2:
            continue
        rf = make_rf().fit(Xo.iloc[:cut], ytr)
        pp = rf.predict_proba(Xo.iloc[cut:])[:, 1]
        rows.append(dict(train_frac=frac, n_train=cut, n_test=len(yte),
                         test_upsets=int(yte.sum()),
                         split_date=str(od.match_date.iloc[cut].date()),
                         roc_auc=roc_auc_score(yte, pp),
                         pr_auc=average_precision_score(yte, pp),
                         base_rate=yte.mean()))
    R.save("rq5_temporal_blocking", pd.DataFrame(rows))
    print("  temporal blocking:")
    show(pd.DataFrame(rows))
    print("  => out-of-time AUC is unstable across split points; with 7-12 upsets")
    print("     per test block that spread is sampling noise, not a usable model.")

    # Grouped, out-of-fold permutation importance.
    # Standard per-feature importance returns EXACTLY ZERO for all 260
    # predictors -- not a bug, but the consequence of extreme redundancy among
    # near-duplicate columns (psi_l1_mean vs psi_l1_median on the same
    # trajectory): permuting one leaves the forest able to reconstruct it.
    def oof_importance(Xm, yv, blocks, n_repeats=20, seed=Config.RF_SEED):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        folds, base = [], np.zeros(len(yv))
        for tr, te in cv.split(Xm, yv):
            m = make_rf(seed).fit(Xm.iloc[tr], yv[tr])
            folds.append((m, te))
            base[te] = m.predict_proba(Xm.iloc[te])[:, 1]
        base_auc = roc_auc_score(yv, base)
        rg = np.random.default_rng(seed)
        rows = []
        for name, cols in blocks.items():
            drops = []
            for _ in range(n_repeats):
                p = np.zeros(len(yv))
                for m, te in folds:
                    Xt = Xm.iloc[te].copy()
                    Xt[cols] = Xt[cols].to_numpy()[rg.permutation(len(te))]
                    p[te] = m.predict_proba(Xt)[:, 1]
                drops.append(base_auc - roc_auc_score(yv, p))
            rows.append(dict(block=name, n_features=len(cols),
                             importance=float(np.mean(drops)),
                             sd=float(np.std(drops))))
        return base_auc, pd.DataFrame(rows).sort_values("importance",
                                                        ascending=False)

    blocks = {mf: [c for c in FEATS if f"__{mf}" in c]
              for mf in Config.MACRO_FEATURES}
    blocks = {k: v for k, v in blocks.items() if v}
    base_auc, imp_macro = oof_importance(X, y, blocks, n_repeats=20)
    R.save("rq5_importance_by_macro_feature", imp_macro)
    print(f"\n  out-of-fold baseline AUC = {base_auc:.3f}")
    print("  RQ5 ranking -- macro-feature blocks, grouped OOF importance:")
    show(imp_macro)

    blocks_o = {f"{mf} l={l}": [c for c in FEATS
                                if f"__{mf}" in c and f"psi_l{l}_" in c]
                for mf in Config.MACRO_FEATURES for l in Config.ORDERS}
    blocks_o = {k: v for k, v in blocks_o.items() if v}
    _, imp_ord = oof_importance(X, y, blocks_o, n_repeats=10)
    R.save("rq5_importance_by_feature_order", imp_ord)

    rf_full = make_rf().fit(X, y)

    # Naive per-feature permutation importance, kept because its DEGENERACY is
    # the finding: it returns exactly zero for all 260 predictors. That is not
    # a bug -- the features are near-duplicates (psi_l1_mean vs psi_l1_median
    # on the same trajectory), so permuting any one leaves the forest able to
    # reconstruct it from its neighbours. This table is the evidence for why
    # importance had to be computed out-of-fold and grouped instead.
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(rf_full, X, y, n_repeats=10,
                                random_state=Config.RF_SEED, n_jobs=-1)
    naive = pd.DataFrame(dict(feature=FEATS, importance=pi.importances_mean,
                              sd=pi.importances_std))
    naive["macro_feature"] = naive.feature.str.extract(r"__(V_[a-z]+)")
    naive["order"] = naive.feature.str.extract(r"psi_l(\d)_").astype(float)
    naive = naive.sort_values("importance", ascending=False)
    R.save("rq5_feature_importance", naive)
    n_zero = int((naive.importance.abs() < 1e-12).sum())
    print(f"\n  naive per-feature permutation importance: {n_zero} of "
          f"{len(naive)} features score EXACTLY zero")
    print("  => not a bug: near-duplicate columns let the forest reconstruct any")
    print("     permuted feature from its neighbours. Hence the grouped OOF form above.")

    mdi = (pd.DataFrame(dict(feature=FEATS, mdi=rf_full.feature_importances_))
           .sort_values("mdi", ascending=False))
    mdi["macro_feature"] = mdi.feature.str.extract(r"__(V_[a-z]+)")
    R.save("rq5_feature_importance_mdi", mdi)
    print("  CAVEAT: MDI is biased toward high-cardinality continuous predictors")
    print("  and is in-sample. The model does not generalise (permutation")
    print(f"  p = {p_auc:.2f}), so no block discriminates giant killings.")
    return MF, X, y, proba, auc, ap, p_auc, null_auc, imp_macro


# --- sensitivity -----------------------------------------------------------
def spec_curve(corpus, y_labels=None):
    """Effect size and p for every analytic specification.

    {summary statistic} x {windowing} x {feature} x {order} x {role} = 324
    cells. An earlier informal pass appeared to find p<0.05 effects; it used
    the MEDIAN as the within-match summary. Nothing in the data justifies that
    over the mean (skewness 0.03, excess kurtosis -0.03), yet it changes the
    answer. Rather than choose, enumerate."""
    d = corpus[corpus.estimate_ok].copy()
    if y_labels is not None:
        d["outcome"] = d.match_id.map(y_labels)
    out = []
    for windowing, dd in [("all windows", d),
                          ("non-overlapping",
                           d[d.window_start % Config.THIN_STEP == 0])]:
        for f in Config.MACRO_FEATURES:
            sub_f = dd[dd.macro_feature == f]
            for l in Config.ORDERS:
                col = f"psi_l{l}"
                for role in ["underdog", "favourite", "both"]:
                    s = sub_f if role == "both" else sub_f[sub_f.role == role]
                    for name, fn in [("mean", "mean"), ("median", "median"),
                                     ("trim10",
                                      lambda x: stats.trim_mean(x.dropna(), .1))]:
                        m = s.groupby(["match_id", "outcome"])[col].agg(fn).reset_index()
                        u = m[m.outcome == "upset"][col].dropna()
                        c = m[m.outcome == "control"][col].dropna()
                        if len(u) < 5 or len(c) < 5:
                            continue
                        out.append(dict(windowing=windowing, feature=f, order=l,
                                        role=role, summary=name,
                                        g=hedges_g(u, c),
                                        p=stats.mannwhitneyu(u, c).pvalue))
    return pd.DataFrame(out)


def psi_at(mid, window, k, features=None, step_sec=None):
    """Recompute order-1 Psi for one match at a given window length and k."""
    features = features or Config.SENS_FEATURES
    step_sec = step_sec or window
    bundle = build_bundle(mid, common_grid=True)
    if bundle is None:
        return None
    rm = roles().get(mid, {})
    rows = []
    for tm, b in bundle.items():
        pos, macro, grid = b["pos"], b["macro"], b["grid"]
        role = ("underdog" if rm.get("underdog") == tm else
                "favourite" if rm.get("favourite") == tm else None)
        for s in range(0, max(0, len(grid) - window), step_sec):
            ok = window_mask(pos, macro["V_all"], s, window=window, lag=1)
            r, meta = window_terms(pos, macro, s, ok=ok, orders=(1,), k=k,
                                   features=features, window=window, lag=1,
                                   seed=mid)
            for f in features:
                if "psi_l1" in r[f]:
                    rows.append(dict(match_id=mid, team=tm, role=role,
                                     window_start=s, macro_feature=f,
                                     psi_l1=r[f]["psi_l1"], **meta))
    return pd.DataFrame(rows)


def sensitivity(corpus, panel, R):
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    banner("Sensitivity analyses")

    # (1) specification curve
    t0 = time.time()
    SPEC = spec_curve(corpus)
    SPEC["p_fdr"] = multipletests(SPEC.p, method=Config.FDR_METHOD)[1]
    R.save("sens_specification_curve", SPEC)
    n_sig = int((SPEC.p < Config.ALPHA).sum())
    step(f"specification curve: {len(SPEC)} cells in {time.time() - t0:.0f}s")
    step(f"  raw p<{Config.ALPHA}: {n_sig} ({n_sig / len(SPEC):.1%}); "
         f"after FDR: {(SPEC.p_fdr < Config.ALPHA).sum()}")
    step(f"  share with g<0: {(SPEC.g < 0).mean():.1%} "
         f"({int((SPEC.g < 0).sum())} of {len(SPEC)})")
    print("  significance rate by within-match summary statistic:")
    show(SPEC.groupby("summary").agg(
        n=("p", "size"), frac_sig=("p", lambda s: (s < Config.ALPHA).mean()),
        median_g=("g", "median")).reset_index())

    # (2) placebo calibration on shuffled labels
    truth = corpus.drop_duplicates("match_id").set_index("match_id").outcome
    rng = np.random.default_rng(7)
    t0, placebo = time.time(), []
    for b in range(CFG.N_PLACEBO):
        sh = pd.Series(rng.permutation(truth.values), index=truth.index)
        sc = spec_curve(corpus, y_labels=sh)
        placebo.append(dict(rep=b, frac_sig=float((sc.p < Config.ALPHA).mean()),
                            n_sig=int((sc.p < Config.ALPHA).sum()),
                            max_abs_g=float(sc.g.abs().max())))
    PLACEBO = R.save("sens_placebo", pd.DataFrame(placebo))
    obs_frac = float((SPEC.p < Config.ALPHA).mean())
    p_emp = float((PLACEBO.frac_sig >= obs_frac).mean() + 1 / CFG.N_PLACEBO)
    step(f"placebo: {CFG.N_PLACEBO} shuffles [{time.time() - t0:.0f}s]")
    step(f"  observed {obs_frac:.1%} vs placebo {PLACEBO.frac_sig.mean():.1%} "
         f"(sd {PLACEBO.frac_sig.std():.1%}) -> empirical p = {p_emp:.3f}")
    print("  => real labels yield ~6x the placebo rate: weak evidence FOR a small")
    print("     real effect, not a 'no better than chance' reading -- while still")
    print("     not licensing any single specification as the headline.")

    # (3) upset-definition robustness and outlier exclusion
    rows = []
    gaps = panel.drop_duplicates("match_id")[["match_id", "outcome"]].merge(
        corpus.drop_duplicates(["match_id", "role"])
              .pivot_table(index="match_id", columns="role", values="team_rank"),
        on="match_id", how="left")
    gaps["rank_gap"] = gaps["underdog"] - gaps["favourite"]
    for thr in [0, 10, 20, 30]:
        keep = set(gaps[gaps.rank_gap >= thr].match_id)
        d = panel[(panel.macro_feature == Config.PRIMARY_FEATURE)
                  & panel.match_id.isin(keep)]
        if d.outcome.nunique() < 2 or d[d.outcome == "upset"].match_id.nunique() < 8:
            continue
        m = smf.mixedlm("psi_l1 ~ upset*time_c + upset*und_c", d,
                        groups=d.match_id).fit()
        rows.append(dict(check=f"rank gap >= {thr}",
                         n_matches=d.match_id.nunique(),
                         n_upset=d[d.outcome == "upset"].match_id.nunique(),
                         coef=m.params["upset"], p=m.pvalues["upset"]))
    for label, mask in [("all windows", panel.index == panel.index),
                        ("drop flagged outliers", ~panel.is_outlier),
                        ("n_eff >= 40", panel.n_eff >= 40),
                        ("drop first/last 10% of match",
                         panel.time_n.between(.1, .9))]:
        d = panel[(panel.macro_feature == Config.PRIMARY_FEATURE) & mask]
        m = smf.mixedlm("psi_l1 ~ upset*time_c + upset*und_c", d,
                        groups=d.match_id).fit()
        rows.append(dict(check=label, n_matches=d.match_id.nunique(),
                         n_upset=d[d.outcome == "upset"].match_id.nunique(),
                         coef=m.params["upset"], p=m.pvalues["upset"]))
    R.save("sens_definition_and_outliers", pd.DataFrame(rows))
    print("\n  definition robustness and outlier exclusion:")
    show(pd.DataFrame(rows), n=10)
    print("  => the upset coefficient is small and non-significant under every variant.")

    # (4) window length and KSG k -- the one place Sprint 5 creates new Psi data
    CFG.sens_dir.mkdir(parents=True, exist_ok=True)
    match_ids = sorted(panel.match_id.unique())
    first = panel.drop_duplicates("match_id").set_index("match_id").outcome
    ups = [m for m in match_ids if first.get(m) == "upset"]
    ctl = [m for m in match_ids if first.get(m) == "control"]
    rg = np.random.default_rng(11)
    n_ctl = min(CFG.SENS_N_CONTROL, len(ctl))
    sens_ids = sorted(ups + list(rg.choice(ctl, n_ctl, replace=False)))
    step(f"sensitivity subsample: {len(sens_ids)} matches "
         f"({len(ups)} upsets, {n_ctl} controls)")

    t0, frames = time.time(), []
    for W, k in CFG.SENS_CONFIGS:
        path = CFG.sens_dir / f"psi_W{W}_k{k}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
        else:
            got = [d for d in (psi_at(m, W, k) for m in sens_ids) if d is not None]
            if not got:
                continue
            df = pd.concat(got, ignore_index=True)
            df.to_parquet(path, index=False)
        df["W"], df["k"] = W, k
        frames.append(df)
        step(f"  W={W:3d} k={k:2d}: {len(df):7,} rows "
             f"({time.time() - t0:5.0f}s cumulative)")
    SENS = pd.concat(frames, ignore_index=True)
    SENS["outcome"] = SENS.match_id.map(first)

    rows = []
    for (W, k, f), d in SENS.groupby(["W", "k", "macro_feature"]):
        d = d.dropna(subset=["psi_l1"]).assign(
            upset=lambda x: (x.outcome == "upset").astype(int),
            und_c=lambda x: np.where(x.role == "underdog", .5, -.5))
        d["time_c"] = d.groupby("match_id").window_start.transform(
            lambda s: (s - s.min()) / max(s.max() - s.min(), 1e-9)) - .5
        try:
            m = smf.mixedlm("psi_l1 ~ upset*time_c + upset*und_c", d,
                            groups=d.match_id).fit()
            sd = np.sqrt(float(m.cov_re.iloc[0, 0]))
            rows.append(dict(W=W, k=k, feature=f, n_rows=len(d),
                             n_matches=d.match_id.nunique(),
                             coef=m.params["upset"], se=m.bse["upset"],
                             p=m.pvalues["upset"],
                             coef_std=m.params["upset"] / sd if sd > 0 else np.nan,
                             median_psi=d.psi_l1.median()))
        except Exception:
            rows.append(dict(W=W, k=k, feature=f, n_rows=len(d),
                             coef=np.nan, p=np.nan))
    sens_wk = R.save("sens_window_and_k", pd.DataFrame(rows))
    print("\n  window length and KSG k:")
    show(sens_wk, n=14)
    print(f"  p<{Config.ALPHA}: {(sens_wk.p < Config.ALPHA).sum()} of {len(sens_wk)}")
    print("  => |Psi| shifts with W and k as expected, but the V_all upset effect")
    print("     stays near zero throughout. V_cvel is stable and reaches nominal")
    print("     significance at W=90 and k=10 -- consistent with a small real")
    print("     effect on the one non-linear macro feature.")

    # (5) power / minimum detectable effect
    def power2(g, n1, n2, alpha=Config.ALPHA):
        se = np.sqrt(1 / n1 + 1 / n2)
        crit = stats.norm.ppf(1 - alpha / 2)
        return stats.norm.cdf(abs(g) / se - crit) + stats.norm.cdf(-abs(g) / se - crit)

    def mde(n1, n2, power=0.8):
        lo, hi = 0.0, 3.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if power2(mid, n1, n2) < power:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    n_up = int(panel[panel.outcome == "upset"].match_id.nunique())
    n_ct = int(panel[panel.outcome == "control"].match_id.nunique())
    pw = R.save("sens_power", pd.DataFrame([
        dict(design=f"this corpus ({n_up} vs {n_ct})", n_upset=n_up,
             n_control=n_ct, mde_80=mde(n_up, n_ct),
             power_at_g_0_3=power2(.3, n_up, n_ct),
             power_at_g_0_5=power2(.5, n_up, n_ct)),
        dict(design="proposal target (50 vs 50)", n_upset=50, n_control=50,
             mde_80=mde(50, 50), power_at_g_0_3=power2(.3, 50, 50),
             power_at_g_0_5=power2(.5, 50, 50)),
        dict(design="proposal target (75 vs 75)", n_upset=75, n_control=75,
             mde_80=mde(75, 75), power_at_g_0_3=power2(.3, 75, 75),
             power_at_g_0_5=power2(.5, 75, 75))]))
    print("\n  power / minimum detectable effect at the match level:")
    show(pw)
    print(f"  => this corpus resolves only |g| >= {mde(n_up, n_ct):.2f} at 80% power.")
    print("     The correct statement is not 'there is no emergence signature' but")
    print("     'if one exists it is smaller than that, and this corpus cannot see it'.")
    return SPEC, PLACEBO, sens_wk, pw


def sprint5_figures(panel, R, rq1, RM, GAP, GLAB, trend, p_perm,
                    y, proba, auc, p_auc, null_auc, imp_macro, SPEC, sens_wk):
    """The eight Sprint-5 diagnostic figures.

    These are analysis figures, deliberately plainer than the Sprint-6
    manuscript figures produced by make_figures.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.metrics import roc_curve

    C_CONTROL, C_UPSET, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
    C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e6e6e3"
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.edgecolor": C_MUTED,
        "axes.linewidth": .8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": C_GRID,
        "grid.linewidth": .7, "axes.axisbelow": True, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold",
        "text.color": C_INK, "axes.labelcolor": C_MUTED,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight"})

    def save(fig, name):
        fig.savefig(CFG.fig_dir / f"{name}.png")
        plt.close(fig)
        step(f"figure -> {name}.png")

    A = Config.ALPHA
    PF, PO = Config.PRIMARY_FEATURE, Config.PRIMARY_ORDER

    # 1. RQ1 forest
    sub = rq1[rq1.term == "upset"].sort_values(["feature", "order"]).copy()
    sub["label"] = sub.feature + "  $\\ell$=" + sub.order.astype(str)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    se_std = ((sub.se / sub.coef.abs()).replace([np.inf, -np.inf], np.nan)
              * sub.coef_std.abs())
    ax.errorbar(sub.coef_std, np.arange(len(sub)), xerr=1.96 * se_std, fmt="o",
                ms=5.5, color=C_UPSET, ecolor=C_MUTED, elinewidth=1,
                capsize=2.5, zorder=3)
    ax.axvline(0, color=C_INK, lw=1.1)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub.label, fontsize=9)
    ax.set_xlabel(r"upset effect, standardised ($\beta_1/\hat\sigma_u$)")
    ax.set_title("RQ1: no upset effect is distinguishable from zero")
    ax.invert_yaxis()
    ax.text(.02, .02, f"0 of {len(rq1)} tests reach p<{A}\n(bars are 95% CI)",
            transform=ax.transAxes, fontsize=9, color=C_MUTED, va="bottom")
    fig.tight_layout()
    save(fig, "rq1_forest")

    # 2. RQ2 interaction
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, role in zip(axes, ["underdog", "favourite"]):
        for grp, col in [("control", C_CONTROL), ("upset", C_UPSET)]:
            s = RM[(RM.role == role) & (RM.outcome == grp)].groupby("order").psi_z
            m, se = s.mean(), s.sem()
            ax.errorbar(m.index, m.values, yerr=1.96 * se.values, marker="o",
                        ms=6, color=col, capsize=3, lw=1.8, label=grp)
        ax.set_xticks(list(Config.ORDERS))
        ax.set_xlabel(r"emergence order $\ell$")
        ax.set_title(role)
    axes[0].set_ylabel(r"$\Psi$ (z-scored within order)")
    axes[0].legend(fontsize=9)
    fig.suptitle("RQ2: the order profile is the same for upsets and controls",
                 y=1.02, fontsize=10.5, color=C_MUTED)
    fig.tight_layout()
    save(fig, "rq2_interaction")

    # 3. RQ3 trajectories
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    for ax, role in zip(axes[:2], ["underdog", "favourite"]):
        cur, lab = trajectories(panel, PF, PO, role=role)
        for grp, col in [("control", C_CONTROL), ("upset", C_UPSET)]:
            Y = np.vstack([cur[m] for m in cur if lab[m] == grp])
            mu, se = Y.mean(0), Y.std(0) / np.sqrt(len(Y))
            ax.plot(GRID, mu, color=col, lw=1.9, label=f"{grp} (n={len(Y)})")
            ax.fill_between(GRID, mu - 1.96 * se, mu + 1.96 * se, color=col,
                            alpha=.18, lw=0)
        ax.set_xlabel("normalised match time")
        ax.set_title(role)
    axes[0].set_ylabel(rf"$\Psi^{{({PO})}}$, {PF} (nats)")
    axes[0].legend(fontsize=9)
    ax = axes[2]
    for grp, col in [("control", C_CONTROL), ("upset", C_UPSET)]:
        Y = np.vstack([GAP[m] for m in GAP if GLAB[m] == grp])
        mu, se = Y.mean(0), Y.std(0) / np.sqrt(len(Y))
        ax.plot(GRID, mu, color=col, lw=1.9, label=grp)
        ax.fill_between(GRID, mu - 1.96 * se, mu + 1.96 * se, color=col,
                        alpha=.18, lw=0)
    ax.axhline(0, color=C_INK, lw=1.1, ls="--")
    ax.set_xlabel("normalised match time")
    ax.set_ylabel(r"$\Psi_{und} - \Psi_{fav}$ (nats)")
    ax.set_title("the tipping-point gap")
    ax.legend(fontsize=9)
    fig.suptitle("RQ3: mean levels are indistinguishable, but the gap DRIFTS "
                 "in opposite directions", y=1.03, fontsize=10.5, color=C_MUTED)
    fig.tight_layout()
    save(fig, "rq3_trajectories")

    # 4. RQ3 gap trend
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    cp = R.tables["rq3_changepoints"]
    for i, (grp, col) in enumerate([("control", C_CONTROL), ("upset", C_UPSET)]):
        v = cp[cp.outcome == grp].gap_trend.to_numpy()
        jit = (np.random.default_rng(3).random(len(v)) - .5) * .26
        axL.scatter(np.full(len(v), i) + jit, v, s=30, color=col, alpha=.72,
                    edgecolor="white", linewidth=.7, zorder=3, label=grp)
        axL.hlines(np.mean(v), i - .3, i + .3, color=col, lw=2.6, zorder=4)
    axL.axhline(0, color=C_INK, lw=1.1, ls="--")
    axL.set_xticks([0, 1])
    axL.set_xticklabels(["control", "upset"])
    axL.set_xlim(-.6, 1.6)
    axL.set_ylabel(r"slope of $\Psi_{und}-\Psi_{fav}$ (nats / match)")
    axL.set_title(f"{PF}, $\\ell$={PO}  (permutation p={p_perm:.3f})")
    srt = trend.sort_values("g").reset_index(drop=True)
    cols = [C_UPSET if q < A else C_MUTED for q in srt.p_fdr]
    axR.barh(np.arange(len(srt)), srt.g, color=cols, edgecolor="white",
             linewidth=.8)
    axR.axvline(0, color=C_INK, lw=1.1)
    axR.set_yticks(np.arange(len(srt)))
    axR.set_yticklabels([f"{r.feature}  $\\ell$={r.order}"
                         for r in srt.itertuples()], fontsize=8)
    axR.set_xlabel("Hedges' g (upset - control) on the gap slope")
    axR.set_title(f"{(srt.p_fdr < A).sum()} of {len(srt)} survive FDR (orange)")
    fig.suptitle("RQ3: the emergence gap trends downward in upsets and upward "
                 "in controls", y=1.03, fontsize=10.5, color=C_MUTED)
    fig.tight_layout()
    save(fig, "rq3_gap_trend")

    # 5. RQ4
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    d = panel[panel.macro_feature == PF]
    ml = d[d.role == "favourite"].groupby(["match_id", "outcome"]).psi_l1.mean().reset_index()
    dc = d.groupby(["match_id", "outcome"]).dc_ic.mean().reset_index()
    for ax, tab, col, lab in [(axes[0], ml, "psi_l1", r"favourite $\Psi^{(1)}$ (nats)"),
                              (axes[1], dc, "dc_ic", "disruption coefficient $dc_{ic}$")]:
        for i, (grp, cc) in enumerate([("control", C_CONTROL), ("upset", C_UPSET)]):
            v = tab[tab.outcome == grp][col].dropna().to_numpy()
            jit = (np.random.default_rng(2).random(len(v)) - .5) * .26
            ax.scatter(np.full(len(v), i) + jit, v, s=28, color=cc, alpha=.7,
                       edgecolor="white", linewidth=.7, zorder=3)
            if len(v):
                ax.hlines(np.median(v), i - .3, i + .3, color=cc, lw=2.4, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["control", "upset"])
        ax.set_xlim(-.6, 1.6)
        ax.set_ylabel(lab)
    axes[0].set_title("Is the favourite's emergence depressed in upsets?")
    axes[1].set_title("Is the underdog disrupting more?")
    fig.suptitle("RQ4: each dot is one match; neither contrast survives correction",
                 y=1.03, fontsize=10.5, color=C_MUTED)
    fig.tight_layout()
    save(fig, "rq4_disruption")

    # 6. RQ5
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fpr, tpr, _ = roc_curve(y, proba)
    axes[0].plot(fpr, tpr, color=C_UPSET, lw=2, label=f"RF (AUC={auc:.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", color=C_MUTED, lw=1, label="chance")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC")
    axes[0].legend(fontsize=9, loc="lower right")
    axes[1].hist(null_auc, bins=26, color=C_CONTROL, alpha=.6,
                 label="shuffled labels")
    axes[1].axvline(auc, color=C_UPSET, lw=2.2, label=f"observed {auc:.3f}")
    axes[1].axvline(np.percentile(null_auc, 95), color=C_MUTED, ls="--", lw=1.2,
                    label="null 95th pct")
    axes[1].set_xlabel("ROC-AUC")
    axes[1].set_ylabel("permutations")
    axes[1].set_title(f"Permutation null (p={p_auc:.2f})")
    axes[1].legend(fontsize=8.5)
    top = imp_macro.iloc[::-1]
    axes[2].barh(np.arange(len(top)), top.importance, xerr=top.sd, color=C_THIRD,
                 edgecolor="white", linewidth=.8,
                 error_kw=dict(ecolor=C_MUTED, lw=.8))
    axes[2].set_yticks(np.arange(len(top)))
    axes[2].set_yticklabels(top.block, fontsize=9)
    axes[2].axvline(0, color=C_INK, lw=1)
    axes[2].set_xlabel("out-of-fold AUC drop when block is permuted")
    axes[2].set_title("Macro-feature importance (grouped, OOF)")
    fig.suptitle("RQ5: classification is indistinguishable from chance", y=1.03,
                 fontsize=10.5, color=C_MUTED)
    fig.tight_layout()
    save(fig, "rq5_classification")

    # 7. specification curve
    S = SPEC.sort_values("g").reset_index(drop=True)
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2.1, 1]})
    sig = S.p < A
    axA.scatter(np.arange(len(S))[~sig], S.g[~sig], s=7, color=C_MUTED,
                alpha=.45, label=f"p >= {A}")
    axA.scatter(np.arange(len(S))[sig], S.g[sig], s=11, color=C_UPSET,
                label=f"p < {A} ({sig.sum()} of {len(S)})")
    axA.axhline(0, color=C_INK, lw=1)
    axA.set_ylabel("Hedges' g (upset - control)")
    axA.set_title("Specification curve: every defensible analytic choice")
    axA.legend(fontsize=9, loc="lower right")
    # orange already means "significant" above, so the rows below are encoded
    # by POSITION only -- reusing hue here would overload the colour channel
    for i, name in enumerate(["mean", "median", "trim10"]):
        mask = (S.summary == name).to_numpy()
        sig_m = mask & sig.to_numpy()
        axB.scatter(np.arange(len(S))[mask & ~sig.to_numpy()],
                    np.full((mask & ~sig.to_numpy()).sum(), i), s=5,
                    color=C_MUTED, alpha=.45)
        axB.scatter(np.arange(len(S))[sig_m], np.full(sig_m.sum(), i), s=9,
                    color=C_UPSET)
    rates = S.groupby("summary").p.apply(lambda z: (z < A).mean())
    axB.set_yticks([0, 1, 2])
    axB.set_yticklabels([f"{n}  ({rates.get(n, 0):.0%} sig)"
                         for n in ["mean", "median", "trim10"]], fontsize=8.5)
    axB.set_xlabel("specification (ordered by effect size)")
    axB.set_ylabel("within-match\nsummary", fontsize=9)
    axB.set_ylim(-.6, 2.6)
    axB.grid(False)
    axB.text(.01, .93, f"{int((S.g < 0).sum())} of {len(S)} specifications point "
                       "the same way (g<0)", transform=axB.transAxes,
             fontsize=8.5, color=C_MUTED, va="top")
    fig.tight_layout()
    save(fig, "sens_specification_curve")

    # 8. window / k sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    w = sens_wk[sens_wk.k == 4].sort_values("W")
    kk = sens_wk[sens_wk.W == 60].sort_values("k")
    for ax, tab, xcol, xlab in [(axes[0], w, "W", "window length W (s)"),
                                (axes[1], kk, "k", "KSG neighbours k")]:
        for f, col in [("V_all", C_CONTROL), ("V_cvel", C_THIRD)]:
            s = tab[tab.feature == f]
            if not len(s):
                continue
            ax.errorbar(s[xcol], s.coef_std,
                        yerr=1.96 * (s.se / s.coef.abs() * s.coef_std.abs()),
                        marker="o", ms=6, lw=1.8, color=col, capsize=3, label=f)
        ax.axhline(0, color=C_INK, lw=1.1)
        ax.set_xlabel(xlab)
        ax.set_ylabel("standardised upset effect")
    axes[0].legend(fontsize=9)
    fig.suptitle("Sensitivity: the null holds across window length and k",
                 y=1.03, fontsize=10.5, color=C_MUTED)
    fig.tight_layout()
    save(fig, "sens_window_k")


def stage_stats() -> None:
    banner("STAGE: stats (Sprint 5) -- RQ1-RQ5 and the sensitivity battery")
    CFG.stat_dir.mkdir(parents=True, exist_ok=True)
    CFG.tab_dir.mkdir(parents=True, exist_ok=True)
    CFG.fig_dir.mkdir(parents=True, exist_ok=True)
    R = Results(CFG.tab_dir)

    corpus, panel, meta = load_corpus_and_panel()

    # The RQ models need outcome variation and enough matches to estimate a
    # between-match variance component. A truncated corpus (e.g. --limit on an
    # earlier stage) produces a singular design; say so instead of surfacing a
    # LinAlgError from deep inside statsmodels.
    counts = panel.groupby("outcome").match_id.nunique().to_dict()
    n_up, n_ct = counts.get("upset", 0), counts.get("control", 0)
    if n_up < 5 or n_ct < 5:
        raise SystemExit(
            f"\nThe stats stage needs at least 5 upsets and 5 controls; this "
            f"corpus has {n_up} upset and {n_ct} control matches.\n"
            f"  {CFG.caus_dir} holds {panel.match_id.nunique()} matches.\n"
            "  If you ran an earlier stage with --limit, the emergence output is a\n"
            "  truncated slice of the corpus. Re-run without --limit, or point\n"
            "  --root at a tree containing the full data/caus_emg/.\n"
            "  The other stages (validate, ksg, emergence, figures) work on any subset.")

    panel, _ = diagnose_clock_void(panel, R)
    rq1, rq1_lrt = rq1_emergence_signatures(panel, R)
    RM, wald = rq2_order_of_emergence(panel, R)
    fda, cp, tt, trend, p_perm, GAP, GLAB = rq3_temporal_dynamics(panel, R)
    rq4 = rq4_disruption(panel, R)
    MF, X, y, proba, auc, ap, p_auc, null_auc, imp_macro = rq5_classification(
        panel, meta, R)
    SPEC, PLACEBO, sens_wk, pw = sensitivity(corpus, panel, R)

    banner("Sprint 5 results summary")
    A = Config.ALPHA
    summary = pd.DataFrame([
        dict(RQ="RQ1", question="emergence signatures",
             method="linear mixed models, random intercept per match",
             n_tests=len(rq1), n_sig_raw=int((rq1.p < A).sum()),
             n_sig_fdr=int((rq1.p_fdr < A).sum()), min_p=float(rq1.p.min()),
             verdict="no effect detected"),
        dict(RQ="RQ2", question="order of emergence",
             method="repeated-measures ANOVA (mixed model), Psi z-scored within order",
             n_tests=int(len(wald)),
             n_sig_raw=int((wald.pvalue < A).sum()),
             n_sig_fdr=int((wald.p_fdr < A).sum()),
             min_p=float(wald.pvalue.min()),
             verdict="order profile does not differ by outcome"),
        dict(RQ="RQ3a", question="temporal dynamics: levels / tipping point",
             method="functional permutation tests + binary-segmentation change points",
             n_tests=int(len(fda) + len(tt)),
             n_sig_raw=int((fda.sup_p < A).sum() + (tt.p < A).sum()),
             n_sig_fdr=int((fda.sup_p_fdr < A).sum() + (tt.p_fdr < A).sum()),
             min_p=float(min(fda.sup_p.min(), tt.p.min())),
             verdict="no tipping point; mean curves indistinguishable"),
        dict(RQ="RQ3b", question="temporal dynamics: gap TREND",
             method="per-match slope of Psi_und - Psi_fav, FDR + permutations",
             n_tests=len(trend), n_sig_raw=int((trend.p < A).sum()),
             n_sig_fdr=int((trend.p_fdr < A).sum()), min_p=float(trend.p.min()),
             verdict=(f"EFFECT DETECTED (perm p={p_perm:.3f}); gap drifts down in upsets"
                      if (p_perm < A and (trend.p_fdr < A).any())
                      else f"no effect detected (perm p={p_perm:.3f})")),
        dict(RQ="RQ4", question="disruption of the favourite",
             method="favourite-only mixed models + disruption coefficient",
             n_tests=len(rq4), n_sig_raw=int((rq4.p < A).sum()),
             n_sig_fdr=int((rq4.p_fdr < A).sum()), min_p=float(rq4.p.min()),
             verdict="no effect; third condition unavailable by design"),
        dict(RQ="RQ5", question="classification / spatial signatures",
             method="random forest, stratified CV + permutation null + temporal blocking",
             n_tests=1, n_sig_raw=int(p_auc < A), n_sig_fdr=int(p_auc < A),
             min_p=float(p_auc), verdict=f"chance-level (AUC {auc:.3f})"),
    ])
    R.save("summary_all_rqs", summary)
    show(summary, n=10)

    banner("Sprint 5 figures")
    sprint5_figures(panel, R, rq1, RM, GAP, GLAB, trend, p_perm, y, proba,
                    auc, p_auc, null_auc, imp_macro, SPEC, sens_wk)

    # manifest of everything written
    man = []
    for p in sorted(glob.glob(str(CFG.stat_dir / "**" / "*.*"), recursive=True)):
        man.append(dict(path=os.path.relpath(p, CFG.root),
                        kind=Path(p).suffix.lstrip("."),
                        size_kb=round(os.path.getsize(p) / 1024, 1)))
    pd.DataFrame(man).to_csv(CFG.stat_dir / "MANIFEST.csv", index=False)
    print(f"\n  {len(man)} files in {CFG.stat_dir}")
    print(f"  results tables: {len(R.tables)}   "
          f"figures: {len(list(CFG.fig_dir.glob('*.png')))}")


# ===========================================================================
# STAGE: figures  (Sprint 6)
#
# The six manuscript figures. Delegates to make_figures.py so there is one
# definition of each figure; that module is the Sprint-6 deliverable and is
# also runnable on its own.
# ===========================================================================

def stage_figures() -> None:
    banner("STAGE: figures (Sprint 6) -- manuscript figures")
    candidates = [CFG.src / "make_figures.py",
                  CFG.src / "sprint6" / "make_figures.py"]
    mod_path = next((p for p in candidates if p.exists()), None)
    if mod_path is None:
        step("make_figures.py not found; skipping the manuscript figures.")
        step(f"looked in: {', '.join(str(c) for c in candidates)}")
        return

    out_dir = mod_path.parent / "manuscript" / "figures"
    step(f"using {mod_path}")
    step(f"output -> {out_dir}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("make_figures", mod_path)
    mf = importlib.util.module_from_spec(spec)
    sys.modules["make_figures"] = mf
    spec.loader.exec_module(mf)

    out_dir.mkdir(parents=True, exist_ok=True)
    mf.set_style()
    panel = pd.read_parquet(CFG.stat_dir / "analysis_panel.parquet")
    step(f"analysis panel: {len(panel):,} rows, {panel.match_id.nunique()} matches")
    mf.fig1_redundancy_floor(panel, out_dir)
    mf.fig2_rq1_forest(CFG.tab_dir, out_dir)
    mf.fig3_gap_trend(CFG.tab_dir, out_dir)
    mf.fig4_gap_trajectory(panel, out_dir)
    mf.fig5_classification(CFG.tab_dir, out_dir)
    mf.fig6_sensitivity(CFG.tab_dir, out_dir)
    print(f"\n  6 manuscript figures (PDF + PNG) -> {out_dir}")


# ===========================================================================
# main
# ===========================================================================

STAGES = ["validate", "preprocess", "ksg", "emergence", "stats", "figures"]
DEFAULT_STAGES = ["validate", "ksg", "emergence", "stats", "figures"]


def main(argv=None):
    global CFG
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(
        description="Giant Killing causal-emergence pipeline (Sprints 2-6).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
stages
  validate    estimator correctness checks (Sprint 3 validation report)
  preprocess  1 Hz tracking + feature bundles  -> data/preprocessed/
  ksg         baseline MI for every window     -> data/ksg/
  emergence   Psi at orders 1,2,3 + EAI/DC/SRR -> data/caus_emg/
  stats       RQ1-RQ5 + sensitivity + figures  -> data/stat_analysis/
  figures     the six manuscript figures       -> manuscript/figures/
  all         validate, ksg, emergence, stats, figures (skips preprocess)

examples
  python pipeline.py --stage all
  python pipeline.py --stage all --limit 8 --quick     # ~10 min smoke test
  python pipeline.py --stage stats figures             # re-run from Sprint-4 data
  python pipeline.py --stage emergence --overwrite     # force recomputation
""")
    ap.add_argument("--stage", nargs="+", default=["all"],
                    choices=STAGES + ["all"],
                    help="stage(s) to run, in the given order (default: all)")
    ap.add_argument("--root", default=str(here.parent),
                    help="project root containing src/, statsbomb-360/, open-data/")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N matches (smoke tests)")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute matches that already have output")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel workers for the emergence stage")
    ap.add_argument("--quick", action="store_true",
                    help="reduced permutation counts; NOT the published numbers")
    ap.add_argument("--clock", choices=["legacy", "corrected"], default="legacy",
                    help="match-clock formula. 'legacy' reproduces the published "
                         "corpus but double-counts the first half, opening a ~42 min "
                         "interpolated void (~29%% of windows). 'corrected' removes "
                         "it and changes every number.")
    args = ap.parse_args(argv)

    Config.CLOCK_MODE = args.clock
    root = Path(args.root).resolve()
    CFG = Config(root, quick=args.quick)
    CFG.mkdirs()

    stages = DEFAULT_STAGES if "all" in args.stage else args.stage

    banner("Giant Killing -- causal emergence pipeline")
    print(f"  root    : {root}")
    print(f"  stages  : {' -> '.join(stages)}")
    print(f"  360 data: {len(local_360_ids())} matches in {CFG.sb_360}")
    print(f"  events  : {len(list(CFG.sb_events.glob('*.json')))} files "
          f"in {CFG.sb_events}")
    if args.limit:
        print(f"  limit   : {args.limit} matches")
    print(f"  clock   : {Config.CLOCK_MODE}"
          + ("  (reproduces the published corpus; see REPRODUCE.md §7)"
             if Config.CLOCK_MODE == "legacy" else
             "  (NOT the published corpus -- artefact removed)"))
    if args.quick:
        print("  QUICK MODE: permutation counts reduced. Results are indicative,")
        print("              not the published numbers.")

    t0 = time.time()
    for s in stages:
        if s == "validate":
            stage_validate()
        elif s == "preprocess":
            stage_preprocess(limit=args.limit)
        elif s == "ksg":
            stage_ksg(limit=args.limit, overwrite=args.overwrite)
        elif s == "emergence":
            stage_emergence(limit=args.limit, overwrite=args.overwrite,
                            n_jobs=args.jobs)
        elif s == "stats":
            stage_stats()
        elif s == "figures":
            stage_figures()

    banner(f"pipeline complete in {(time.time() - t0) / 60:.1f} min")
    print(f"  tables  : {CFG.tab_dir}")
    print(f"  figures : {CFG.fig_dir}")
    print(f"  panel   : {CFG.stat_dir / 'analysis_panel.parquet'}")


if __name__ == "__main__":
    main()
