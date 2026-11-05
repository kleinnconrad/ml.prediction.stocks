"""
Quarterly variable review: champion-challenger test of the Step 1 feature set.

The champion is the current `step1_features` list, evaluated with the production model
(pooled L2 logistic regression on within-date ranks, src/modeling/step1_pooled.py). The
challengers are feature lists produced by fixed selection procedures from a pre-registered
candidate pool (the feature registry of src/processing/features.py, or the list in
`review_candidate_features`):

- `stepwise`: forward selection on the mean per-date AUC of inner validation blocks, with
  an ANOVA prefilter refitted inside each inner training split;
- `lasso`: L1 logistic regression over the pool, its penalty chosen on the same inner
  splits; the production model is refitted on the nonzero coefficients (relaxed lasso);
- `fm_screen`: per-date rank correlations with the label, kept if the Newey-West
  t-statistic of their mean exceeds `review_fm_t_hurdle`;
- `all_candidates`: the whole pool;
- `manual:<name>`: the fixed lists in `review_manual_sets`.

Every procedure runs inside each yearly walk-forward training window (the protocol of the
production validation), so the comparison includes the optimism of the selection itself.
All lists are scored with the production model on the same test rows. The paired per-date
AUC differences against the champion feed the switch rule: their mean, their block-bootstrap
interval at a Bonferroni level corrected for every challenger comparison since the champion
was adopted (m_cum, counted from the committed reports of the earlier reviews), the test
years in which the challenger is ahead, the fresh test dates after `step1_features_adopted`
and the top-share hit rates. A switch only proposes a settings change; the user adopts it by
merging a pull request (.github/workflows/05_review_variables.yml).
"""
import json
import logging
import math
import multiprocessing
import os
import re
import textwrap
import time
import warnings
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression

from ..ingestion.market_api import PricePanel
from ..processing.features import (FEATURE_FAMILIES, FEATURE_FUNCTIONS, StockSeries, check_market_index,
                                   label_column, month_end_dates, rank_long_panel, stock_panel_rows,
                                   validate_feature_names)
from .step1_pooled import (block_bootstrap_ci, fit_pooled_model, per_date_table, predict_probability,
                           walk_forward_windows)

logger = logging.getLogger(__name__)

CHAMPION = 'champion'
SELECTION_PROCEDURES = ('stepwise', 'lasso', 'fm_screen', 'all_candidates')
MANUAL_PREFIX = 'manual:'
FINAL_WINDOW = 'final'
DECISION_KEEP = 'keep'
DECISION_SWITCH = 'switch'
# The lasso uses liblinear's coordinate descent; its seed (review_lasso_seed) fixes the
# coordinate order and with it the result
LASSO_SOLVER = 'liblinear'
# AUC of a model without features (constant scores)
NO_SKILL_AUC = 0.5
# Worker processes use one BLAS thread each; several threads per worker oversubscribe the CPUs
WORKER_THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
# Switch rule: (b) needs a lower bound above, (d) a fresh mean difference not below this value
NO_DIFFERENCE = 0.0
# Status of condition (d) while fewer than review_fresh_check_min_dates fresh test dates exist
FRESH_NOT_YET_AVAILABLE = 'not yet available'
CONDITION_LABELS = {
    'a_historical_mean_d': '(a) historical mean d >= review_auc_hurdle',
    'b_ci_lower_above_zero': '(b) CI lower bound > 0 at the level 1 - review_family_alpha / m_cum',
    'c_winning_years': '(c) mean d positive in at least review_min_winning_years test years',
    'd_fresh_evidence': '(d) fresh mean d >= 0 and fresh top-share hit rate not below the champion',
    'e_hit_rate': '(e) top-share hit rate on all test dates not below the champion',
    'f_list_differs': 'final list differs from the champion',
}
REVIEW_PARAMETER_KEYS = (
    'horizon_days', 'threshold', 'step1_features', 'step1_features_adopted', 'step1_history_years',
    'feature_warmup_years', 'step1_l2_c', 'step1_solver_tol', 'step1_max_iter', 'step1_up_share',
    'step1_min_price_eur', 'step1_min_history_days', 'step1_ffill_limit_days', 'step1_min_stocks_per_date',
    'step1_bootstrap_block_months', 'step1_bootstrap_replications', 'step1_bootstrap_seed', 'market_index_ticker',
    'review_candidate_features', 'review_validation_years', 'review_inner_splits', 'review_inner_validation_months',
    'review_inner_min_train_months', 'review_anova_k', 'review_stepwise_min_gain', 'review_max_features',
    'review_lasso_c_grid', 'review_lasso_max_features', 'review_lasso_seed', 'review_fm_t_hurdle',
    'review_fm_nw_lag', 'review_manual_sets', 'review_auc_hurdle', 'review_family_alpha', 'review_min_winning_years',
    'review_fresh_check_min_dates',
)
# The committed reports of the reviews (data/processed/variable_reviews/review_YYYY-MM-DD.json)
REPORT_PATTERN = 'review_*.json'
FEATURE_NAME_PATTERN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
SETTINGS_FEATURES_LINE = re.compile(r'^(step1_features:[ \t]*)\[[^\]\n]*\](.*)$')
SETTINGS_ADOPTED_LINE = re.compile(r'^(step1_features_adopted:[ \t]*)("[^"\n]*"|\'[^\'\n]*\'|[^\s#]+)(.*)$')


def _iso(value: Any) -> Optional[str]:
    """Returns the ISO date of a timestamp-like value, or None for a missing value."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _num(value: Any, digits: Optional[int] = None) -> Optional[float]:
    """Returns a finite Python float (optionally rounded), or None."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


def _finite_mean(values: Iterable[float]) -> float:
    """Mean of the finite values, NaN if there are none."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float('nan')


@contextmanager
def _single_threaded_workers() -> Iterator[None]:
    """Sets one BLAS thread for the processes started inside the block and restores the environment after it."""
    previous = {name: os.environ.get(name) for name in WORKER_THREAD_VARIABLES}
    os.environ.update({name: '1' for name in WORKER_THREAD_VARIABLES})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _worker_peak_mb() -> Optional[float]:
    """
    Peak resident memory of the current process in MB (None where the platform reports none).

    On Linux this is VmHWM of /proc/self/status, the peak of the process's own address
    space. ru_maxrss would also count the parent's memory at the moment a spawned worker
    was forked.
    """
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmHWM:'):
                return round(int(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource
    except ImportError:
        return None
    # ru_maxrss is in kilobytes on Linux and in bytes on macOS; this fallback assumes kilobytes
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _parallel_map(function: Callable[[Any], Any], tasks: Sequence[Any], n_jobs: int,
                  initializer: Callable[..., None], initargs: Tuple[Any, ...],
                  describe: Optional[Callable[[Any], str]] = None) -> List[Any]:
    """
    Maps `function` over `tasks` in worker processes, or in this process if n_jobs <= 1.

    The workers are started with the 'spawn' method and one BLAS thread each, so their
    numerical libraries do not inherit the thread pools of this process. Progress is
    logged by this process as the results arrive.

    Args:
        function (Callable[[Any], Any]): Module-level task function.
        tasks (Sequence[Any]): Task arguments.
        n_jobs (int): Number of worker processes.
        initializer (Callable[..., None]): Sets up the shared data of a worker.
        initargs (Tuple[Any, ...]): Arguments of the initializer.
        describe (Optional[Callable[[Any], str]]): Log line of a finished task (None: no log).

    Returns:
        List[Any]: The results in task order.
    """
    results: List[Any] = [None] * len(tasks)
    if n_jobs <= 1 or len(tasks) <= 1:
        initializer(*initargs)
        for i, task in enumerate(tasks):
            results[i] = function(task)
            if describe:
                logger.info(f"[{i + 1}/{len(tasks)}] {describe(results[i])}")
        return results
    with _single_threaded_workers(), ProcessPoolExecutor(
            max_workers=min(n_jobs, len(tasks)), mp_context=multiprocessing.get_context('spawn'),
            initializer=initializer, initargs=initargs) as executor:
        futures: Dict[Future, int] = {executor.submit(function, task): i for i, task in enumerate(tasks)}
        finished = False
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if describe:
                    logger.info(f"[{done}/{len(tasks)}] {describe(results[futures[future]])}")
            finished = True
        finally:
            # After a failed task, the tasks that have not started yet are dropped
            if not finished:
                for future in futures:
                    future.cancel()
    return results


# ----------------------------------------------------------------------------------------
# Candidate pool and long panel
# ----------------------------------------------------------------------------------------
def candidate_pool(settings: Dict[str, Any], override: Optional[Sequence[str]] = None) -> List[str]:
    """
    The pre-registered candidate pool of the review.

    Args:
        settings (Dict[str, Any]): Loaded settings (review_candidate_features: 'all' for the
            whole registry, or a list of names).
        override (Optional[Sequence[str]]): Names that replace the pool (local testing).

    Returns:
        List[str]: The candidates in registry order (or in the given order).

    Raises:
        ValueError: On unknown or duplicate names, or an invalid setting.
    """
    if override:
        return validate_feature_names(override)
    value = settings['review_candidate_features']
    if isinstance(value, str):
        if value != 'all':
            raise ValueError(f"review_candidate_features must be 'all' or a list of feature names, not {value!r}.")
        return list(FEATURE_FUNCTIONS)
    return validate_feature_names(value)


def manual_sets(settings: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    The fixed challenger lists of `review_manual_sets`.

    Args:
        settings (Dict[str, Any]): Loaded settings.

    Returns:
        Dict[str, List[str]]: Feature list per set name.

    Raises:
        ValueError: On a malformed setting or unknown feature names.
    """
    value = settings.get('review_manual_sets') or {}
    if not isinstance(value, dict):
        raise ValueError("review_manual_sets must map a name to a list of feature names.")
    return {str(name): validate_feature_names(features) for name, features in value.items()}


@dataclass
class ReviewPanel:
    """
    Long month-end panel of the review: the eligible rows with the within-date ranks of all computed features.

    Attributes:
        ranks (np.ndarray): Rank values, rows x features (0 for a missing value).
        feature_names (List[str]): Features of the rank columns.
        dates (pd.DatetimeIndex): Date of every row.
        tickers (np.ndarray): Ticker of every row.
        label_end (np.ndarray): Label end date of every row (datetime64[ns]; NaT if unknown).
        y (np.ndarray): Labels (NaN if unknown).
        fwd (np.ndarray): Forward returns (NaN if unknown).
        scoring_date (pd.Timestamp): Last date of the union calendar.
        missing_share (Dict[str, float]): Share of rows without a value per feature.
        n_tickers (int): Tickers with at least one eligible row.
        worker_peak_mb (Optional[float]): Largest peak memory of a process that computed rows.
    """
    ranks: np.ndarray
    feature_names: List[str]
    dates: pd.DatetimeIndex
    tickers: np.ndarray
    label_end: np.ndarray
    y: np.ndarray
    fwd: np.ndarray
    scoring_date: pd.Timestamp
    missing_share: Dict[str, float]
    n_tickers: int
    worker_peak_mb: Optional[float] = None

    def columns(self, names: Sequence[str]) -> List[int]:
        """
        Column positions of features in `ranks`.

        Args:
            names (Sequence[str]): Feature names.

        Returns:
            List[int]: Positions in the order of `names`.
        """
        position = {name: i for i, name in enumerate(self.feature_names)}
        return [position[name] for name in names]

    def label_frame(self, rows: np.ndarray) -> pd.DataFrame:
        """
        The labels and forward returns of some rows, indexed by (date, ticker).

        Args:
            rows (np.ndarray): Row positions.

        Returns:
            pd.DataFrame: Columns y and fwd.
        """
        index = pd.MultiIndex.from_arrays([self.dates[rows], self.tickers[rows]], names=['date', 'ticker'])
        return pd.DataFrame({'y': self.y[rows], 'fwd': self.fwd[rows]}, index=index)


_PANEL_CONTEXT: Dict[str, Any] = {}


def _init_panel_worker(context: Dict[str, Any]) -> None:
    """Stores the data shared by all panel tasks of a worker process."""
    _PANEL_CONTEXT.clear()
    _PANEL_CONTEXT.update(context)


def _panel_rows_task(task: Tuple[str, pd.Series, Optional[pd.Series], Optional[pd.Series], Optional[pd.Series],
                                 Optional[pd.Series]]) -> Tuple[str, pd.DataFrame, Optional[float]]:
    """Computes the eligible month-end rows of one stock (worker task); also returns the worker's peak memory."""
    ticker, close, volume, open_, high, low = task
    context = _PANEL_CONTEXT
    series = StockSeries.build(close, volume=volume, open_=open_, high=high, low=low, market=context['market'])
    rows = stock_panel_rows(series, context['names'], context['calendar'], context['dates'], context['settings'])
    return ticker, rows, _worker_peak_mb()


def build_review_panel(price_panel: PricePanel, feature_names: Sequence[str], settings: Dict[str, Any],
                       n_jobs: int = 1) -> ReviewPanel:
    """
    Builds the long panel of the eligible month-end rows for all features at once.

    Each stock's own-day features are computed and only its eligible rows at the month-ends
    of the union calendar (the last one is the scoring date) are kept, so no wide frame per
    feature is built. The ranks are the within-date percentile ranks minus 0.5 over the
    eligible rows that have a value; a missing value gets the neutral rank 0. For the
    `step1_features` this reproduces the long panel of the production model.

    Args:
        price_panel (PricePanel): Prices, volumes and the market index.
        feature_names (Sequence[str]): Features to compute (pool, champion and manual sets).
        settings (Dict[str, Any]): Loaded settings.
        n_jobs (int): Worker processes for the per-stock computation.

    Returns:
        ReviewPanel: The ranked long panel.

    Raises:
        ValueError: If the panel holds no prices or no eligible rows, or the market index
            of the market-relative features is missing.
    """
    names = validate_feature_names(feature_names)
    check_market_index(price_panel, names)
    close = price_panel.close.sort_index()
    calendar = pd.DatetimeIndex(close.dropna(how='all').index)
    if calendar.empty:
        raise ValueError("The price panel contains no prices; the variable review cannot run.")
    dates = month_end_dates(calendar)

    def column(frame: Optional[pd.DataFrame], ticker: str) -> Optional[pd.Series]:
        return frame[ticker] if frame is not None and ticker in frame.columns else None

    tasks = [(ticker, close[ticker], column(price_panel.volume, ticker), column(price_panel.open, ticker),
              column(price_panel.high, ticker), column(price_panel.low, ticker))
             for ticker in sorted(close.columns) if close[ticker].notna().any()]
    context = {'names': names, 'calendar': calendar, 'dates': dates, 'settings': settings,
               'market': price_panel.market}
    logger.info(f"Computing {len(names)} features of {len(tasks)} stocks at {len(dates)} month-ends "
                f"with {max(1, n_jobs)} processes.")
    results = _parallel_map(_panel_rows_task, tasks, n_jobs, _init_panel_worker, (context,))
    frames = [rows.assign(ticker=ticker) for ticker, rows, _ in results if not rows.empty]
    if not frames:
        raise ValueError("No stock is eligible on any month-end; the variable review cannot run.")
    long = pd.concat(frames)
    long.index.name = 'date'
    long = long.set_index('ticker', append=True).sort_index()
    raw = long[names]
    missing_share = {name: float(share) for name, share in raw.isna().mean().items()}
    ranks = rank_long_panel(raw).to_numpy(dtype=float)
    fwd = long['fwd'].to_numpy(dtype=float)
    dates_index = pd.DatetimeIndex(long.index.get_level_values('date'))
    tickers = long.index.get_level_values('ticker').to_numpy()
    logger.info(f"Review panel: {len(long)} eligible month-end rows of {len(set(tickers))} stocks on "
                f"{dates_index.nunique()} dates, {len(names)} features.")
    return ReviewPanel(ranks=ranks, feature_names=names, dates=dates_index, tickers=tickers,
                       label_end=pd.to_datetime(long['label_end']).to_numpy(dtype='datetime64[ns]'),
                       y=label_column(fwd, float(settings['threshold'])), fwd=fwd, scoring_date=calendar[-1],
                       missing_share=missing_share, n_tickers=len(set(tickers)),
                       worker_peak_mb=max((peak for _, _, peak in results if peak is not None), default=None))


# ----------------------------------------------------------------------------------------
# Per-date AUC, inner splits and Newey-West t-statistic
# ----------------------------------------------------------------------------------------
class DateGroups:
    """
    Groups rows by date for fast per-date cross-sectional AUCs (Mann-Whitney with average ranks).

    A date counts if it has at least `min_stocks` rows and both classes, as in the
    production validation.

    Args:
        dates (np.ndarray): Date of every row.
        y (np.ndarray): 0/1 labels.
        min_stocks (int): Minimum rows of a date (`step1_min_stocks_per_date`).
    """

    def __init__(self, dates: np.ndarray, y: np.ndarray, min_stocks: int) -> None:
        codes, uniques = pd.factorize(np.asarray(dates), sort=True)
        self.codes = codes
        self.n_dates = len(uniques)
        self.y = np.asarray(y, dtype=float)
        count = np.bincount(codes, minlength=self.n_dates)
        self.positives = np.bincount(codes, weights=self.y, minlength=self.n_dates)
        self.negatives = count - self.positives
        self.valid = (count >= min_stocks) & (self.positives > 0) & (self.negatives > 0)

    @property
    def n_valid(self) -> int:
        """Number of dates with a defined AUC."""
        return int(self.valid.sum())

    def auc(self, scores: np.ndarray) -> np.ndarray:
        """
        Per-date AUC of the valid dates.

        Args:
            scores (np.ndarray): Score of every row (higher = more likely positive).

        Returns:
            np.ndarray: AUC of every valid date, in date order.
        """
        ranks = pd.Series(np.asarray(scores, dtype=float)).groupby(self.codes).rank(method='average').to_numpy()
        rank_sum = np.bincount(self.codes, weights=ranks * self.y, minlength=self.n_dates)
        positives, negatives = self.positives[self.valid], self.negatives[self.valid]
        return (rank_sum[self.valid] - positives * (positives + 1.0) / 2.0) / (positives * negatives)


def inner_date_splits(dates: np.ndarray, label_end: np.ndarray,
                      settings: Dict[str, Any]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """
    Expanding date-block splits inside one training window.

    The validation blocks are the last k blocks of `review_inner_validation_months`
    month-ends. The inner training rows of a block are dated before its first month-end and
    have a label that ended before it (label-end purge). k is the largest number up to
    `review_inner_splits` for which the first block keeps at least
    `review_inner_min_train_months` inner training month-ends. If not even one full block
    does, a single shorter block is used: the month-ends after the first
    `review_inner_min_train_months` inner training month-ends and their purge.

    Args:
        dates (np.ndarray): Date of every training row.
        label_end (np.ndarray): Label end date of every training row.
        settings (Dict[str, Any]): Loaded settings.

    Returns:
        Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]: (inner training rows,
        validation rows) per split, and a description of the splits. The list is empty if
        the window is too short for any split.
    """
    block = int(settings['review_inner_validation_months'])
    max_splits = int(settings['review_inner_splits'])
    min_train = int(settings['review_inner_min_train_months'])
    dates = np.asarray(dates, dtype='datetime64[ns]')
    label_end = np.asarray(label_end, dtype='datetime64[ns]')
    unique = np.unique(dates)

    def train_rows(start: np.datetime64) -> np.ndarray:
        return (dates < start) & (label_end < start)

    def n_train_dates(start: np.datetime64) -> int:
        return int(np.unique(dates[train_rows(start)]).size)

    bounds: List[Tuple[int, int]] = []
    for k in range(max_splits, 0, -1):
        if unique.size < k * block + 1:
            continue
        if n_train_dates(unique[unique.size - k * block]) >= min_train:
            bounds = [(unique.size - (k - i) * block, unique.size - (k - i - 1) * block) for i in range(k)]
            break
    shortened = False
    if not bounds:
        for start in range(1, unique.size):
            if n_train_dates(unique[start]) >= min_train:
                bounds, shortened = [(start, unique.size)], True
                break
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    blocks: List[Dict[str, Any]] = []
    for low, high in bounds:
        first, last = unique[low], unique[high - 1]
        train = np.flatnonzero(train_rows(first))
        validation = np.flatnonzero((dates >= first) & (dates <= last))
        splits.append((train, validation))
        blocks.append({'validation_first': _iso(first), 'validation_last': _iso(last),
                       'n_validation_dates': int(high - low), 'n_train_dates': n_train_dates(first),
                       'n_train': int(train.size), 'n_validation': int(validation.size),
                       'max_train_label_end': _iso(label_end[train].max()) if train.size else None})
    return splits, {'n_splits': len(splits), 'shortened': shortened, 'blocks': blocks}


def newey_west_t(values: np.ndarray, lag: int) -> Tuple[float, float]:
    """
    Mean and Newey-West (Bartlett kernel) t-statistic of a date-ordered series.

    Args:
        values (np.ndarray): The series; non-finite values are dropped.
        lag (int): Number of lags of the long-run variance.

    Returns:
        Tuple[float, float]: (mean, t-statistic); NaN with fewer than two values or no variation.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (float(x.mean()) if n else float('nan')), float('nan')
    errors = x - x.mean()
    variance = errors @ errors / n
    for k in range(1, min(lag, n - 1) + 1):
        variance += 2.0 * (1.0 - k / (lag + 1.0)) * (errors[k:] @ errors[:-k]) / n
    standard_error = math.sqrt(variance / n) if variance > 0 else 0.0
    return float(x.mean()), (float(x.mean() / standard_error) if standard_error > 0 else float('nan'))


# ----------------------------------------------------------------------------------------
# Selection procedures
# ----------------------------------------------------------------------------------------
@dataclass
class TrainingData:
    """
    The training rows of one walk-forward window, restricted to the candidate pool.

    Attributes:
        X (np.ndarray): Rank columns of the candidates, rows x candidates.
        y (np.ndarray): 0/1 labels.
        dates (np.ndarray): Date of every row (datetime64[ns]).
        label_end (np.ndarray): Label end date of every row (datetime64[ns]).
        names (List[str]): Candidate names of the columns of X.
    """
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    label_end: np.ndarray
    names: List[str]


@dataclass
class Selection:
    """
    The feature list chosen by a procedure and a description of how it was chosen.

    Attributes:
        features (List[str]): Selected features.
        info (Dict[str, Any]): Procedure details (inner splits, paths, scores, fallbacks).
    """
    features: List[str]
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _InnerSplit:
    """Arrays of one inner split of a training window."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    groups: DateGroups


def _inner_splits(data: TrainingData, settings: Dict[str, Any]) -> Tuple[List[_InnerSplit], Dict[str, Any]]:
    """Builds the inner splits of a training window with their validation date groups."""
    splits, info = inner_date_splits(data.dates, data.label_end, settings)
    min_stocks = int(settings['step1_min_stocks_per_date'])
    inner = [_InnerSplit(data.X[train], data.y[train], data.X[validation],
                         DateGroups(data.dates[validation], data.y[validation], min_stocks))
             for train, validation in splits]
    for block, split in zip(info['blocks'], inner):
        block['n_validation_auc_dates'] = split.groups.n_valid
    return inner, info


def _linear_scores(X_train: np.ndarray, y_train: np.ndarray, X_score: np.ndarray,
                   settings: Dict[str, Any]) -> np.ndarray:
    """Fits the production model and returns its linear scores (log-odds) for other rows."""
    model = fit_pooled_model(X_train, y_train, settings)
    return X_score @ model.coef_[0] + model.intercept_[0]


def select_stepwise(data: TrainingData, settings: Dict[str, Any]) -> Selection:
    """
    Forward stepwise selection on the inner mean per-date AUC (the honest variant).

    In each inner training split, the ANOVA F-test (f_classif) keeps the `review_anova_k`
    candidates with the highest F values in that split. A step adds the candidate with the
    largest increase of the mean per-date AUC over all inner validation dates; in a split
    whose prefilter dropped the candidate, the split keeps its current model. The
    production model (L2, `step1_l2_c`) is refitted for every trial. The selection stops
    when the gain is below `review_stepwise_min_gain` or at `review_max_features` features;
    ties go to the candidate that comes first in the pool.

    Args:
        data (TrainingData): Training rows of the window.
        settings (Dict[str, Any]): Loaded settings.

    Returns:
        Selection: The selected features in the order of selection (empty if none adds the gain).
    """
    splits, split_info = _inner_splits(data, settings)
    info: Dict[str, Any] = {'inner_splits': split_info}
    if not splits:
        info['stop_reason'] = 'no_inner_split'
        return Selection([], info)
    k = int(settings['review_anova_k'])
    min_gain = float(settings['review_stepwise_min_gain'])
    max_features = int(settings['review_max_features'])
    pools: List[set] = []
    current: List[np.ndarray] = []
    columns: List[List[int]] = []
    for split in splits:
        with warnings.catch_warnings():
            # Constant columns (e.g. a feature missing in the whole split) give NaN F values
            warnings.simplefilter('ignore', RuntimeWarning)
            warnings.simplefilter('ignore', UserWarning)
            scores, _ = f_classif(split.X_train, split.y_train)
        order = np.argsort(-np.nan_to_num(scores, nan=0.0), kind='mergesort')
        pools.append(set(order[:k].tolist()))
        current.append(np.full(split.groups.n_valid, NO_SKILL_AUC))
        columns.append([])
    info['anova_pools'] = [[data.names[j] for j in sorted(pool)] for pool in pools]
    n_validation = sum(len(values) for values in current)
    if n_validation == 0:
        info['stop_reason'] = 'no_validation_auc_dates'
        return Selection([], info)
    union = sorted(set().union(*pools))
    selected: List[int] = []
    score = NO_SKILL_AUC
    path: List[Dict[str, Any]] = []
    n_fits = 0
    info['stop_reason'] = 'max_features'
    while len(selected) < max_features:
        leader: Optional[Tuple[int, float, List[Optional[Tuple[List[int], np.ndarray]]]]] = None
        for j in union:
            if j in selected:
                continue
            total = 0.0
            trial: List[Optional[Tuple[List[int], np.ndarray]]] = []
            for split, pool, values, cols in zip(splits, pools, current, columns):
                if j not in pool:
                    total += float(values.sum())
                    trial.append(None)
                    continue
                trial_cols = cols + [j]
                auc = split.groups.auc(_linear_scores(split.X_train[:, trial_cols], split.y_train,
                                                      split.X_validation[:, trial_cols], settings))
                n_fits += 1
                total += float(auc.sum())
                trial.append((trial_cols, auc))
            trial_score = total / n_validation
            if leader is None or trial_score > leader[1]:
                leader = (j, trial_score, trial)
        if leader is None or leader[1] - score < min_gain:
            path.append({'step': len(selected) + 1, 'stopped': True,
                         'next_candidate': data.names[leader[0]] if leader else None,
                         'score': _num(leader[1], 6) if leader else None,
                         'gain': _num(leader[1] - score, 6) if leader else None})
            info['stop_reason'] = 'gain_below_minimum' if leader else 'no_candidate_left'
            break
        selected.append(leader[0])
        for i, state in enumerate(leader[2]):
            if state is not None:
                columns[i], current[i] = state
        path.append({'step': len(selected), 'feature': data.names[leader[0]], 'score': _num(leader[1], 6),
                     'gain': _num(leader[1] - score, 6)})
        score = leader[1]
    info.update({'path': path, 'inner_score': _num(score, 6), 'n_fits': n_fits, 'union_size': len(union)})
    return Selection([data.names[j] for j in selected], info)


def _fit_l1(X: np.ndarray, y: np.ndarray, C: float, settings: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
    """Fits the L1 logistic regression (seed `review_lasso_seed`); returns (coefficients, converged)."""
    max_iter = int(settings['step1_max_iter'])
    model = LogisticRegression(C=C, l1_ratio=1.0, solver=LASSO_SOLVER, tol=float(settings['step1_solver_tol']),
                               max_iter=max_iter, random_state=int(settings['review_lasso_seed']))
    with warnings.catch_warnings():
        # Counted and reported through the selection info instead of a warning per fit
        warnings.simplefilter('ignore', ConvergenceWarning)
        model.fit(X, y.astype(int))
    return model.coef_[0], bool(int(np.max(model.n_iter_)) < max_iter)


def _largest_nonzero(coefficients: np.ndarray, max_features: int) -> Tuple[List[int], bool]:
    """Columns with a nonzero coefficient, capped at the `max_features` largest absolute ones (in column order)."""
    nonzero = np.flatnonzero(coefficients != 0)
    if nonzero.size <= max_features:
        return nonzero.tolist(), False
    order = np.argsort(-np.abs(coefficients[nonzero]), kind='mergesort')
    return sorted(nonzero[order[:max_features]].tolist()), True


def select_lasso(data: TrainingData, settings: Dict[str, Any]) -> Selection:
    """
    Relaxed lasso: L1 logistic regression over the pool, the production model refitted on its nonzero coefficients.

    For every C of `review_lasso_c_grid` and every inner split, the L1 model is fitted on
    the inner training rows; the production model (L2, `step1_l2_c`) is refitted on its
    nonzero coefficients (at most `review_lasso_max_features`, the largest absolute ones)
    and scored on the validation blocks. The C with the highest mean per-date AUC is chosen
    (ties go to the smaller C), and its L1 model on the whole window gives the selection.

    Args:
        data (TrainingData): Training rows of the window.
        settings (Dict[str, Any]): Loaded settings.

    Returns:
        Selection: The nonzero coefficients in pool order (empty if all are zero).
    """
    grid = sorted(float(c) for c in settings['review_lasso_c_grid'])
    max_features = int(settings['review_lasso_max_features'])
    splits, split_info = _inner_splits(data, settings)
    info: Dict[str, Any] = {'inner_splits': split_info}
    if not splits:
        info['stop_reason'] = 'no_inner_split'
        return Selection([], info)
    per_c: Dict[float, List[np.ndarray]] = {C: [] for C in grid}
    nonzero_inner: Dict[float, List[int]] = {C: [] for C in grid}
    n_unconverged = 0
    for split in splits:
        for C in grid:
            coefficients, converged = _fit_l1(split.X_train, split.y_train, C, settings)
            n_unconverged += int(not converged)
            cols, _ = _largest_nonzero(coefficients, max_features)
            nonzero_inner[C].append(int(np.count_nonzero(coefficients)))
            if cols:
                auc = split.groups.auc(_linear_scores(split.X_train[:, cols], split.y_train,
                                                      split.X_validation[:, cols], settings))
            else:
                auc = np.full(split.groups.n_valid, NO_SKILL_AUC)
            per_c[C].append(auc)
    scores = {C: _finite_mean(np.concatenate(values)) for C, values in per_c.items()}
    chosen = grid[0]
    for C in grid:
        if np.isfinite(scores[C]) and (not np.isfinite(scores[chosen]) or scores[C] > scores[chosen]):
            chosen = C
    coefficients, converged = _fit_l1(data.X, data.y, chosen, settings)
    n_unconverged += int(not converged)
    cols, capped = _largest_nonzero(coefficients, max_features)
    info.update({'inner_scores': {str(C): _num(score, 6) for C, score in scores.items()},
                 'nonzero_inner': {str(C): counts for C, counts in nonzero_inner.items()},
                 'chosen_c': chosen, 'n_nonzero': int(np.count_nonzero(coefficients)), 'capped': capped,
                 'n_unconverged_fits': n_unconverged,
                 'coefficients': {data.names[j]: _num(coefficients[j], 6) for j in cols}})
    if capped:
        logger.info(f"Lasso: {info['n_nonzero']} nonzero coefficients; kept the {max_features} largest.")
    return Selection([data.names[j] for j in cols], info)


def select_fm_screen(data: TrainingData, settings: Dict[str, Any]) -> Selection:
    """
    Fama-MacBeth screening with a multiple-testing hurdle.

    On every training date with at least `step1_min_stocks_per_date` rows and both classes,
    the correlation of each candidate's rank with the label is computed (a rank correlation,
    since the inputs are ranks). A candidate is kept if the absolute Newey-West t-statistic
    (`review_fm_nw_lag` lags) of the mean correlation exceeds `review_fm_t_hurdle`.

    Args:
        data (TrainingData): Training rows of the window.
        settings (Dict[str, Any]): Loaded settings.

    Returns:
        Selection: The kept candidates in pool order (empty if none passes).
    """
    hurdle = float(settings['review_fm_t_hurdle'])
    lag = int(settings['review_fm_nw_lag'])
    groups = DateGroups(data.dates, data.y, int(settings['step1_min_stocks_per_date']))
    order = np.argsort(groups.codes, kind='mergesort')
    bounds = np.concatenate([[0], np.cumsum(np.bincount(groups.codes, minlength=groups.n_dates))])
    correlations = np.full((groups.n_valid, data.X.shape[1]), np.nan)
    for row, date in enumerate(np.flatnonzero(groups.valid)):
        rows = order[bounds[date]:bounds[date + 1]]
        X, y = data.X[rows], data.y[rows]
        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean()
        denominator = np.sqrt((X_centered ** 2).sum(axis=0) * (y_centered ** 2).sum())
        with np.errstate(invalid='ignore', divide='ignore'):
            correlations[row] = np.where(denominator > 0, X_centered.T @ y_centered / denominator, np.nan)
    statistics = [newey_west_t(correlations[:, j], lag) for j in range(data.X.shape[1])]
    kept = [j for j, (_, t) in enumerate(statistics) if np.isfinite(t) and abs(t) > hurdle]
    info = {'n_dates': groups.n_valid,
            't_statistics': {data.names[j]: _num(t, 3) for j, (_, t) in enumerate(statistics)},
            'mean_correlations': {data.names[j]: _num(mean, 5) for j, (mean, _) in enumerate(statistics)}}
    return Selection([data.names[j] for j in kept], info)


def select_features(procedure: str, data: TrainingData, settings: Dict[str, Any], champion: Sequence[str],
                    fixed_sets: Dict[str, List[str]]) -> Selection:
    """
    Runs one procedure on the training rows of a window.

    The stepwise, lasso and fm_screen procedures fall back to the champion list when they
    select nothing (recorded as `fallback_to_champion`).

    Args:
        procedure (str): 'champion', a selection procedure or 'manual:<name>'.
        data (TrainingData): Training rows of the window.
        settings (Dict[str, Any]): Loaded settings.
        champion (Sequence[str]): The current `step1_features`.
        fixed_sets (Dict[str, List[str]]): The `review_manual_sets`.

    Returns:
        Selection: The feature list.

    Raises:
        ValueError: On an unknown procedure.
    """
    if procedure == CHAMPION:
        return Selection(list(champion))
    if procedure.startswith(MANUAL_PREFIX):
        return Selection(list(fixed_sets[procedure[len(MANUAL_PREFIX):]]))
    if procedure == 'all_candidates':
        return Selection(list(data.names))
    procedures: Dict[str, Callable[[TrainingData, Dict[str, Any]], Selection]] = {
        'stepwise': select_stepwise, 'lasso': select_lasso, 'fm_screen': select_fm_screen}
    if procedure not in procedures:
        raise ValueError(f"Unknown review procedure {procedure!r}.")
    selection = procedures[procedure](data, settings)
    if not selection.features:
        return Selection(list(champion), {**selection.info, 'fallback_to_champion': True})
    return Selection(selection.features, {**selection.info, 'fallback_to_champion': False})


# ----------------------------------------------------------------------------------------
# Walk-forward evaluation
# ----------------------------------------------------------------------------------------
@dataclass
class ReviewWindow:
    """
    A training window of the review: one walk-forward test year, or the final window.

    Attributes:
        label (str): The test year as text, or 'final'.
        status (str): 'ok' or why the window cannot be fitted (as in the production validation).
        train (np.ndarray): Row positions of the training rows.
        test (np.ndarray): Row positions of the test rows (empty for the final window).
        test_start (Optional[pd.Timestamp]): First test date.
    """
    label: str
    status: str
    train: np.ndarray
    test: np.ndarray
    test_start: Optional[pd.Timestamp] = None


def review_windows(panel: ReviewPanel, settings: Dict[str, Any]) -> List[ReviewWindow]:
    """
    The yearly walk-forward windows and the final window of the review.

    The test years are the `review_validation_years` calendar years ending with the year of
    the scoring date, with the training rows of the production validation. The final window
    holds the training rows of the production model on the scoring date.

    Args:
        panel (ReviewPanel): The long panel.
        settings (Dict[str, Any]): Loaded settings.

    Returns:
        List[ReviewWindow]: The test-year windows in order, then the final window.
    """
    history_years = int(settings['step1_history_years'])
    windows = [ReviewWindow(str(w.year), w.status, np.flatnonzero(w.train), np.flatnonzero(w.test), w.test_start)
               for w in walk_forward_windows(panel.dates, panel.label_end, panel.y, panel.scoring_date,
                                             int(settings['review_validation_years']), history_years)]
    scoring = np.datetime64(panel.scoring_date)
    dates = panel.dates
    train = (~np.isnan(panel.y) & (panel.label_end < scoring) & np.asarray(dates < panel.scoring_date)
             & np.asarray(dates >= panel.scoring_date - pd.DateOffset(years=history_years)))
    status = 'ok'
    if not train.any():
        status = 'no_training_rows'
    elif np.unique(panel.y[train]).size < 2:
        status = 'single_class_training'
    windows.append(ReviewWindow(FINAL_WINDOW, status, np.flatnonzero(train), np.array([], dtype=int)))
    return windows


_TASK_CONTEXT: Dict[str, Any] = {}


def _init_task_worker(context: Dict[str, Any]) -> None:
    """Stores the data shared by all procedure tasks of a worker process."""
    _TASK_CONTEXT.clear()
    _TASK_CONTEXT.update(context)


def _procedure_task(task: Tuple[str, int]) -> Dict[str, Any]:
    """
    Runs one procedure in one window and scores its list on the window's test rows (worker task).

    Args:
        task (Tuple[str, int]): (procedure, window position).

    Returns:
        Dict[str, Any]: procedure, window, selected, info, runtime_s and, for a test year,
        the test-row probabilities of the production model on the selected list.
    """
    procedure, position = task
    context = _TASK_CONTEXT
    panel: ReviewPanel = context['panel']
    settings = context['settings']
    window: ReviewWindow = context['windows'][position]
    started = time.monotonic()
    pool_columns = panel.columns(context['pool'])
    data = TrainingData(X=panel.ranks[np.ix_(window.train, pool_columns)], y=panel.y[window.train],
                        dates=panel.dates.to_numpy(dtype='datetime64[ns]')[window.train],
                        label_end=panel.label_end[window.train], names=list(context['pool']))
    selection = select_features(procedure, data, settings, context['champion'], context['manual_sets'])
    result: Dict[str, Any] = {'procedure': procedure, 'window': window.label, 'selected': selection.features,
                              'info': selection.info, 'test_rows': window.test}
    columns = panel.columns(selection.features)
    model = fit_pooled_model(panel.ranks[np.ix_(window.train, columns)], panel.y[window.train], settings)
    result['coefficients'] = {name: _num(coef, 6) for name, coef in zip(selection.features, model.coef_[0])}
    if window.test.size:
        result['probability'] = predict_probability(model, panel.ranks[np.ix_(window.test, columns)])
    result['runtime_s'] = round(time.monotonic() - started, 1)
    result['worker_peak_mb'] = _worker_peak_mb()
    return result


def _describe_task(output: Dict[str, Any]) -> str:
    """Log line of a finished procedure task."""
    fallback = ' (fallback to the champion)' if output['info'].get('fallback_to_champion') else ''
    return (f"{output['procedure']} {output['window']}: {len(output['selected'])} features{fallback} "
            f"in {output['runtime_s']} s")


def _pooled_hit_rate(table: pd.DataFrame) -> Optional[float]:
    """Top-share hit rate pooled over the dates of a per-date table."""
    top_rows = float(table['n_top'].sum())
    return _num((table['top_hit_rate'] * table['n_top']).sum() / top_rows) if top_rows else None


def earlier_comparisons(report_dir: Optional[Path], adopted: pd.Timestamp,
                        review_date: pd.Timestamp) -> Tuple[int, List[str]]:
    """
    Counts the challenger comparisons of the earlier reviews of the current champion.

    The reports in `report_dir` (review_*.json, committed by the review workflow) of the
    reviews dated after the adoption date of the champion and before this review are
    counted, each with the number of its challengers. A report dated on or before the
    adoption date compared an earlier champion (a switch starts a new count). A report
    dated on the review date itself belongs to a run that this review replaces.

    Args:
        report_dir (Optional[Path]): Directory of the review reports (None or a missing
            directory: no earlier review).
        adopted (pd.Timestamp): `step1_features_adopted`.
        review_date (pd.Timestamp): Date of this review.

    Returns:
        Tuple[int, List[str]]: The number of earlier comparisons and the file names of the
        counted reports.

    Raises:
        ValueError: If a report cannot be read or has no review date or challenger list.
    """
    if report_dir is None or not Path(report_dir).is_dir():
        return 0, []
    adopted, review_date = pd.Timestamp(adopted).normalize(), pd.Timestamp(review_date).normalize()
    total, counted = 0, []
    for path in sorted(Path(report_dir).glob(REPORT_PATTERN)):
        try:
            report = json.loads(path.read_text(encoding='utf-8'))
            date = pd.Timestamp(report['review_date'])
            challengers = list(report['challengers'])
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"The review report {path} cannot be read: {e}") from e
        if pd.isna(date):
            raise ValueError(f"The review report {path} has no review date.")
        if adopted < date.normalize() < review_date:
            total += len(challengers)
            counted.append(path.name)
    return total, counted


def multiple_testing_correction(settings: Dict[str, Any], n_challengers: int, earlier: int,
                                earlier_reports: Sequence[str]) -> Dict[str, Any]:
    """
    The Bonferroni correction of condition (b) over all comparisons since the champion was adopted.

    Args:
        settings (Dict[str, Any]): Loaded settings (review_family_alpha).
        n_challengers (int): Challengers of this review.
        earlier (int): Challenger comparisons of the earlier reviews (earlier_comparisons()).
        earlier_reports (Sequence[str]): The counted report files.

    Returns:
        Dict[str, Any]: m_cum (this review's and the earlier comparisons), its parts, the
        family alpha and the two-sided CI level 1 - review_family_alpha / m_cum.

    Raises:
        ValueError: If review_family_alpha is not in (0, 1] or there is no comparison.
    """
    family_alpha = float(settings['review_family_alpha'])
    if not 0.0 < family_alpha <= 1.0:
        raise ValueError(f"review_family_alpha must be in (0, 1], not {family_alpha}.")
    m_cum = int(n_challengers) + int(earlier)
    if m_cum < 1:
        raise ValueError("The variable review has no challenger to compare.")
    return {'m_cum': m_cum, 'this_review': int(n_challengers), 'earlier_comparisons': int(earlier),
            'earlier_reports': list(earlier_reports), 'family_alpha': family_alpha,
            'ci_level': 1.0 - family_alpha / m_cum}


def compare_to_champion(champion: pd.DataFrame, challenger: pd.DataFrame, adopted: pd.Timestamp,
                        settings: Dict[str, Any], m_cum: int) -> Dict[str, Any]:
    """
    Paired comparison of a challenger with the champion on the same test dates.

    d_t = AUC_challenger(t) - AUC_champion(t). The historical interval of the mean d is a
    moving-block bootstrap (step1_bootstrap_* settings) at the two-sided level
    1 - review_family_alpha / m_cum (Bonferroni over the challenger comparisons since the
    champion was adopted). A test year is won if the mean d of its dates is positive. The
    fresh dates are the test dates after the adoption date of the champion.

    Args:
        champion (pd.DataFrame): Per-date table of the champion (per_date_table columns).
        challenger (pd.DataFrame): Per-date table of the challenger on the same test rows.
        adopted (pd.Timestamp): `step1_features_adopted`.
        settings (Dict[str, Any]): Loaded settings.
        m_cum (int): Number of comparisons of the Bonferroni correction.

    Returns:
        Dict[str, Any]: auc_mean, historical (mean_d, ci, ci_level, n_dates, mean_d_per_year,
        years_won, n_years), fresh (n_dates, mean_d) and hit_rate (all and fresh, challenger
        and champion).

    Raises:
        ValueError: If the two tables do not cover the same dates.
    """
    if not champion['date'].reset_index(drop=True).equals(challenger['date'].reset_index(drop=True)):
        raise ValueError("The challenger was not scored on the champion's test dates.")
    alpha = float(settings['review_family_alpha']) / max(1, int(m_cum))
    d = challenger['auc'].to_numpy(dtype=float) - champion['auc'].to_numpy(dtype=float)
    ci = block_bootstrap_ci(d, int(settings['step1_bootstrap_block_months']),
                            int(settings['step1_bootstrap_replications']), int(settings['step1_bootstrap_seed']),
                            percentiles=(100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)))
    # Mean per-date d of every test year with at least one date with an AUC
    yearly = pd.Series(d, index=champion['year'].to_numpy()).groupby(level=0).mean().dropna()
    fresh = (champion['date'] > adopted).to_numpy()
    fresh_d = d[fresh]
    fresh_d = fresh_d[np.isfinite(fresh_d)]
    ci_bounds = [_num(ci['ci_lower']), _num(ci['ci_upper'])]
    return {
        'auc_mean': _num(_finite_mean(challenger['auc'])),
        'historical': {'mean_d': _num(ci['mean']), 'ci': ci_bounds if None not in ci_bounds else None,
                       'ci_level': 1.0 - alpha, 'n_dates': int(ci['n']),
                       'mean_d_per_year': {str(year): _num(value) for year, value in yearly.items()},
                       'years_won': int((yearly > 0).sum()), 'n_years': int(yearly.size)},
        'fresh': {'n_dates': int(fresh_d.size), 'mean_d': _num(fresh_d.mean()) if fresh_d.size else None},
        'hit_rate': {'all': {'challenger': _pooled_hit_rate(challenger), 'champion': _pooled_hit_rate(champion)},
                     'fresh': {'challenger': _pooled_hit_rate(challenger[fresh]),
                               'champion': _pooled_hit_rate(champion[fresh])}},
    }


def _not_lower(challenger: Optional[float], champion: Optional[float]) -> bool:
    """True if both hit rates exist and the challenger's is not lower."""
    return challenger is not None and champion is not None and challenger >= champion


def evaluate_switch_rule(comparisons: Dict[str, Dict[str, Any]], final_lists: Dict[str, List[str]],
                         champion: Sequence[str], settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies the switch rule to every challenger and picks the winner.

    A challenger qualifies if all of these hold:

    - (a) its historical mean d >= `review_auc_hurdle`;
    - (b) the lower bound of its interval at the level 1 - review_family_alpha / m_cum
      (compare_to_champion()) is above 0;
    - (c) its mean per-date d is positive in at least `review_min_winning_years` test years;
    - (d) the fresh evidence does not contradict: with at least
      `review_fresh_check_min_dates` fresh test dates, the fresh mean d is >= 0 and the
      fresh top-share hit rate is not lower than the champion's. With fewer fresh dates
      the check is not yet available and passes;
    - (e) its top-share hit rate on all test dates is not lower than the champion's;
    - its final list differs from the champion.

    Among several qualifying challengers, the largest lower bound wins, then the larger
    historical mean d, then the order of the challengers.

    Args:
        comparisons (Dict[str, Dict[str, Any]]): compare_to_champion() per challenger, in challenger order.
        final_lists (Dict[str, List[str]]): Final list of every challenger.
        champion (Sequence[str]): The current `step1_features`.
        settings (Dict[str, Any]): Loaded settings.

    Returns:
        Dict[str, Any]: decision ('keep' or 'switch'), winner, adopted_features, reasons,
        fresh_check (status and number of fresh dates) and the conditions per challenger.
    """
    hurdle = float(settings['review_auc_hurdle'])
    min_years = int(settings['review_min_winning_years'])
    fresh_min = int(settings['review_fresh_check_min_dates'])
    results: Dict[str, Dict[str, Any]] = {}
    n_fresh = 0
    for name, comparison in comparisons.items():
        historical, fresh, hit = comparison['historical'], comparison['fresh'], comparison['hit_rate']
        mean_d = historical['mean_d']
        ci_lower = historical['ci'][0] if historical['ci'] else None
        n_fresh = int(fresh['n_dates'])
        if n_fresh >= fresh_min:
            fresh_passed = (fresh['mean_d'] is not None and fresh['mean_d'] >= NO_DIFFERENCE
                            and _not_lower(hit['fresh']['challenger'], hit['fresh']['champion']))
            fresh_status = 'passed' if fresh_passed else 'failed'
        else:
            fresh_passed, fresh_status = True, FRESH_NOT_YET_AVAILABLE
        conditions = {
            'a_historical_mean_d': {'passed': mean_d is not None and mean_d >= hurdle, 'value': mean_d,
                                    'threshold': hurdle},
            'b_ci_lower_above_zero': {'passed': ci_lower is not None and ci_lower > NO_DIFFERENCE, 'value': ci_lower,
                                      'threshold': NO_DIFFERENCE, 'ci_level': historical['ci_level']},
            'c_winning_years': {'passed': int(historical['years_won']) >= min_years,
                                'value': int(historical['years_won']), 'n_years': int(historical['n_years']),
                                'threshold': min_years},
            'd_fresh_evidence': {'passed': fresh_passed, 'status': fresh_status,
                                 'value': {'n_dates': n_fresh, 'mean_d': fresh['mean_d'], 'hit_rate': hit['fresh']},
                                 'threshold': {'min_dates': fresh_min, 'mean_d': NO_DIFFERENCE}},
            'e_hit_rate': {'passed': _not_lower(hit['all']['challenger'], hit['all']['champion']),
                           'value': hit['all']},
            'f_list_differs': {'passed': set(final_lists.get(name, [])) != set(champion),
                               'value': len(final_lists.get(name, []))},
        }
        failed = [key for key, condition in conditions.items() if not condition['passed']]
        results[name] = {'qualifies': not failed, 'failed': failed, 'conditions': conditions}
    order = list(comparisons)
    qualifying = [name for name in order if results[name]['qualifies']]
    winner = None
    if qualifying:
        winner = max(qualifying, key=lambda name: (comparisons[name]['historical']['ci'][0],
                                                   comparisons[name]['historical']['mean_d'], -order.index(name)))
    if winner:
        reasons = [f"{winner} meets every condition of the switch rule"
                   + (f" (qualifying: {', '.join(qualifying)}; the largest corrected lower bound wins)"
                      if len(qualifying) > 1 else "") + "."]
    else:
        reasons = [f"{name}: fails {', '.join(CONDITION_LABELS[key] for key in results[name]['failed'])}"
                   for name in order] or ["No challenger was evaluated."]
    fresh_check = {'status': 'checked' if n_fresh >= fresh_min else FRESH_NOT_YET_AVAILABLE,
                   'n_fresh': n_fresh, 'min_dates': fresh_min}
    return {'decision': DECISION_SWITCH if winner else DECISION_KEEP, 'winner': winner,
            'adopted_features': list(final_lists[winner]) if winner else None,
            'reasons': reasons, 'fresh_check': fresh_check, 'challengers': results}


@dataclass
class ReviewResult:
    """
    Outcome of a variable review.

    Attributes:
        review_date (pd.Timestamp): Date of the review (data end, exclusive).
        scoring_date (pd.Timestamp): Last trading day of the data.
        pool (List[str]): Candidate pool.
        champion (List[str]): The current `step1_features`.
        challengers (List[str]): Evaluated challengers, in order.
        windows (List[ReviewWindow]): Test-year windows and the final window.
        selections (Dict[str, Dict[str, Dict[str, Any]]]): Per procedure and window: selected
            list, coefficients, info and runtime.
        per_date (Dict[str, pd.DataFrame]): Per-date test metrics per procedure (incl. the champion).
        comparisons (Dict[str, Dict[str, Any]]): Paired comparison per challenger.
        decision (Dict[str, Any]): Result of the switch rule.
        correction (Dict[str, Any]): The multiple-testing correction of condition (b): m_cum,
            the counted earlier reports and the CI level.
        panel_summary (Dict[str, Any]): Size of the long panel and missing shares.
        memory (Dict[str, Any]): Worker processes and the largest peak memory of a worker in MB.
        timings (Dict[str, Any]): Runtime per stage in seconds (procedures_cpu_s: summed
            task runtime per procedure).
    """
    review_date: pd.Timestamp
    scoring_date: pd.Timestamp
    pool: List[str]
    champion: List[str]
    challengers: List[str]
    windows: List[ReviewWindow]
    selections: Dict[str, Dict[str, Dict[str, Any]]]
    per_date: Dict[str, pd.DataFrame]
    comparisons: Dict[str, Dict[str, Any]]
    decision: Dict[str, Any]
    correction: Dict[str, Any]
    panel_summary: Dict[str, Any]
    timings: Dict[str, Any]
    memory: Dict[str, Any] = field(default_factory=dict)


def truncate_price_panel(price_panel: PricePanel, review_date: pd.Timestamp) -> PricePanel:
    """
    Drops the rows dated on or after the review date, as the download does for today.

    Args:
        price_panel (PricePanel): Downloaded or cached prices.
        review_date (pd.Timestamp): The review date.

    Returns:
        PricePanel: The prices before the review date.
    """
    cut = pd.Timestamp(review_date)

    def before(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        return None if frame is None else frame.loc[frame.index < cut]

    close = before(price_panel.close).dropna(how='all')
    market = None if price_panel.market is None else price_panel.market.loc[price_panel.market.index < cut]

    def aligned(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        return None if frame is None else before(frame).reindex(index=close.index, columns=close.columns)

    return PricePanel(close=close, volume=aligned(price_panel.volume), missing=list(price_panel.missing),
                      open=aligned(price_panel.open), high=aligned(price_panel.high), low=aligned(price_panel.low),
                      market=market)


def run_variable_review(price_panel: PricePanel, settings: Dict[str, Any], review_date: pd.Timestamp,
                        candidates: Optional[Sequence[str]] = None, n_jobs: int = 1,
                        report_dir: Optional[Path] = None) -> ReviewResult:
    """
    Runs the champion-challenger review.

    Builds the long panel of the pool, the champion and the manual sets, runs every
    procedure in every walk-forward window and in the final window (in parallel with
    `n_jobs` processes), scores all lists with the production model on the same test rows,
    compares each challenger with the champion at the Bonferroni level of m_cum (this
    review's challengers plus those of the earlier reviews since the champion was adopted,
    counted from the reports in `report_dir`) and applies the switch rule.

    Args:
        price_panel (PricePanel): Prices, volumes and the market index.
        settings (Dict[str, Any]): Loaded settings.
        review_date (pd.Timestamp): The review date; rows on or after it are dropped.
        candidates (Optional[Sequence[str]]): Restricts the pool (local testing).
        n_jobs (int): Worker processes.
        report_dir (Optional[Path]): Directory of the earlier review reports (None: no
            earlier review is counted).

    Returns:
        ReviewResult: Selections, per-date metrics, comparisons, the correction and the decision.

    Raises:
        ValueError: On invalid settings, unreadable earlier reports, a missing market index
            for market-relative features or data without test dates.
    """
    timings: Dict[str, Any] = {}
    review_date = pd.Timestamp(review_date).normalize()
    champion = validate_feature_names(settings['step1_features'])
    pool = candidate_pool(settings, candidates)
    fixed_sets = manual_sets(settings)
    adopted = pd.Timestamp(str(settings['step1_features_adopted'])).normalize()
    challengers = list(SELECTION_PROCEDURES) + [f"{MANUAL_PREFIX}{name}" for name in fixed_sets]
    computed = list(dict.fromkeys(pool + champion + [f for features in fixed_sets.values() for f in features]))
    earlier, earlier_reports = earlier_comparisons(report_dir, adopted, review_date)
    correction = multiple_testing_correction(settings, len(challengers), earlier, earlier_reports)
    logger.info(f"Bonferroni correction: m_cum = {correction['m_cum']} ({len(challengers)} challengers of this "
                f"review, {earlier} of {len(earlier_reports)} earlier reviews since {adopted.date()}); "
                f"CI level {100.0 * correction['ci_level']:.2f}%.")

    started = time.monotonic()
    panel = build_review_panel(truncate_price_panel(price_panel, review_date), computed, settings, n_jobs)
    timings['panel_s'] = round(time.monotonic() - started, 1)

    windows = review_windows(panel, settings)
    procedures = [CHAMPION] + challengers
    # The expensive procedures first, so the workers finish at about the same time
    cost = {'stepwise': 0, 'lasso': 1}
    tasks = sorted([(procedure, i) for procedure in procedures for i, window in enumerate(windows)
                    if window.status == 'ok'], key=lambda task: (cost.get(task[0], 2), task[1]))
    context = {'panel': panel, 'windows': windows, 'settings': settings, 'pool': pool, 'champion': champion,
               'manual_sets': fixed_sets}
    logger.info(f"Running {len(procedures)} lists ({', '.join(procedures)}) in {len(windows)} windows "
                f"({len(tasks)} tasks) with {max(1, n_jobs)} processes; candidate pool: {len(pool)} features.")
    started = time.monotonic()
    outputs = _parallel_map(_procedure_task, tasks, n_jobs, _init_task_worker, (context,), describe=_describe_task)
    timings['procedures_s'] = round(time.monotonic() - started, 1)
    timings['procedures_cpu_s'] = {
        procedure: round(sum(o['runtime_s'] for o in outputs if o['procedure'] == procedure), 1)
        for procedure in procedures}

    started = time.monotonic()
    selections: Dict[str, Dict[str, Dict[str, Any]]] = {procedure: {} for procedure in procedures}
    for output in outputs:
        selections[output['procedure']][output['window']] = {
            key: output[key] for key in ('selected', 'coefficients', 'info', 'runtime_s')}
    up_share = float(settings['step1_up_share'])
    min_stocks = int(settings['step1_min_stocks_per_date'])
    tables: Dict[str, List[pd.DataFrame]] = {procedure: [] for procedure in procedures}
    by_task = {(output['procedure'], output['window']): output for output in outputs}
    for window in windows:
        if window.label == FINAL_WINDOW or window.status != 'ok':
            continue
        labels = panel.label_frame(window.test)
        for procedure in procedures:
            output = by_task[(procedure, window.label)]
            # Every list is scored on exactly the same test rows
            if not np.array_equal(output['test_rows'], window.test) or len(output['probability']) != len(window.test):
                raise ValueError(f"{procedure} was not scored on the test rows of {window.label}.")
            tables[procedure].append(per_date_table(labels.assign(probability=output['probability']),
                                                    int(window.label), up_share, min_stocks))
    if not tables[CHAMPION]:
        raise ValueError("The review has no test year with training and test rows.")
    per_date = {procedure: pd.concat(parts, ignore_index=True).sort_values('date').reset_index(drop=True)
                for procedure, parts in tables.items()}
    comparisons = {name: compare_to_champion(per_date[CHAMPION], per_date[name], adopted, settings,
                                             correction['m_cum'])
                   for name in challengers}
    final_lists = {name: selections[name][FINAL_WINDOW]['selected'] for name in challengers
                   if FINAL_WINDOW in selections[name]}
    decision = evaluate_switch_rule(comparisons, final_lists, champion, settings)
    timings['evaluation_s'] = round(time.monotonic() - started, 1)
    labeled = ~np.isnan(panel.y)
    panel_summary = {
        'n_rows': int(len(panel.y)), 'n_labeled_rows': int(labeled.sum()), 'n_tickers': panel.n_tickers,
        'n_dates': int(panel.dates.nunique()), 'first_date': _iso(panel.dates.min()),
        'scoring_date': _iso(panel.scoring_date), 'n_features_computed': len(panel.feature_names),
        'missing_share': {name: _num(share, 4) for name, share in panel.missing_share.items()},
        'market_index': None if price_panel.market is None else str(price_panel.market.name),
        'no_price_data': sorted(price_panel.missing),
    }
    logger.info(f"Variable review {review_date.date()}: decision {decision['decision']}"
                f"{' (' + decision['winner'] + ')' if decision['winner'] else ''}.")
    workers = min(n_jobs, len(tasks)) if n_jobs > 1 else 0
    peaks = [peak for peak in [panel.worker_peak_mb] + [o['worker_peak_mb'] for o in outputs] if peak is not None]
    memory = {'workers': workers, 'largest_worker': max(peaks) if workers and peaks else None}
    return ReviewResult(review_date=review_date, scoring_date=panel.scoring_date, pool=pool, champion=champion,
                        challengers=challengers, windows=windows, selections=selections, per_date=per_date,
                        comparisons=comparisons, decision=decision, correction=correction,
                        panel_summary=panel_summary, timings=timings, memory=memory)


# ----------------------------------------------------------------------------------------
# Report, settings update and pull request texts
# ----------------------------------------------------------------------------------------
def _year_table(result: ReviewResult) -> List[Dict[str, Any]]:
    """Mean per-date AUC per test year and procedure."""
    rows = []
    for window in result.windows:
        if window.label == FINAL_WINDOW:
            continue
        row: Dict[str, Any] = {'year': int(window.label), 'status': window.status}
        if window.status == 'ok':
            for procedure, table in result.per_date.items():
                year = table[table['year'] == int(window.label)]
                row[procedure] = _num(_finite_mean(year['auc']))
            row['n_dates'] = int(result.per_date[CHAMPION]['year'].eq(int(window.label)).sum())
        rows.append(row)
    return rows


def build_report(result: ReviewResult, settings: Dict[str, Any], timings: Dict[str, Any],
                 memory: Optional[Dict[str, Optional[float]]] = None) -> Dict[str, Any]:
    """
    Builds the JSON report of a review (plain JSON types only: no Timestamps, no NaN).

    Args:
        result (ReviewResult): The review.
        settings (Dict[str, Any]): Loaded settings (the parameters are copied).
        timings (Dict[str, Any]): Runtime per stage, including the stages outside the review.
        memory (Optional[Dict[str, Optional[float]]]): Peak memory of the main process in MB
            (peak_memory_mb()); the worker peaks come from the result.

    Returns:
        Dict[str, Any]: The report.
    """
    champion = result.per_date[CHAMPION]
    ci = block_bootstrap_ci(champion['auc'].to_numpy(dtype=float), int(settings['step1_bootstrap_block_months']),
                            int(settings['step1_bootstrap_replications']), int(settings['step1_bootstrap_seed']))
    adopted = pd.Timestamp(str(settings['step1_features_adopted'])).normalize()
    families: Dict[str, List[str]] = {}
    for name in result.pool:
        families.setdefault(FEATURE_FAMILIES[name], []).append(name)
    procedures: Dict[str, Any] = {}
    for procedure, windows in result.selections.items():
        procedures[procedure] = {
            'final_list': windows.get(FINAL_WINDOW, {}).get('selected'),
            'final_coefficients': windows.get(FINAL_WINDOW, {}).get('coefficients'),
            'selected_per_year': {label: entry['selected'] for label, entry in windows.items()
                                  if label != FINAL_WINDOW},
            'info': {label: entry['info'] for label, entry in windows.items()},
            'runtime_s': {label: entry['runtime_s'] for label, entry in windows.items()},
        }
    per_date = {procedure: [{'date': _iso(row.date), 'year': int(row.year), 'n': int(row.n), 'auc': _num(row.auc),
                             'top_hit_rate': _num(row.top_hit_rate), 'base_rate': _num(row.base_rate),
                             'top_minus_universe_fwd': _num(row.top_minus_universe_fwd)}
                            for row in table.itertuples(index=False)]
                for procedure, table in result.per_date.items()}
    ci_bounds = [_num(ci['ci_lower']), _num(ci['ci_upper'])]
    report = {
        'review_date': _iso(result.review_date),
        'scoring_date': _iso(result.scoring_date),
        'decision': result.decision['decision'],
        'winner': result.decision['winner'],
        'adopted_features': result.decision['adopted_features'],
        'reasons': result.decision['reasons'],
        'correction': result.correction,
        'fresh_check': result.decision['fresh_check'],
        'champion': {
            'features': result.champion, 'adopted': _iso(adopted), 'auc_mean': _num(ci['mean']),
            'auc_ci_95': ci_bounds if None not in ci_bounds else None, 'n_dates': int(ci['n']),
            'n_fresh_dates': int((champion['date'] > adopted).sum()),
            'top_hit_rate': _pooled_hit_rate(champion),
            'base_rate': _num(_finite_mean(champion['base_rate'])),
        },
        'challengers': result.challengers,
        'comparisons': {name: {**comparison, **result.decision['challengers'][name]}
                        for name, comparison in result.comparisons.items()},
        'auc_per_year': _year_table(result),
        'procedures': procedures,
        'pool': {'size': len(result.pool), 'families': families, 'names': result.pool},
        'data': result.panel_summary,
        'runtime_s': timings,
        'peak_memory_mb': _memory_summary(memory or {}, result.memory),
        'parameters': {key: settings[key] for key in REVIEW_PARAMETER_KEYS if key in settings},
        'per_date': per_date,
    }
    return report


def _memory_summary(main: Dict[str, Optional[float]], workers: Dict[str, Any]) -> Dict[str, Any]:
    """Peak memory of the main process and the workers, and their sum as an upper bound of the total."""
    summary: Dict[str, Any] = {'main': main.get('main'), 'workers': workers.get('workers', 0),
                               'largest_worker': workers.get('largest_worker')}
    if summary['main'] is not None:
        workers_total = summary['workers'] * (summary['largest_worker'] or 0.0)
        summary['total_upper_bound'] = round(summary['main'] + workers_total, 1)
    return summary


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    """Formats a number for the markdown tables ('-' if missing)."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return '-'
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    return f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"


def _interval(bounds: Optional[Sequence[Optional[float]]], signed: bool = False, digits: int = 3) -> str:
    """Formats an interval as [lower, upper]."""
    if not bounds:
        return '-'
    return f"[{_fmt(bounds[0], digits, signed)}, {_fmt(bounds[1], digits, signed)}]"


def _fresh_text(report: Dict[str, Any], comparison: Optional[Dict[str, Any]] = None) -> str:
    """Describes the fresh evidence of condition (d), for the review or for one challenger, in words."""
    check, adopted = report['fresh_check'], report['champion']['adopted']
    if check['status'] == FRESH_NOT_YET_AVAILABLE:
        return (f"{FRESH_NOT_YET_AVAILABLE} ({check['n_fresh']} fresh test dates after {adopted}; "
                f"the check starts at {check['min_dates']})")
    if comparison is None:
        return f"checked on {check['n_fresh']} fresh test dates after {adopted}"
    fresh, hit = comparison['fresh'], comparison['hit_rate']['fresh']
    return (f"{comparison['conditions']['d_fresh_evidence']['status']} on {fresh['n_dates']} fresh test dates "
            f"after {adopted}: mean difference {_fmt(fresh['mean_d'], 4, True)}, top-share hit rate "
            f"{_fmt(hit['challenger'])} against {_fmt(hit['champion'])}")


def _correction_text(report: Dict[str, Any]) -> str:
    """Describes the Bonferroni correction of condition (b) in words."""
    correction = report['correction']
    counted = ', '.join(correction['earlier_reports']) or 'none'
    return (f"m_cum = {correction['m_cum']} ({correction['this_review']} challengers of this review plus "
            f"{correction['earlier_comparisons']} comparisons of the earlier reviews since the adoption on "
            f"{report['champion']['adopted']}; counted reports: {counted}). Condition (b) uses the two-sided "
            f"{_fmt(100 * correction['ci_level'], 2)}% interval (1 - {correction['family_alpha']} / m_cum).")


def _anchor(title: str) -> str:
    """GitHub anchor of a markdown heading."""
    return '#' + re.sub(r'[^a-z0-9 -]', '', title.lower()).replace(' ', '-')


def _toc(titles: Sequence[str]) -> List[str]:
    """Table of contents lines for the given section titles."""
    return ['## Table of Contents'] + [f"- [{title}]({_anchor(title)})" for title in titles] + ['']


def _comparison_table(report: Dict[str, Any]) -> List[str]:
    """Markdown table of the paired comparison."""
    level = report['correction']['ci_level']
    lines = [f"| Challenger | Mean AUC | Mean d | CI of d ({_fmt(100 * level, 2)}%) | Years with mean d > 0 "
             "| Hit rate (challenger / champion) | Fresh dates | Fresh mean d "
             "| Fresh hit rate (challenger / champion) | Fresh check (d) | Qualifies |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, comparison in report['comparisons'].items():
        historical, fresh, hit = comparison['historical'], comparison['fresh'], comparison['hit_rate']
        lines.append(
            f"| `{name}` | {_fmt(comparison['auc_mean'])} | {_fmt(historical['mean_d'], 4, True)} "
            f"| {_interval(historical['ci'], True, 4)} | {historical['years_won']} of {historical['n_years']} "
            f"| {_fmt(hit['all']['challenger'])} / {_fmt(hit['all']['champion'])} "
            f"| {fresh['n_dates']} | {_fmt(fresh['mean_d'], 4, True)} "
            f"| {_fmt(hit['fresh']['challenger'])} / {_fmt(hit['fresh']['champion'])} "
            f"| {comparison['conditions']['d_fresh_evidence']['status']} "
            f"| {'yes' if comparison['qualifies'] else 'no'} |")
    return lines + ['']


def _yearly_d_table(report: Dict[str, Any]) -> List[str]:
    """Markdown table of the mean d per test year (condition (c))."""
    years = sorted({year for comparison in report['comparisons'].values()
                    for year in comparison['historical']['mean_d_per_year']})
    lines = ['| Challenger | ' + ' | '.join(years) + ' |', '|---|' + '---|' * len(years)]
    for name, comparison in report['comparisons'].items():
        per_year = comparison['historical']['mean_d_per_year']
        lines.append(f"| `{name}` | " + ' | '.join(_fmt(per_year.get(year), 4, True) for year in years) + ' |')
    return lines + ['']


def render_markdown(report: Dict[str, Any]) -> str:
    """
    Renders the human-readable review report.

    Args:
        report (Dict[str, Any]): build_report() output.

    Returns:
        str: Markdown text with a table of contents.
    """
    titles = ['Decision', 'Champion', 'Paired Comparison', 'AUC per Year', 'Selected Features',
              'Candidate Pool', 'Data and Runtime', 'Parameters']
    champion = report['champion']
    lines = [f"# Variable Review {report['review_date']}", '',
             f"Quarterly champion-challenger review of the Step 1 feature set on the data up to "
             f"{report['scoring_date']}. Generated by scripts/05_review_variables.py.", '']
    lines += _toc(titles)
    if report['decision'] == DECISION_SWITCH:
        verdict = (f"**switch** to the final list of `{report['winner']}` "
                   f"({len(report['adopted_features'])} features). The change is proposed in a pull request; "
                   "it takes effect only when the pull request is merged.")
    else:
        verdict = "**keep** the current `step1_features`."
    lines += ['## Decision', '', f"Decision: {verdict}", '',
              f"- Multiple-testing correction: {_correction_text(report)}",
              f"- Fresh evidence (d): {_fresh_text(report)}.", '', 'Reasons:', '']
    lines += [f"- {reason}" for reason in report['reasons']] + ['']
    lines += ['## Champion', '',
              f"- Features ({len(champion['features'])}): {', '.join(f'`{f}`' for f in champion['features'])}",
              f"- Adopted: {champion['adopted']}; fresh test dates since then: {champion['n_fresh_dates']}",
              f"- Mean per-date AUC: {_fmt(champion['auc_mean'], 4)} {_interval(champion['auc_ci_95'])} "
              f"(95% block-bootstrap CI, {champion['n_dates']} test dates)",
              f"- Top-share hit rate: {_fmt(champion['top_hit_rate'])} (base rate {_fmt(champion['base_rate'])})",
              '']
    lines += ['## Paired Comparison', '',
              "d is the per-date AUC of the challenger minus that of the champion on the same test rows. The "
              "interval is a moving-block bootstrap at the level 1 - review_family_alpha / m_cum. A test year "
              "counts for condition (c) if the mean d of its dates is positive. Fresh dates are the test dates "
              "after the adoption date of the champion; condition (d) checks them once "
              f"{report['fresh_check']['min_dates']} exist.", '']
    lines += _comparison_table(report)
    lines += ['Mean d per test year:', ''] + _yearly_d_table(report)
    procedures = ['champion'] + list(report['challengers'])
    lines += ['## AUC per Year', '', '| Year | Test dates | ' + ' | '.join(f'`{p}`' for p in procedures) + ' |',
              '|---|---|' + '---|' * len(procedures)]
    for row in report['auc_per_year']:
        if row['status'] != 'ok':
            lines.append(f"| {row['year']} | {row['status']} |" + ' |' * len(procedures))
            continue
        lines.append(f"| {row['year']} | {row['n_dates']} | "
                     + ' | '.join(_fmt(row.get(p), 4) for p in procedures) + ' |')
    lines += ['', '## Selected Features', '']
    pool_names = list(report['pool']['names'])

    def listing(features: Sequence[str], code: bool) -> str:
        if sorted(features) == sorted(pool_names):
            return f"the whole candidate pool ({len(features)} features)"
        return ', '.join(f'`{f}`' if code else f for f in features)

    for procedure in report['challengers']:
        entry = report['procedures'][procedure]
        final = entry['final_list'] or []
        lines += [f"### `{procedure}`", '', f"Final list ({len(final)}): {listing(final, True)}", '',
                  '| Year | Features | Selected |', '|---|---|---|']
        for label, selected in entry['selected_per_year'].items():
            lines.append(f"| {label} | {len(selected)} | {listing(selected, False)} |")
        lines.append('')
    pool = report['pool']
    lines += ['## Candidate Pool', '', f"{pool['size']} candidates in {len(pool['families'])} families:", '',
              '| Family | Candidates | Names |', '|---|---|---|']
    for family, names in pool['families'].items():
        lines.append(f"| {family} | {len(names)} | {', '.join(names)} |")
    data, runtime = report['data'], report['runtime_s']
    memory = report.get('peak_memory_mb') or {}
    lines += ['', '## Data and Runtime', '',
              f"- Eligible month-end rows: {data['n_rows']} ({data['n_labeled_rows']} with a known label), "
              f"{data['n_tickers']} stocks, {data['n_dates']} dates from {data['first_date']} "
              f"to {data['scoring_date']}",
              f"- Market index: {data['market_index']}; tickers without price data: {len(data['no_price_data'])}",
              '- Runtime per stage (s): ' + ', '.join(f"{key} {value}" for key, value in runtime.items()
                                                      if not isinstance(value, dict)),
              '- Peak memory (MB): ' + (', '.join(f"{key} {_fmt(value, 0)}" for key, value in memory.items())
                                        or '-'), '']
    lines += ['## Parameters', '', '| Key | Value |', '|---|---|']
    lines += [f"| `{key}` | {value} |" for key, value in report['parameters'].items()]
    return '\n'.join(lines) + '\n'


def render_pr_body(report: Dict[str, Any], report_path: str) -> str:
    """
    Renders the pull request body of a switch.

    Args:
        report (Dict[str, Any]): build_report() output with decision 'switch'.
        report_path (str): Repository path of the markdown report.

    Returns:
        str: Markdown text.
    """
    winner = report['winner']
    comparison = report['comparisons'][winner]
    historical, hit = comparison['historical'], comparison['hit_rate']['all']
    current, proposed = report['champion']['features'], report['adopted_features']
    titles = ['Summary', 'Evidence', 'Feature Lists', 'Before Merging']
    lines = [f"Variable review of {report['review_date']}: adopt the final list of `{winner}`.", '']
    lines += _toc(titles)
    lines += ['## Summary', '',
              f"The quarterly variable review qualifies `{winner}` under the switch rule. This pull request sets "
              f"`step1_features` to its final list ({len(proposed)} features) and "
              f"`step1_features_adopted` to {report['review_date']}. Nothing changes until it is merged.", '',
              f"- Historical mean AUC difference: {_fmt(historical['mean_d'], 4, True)} "
              f"{_interval(historical['ci'], True, 4)} ({_fmt(100 * historical['ci_level'], 2)}% CI, "
              f"m_cum = {report['correction']['m_cum']}, {historical['n_dates']} test dates)",
              f"- Test years with a positive mean difference: {historical['years_won']} of {historical['n_years']}",
              f"- Top-share hit rate on all test dates: {_fmt(hit['challenger'])} against "
              f"{_fmt(hit['champion'])} for the champion",
              f"- Fresh evidence: {_fresh_text(report, comparison)}",
              f"- Champion AUC {_fmt(report['champion']['auc_mean'], 4)}, "
              f"challenger AUC {_fmt(comparison['auc_mean'], 4)}",
              f"- Multiple-testing correction: {_correction_text(report)}", '']
    lines += ['## Evidence', ''] + _comparison_table(report)
    lines += ['Mean d per test year:', ''] + _yearly_d_table(report)
    lines += ['## Feature Lists', '',
              f"- Current ({len(current)}): {', '.join(f'`{f}`' for f in current)}",
              f"- Proposed ({len(proposed)}): {', '.join(f'`{f}`' for f in proposed)}", '']
    lines += ['## Before Merging', '',
              "- This pull request was created with the workflow's `GITHUB_TOKEN`. GitHub does not start the "
              "jobs of the `CI` run that such a pull request triggers. Start that run from the Actions tab, or "
              "close and reopen the pull request, to run the checks before merging.",
              f"- The full report is `{report_path}` on `main`.",
              f"- Output size: the proposed list has {len(proposed)} features against {len(current)} now. The "
              "prediction payloads (`feature_percentiles`, `logit_contributions`), the feature diagnostics and "
              "the daily batch report, which is committed and archived, grow about linearly with the number "
              "of features.",
              "- SQL reference: `docs/feature_engineering.sql` documents the computation of the current list and "
              "is not regenerated by the review. `docs/generate_sql.py` stops with an error for features without "
              "a SQL template; add templates to `FEATURE_SQL` to document the new list.",
              "- After the merge, the next run of `2. Execute Pipeline (Step 1 & 2)` uses the new list. The next "
              "review counts the test dates after the new adoption date as fresh evidence and starts a new "
              "count of m_cum.", '']
    return '\n'.join(lines)


def render_commit_body(report: Dict[str, Any], report_path: str) -> str:
    """
    Renders the commit message body of a switch (plain text).

    Args:
        report (Dict[str, Any]): build_report() output with decision 'switch'.
        report_path (str): Repository path of the markdown report.

    Returns:
        str: Plain text lines.
    """
    winner = report['winner']
    comparison = report['comparisons'][winner]
    historical, hit = comparison['historical'], comparison['hit_rate']
    paragraphs = [
        f"The variable review of {report['review_date']} qualifies {winner} under the switch rule. "
        f"Historical mean AUC difference {_fmt(historical['mean_d'], 4, True)}, "
        f"{_fmt(100 * historical['ci_level'], 2)}% CI {_interval(historical['ci'], True, 4)} "
        f"(m_cum = {report['correction']['m_cum']}), {historical['n_dates']} test dates, positive in "
        f"{historical['years_won']} of {historical['n_years']} test years. Top-share hit rate "
        f"{_fmt(hit['all']['challenger'])} against {_fmt(hit['all']['champion'])} for the champion. "
        f"Fresh evidence: {_fresh_text(report, comparison)}.",
        f"Adopted list ({len(report['adopted_features'])}): {', '.join(report['adopted_features'])}.",
        f"Report: {report_path}",
    ]
    return '\n\n'.join(textwrap.fill(paragraph, width=72, break_long_words=False, break_on_hyphens=False)
                       for paragraph in paragraphs) + '\n'


def update_settings_text(text: str, features: Sequence[str], adopted: str) -> str:
    """
    Replaces the `step1_features` and `step1_features_adopted` lines of the settings file.

    Only these two lines change; comments, blank lines and the key order are kept. The
    result is parsed and compared with the original: every other key must be unchanged.

    Args:
        text (str): Content of config/settings.yaml.
        features (Sequence[str]): The adopted feature list.
        adopted (str): ISO date of the adoption.

    Returns:
        str: The updated content.

    Raises:
        ValueError: If a line is missing or appears twice, a feature name cannot be written
            as a plain YAML scalar, or the verification fails.
    """
    names = validate_feature_names(features)
    invalid = [name for name in names if not FEATURE_NAME_PATTERN.match(name)]
    if invalid:
        raise ValueError(f"Feature names that cannot be written to the settings file: {invalid}")
    adopted = pd.Timestamp(adopted).date().isoformat()
    lines = text.splitlines(keepends=True)
    replacements = {
        'step1_features': (SETTINGS_FEATURES_LINE, lambda m: f"{m.group(1)}[{', '.join(names)}]{m.group(2)}"),
        'step1_features_adopted': (SETTINGS_ADOPTED_LINE, lambda m: f'{m.group(1)}"{adopted}"{m.group(3)}'),
    }
    for key, (pattern, replace) in replacements.items():
        matches = [i for i, line in enumerate(lines) if pattern.match(line.rstrip('\r\n'))]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one '{key}:' line in the settings file, found {len(matches)}.")
        i = matches[0]
        ending = lines[i][len(lines[i].rstrip('\r\n')):]
        lines[i] = pattern.sub(replace, lines[i].rstrip('\r\n')) + ending
    updated = ''.join(lines)
    try:
        before, after = yaml.safe_load(text), yaml.safe_load(updated)
    except yaml.YAMLError as e:
        raise ValueError(f"The updated settings file does not parse: {e}") from e
    if list(after) != list(before):
        raise ValueError("The settings update changed the keys or their order.")
    if after['step1_features'] != names or str(after['step1_features_adopted']) != adopted:
        raise ValueError("The settings update did not write the adopted list and date.")
    changed = [key for key in before if key not in replacements and before[key] != after[key]]
    if changed:
        raise ValueError(f"The settings update changed other keys: {changed}")
    return updated


def peak_memory_mb() -> Dict[str, Optional[float]]:
    """
    Peak resident memory of this process.

    The worker processes report their own peaks (ReviewResult.memory); build_report()
    combines both.

    Returns:
        Dict[str, Optional[float]]: 'main' in MB (None where the platform does not report it).
    """
    return {'main': _worker_peak_mb()}
