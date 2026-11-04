"""
Step 1: one pooled cross-sectional logistic regression for all stocks.

Complete pooling: a single L2-regularized logistic regression with one shared intercept
and shared coefficients is fitted on the month-end rows of all stocks. Its inputs are the
within-date percentile ranks of the features in `step1_features`, so the model ranks the
stocks of a date against each other; it does not forecast the direction of the market.
On the scoring date (the last closed trading day), the top `step1_up_share` of the ranked
stocks are UP.

A walk-forward validation is recomputed on every run. Each of the last
`step1_validation_years` calendar years is scored by a model refitted on the rows whose
forward return ended before the first test month-end of that year. The mean per-date
cross-sectional AUC and its moving-block bootstrap CI form the model gate: if the lower
CI bound is at or below `step1_min_validation_auc_lower`, no stock is UP.
"""
import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..ingestion.market_api import PricePanel
from ..processing.features import (FeaturePanel, build_feature_panel, month_end_dates, rank_columns,
                                   to_long_panel, validate_feature_names)

logger = logging.getLogger(__name__)

MODEL_NAME = 'pooled_cross_sectional_logit'
SOLVER = 'lbfgs'                    # fixed design decision; C, tol and max_iter are settings
CI_PERCENTILES = (2.5, 97.5)        # two-sided 95% percentile interval of the bootstrap

STATUS_OK = 'ok'
STATUS_GATE_FAILED = 'model_gate_failed'
NOT_ELIGIBLE = 'not_eligible'
# Precedence of the reasons when several apply
INELIGIBLE_REASONS = ('stale_price', 'insufficient_history', 'price_at_or_below_min', 'no_traded_value')

GATE_INSUFFICIENT_DATES = 'insufficient_validation_dates'
GATE_LOW_AUC = 'auc_ci_lower_at_or_below_threshold'
GATE_NO_TRAINING_ROWS = 'no_training_rows'

# Settings that define the model; copied into the model summary
SUMMARY_PARAMETER_KEYS = (
    'horizon_days', 'threshold', 'step1_features', 'step1_history_years', 'feature_warmup_years',
    'step1_l2_c', 'step1_solver_tol', 'step1_max_iter', 'step1_up_share', 'step1_min_price_eur',
    'step1_ffill_limit_days', 'step1_validation_years', 'step1_min_stocks_per_date',
    'step1_bootstrap_block_months', 'step1_bootstrap_replications', 'step1_bootstrap_seed',
    'step1_min_validation_auc_lower',
)


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


def _mean(values: Sequence[float]) -> float:
    """Mean of the finite values, NaN if there are none."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float('nan')


def fit_pooled_model(X: Any, y: Any, settings: Dict[str, Any]) -> LogisticRegression:
    """
    Fits the pooled L2 logistic regression (one shared intercept and shared coefficients).

    Args:
        X (Any): Rank features, rows x features (array or DataFrame).
        y (Any): Binary labels (0/1).
        settings (Dict[str, Any]): Loaded settings (step1_l2_c, step1_solver_tol, step1_max_iter).

    Returns:
        LogisticRegression: The fitted model. Class weights are not used.

    Raises:
        ValueError: If the labels contain a single class.
    """
    max_iter = int(settings['step1_max_iter'])
    model = LogisticRegression(C=float(settings['step1_l2_c']), solver=SOLVER,
                               tol=float(settings['step1_solver_tol']), max_iter=max_iter)
    with warnings.catch_warnings():
        # Reported once through the log below instead of a warning per refit
        warnings.simplefilter('ignore', ConvergenceWarning)
        model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float).astype(int))
    if int(np.max(model.n_iter_)) >= max_iter:
        logger.warning(f"Step 1 model did not converge within {max_iter} iterations.")
    return model


def predict_probability(model: LogisticRegression, X: Any) -> np.ndarray:
    """
    Returns the predicted probability of class 1 (the +threshold target).

    Args:
        model (LogisticRegression): A fitted model.
        X (Any): Rank features, rows x features.

    Returns:
        np.ndarray: Probabilities.
    """
    return model.predict_proba(np.asarray(X, dtype=float))[:, 1]


def per_date_auc(df: pd.DataFrame, prob_col: str, label_col: str, min_stocks: int) -> pd.Series:
    """
    Computes the cross-sectional ROC AUC of every date.

    Dates with fewer than `min_stocks` labelled rows or a single class are skipped.

    Args:
        df (pd.DataFrame): Rows indexed by (date, ticker).
        prob_col (str): Column with the predicted probabilities.
        label_col (str): Column with the 0/1 labels (NaN rows are ignored).
        min_stocks (int): Minimum labelled rows of a date.

    Returns:
        pd.Series: AUC per date (sorted by date), only for the dates that qualify.
    """
    values: Dict[pd.Timestamp, float] = {}
    for date, group in df.groupby(level='date', sort=True):
        y = group[label_col].to_numpy(dtype=float)
        p = group[prob_col].to_numpy(dtype=float)
        keep = ~np.isnan(y) & ~np.isnan(p)
        y, p = y[keep], p[keep]
        if y.size < min_stocks or np.unique(y).size < 2:
            continue
        values[pd.Timestamp(date)] = float(roc_auc_score(y, p))
    return pd.Series(values, dtype=float, name='auc').sort_index()


def block_bootstrap_ci(values: Sequence[float], block: int, replications: int, seed: int) -> Dict[str, float]:
    """
    Moving-block bootstrap of the mean of a date-ordered series.

    Every replication concatenates randomly chosen blocks of `block` consecutive values
    (overlapping block starts, drawn with replacement) until the series length is reached.
    The interval is the 2.5% and 97.5% percentile of the replicated means.

    Args:
        values (Sequence[float]): The series in date order; non-finite values are dropped.
        block (int): Block length (consecutive dates).
        replications (int): Number of bootstrap replications.
        seed (int): Seed of numpy's default random generator.

    Returns:
        Dict[str, float]: mean, n, ci_lower, ci_upper and boot_sd. The interval is NaN if
        the series has fewer than two blocks of values.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    result: Dict[str, float] = {'mean': float(x.mean()) if n else float('nan'), 'n': n,
                                'ci_lower': float('nan'), 'ci_upper': float('nan'), 'boot_sd': float('nan')}
    if block < 1 or replications < 1 or n < 2 * block:
        return result
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(n - block + 1)
    offsets = np.arange(block)
    means = np.empty(replications)
    for i in range(replications):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        idx = (chosen[:, None] + offsets[None, :]).ravel()[:n]
        means[i] = x[idx].mean()
    result['ci_lower'] = float(np.percentile(means, CI_PERCENTILES[0]))
    result['ci_upper'] = float(np.percentile(means, CI_PERCENTILES[1]))
    result['boot_sd'] = float(np.std(means))
    return result


def n_up_for(n_ranked: int, up_share: float) -> int:
    """
    Number of UP stocks among `n_ranked` ranked stocks: max(1, floor(up_share * n + 0.5)).

    Args:
        n_ranked (int): Number of ranked stocks.
        up_share (float): Share of the ranking marked UP (`step1_up_share`).

    Returns:
        int: The top-share size (0 if nothing is ranked).
    """
    if n_ranked <= 0:
        return 0
    return min(n_ranked, max(1, int(math.floor(up_share * n_ranked + 0.5))))


def rank_by_probability(probability: pd.Series) -> pd.Series:
    """
    Ranks stocks by probability: 1 = highest; ties are broken by ticker, ascending.

    Args:
        probability (pd.Series): Probabilities indexed by ticker.

    Returns:
        pd.Series: Integer ranks with the index of `probability`.
    """
    order = pd.DataFrame({'p': probability.to_numpy(dtype=float),
                          't': [str(t) for t in probability.index]})
    order = order.sort_values(['p', 't'], ascending=[False, True], kind='mergesort')
    ranks = np.empty(len(order), dtype=int)
    ranks[order.index.to_numpy()] = np.arange(1, len(order) + 1)
    return pd.Series(ranks, index=probability.index, name='rank')


def top_share_mask(probability: pd.Series, up_share: float) -> np.ndarray:
    """
    Marks the top-share stocks of one date (same rule as the UP decision).

    Args:
        probability (pd.Series): Probabilities indexed by ticker.
        up_share (float): Share of the ranking to mark.

    Returns:
        np.ndarray: Boolean mask aligned with `probability`.
    """
    ranks = rank_by_probability(probability).to_numpy()
    return ranks <= n_up_for(len(ranks), up_share)


@dataclass
class ValidationResult:
    """
    Result of the walk-forward validation.

    Attributes:
        years (List[Dict[str, Any]]): One entry per test year (fitted or skipped with a status).
        overall (Dict[str, Any]): Metrics over all test dates, including the bootstrap CI.
        per_date (pd.DataFrame): One row per test date (date, year, n, auc, n_top,
            top_hit_rate, base_rate, top_fwd, universe_fwd, top_minus_universe_fwd).
    """
    years: List[Dict[str, Any]]
    overall: Dict[str, Any]
    per_date: pd.DataFrame


PER_DATE_COLUMNS = ['date', 'year', 'n', 'auc', 'n_top', 'top_hit_rate', 'base_rate',
                    'top_fwd', 'universe_fwd', 'top_minus_universe_fwd']


def _per_date_table(test: pd.DataFrame, year: int, up_share: float, min_stocks: int) -> pd.DataFrame:
    """
    Per-date metrics of the scored test rows of one year.

    Args:
        test (pd.DataFrame): Test rows indexed by (date, ticker) with y, fwd and probability.
        year (int): The test year.
        up_share (float): Top share used for the hit rate and the return spread.
        min_stocks (int): Minimum labelled rows of a date for its AUC.

    Returns:
        pd.DataFrame: One row per date with the PER_DATE_COLUMNS.
    """
    auc = per_date_auc(test, 'probability', 'y', min_stocks)
    rows = []
    for date, group in test.groupby(level='date', sort=True):
        prob = pd.Series(group['probability'].to_numpy(), index=group.index.get_level_values('ticker'))
        top = top_share_mask(prob, up_share)
        y = group['y'].to_numpy(dtype=float)
        fwd = group['fwd'].to_numpy(dtype=float)
        top_fwd = float(fwd[top].mean())
        universe_fwd = float(fwd.mean())
        rows.append({
            'date': pd.Timestamp(date), 'year': year, 'n': int(len(group)),
            'auc': float(auc.get(pd.Timestamp(date), float('nan'))), 'n_top': int(top.sum()),
            'top_hit_rate': float(y[top].mean()), 'base_rate': float(y.mean()),
            'top_fwd': top_fwd, 'universe_fwd': universe_fwd,
            'top_minus_universe_fwd': top_fwd - universe_fwd,
        })
    return pd.DataFrame(rows, columns=PER_DATE_COLUMNS)


def walk_forward_validation(long_panel: pd.DataFrame, feature_cols: Sequence[str],
                            scoring_date: pd.Timestamp, settings: Dict[str, Any]) -> ValidationResult:
    """
    Validates the pooled model with yearly walk-forward refits.

    The test years are the `step1_validation_years` calendar years ending with the year of
    the scoring date; the current year is a partial block. A test row needs a known label
    (label_end <= scoring_date). For each year, the model is refitted on the rows with a
    known label, label_end before the year's first test date, a date before it and a date
    within `step1_history_years` of it, so no training label overlaps the test period.
    A year without training or test rows is skipped and recorded with a status.

    Args:
        long_panel (pd.DataFrame): Month-end rows indexed by (date, ticker) with the rank
            columns, fwd, label_end and y.
        feature_cols (Sequence[str]): Rank columns used as model inputs.
        scoring_date (pd.Timestamp): Last date of the union calendar.
        settings (Dict[str, Any]): Loaded settings.

    Returns:
        ValidationResult: Per-year metrics, overall metrics with the bootstrap CI, and the per-date table.
    """
    scoring_date = pd.Timestamp(scoring_date)
    history = pd.DateOffset(years=int(settings['step1_history_years']))
    up_share = float(settings['step1_up_share'])
    min_stocks = int(settings['step1_min_stocks_per_date'])
    n_years = int(settings['step1_validation_years'])
    cols = list(feature_cols)

    dates = pd.DatetimeIndex(long_panel.index.get_level_values('date'))
    label_end = long_panel['label_end'].to_numpy(dtype='datetime64[ns]')
    y = long_panel['y'].to_numpy(dtype=float)
    X = long_panel[cols].to_numpy(dtype=float)
    known = ~np.isnan(y) & (label_end <= np.datetime64(scoring_date)) & np.asarray(dates <= scoring_date)

    years: List[Dict[str, Any]] = []
    tables: List[pd.DataFrame] = []
    pooled_top_hits, pooled_top_rows, pooled_y_sum, pooled_rows = 0.0, 0, 0.0, 0
    for year in range(scoring_date.year - n_years + 1, scoring_date.year + 1):
        test = known & np.asarray(dates.year == year)
        if not test.any():
            years.append({'year': year, 'status': 'no_test_rows'})
            continue
        test_start = dates[test].min()
        train = (known & (label_end < np.datetime64(test_start)) & np.asarray(dates < test_start)
                 & np.asarray(dates >= test_start - history))
        if not train.any():
            years.append({'year': year, 'status': 'no_training_rows'})
            continue
        if np.unique(y[train]).size < 2:
            years.append({'year': year, 'status': 'single_class_training'})
            continue

        model = fit_pooled_model(X[train], y[train], settings)
        p_train = predict_probability(model, X[train])
        p_test = predict_probability(model, X[test])
        scored = long_panel.loc[test, ['y', 'fwd']].copy()
        scored['probability'] = p_test
        table = _per_date_table(scored, year, up_share, min_stocks)
        tables.append(table)

        y_test = y[test]
        top_rows = int(table['n_top'].sum())
        top_hits = float((table['top_hit_rate'] * table['n_top']).sum())
        pooled_top_hits += top_hits
        pooled_top_rows += top_rows
        pooled_y_sum += float(y_test.sum())
        pooled_rows += int(y_test.size)
        train_dates = dates[train]
        years.append({
            'year': year, 'status': 'ok',
            'n_train': int(train.sum()), 'n_train_dates': int(train_dates.nunique()),
            'train_first': _iso(train_dates.min()), 'train_last': _iso(train_dates.max()),
            'max_label_end': _iso(pd.Timestamp(label_end[train].max())),
            'test_start': _iso(test_start),
            'n_test': int(test.sum()), 'n_test_dates': int(table.shape[0]),
            'n_auc_dates': int(table['auc'].notna().sum()),
            'auc_mean': _num(_mean(table['auc'])),
            'is_auc': _num(roc_auc_score(y[train], p_train)),
            'oos_auc': _num(roc_auc_score(y_test, p_test)) if np.unique(y_test).size > 1 else None,
            'top_hit_rate': _num(top_hits / top_rows) if top_rows else None,
            'base_rate': _num(y_test.mean()),
            'top_minus_universe_fwd': _num(_mean(table['top_minus_universe_fwd'])),
            'intercept': _num(model.intercept_[0]),
            'coefficients': {col: _num(coef) for col, coef in zip(cols, model.coef_[0])},
        })
        logger.info(f"Validation {year}: {years[-1]['n_train']} training rows, {years[-1]['n_test_dates']} test dates, "
                    f"mean per-date AUC {years[-1]['auc_mean']}")

    per_date = (pd.concat(tables, ignore_index=True) if tables
                else pd.DataFrame(columns=PER_DATE_COLUMNS)).sort_values('date').reset_index(drop=True)
    auc_series = per_date['auc'].to_numpy(dtype=float)
    ci = block_bootstrap_ci(auc_series, int(settings['step1_bootstrap_block_months']),
                            int(settings['step1_bootstrap_replications']), int(settings['step1_bootstrap_seed']))
    valid_auc = auc_series[np.isfinite(auc_series)]
    fitted = [entry for entry in years if entry['status'] == 'ok']
    ci_bounds = [_num(ci['ci_lower']), _num(ci['ci_upper'])]
    overall = {
        'auc_mean': _num(ci['mean']),
        'auc_ci': ci_bounds if all(b is not None for b in ci_bounds) else None,
        'n_dates': int(valid_auc.size),
        'share_dates_above_half': _num((valid_auc > 0.5).mean()) if valid_auc.size else None,
        'top_hit_rate': _num(pooled_top_hits / pooled_top_rows) if pooled_top_rows else None,
        'base_rate': _num(pooled_y_sum / pooled_rows) if pooled_rows else None,
        'top_minus_universe_fwd': _num(_mean(per_date['top_minus_universe_fwd'])),
        'is_auc_mean': _num(_mean([entry['is_auc'] for entry in fitted if entry['is_auc'] is not None])),
        'oos_auc_mean': _num(_mean([entry['oos_auc'] for entry in fitted if entry['oos_auc'] is not None])),
        'n_years_fitted': len(fitted),
        'n_test_rows': pooled_rows,
    }
    return ValidationResult(years=years, overall=overall, per_date=per_date)


def evaluate_gate(validation: ValidationResult, settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies the model gate: the lower bound of the bootstrap CI of the mean per-date AUC
    must exceed `step1_min_validation_auc_lower`.

    Args:
        validation (ValidationResult): The walk-forward validation.
        settings (Dict[str, Any]): Loaded settings.

    Returns:
        Dict[str, Any]: passed, ci_lower, threshold and reason (None if passed).
    """
    threshold = float(settings['step1_min_validation_auc_lower'])
    ci = validation.overall.get('auc_ci')
    if not ci or ci[0] is None:
        return {'passed': False, 'ci_lower': None, 'threshold': threshold, 'reason': GATE_INSUFFICIENT_DATES}
    passed = bool(ci[0] > threshold)
    return {'passed': passed, 'ci_lower': ci[0], 'threshold': threshold, 'reason': None if passed else GATE_LOW_AUC}


def score_cross_section(model: LogisticRegression, cross_section: pd.DataFrame, settings: Dict[str, Any],
                        gate_passed: bool) -> pd.DataFrame:
    """
    Scores and ranks the eligible stocks of the scoring date.

    Args:
        model (LogisticRegression): The final model.
        cross_section (pd.DataFrame): Eligible rows of the scoring date, indexed by ticker
            or by (date, ticker), with the rank columns of `step1_features`.
        settings (Dict[str, Any]): Loaded settings (step1_features, step1_up_share).
        gate_passed (bool): Result of the model gate; if False, no stock is UP.

    Returns:
        pd.DataFrame: Indexed by ticker with probability, rank (1 = highest probability,
        ties by ticker), n_ranked, top_percent (100 * rank / n_ranked) and predicted_class.
    """
    frame = cross_section
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.droplevel('date')
    cols = rank_columns(settings['step1_features'])
    columns = ['probability', 'rank', 'n_ranked', 'top_percent', 'predicted_class']
    if frame.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name='ticker'))
    probability = pd.Series(predict_probability(model, frame[cols]), index=frame.index)
    ranks = rank_by_probability(probability)
    n_ranked = len(frame)
    n_up = n_up_for(n_ranked, float(settings['step1_up_share']))
    scored = pd.DataFrame({
        'probability': probability, 'rank': ranks, 'n_ranked': n_ranked,
        'top_percent': 100.0 * ranks / n_ranked,
        'predicted_class': np.where(gate_passed & (ranks.to_numpy() <= n_up), 'UP', 'NOT_UP'),
    }, index=frame.index)
    scored.index.name = 'ticker'
    return scored


def not_eligible_reasons(feature_panel: FeaturePanel, date: pd.Timestamp, min_price: float) -> pd.Series:
    """
    Explains why stocks are not eligible on a date.

    Args:
        feature_panel (FeaturePanel): The aligned frames.
        date (pd.Timestamp): The date (normally the scoring date).
        min_price (float): `step1_min_price_eur`.

    Returns:
        pd.Series: Reason per ticker (one of INELIGIBLE_REASONS), None for eligible stocks.
    """
    price = feature_panel.price.loc[date]
    traded_value = feature_panel.traded_value.loc[date]
    missing = traded_value.isna()
    for frame in feature_panel.features.values():
        missing |= frame.loc[date].isna()
    reasons = pd.Series(None, index=feature_panel.price.columns, dtype=object)
    finite_tv = np.isfinite(traded_value.to_numpy(dtype=float))
    checks = [
        ('stale_price', price.isna().to_numpy()),
        ('insufficient_history', missing.to_numpy()),
        ('price_at_or_below_min', (price <= min_price).to_numpy()),
        ('no_traded_value', ~(finite_tv & (np.nan_to_num(traded_value.to_numpy(dtype=float), nan=-1.0) > 0))),
    ]
    eligible = feature_panel.eligible.loc[date].to_numpy()
    for reason, mask in reversed(checks):  # earlier reasons take precedence
        reasons[mask & ~eligible] = reason
    reasons[eligible] = None
    return reasons


@dataclass
class Step1Result:
    """
    Result of the pooled Step 1.

    Attributes:
        results (pd.DataFrame): One row per ticker with price data, indexed by ticker:
            latest_price, status, predicted_class, probability, rank, top_percent, the
            raw features and the rank columns r_<feature> on the scoring date.
        summary (Dict[str, Any]): The model summary (report key `step1_model`).
        validation (ValidationResult): The walk-forward validation.
    """
    results: pd.DataFrame
    summary: Dict[str, Any]
    validation: ValidationResult

    @property
    def feature_names(self) -> List[str]:
        """The model features."""
        return list(self.summary['features'])

    @property
    def up_tickers(self) -> List[str]:
        """Tickers predicted UP, sorted."""
        return sorted(self.results.index[self.results['predicted_class'] == 'UP'])


def execute_step1_pooled(price_panel: PricePanel, settings: Dict[str, Any]) -> Step1Result:
    """
    Runs the pooled Step 1: features, month-end panel, walk-forward validation, gate,
    final fit and scoring of the last closed trading day.

    Args:
        price_panel (PricePanel): Closes and volumes of the universe.
        settings (Dict[str, Any]): Loaded settings.

    Returns:
        Step1Result: Per-ticker results, model summary and validation.

    Raises:
        ValueError: If the panel holds no prices or the feature list is invalid.
    """
    feature_names = validate_feature_names(settings['step1_features'])
    feature_panel = build_feature_panel(price_panel, feature_names, settings)
    if feature_panel.price.empty or len(feature_panel.calendar) == 0:
        raise ValueError("The price panel contains no prices; Step 1 cannot run.")
    cols = rank_columns(feature_names)
    calendar = feature_panel.calendar
    scoring_date = calendar[-1]
    history = pd.DateOffset(years=int(settings['step1_history_years']))
    up_share = float(settings['step1_up_share'])
    min_price = float(settings['step1_min_price_eur'])

    panel = to_long_panel(feature_panel, month_end_dates(calendar), settings)
    validation = walk_forward_validation(panel, cols, scoring_date, settings)
    gate = evaluate_gate(validation, settings)

    dates = pd.DatetimeIndex(panel.index.get_level_values('date'))
    label_end = panel['label_end'].to_numpy(dtype='datetime64[ns]')
    train = (panel['y'].notna().to_numpy() & (label_end < np.datetime64(scoring_date))
             & np.asarray(dates < scoring_date) & np.asarray(dates >= scoring_date - history))
    model: Optional[LogisticRegression] = None
    if train.any() and panel.loc[train, 'y'].nunique() > 1:
        model = fit_pooled_model(panel.loc[train, cols], panel.loc[train, 'y'], settings)
    else:
        gate = {**gate, 'passed': False, 'reason': GATE_NO_TRAINING_ROWS}
    if gate['passed']:
        logger.info(f"Model gate passed: lower CI bound {gate['ci_lower']:.4f} > {gate['threshold']}.")
    else:
        logger.warning(f"Model gate failed ({gate['reason']}); no stock is UP today.")

    cross_section = to_long_panel(feature_panel, [scoring_date], settings).droplevel('date')
    if model is not None:
        scored = score_cross_section(model, cross_section, settings, gate['passed'])
    else:
        scored = pd.DataFrame(index=cross_section.index, columns=['probability', 'rank', 'top_percent'],
                              dtype=float).assign(predicted_class='NOT_UP')
    n_ranked = int(len(cross_section))
    n_up = n_up_for(n_ranked, up_share)

    tickers = feature_panel.tickers
    results = pd.DataFrame(index=pd.Index(tickers, name='ticker'))
    results['latest_price'] = price_panel.close.reindex(columns=tickers).ffill().iloc[-1]
    reasons = not_eligible_reasons(feature_panel, scoring_date, min_price)
    eligible_status = STATUS_OK if gate['passed'] else STATUS_GATE_FAILED
    results['status'] = [f"{NOT_ELIGIBLE}: {reasons[t]}" if reasons[t] else eligible_status for t in tickers]
    results['predicted_class'] = scored['predicted_class'].reindex(tickers).fillna('NOT_UP')
    for col in ('probability', 'rank', 'top_percent'):
        results[col] = pd.to_numeric(scored[col].reindex(tickers), errors='coerce')
    for name, rank_col in zip(feature_names, cols):
        results[name] = feature_panel.features[name].loc[scoring_date].reindex(tickers)
        results[rank_col] = cross_section[rank_col].reindex(tickers)

    coefficients = ({name: _num(coef) for name, coef in zip(feature_names, model.coef_[0])}
                    if model is not None else {})
    train_dates = dates[train]
    not_eligible_counts = reasons.value_counts()
    summary = {
        'model': MODEL_NAME,
        'scoring_date': _iso(scoring_date),
        'features': feature_names,
        'coefficients': coefficients,
        'intercept': _num(model.intercept_[0]) if model is not None else None,
        'training': {
            'n_rows': int(train.sum()), 'n_dates': int(train_dates.nunique()),
            'first_date': _iso(train_dates.min()) if train.any() else None,
            'last_date': _iso(train_dates.max()) if train.any() else None,
            'max_label_end': _iso(pd.Timestamp(label_end[train].max())) if train.any() else None,
        },
        'cross_section': {
            'n_universe': len(tickers), 'n_ranked': n_ranked, 'n_up': n_up, 'up_share': up_share,
            'n_predicted_up': int((results['predicted_class'] == 'UP').sum()),
            'not_eligible': {reason: int(not_eligible_counts.get(reason, 0)) for reason in INELIGIBLE_REASONS},
            'no_price_data': sorted(price_panel.missing),
        },
        'validation': {'years': validation.years, **validation.overall},
        'gate': gate,
        'parameters': {key: settings[key] for key in SUMMARY_PARAMETER_KEYS if key in settings},
    }
    logger.info(f"Step 1 scored {n_ranked} of {len(tickers)} stocks on {summary['scoring_date']}; "
                f"{summary['cross_section']['n_predicted_up']} UP (top {n_up}), validation AUC "
                f"{validation.overall['auc_mean']} CI {validation.overall['auc_ci']}.")
    return Step1Result(results=results, summary=summary, validation=validation)


def ticker_step1_model(result: Step1Result, ticker: str) -> Dict[str, Any]:
    """
    Builds the per-ticker `step1_model` payload entry.

    Model-level metrics (validation, gate, coefficients) are not repeated here; they are
    published once in the report key `step1_model`.

    Args:
        result (Step1Result): The pooled Step 1 result.
        ticker (str): A ticker of `result.results`.

    Returns:
        Dict[str, Any]: model, status, predicted_class, probability, rank, n_ranked,
        top_percent, up_share, n_up, scoring_date, feature_percentiles and
        logit_contributions (feature percentiles and contributions are None for
        stocks that are not ranked).
    """
    row = result.results.loc[ticker]
    cross = result.summary['cross_section']
    ranked = pd.notna(row['rank'])
    percentiles: Optional[Dict[str, Optional[float]]] = None
    contributions: Optional[Dict[str, Optional[float]]] = None
    if ranked:
        names = result.feature_names
        percentiles = {name: _num((row[f"r_{name}"] + 0.5) * 100.0, 2) for name in names}
        coefficients = result.summary['coefficients']
        contributions = {name: _num(coefficients.get(name, float('nan')) * row[f"r_{name}"], 4) for name in names}
    return {
        'model': MODEL_NAME,
        'status': str(row['status']),
        'predicted_class': str(row['predicted_class']),
        'probability': _num(row['probability'], 6) if ranked else None,
        'rank': int(row['rank']) if ranked else None,
        'n_ranked': int(cross['n_ranked']),
        'top_percent': _num(row['top_percent'], 2) if ranked else None,
        'up_share': float(cross['up_share']),
        'n_up': int(cross['n_up']),
        'scoring_date': result.summary['scoring_date'],
        'feature_percentiles': percentiles,
        'logit_contributions': contributions,
    }


def ticker_feature_diagnostics(result: Step1Result, ticker: str) -> Dict[str, Any]:
    """
    Builds the per-ticker Step 1 feature diagnostics (diagnostics key `step1_pooled`).

    Args:
        result (Step1Result): The pooled Step 1 result.
        ticker (str): A ticker of `result.results`.

    Returns:
        Dict[str, Any]: features, feature_values (raw values on the scoring date) and
        feature_percentiles (None for stocks that are not ranked).
    """
    row = result.results.loc[ticker]
    names = result.feature_names
    ranked = pd.notna(row['rank'])
    return {
        'features': names,
        'feature_values': {name: _num(row[name]) for name in names},
        'feature_percentiles': ({name: _num((row[f"r_{name}"] + 0.5) * 100.0, 2) for name in names}
                                if ranked else None),
    }
