"""
Offline tests of the quarterly variable review (src/modeling/variable_review.py): the inner
date splits and their purge, the per-date AUC, the Newey-West t-statistic, the selection
procedures on synthetic data with a planted signal, the long panel against the production
panel, the paired comparison, the cumulative Bonferroni correction (m_cum), the switch rule,
the settings update and the report.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from config.settings import load_settings
from src.ingestion.market_api import PricePanel
from src.modeling import step1_pooled as S
from src.modeling import variable_review as R
from src.processing import features as F
from tests.conftest import (DEFAULT_STEP1_FEATURES, PINNED_STEP1_SETTINGS, REVIEW_TEST_SETTINGS, PanelFactory,
                            ohlc_panel)

Settings = Dict[str, Any]
REPO = Path(__file__).resolve().parents[1]
PLANTED = 'mom_12_1'
NOISE = ['ret_5', 'vol_21', 'beta_252', 'rsi_14', 'seas_1y']


@pytest.fixture
def review_settings(step1_settings: Settings) -> Settings:
    """Production settings with the pinned model parameters and small review parameters."""
    return {**step1_settings, **REVIEW_TEST_SETTINGS}


def _training_data(signal: float = 2.5, n_dates: int = 72, n_stocks: int = 150, seed: int = 1) -> R.TrainingData:
    """
    Month-end training rows with within-date ranks; the label depends on the first column only.

    With the default seed, the largest absolute Newey-West t-statistic of a noise column is
    1.3 (1.6 without the signal), clearly below the fm_screen hurdle of 3; other seeds can
    exceed it by chance (the hurdle limits false positives, it does not exclude them).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2015-01-31', periods=n_dates, freq='ME')
    names = [PLANTED] + NOISE
    raw = rng.standard_normal((n_dates, n_stocks, len(names)))
    ranks = (raw.argsort(axis=1).argsort(axis=1) + 1.0) / n_stocks - 0.5
    X = ranks.reshape(n_dates * n_stocks, len(names))
    probability = 1.0 / (1.0 + np.exp(-(-0.7 + signal * X[:, 0])))
    y = (rng.random(len(X)) < probability).astype(float)
    row_dates = np.repeat(dates.to_numpy(dtype='datetime64[ns]'), n_stocks)
    label_end = np.repeat((dates + pd.DateOffset(months=6)).to_numpy(dtype='datetime64[ns]'), n_stocks)
    return R.TrainingData(X=X, y=y, dates=row_dates, label_end=label_end, names=names)


# ----------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------
def test_inner_splits_purge_the_labels_and_keep_enough_training_dates(review_settings: Settings) -> None:
    data = _training_data()
    splits, info = R.inner_date_splits(data.dates, data.label_end, review_settings)
    assert info['n_splits'] == 2 and not info['shortened']
    unique = np.unique(data.dates)
    for (train, validation), block in zip(splits, info['blocks']):
        start = data.dates[validation].min()
        assert data.label_end[train].max() < start and data.dates[train].max() < start
        assert np.unique(data.dates[validation]).size == review_settings['review_inner_validation_months']
        assert block['n_train_dates'] >= review_settings['review_inner_min_train_months']
    # Expanding: the second block starts where the first ends, the last block ends with the window
    assert data.dates[splits[0][1]].max() < data.dates[splits[1][1]].min()
    assert data.dates[splits[1][1]].max() == unique[-1]
    # Fewer splits when the window is short: five blocks would leave too few training dates
    splits, info = R.inner_date_splits(data.dates, data.label_end, {**review_settings, 'review_inner_splits': 5})
    assert info['n_splits'] == 4 and np.unique(data.dates[splits[0][0]]).size >= 12


def test_a_short_window_gets_one_shortened_validation_block(review_settings: Settings) -> None:
    data = _training_data(n_dates=30)
    settings = {**review_settings, 'review_inner_validation_months': 24}
    splits, info = R.inner_date_splits(data.dates, data.label_end, settings)
    assert info['n_splits'] == 1 and info['shortened']
    train, validation = splits[0]
    assert np.unique(data.dates[train]).size == settings['review_inner_min_train_months']
    assert data.label_end[train].max() < data.dates[validation].min()
    assert 0 < info['blocks'][0]['n_validation_dates'] < 24
    # Too short for any split: the procedures fall back to the champion
    # 15 month-ends: 12 training month-ends plus the 6-month purge leave no validation date
    none, info = R.inner_date_splits(data.dates[:15 * 150], data.label_end[:15 * 150], settings)
    assert none == [] and info['n_splits'] == 0


def test_the_fast_per_date_auc_equals_sklearn() -> None:
    rng = np.random.default_rng(3)
    dates = np.repeat(pd.date_range('2020-01-31', periods=5, freq='ME').to_numpy(), 40)
    y = (rng.random(200) < 0.3).astype(float)
    y[40:80] = 0.0                                   # a single-class date is skipped
    scores = np.round(rng.random(200), 1)            # with ties
    groups = R.DateGroups(dates, y, min_stocks=30)
    expected = [roc_auc_score(y[i:i + 40], scores[i:i + 40]) for i in (0, 80, 120, 160)]
    assert groups.n_valid == 4
    np.testing.assert_allclose(groups.auc(scores), expected, rtol=1e-12)


def test_newey_west_t_statistic() -> None:
    x = np.array([0.02, -0.01, 0.03, 0.01, 0.00, 0.04, -0.02, 0.02])
    errors = x - x.mean()
    plain = x.mean() / np.sqrt(errors @ errors / x.size / x.size)
    assert R.newey_west_t(x, lag=0) == pytest.approx((x.mean(), plain))
    gamma1 = errors[1:] @ errors[:-1] / x.size
    variance = errors @ errors / x.size + 2 * (1 - 1 / 2) * gamma1
    assert R.newey_west_t(x, lag=1)[1] == pytest.approx(x.mean() / np.sqrt(variance / x.size))
    assert np.isnan(R.newey_west_t(np.array([0.1]), lag=6)[1])


# ----------------------------------------------------------------------------------------
# Selection procedures on a planted signal
# ----------------------------------------------------------------------------------------
def test_stepwise_finds_the_planted_feature_and_stops_on_noise(review_settings: Settings) -> None:
    selection = R.select_features('stepwise', _training_data(), review_settings, DEFAULT_STEP1_FEATURES, {})
    assert selection.features == [PLANTED]
    assert selection.info['stop_reason'] == 'gain_below_minimum' and not selection.info['fallback_to_champion']
    path = selection.info['path']
    assert path[0]['feature'] == PLANTED and path[0]['gain'] > 0.1
    assert path[-1]['stopped'] and path[-1]['gain'] < review_settings['review_stepwise_min_gain']
    # The prefilter is refitted in every inner split and keeps review_anova_k candidates
    assert all(len(pool) == review_settings['review_anova_k'] for pool in selection.info['anova_pools'])


def test_stepwise_without_any_gain_falls_back_to_the_champion(review_settings: Settings) -> None:
    settings = {**review_settings, 'review_stepwise_min_gain': 0.5}
    selection = R.select_features('stepwise', _training_data(), settings, DEFAULT_STEP1_FEATURES, {})
    assert selection.features == DEFAULT_STEP1_FEATURES and selection.info['fallback_to_champion']


def test_lasso_keeps_the_planted_feature(review_settings: Settings) -> None:
    selection = R.select_features('lasso', _training_data(), review_settings, DEFAULT_STEP1_FEATURES, {})
    assert PLANTED in selection.features
    info = selection.info
    assert info['chosen_c'] in review_settings['review_lasso_c_grid'] and info['n_unconverged_fits'] == 0
    assert info['inner_scores'][str(info['chosen_c'])] == max(v for v in info['inner_scores'].values())
    assert not info['capped']
    capped = R.select_features('lasso', _training_data(), {**review_settings, 'review_lasso_max_features': 1,
                                                           'review_lasso_c_grid': [1.0]}, DEFAULT_STEP1_FEATURES, {})
    assert capped.features == [PLANTED] and capped.info['capped'] and capped.info['n_nonzero'] > 1


def test_fm_screen_keeps_the_planted_feature_and_rejects_noise(review_settings: Settings) -> None:
    selection = R.select_features('fm_screen', _training_data(), review_settings, DEFAULT_STEP1_FEATURES, {})
    assert selection.features == [PLANTED]
    t = selection.info['t_statistics']
    assert t[PLANTED] > 10 and max(abs(t[name]) for name in NOISE) < review_settings['review_fm_t_hurdle']
    noise = R.select_features('fm_screen', _training_data(signal=0.0), review_settings, DEFAULT_STEP1_FEATURES, {})
    assert noise.features == DEFAULT_STEP1_FEATURES and noise.info['fallback_to_champion']


def test_all_candidates_and_manual_sets_are_fixed_lists(review_settings: Settings) -> None:
    data = _training_data()
    assert R.select_features('all_candidates', data, review_settings, DEFAULT_STEP1_FEATURES, {}).features == data.names
    manual = R.select_features('manual:short', data, review_settings, DEFAULT_STEP1_FEATURES, {'short': ['ret_21']})
    assert manual.features == ['ret_21']
    with pytest.raises(ValueError):
        R.select_features('ridge', data, review_settings, DEFAULT_STEP1_FEATURES, {})


def test_candidate_pool_and_manual_sets_come_from_the_settings(review_settings: Settings) -> None:
    assert R.candidate_pool(review_settings) == list(F.FEATURE_FUNCTIONS)
    listed = {**review_settings, 'review_candidate_features': ['ret_21', 'vol_21']}
    assert R.candidate_pool(listed) == ['ret_21', 'vol_21']
    assert R.candidate_pool(review_settings, ['ret_5']) == ['ret_5']
    with pytest.raises(ValueError):
        R.candidate_pool({**review_settings, 'review_candidate_features': 'some'})
    with pytest.raises(ValueError):
        R.manual_sets({**review_settings, 'review_manual_sets': {'bad': ['not_a_feature']}})


def test_candidate_pool_by_family_is_checked_and_returned_in_registry_order(review_settings: Settings) -> None:
    def pool(value: object) -> List[str]:
        return R.candidate_pool({**review_settings, 'review_candidate_features': value})

    assert pool({'volatility': ['vol_21'], 'returns': ['ret_63', 'ret_21'], 'seasonality': None}) == \
        ['ret_21', 'ret_63', 'vol_21']
    for invalid, message in (({'returns': ['vol_21']}, 'family volatility'),
                             ({'returns': ['ret_21'], 'volatility': ['ret_21']}, 'family returns'),
                             ({'returns': ['ret_21', 'ret_21']}, 'twice'),
                             ({'returns': ['ret_21', 'not_a_feature']}, 'unknown'),
                             ({'returns': 'ret_21'}, 'must list'),
                             ({'returns': []}, 'empty'),
                             (42, 'by family')):
        with pytest.raises(ValueError, match=message):
            pool(invalid)


def test_the_pool_in_the_settings_lists_every_registered_feature_in_its_family() -> None:
    # A feature added to the registry must also be listed in config/settings.yaml (or the
    # registry and this test changed together), so the pool never grows unnoticed
    listed = load_settings()['review_candidate_features']
    assert isinstance(listed, dict)
    by_family = {name: family for family, names in listed.items() for name in names or []}
    assert by_family == F.FEATURE_FAMILIES
    assert R.candidate_pool(load_settings()) == list(F.FEATURE_FUNCTIONS)


# ----------------------------------------------------------------------------------------
# Long panel and walk-forward evaluation
# ----------------------------------------------------------------------------------------
def test_the_review_panel_reproduces_the_production_panel(simulate_ohlc_panel: PanelFactory,
                                                          review_settings: Settings) -> None:
    prices = simulate_ohlc_panel(n_stocks=30, n_days=1100, seed=2)
    production = F.build_feature_panel(prices, DEFAULT_STEP1_FEATURES, review_settings)
    expected = F.to_long_panel(production, F.month_end_dates(production.calendar), review_settings)
    panel = R.build_review_panel(prices, DEFAULT_STEP1_FEATURES + ['beta_252', 'parkinson_21'], review_settings)

    index = pd.MultiIndex.from_arrays([panel.dates, panel.tickers], names=['date', 'ticker'])
    assert index.equals(expected.index)
    columns = panel.columns(DEFAULT_STEP1_FEATURES)
    np.testing.assert_allclose(panel.ranks[:, columns], expected[F.rank_columns(DEFAULT_STEP1_FEATURES)].to_numpy(),
                               rtol=0, atol=1e-15)
    np.testing.assert_array_equal(panel.y, expected['y'].to_numpy())
    np.testing.assert_array_equal(panel.label_end, expected['label_end'].to_numpy(dtype='datetime64[ns]'))
    assert panel.scoring_date == production.calendar[-1]
    assert 0 <= panel.missing_share['beta_252'] < 1


def test_market_relative_candidates_need_the_market_index(simulate_ohlc_panel: PanelFactory,
                                                          review_settings: Settings) -> None:
    prices = simulate_ohlc_panel(n_stocks=5, n_days=400, seed=2)
    prices.market = None
    with pytest.raises(ValueError, match='beta_252'):
        R.build_review_panel(prices, ['ret_21', 'beta_252'], review_settings)
    with pytest.raises(ValueError, match='market index'):
        F.build_feature_panel(prices, ['ret_21', 'rel_ret_21'], review_settings)
    # Without market-relative features the index is not needed
    assert F.build_feature_panel(prices, ['ret_21', 'parkinson_21'], review_settings).feature_names == \
        ['ret_21', 'parkinson_21']


@pytest.fixture(scope='module')
def review_run() -> Tuple[R.ReviewResult, Settings, PricePanel]:
    """One small review on a synthetic universe with a drift signal (shared by several tests)."""
    settings = {**load_settings(), **PINNED_STEP1_SETTINGS, **REVIEW_TEST_SETTINGS,
                'step1_features': list(DEFAULT_STEP1_FEATURES),
                'review_manual_sets': {'momentum': ['mom_12_1', 'ret_252']}}
    prices = ohlc_panel(n_stocks=40, n_days=1800, seed=5)
    pool = ['ret_21', 'ret_63', 'ret_252', 'mom_12_1', 'vol_21', 'beta_252', 'parkinson_21', 'rsi_14']
    result = R.run_variable_review(prices, settings, pd.Timestamp('2026-11-05'), pool, n_jobs=1)
    return result, settings, prices


def test_every_list_is_scored_on_the_same_test_rows(review_run: Tuple[R.ReviewResult, Settings, PricePanel]) -> None:
    result, settings, _ = review_run
    reference = result.per_date[R.CHAMPION][['date', 'year', 'n']]
    for procedure, table in result.per_date.items():
        pd.testing.assert_frame_equal(table[['date', 'year', 'n']], reference)
    assert set(result.per_date) == {R.CHAMPION, 'stepwise', 'lasso', 'fm_screen', 'all_candidates', 'manual:momentum'}
    years = [w.label for w in result.windows]
    last = result.scoring_date.year
    assert years == [str(last - 2), str(last - 1), str(last), R.FINAL_WINDOW]
    for procedure in result.selections:
        assert set(result.selections[procedure]) == set(years)


def test_the_champion_is_scored_like_the_production_validation(
        review_run: Tuple[R.ReviewResult, Settings, PricePanel]) -> None:
    result, settings, prices = review_run
    production = F.build_feature_panel(R.truncate_price_panel(prices, result.review_date),
                                       settings['step1_features'], settings)
    panel = F.to_long_panel(production, F.month_end_dates(production.calendar), settings)
    validation = S.walk_forward_validation(panel, F.rank_columns(settings['step1_features']), production.calendar[-1],
                                           {**settings, 'step1_validation_years': settings['review_validation_years']})
    np.testing.assert_allclose(result.per_date[R.CHAMPION]['auc'].to_numpy(), validation.per_date['auc'].to_numpy(),
                               rtol=1e-12, equal_nan=True)


def test_the_review_does_not_depend_on_the_number_of_processes(
        review_run: Tuple[R.ReviewResult, Settings, PricePanel]) -> None:
    result, settings, prices = review_run
    parallel = R.run_variable_review(prices, settings, result.review_date, result.pool, n_jobs=2)
    for procedure, table in result.per_date.items():
        pd.testing.assert_frame_equal(parallel.per_date[procedure], table)
    assert parallel.decision == result.decision
    assert {p: {w: s['selected'] for w, s in v.items()} for p, v in parallel.selections.items()} == \
           {p: {w: s['selected'] for w, s in v.items()} for p, v in result.selections.items()}


def _walk(value: Any) -> None:
    """Asserts that a structure holds only plain JSON types and finite floats."""
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)
        for item in value.values():
            _walk(item)
    elif isinstance(value, list):
        for item in value:
            _walk(item)
    else:
        assert value is None or type(value) in (str, bool, int, float), type(value)
        if isinstance(value, float):
            assert np.isfinite(value)


def test_the_report_is_plain_json_and_the_markdown_is_generated(
        review_run: Tuple[R.ReviewResult, Settings, PricePanel]) -> None:
    result, settings, _ = review_run
    report = R.build_report(result, settings, {'download_s': 1.0, **result.timings}, R.peak_memory_mb())
    _walk(report)
    json.dumps(report, allow_nan=False)
    assert report['review_date'] == '2026-11-05' and report['decision'] in (R.DECISION_KEEP, R.DECISION_SWITCH)
    assert report['pool']['size'] == 8 and sum(len(v) for v in report['pool']['families'].values()) == 8
    assert set(report['comparisons']) == set(result.challengers)
    assert report['champion']['auc_ci_95'] is not None and report['champion']['n_fresh_dates'] == 0
    # No earlier report: m_cum is the number of challengers of this review (4 procedures and 1 manual set)
    assert report['correction']['m_cum'] == 5 and report['correction']['earlier_reports'] == []
    assert report['correction']['ci_level'] == pytest.approx(1 - 0.05 / 5)
    assert all(c['historical']['ci_level'] == report['correction']['ci_level'] for c in report['comparisons'].values())
    assert report['fresh_check']['status'] == R.FRESH_NOT_YET_AVAILABLE
    last = result.scoring_date.year
    assert set(report['procedures']['stepwise']['selected_per_year']) == {str(last - 2), str(last - 1), str(last)}
    assert set(report['parameters']) == {key for key in R.REVIEW_PARAMETER_KEYS if key in settings}

    markdown = R.render_markdown(report)
    assert markdown.startswith('# Variable Review 2026-11-05')
    for title in ('Table of Contents', 'Decision', 'Champion', 'Paired Comparison', 'AUC per Year',
                  'Selected Features', 'Candidate Pool', 'Data and Runtime', 'Parameters'):
        assert f"## {title}" in markdown
    assert '](#paired-comparison)' in markdown and '`lasso`' in markdown
    assert 'm_cum = 5' in markdown and 'Fresh evidence (d): not yet available' in markdown


# ----------------------------------------------------------------------------------------
# Paired comparison and switch rule
# ----------------------------------------------------------------------------------------
def _per_date(auc: List[float], hit: List[float], start: str = '2025-01-31') -> pd.DataFrame:
    """A per-date table with 10 top-share stocks per date."""
    dates = pd.date_range(start, periods=len(auc), freq='ME')
    return pd.DataFrame({'date': dates, 'year': dates.year, 'n': 100, 'auc': auc, 'n_top': 10,
                         'top_hit_rate': hit, 'base_rate': 0.3, 'top_fwd': 0.0, 'universe_fwd': 0.0,
                         'top_minus_universe_fwd': 0.0})


def test_the_paired_comparison_uses_the_dates_after_adoption_as_fresh_evidence(review_settings: Settings) -> None:
    rng = np.random.default_rng(0)
    champion_auc = 0.55 + 0.02 * rng.standard_normal(36)
    champion = _per_date(list(champion_auc), [0.4] * 36)
    gain = np.full(36, 0.02)
    gain[12:24] = -0.01                                     # the challenger is behind in 2026
    challenger = _per_date(list(champion_auc + gain), [0.5] * 36)
    adopted = pd.Timestamp('2027-07-15')                    # the last 6 of 36 month-ends are fresh
    one = R.compare_to_champion(champion, challenger, adopted, review_settings, m_cum=1)
    four = R.compare_to_champion(champion, challenger, adopted, review_settings, m_cum=4)

    assert one['historical']['mean_d'] == pytest.approx(gain.mean()) and one['historical']['n_dates'] == 36
    # A year is won if the mean d of its dates is positive
    assert one['historical']['mean_d_per_year'] == {'2025': pytest.approx(0.02), '2026': pytest.approx(-0.01),
                                                    '2027': pytest.approx(0.02)}
    assert one['historical']['years_won'] == 2 and one['historical']['n_years'] == 3
    assert one['fresh']['n_dates'] == 6 and one['fresh']['mean_d'] == pytest.approx(0.02)
    assert one['hit_rate']['all'] == {'challenger': pytest.approx(0.5), 'champion': pytest.approx(0.4)}
    assert one['hit_rate']['fresh'] == {'challenger': pytest.approx(0.5), 'champion': pytest.approx(0.4)}
    # Bonferroni: level 1 - alpha / m_cum, a wider interval for more comparisons
    assert one['historical']['ci_level'] == pytest.approx(0.95)
    assert four['historical']['ci_level'] == pytest.approx(1 - 0.05 / 4)
    assert four['historical']['ci'][0] <= one['historical']['ci'][0] <= one['historical']['ci'][1] \
        <= four['historical']['ci'][1]
    with pytest.raises(ValueError):
        R.compare_to_champion(champion, challenger.iloc[1:], adopted, review_settings, 1)


def _write_report(folder: Path, date: str, n_challengers: int = 4) -> None:
    """A minimal review report as the workflow commits it."""
    challengers = ['stepwise', 'lasso', 'fm_screen', 'all_candidates', 'manual:a', 'manual:b'][:n_challengers]
    (folder / f'review_{date}.json').write_text(json.dumps({'review_date': date, 'challengers': challengers}))


def test_m_cum_counts_the_reviews_after_the_adoption_and_before_this_review(tmp_path: Path) -> None:
    adopted, review_date = pd.Timestamp('2026-11-05'), pd.Timestamp('2027-11-01')
    _write_report(tmp_path, '2026-08-01')                   # compared the previous champion
    _write_report(tmp_path, '2026-11-05')                   # the review that adopted the champion
    _write_report(tmp_path, '2027-02-01')
    _write_report(tmp_path, '2027-05-01', n_challengers=5)  # with a manual set
    _write_report(tmp_path, '2027-08-01')
    _write_report(tmp_path, '2027-11-01')                   # an earlier run of this review, replaced by it
    (tmp_path / 'review_2027-11-01.md').write_text('# not a report')

    earlier, counted = R.earlier_comparisons(tmp_path, adopted, review_date)
    assert earlier == 13
    assert counted == ['review_2027-02-01.json', 'review_2027-05-01.json', 'review_2027-08-01.json']
    correction = R.multiple_testing_correction({'review_family_alpha': 0.05}, 4, earlier, counted)
    assert correction['m_cum'] == 17 and correction['ci_level'] == pytest.approx(1 - 0.05 / 17)
    # A switch moves the adoption date and starts a new count
    assert R.earlier_comparisons(tmp_path, pd.Timestamp('2027-08-01'), review_date) == (0, [])
    assert R.earlier_comparisons(tmp_path / 'missing', adopted, review_date) == (0, [])
    assert R.earlier_comparisons(None, adopted, review_date) == (0, [])


def test_m_cum_sets_the_bonferroni_level_and_rejects_unreadable_reports(tmp_path: Path) -> None:
    # The first review with the 4 default challengers, and after four quarterly reviews
    assert R.multiple_testing_correction({'review_family_alpha': 0.05}, 4, 0, [])['ci_level'] == \
        pytest.approx(0.9875)
    correction = R.multiple_testing_correction({'review_family_alpha': 0.05}, 4, 8, ['a', 'b'])
    assert correction['m_cum'] == 12 and correction['ci_level'] == pytest.approx(1 - 0.05 / 12)
    assert R.multiple_testing_correction({'review_family_alpha': 0.05}, 4, 12, [])['ci_level'] == \
        pytest.approx(0.996875)
    for alpha in (0.0, 1.5):
        with pytest.raises(ValueError):
            R.multiple_testing_correction({'review_family_alpha': alpha}, 4, 0, [])
    (tmp_path / 'review_2027-02-01.json').write_text('{"review_date": "2027-02-01"')
    with pytest.raises(ValueError, match='review_2027-02-01.json'):
        R.earlier_comparisons(tmp_path, pd.Timestamp('2026-11-05'), pd.Timestamp('2027-05-01'))


def _comparison(mean_d: float = 0.02, ci_lower: Optional[float] = 0.005, years_won: int = 6, n_years: int = 8,
                n_fresh: int = 0, fresh_d: Optional[float] = None, hit: Tuple[float, float] = (0.42, 0.40),
                fresh_hit: Tuple[Optional[float], Optional[float]] = (None, None)) -> Dict[str, Any]:
    """A compare_to_champion() result of a first review (no fresh dates) that passes every condition."""
    return {'auc_mean': 0.58,
            'historical': {'mean_d': mean_d, 'ci': None if ci_lower is None else [ci_lower, mean_d + 0.02],
                           'ci_level': 0.9875, 'n_dates': 96, 'years_won': years_won, 'n_years': n_years},
            'fresh': {'n_dates': n_fresh, 'mean_d': fresh_d},
            'hit_rate': {'all': {'challenger': hit[0], 'champion': hit[1]},
                         'fresh': {'challenger': fresh_hit[0], 'champion': fresh_hit[1]}}}


def _decide(review_settings: Settings, **change: Any) -> Dict[str, Any]:
    """The switch rule for one challenger (lasso) with a list that differs from the champion."""
    return R.evaluate_switch_rule({'lasso': _comparison(**change)}, {'lasso': ['ret_21', 'vol_21']},
                                  DEFAULT_STEP1_FEATURES, review_settings)


@pytest.mark.parametrize('change,failed', [
    ({}, None),
    ({'mean_d': 0.014}, 'a_historical_mean_d'),
    ({'ci_lower': 0.0}, 'b_ci_lower_above_zero'),
    ({'ci_lower': None}, 'b_ci_lower_above_zero'),
    ({'years_won': 5}, 'c_winning_years'),
    ({'n_fresh': 6, 'fresh_d': -0.001, 'fresh_hit': (0.42, 0.40)}, 'd_fresh_evidence'),
    ({'n_fresh': 6, 'fresh_d': 0.01, 'fresh_hit': (0.39, 0.40)}, 'd_fresh_evidence'),
    ({'hit': (0.399, 0.40)}, 'e_hit_rate'),
])
def test_each_condition_of_the_switch_rule_decides_on_its_own(change: Dict[str, Any], failed: Optional[str],
                                                              review_settings: Settings) -> None:
    decision = _decide(review_settings, **change)
    if failed is None:
        assert decision['decision'] == R.DECISION_SWITCH and decision['winner'] == 'lasso'
        assert decision['adopted_features'] == ['ret_21', 'vol_21']
    else:
        assert decision['decision'] == R.DECISION_KEEP and decision['winner'] is None
        assert decision['challengers']['lasso']['failed'] == [failed]
        assert decision['reasons'][0].startswith('lasso: fails')


def test_the_first_review_can_switch_and_the_fresh_check_waits_for_enough_dates(review_settings: Settings) -> None:
    first = _decide(review_settings)
    assert first['decision'] == R.DECISION_SWITCH
    assert first['fresh_check'] == {'status': R.FRESH_NOT_YET_AVAILABLE, 'n_fresh': 0, 'min_dates': 6}
    assert first['challengers']['lasso']['conditions']['d_fresh_evidence']['status'] == R.FRESH_NOT_YET_AVAILABLE
    # Five fresh dates: contradicting fresh evidence is not checked yet
    early = _decide(review_settings, n_fresh=5, fresh_d=-0.03, fresh_hit=(0.20, 0.40))
    assert early['decision'] == R.DECISION_SWITCH
    assert early['challengers']['lasso']['conditions']['d_fresh_evidence']['status'] == R.FRESH_NOT_YET_AVAILABLE
    # From six fresh dates on, a fresh mean d of 0 and an equal hit rate do not contradict; the
    # fresh mean d does not have to reach the hurdle
    checked = _decide(review_settings, n_fresh=6, fresh_d=0.0, fresh_hit=(0.40, 0.40))
    assert checked['decision'] == R.DECISION_SWITCH and checked['fresh_check']['status'] == 'checked'
    assert checked['challengers']['lasso']['conditions']['d_fresh_evidence']['status'] == 'passed'
    # The stability condition counts the test years that have dates (7 in a February review)
    assert _decide(review_settings, years_won=6, n_years=7)['decision'] == R.DECISION_SWITCH


def test_no_switch_to_a_list_equal_to_the_champion(review_settings: Settings) -> None:
    same = R.evaluate_switch_rule({'fm_screen': _comparison()}, {'fm_screen': list(reversed(DEFAULT_STEP1_FEATURES))},
                                  DEFAULT_STEP1_FEATURES, review_settings)
    assert same['decision'] == R.DECISION_KEEP and same['challengers']['fm_screen']['failed'] == ['f_list_differs']


def test_the_largest_corrected_lower_bound_wins_ties_go_to_the_mean_and_the_order(review_settings: Settings) -> None:
    lists = {name: ['ret_21'] for name in ('stepwise', 'lasso', 'fm_screen')}
    by_bound = R.evaluate_switch_rule({'stepwise': _comparison(ci_lower=0.004), 'lasso': _comparison(ci_lower=0.009),
                                       'fm_screen': _comparison(ci_lower=0.006)}, lists, DEFAULT_STEP1_FEATURES,
                                      review_settings)
    assert by_bound['winner'] == 'lasso'
    by_mean = R.evaluate_switch_rule({'stepwise': _comparison(mean_d=0.02), 'lasso': _comparison(mean_d=0.03)},
                                     lists, DEFAULT_STEP1_FEATURES, review_settings)
    assert by_mean['winner'] == 'lasso'
    by_order = R.evaluate_switch_rule({'fm_screen': _comparison(), 'stepwise': _comparison()}, lists,
                                      DEFAULT_STEP1_FEATURES, review_settings)
    assert by_order['winner'] == 'fm_screen'


# ----------------------------------------------------------------------------------------
# Settings update
# ----------------------------------------------------------------------------------------
def test_the_settings_update_changes_only_the_two_lines() -> None:
    text = (REPO / 'config' / 'settings.yaml').read_text(encoding='utf-8')
    updated = R.update_settings_text(text, ['ret_21', 'dist_low_252', 'mkt_r2_252'], '2027-08-01')

    before, after = text.splitlines(), updated.splitlines()
    assert len(before) == len(after)
    changed = [(old, new) for old, new in zip(before, after) if old != new]
    assert [new.split(':')[0] for _, new in changed] == ['step1_features', 'step1_features_adopted']
    for old, new in changed:
        assert old.split('#', 1)[1] == new.split('#', 1)[1]          # the comment is kept
    assert 'step1_features: [ret_21, dist_low_252, mkt_r2_252]  #' in updated
    assert 'step1_features_adopted: "2027-08-01"  #' in updated
    parsed, original = yaml.safe_load(updated), yaml.safe_load(text)
    assert list(parsed) == list(original)
    assert parsed['step1_features'] == ['ret_21', 'dist_low_252', 'mkt_r2_252']
    assert parsed['step1_features_adopted'] == '2027-08-01'


def test_the_settings_update_rejects_unknown_names_and_missing_lines() -> None:
    text = (REPO / 'config' / 'settings.yaml').read_text(encoding='utf-8')
    with pytest.raises(ValueError):
        R.update_settings_text(text, ['not_a_feature'], '2027-08-01')
    without = '\n'.join(line for line in text.splitlines() if not line.startswith('step1_features_adopted'))
    with pytest.raises(ValueError, match='step1_features_adopted'):
        R.update_settings_text(without, ['ret_21'], '2027-08-01')
    twice = text + '\nstep1_features: [ret_21]\n'
    with pytest.raises(ValueError, match="'step1_features:'"):
        R.update_settings_text(twice, ['ret_21'], '2027-08-01')


def test_the_pull_request_texts_state_the_evidence(review_run: Tuple[R.ReviewResult, Settings, PricePanel]) -> None:
    result, settings, _ = review_run
    report = R.build_report(result, settings, result.timings)
    report = {**report, 'decision': R.DECISION_SWITCH, 'winner': 'lasso',
              'adopted_features': report['procedures']['lasso']['final_list']}
    body = R.render_pr_body(report, 'data/processed/variable_reviews/review_2026-11-05.md')
    assert '## Table of Contents' in body and '`lasso`' in body and 'GITHUB_TOKEN' in body
    assert 'data/processed/variable_reviews/review_2026-11-05.md' in body
    assert 'm_cum = 5' in body and 'not yet available' in body
    assert 'Output size' in body and 'docs/feature_engineering.sql' in body
    commit = R.render_commit_body(report, 'data/processed/variable_reviews/review_2026-11-05.md')
    assert commit.startswith('The variable review of 2026-11-05 qualifies lasso') and '#' not in commit
    assert 'm_cum = 5' in commit
