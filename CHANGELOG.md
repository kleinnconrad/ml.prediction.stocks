# Changelog

## [3.1.0](https://github.com/kleinnconrad/ml.prediction.stocks/compare/v3.0.1...v3.1.0) (2026-11-05)


### Features

* **review:** add a quarterly review that tests 237 candidate variables against the Step 1 set and proposes a switch by pull request ([959137b](https://github.com/kleinnconrad/ml.prediction.stocks/commit/959137bf12bbab024f382212d12b678edd707101))

## [3.0.1](https://github.com/kleinnconrad/ml.prediction.stocks/compare/v3.0.0...v3.0.1) (2026-11-05)


### Bug Fixes

* **step1:** address the review findings of the pooled model ([c080954](https://github.com/kleinnconrad/ml.prediction.stocks/commit/c080954bc51da44a33c14f3ae4cb75c69bd53f03))


### Documentation

* **readme:** remove two filler adverbs from the paradigm section ([5cc4c72](https://github.com/kleinnconrad/ml.prediction.stocks/commit/5cc4c72292e3245df4a12e390e61ef210aa77111))

## [3.0.0](https://github.com/kleinnconrad/ml.prediction.stocks/compare/v2.1.1...v3.0.0) (2026-11-04)


### ⚠ BREAKING CHANGES

* **step1:** Step 1 changes from about 420 per-stock models to one pooled cross-sectional model. The per-stock logistic regressions with ANOVA pre-filter and stepwise selection over about 400 macro and technical features were overfitted (in-sample AUC 0.97, out-of-sample 0.51, the same as on data without macro information). Step 1 now fits one L2 logistic regression on the month-end rows of all stocks with 10 fixed price and volume features ranked within each date. It validates the model by yearly walk-forward refits (out-of-sample cross-sectional AUC 0.56 [0.545, 0.575]) and marks the top 10% of the daily ranking as UP when the lower bound of that interval exceeds 0.5, so UP now means a top-10% rank among all stocks, not a per-stock timing forecast. The ANOVA pre-filter and the stepwise feature selection are dropped: tested honestly on 53 stock-level candidates, neither they nor lasso, elastic net or significance screening beat the fixed features out of sample, and with macro candidates the selection reduced the ranking to chance. Removed: the macro and FRED data pipeline, the per-stock KS cutoffs, CV gates and charts, and the settings features_to_select, anova_k, quantiles, cv_splits, sfs_cv_splits, min_cv_train_rows, min_cv_accuracy and fred_*. New step1_* settings configure the model. The payload field step1_model, the diagnostics key (step1_pooled) and the Step 1 state file (data/state/step1_passed.csv) change. The report gains a top-level step1_model summary. See the README section "Paradigm Change in 3.0.0".

### Features

* **step1:** replace the per-stock models with one pooled cross-sectional model ([7de20b0](https://github.com/kleinnconrad/ml.prediction.stocks/commit/7de20b058054e9d755e96e2c7b02291844796940))

## [2.1.1](https://github.com/kleinnconrad/ml.prediction.stocks/compare/v2.1.0...v2.1.1) (2026-11-03)


### Bug Fixes

* read the Gemini model from settings.yaml and switch to gemini-3.… ([34a6e57](https://github.com/kleinnconrad/ml.prediction.stocks/commit/34a6e5732094c9d225082a66e734fe7425e2b357))
* read the Gemini model from settings.yaml and switch to gemini-3.5-flash-lite ([096e789](https://github.com/kleinnconrad/ml.prediction.stocks/commit/096e789f05c1da47e686fbc079385574205e638d))

## [2.1.0](https://github.com/kleinnconrad/ml.prediction.stocks/compare/v2.0.0...v2.1.0) (2026-10-09)


### Features

* add pre merge pipeline ([e115259](https://github.com/kleinnconrad/ml.prediction.stocks/commit/e115259cddf4c989f6b8f7584c05b6b1cbb7964b))
* add pre merge pipeline ([67cb110](https://github.com/kleinnconrad/ml.prediction.stocks/commit/67cb110c08756fcaa7af7b3eff675d6391a56ad5))

## [2.0.0](https://github.com/kleinnconrad/ml.prediction.stocks/compare/v1.5.0...v2.0.0) (2026-10-09)


### ⚠ BREAKING CHANGES

* **step2:** Predictions are not comparable with 1.x. Step 1 trains on 10 years of history with purged cross-validation and walk-forward KS cutoffs, rejects tickers it cannot validate with a cv_status (listings younger than about 3.3 years are rejected), and uses a changed macro feature set (publication-lagged FRED data, copper, gold and crude oil added, discontinued series removed). Step 2 compares against the same period one year earlier and scores only applicable rules with a pro-rated threshold. settings.yaml gained new keys and no longer contains step2_history_years.

### Features

* **ci:** run the unit test suite on pull requests ([96e896d](https://github.com/kleinnconrad/ml.prediction.stocks/commit/96e896da4d05df6b67a2c0e054e19f5bd881da2d))
* **step2:** compare fundamentals year over year and score only applicable rules ([b88c22e](https://github.com/kleinnconrad/ml.prediction.stocks/commit/b88c22efca9585714f07016a02a10d380617e87f))


### Bug Fixes

* **ci:** find the Step 2 prediction payloads in the downloaded artifact layout ([cb32a1c](https://github.com/kleinnconrad/ml.prediction.stocks/commit/cb32a1c8e207fffa6ff9fdf0fb34b111616e90bf))
* **ci:** pin actions, lock uv, set explicit permissions and let Dependabot update uv.lock ([76e6ada](https://github.com/kleinnconrad/ml.prediction.stocks/commit/76e6adad50db8d6a161e70df4b9563acf13f7d71))
* **ci:** run the T7 refresh before the Monday pipeline and use the committed ticker list ([d083011](https://github.com/kleinnconrad/ml.prediction.stocks/commit/d0830116c8626ac33e361bb25303d41bb7dd01a7))
* **config:** honor settings.yaml parameters including the 10-year Step 1 history ([53cd5dc](https://github.com/kleinnconrad/ml.prediction.stocks/commit/53cd5dce09e4eabcafb2236439f6ef0b9a82ff54))
* **dashboard:** escape data-derived HTML and pin the Chart.js version ([bbdcfac](https://github.com/kleinnconrad/ml.prediction.stocks/commit/bbdcfaca2cb483f755af011f95a34459040e5bb2))
* **dashboard:** show UTC execution time and handle missing uplift and non-applicable rules ([930e72c](https://github.com/kleinnconrad/ml.prediction.stocks/commit/930e72c5d8bad94245d042405c6cf23f4b1d2dff))
* **deps:** declare pydantic as a direct dependency and add pytest for development ([6ad428f](https://github.com/kleinnconrad/ml.prediction.stocks/commit/6ad428febf3353ff499cdde223a6362ebd5107a9))
* **devcontainer:** move the config to .devcontainer and install with uv on Python 3.12 ([3e58ad9](https://github.com/kleinnconrad/ml.prediction.stocks/commit/3e58ad95e4cdf6bd1a1e0a590f00763448a2c25c))
* **diagnostics:** render diagnostic charts with the non-interactive Agg backend ([661ccd2](https://github.com/kleinnconrad/ml.prediction.stocks/commit/661ccd25e91e80e18d75caf111cc33006e37fa1f))
* **docs:** align README, module docs and SQL reference with the implemented pipeline ([625be77](https://github.com/kleinnconrad/ml.prediction.stocks/commit/625be771fbe17e16e2e470946f18b3d6887aa36d))
* **fundamentals:** warn when the fundamentals cache is older than its refresh interval ([3d1cf9c](https://github.com/kleinnconrad/ml.prediction.stocks/commit/3d1cf9c5a28a257e084ddea9f0f18a35a84ec79f))
* **history:** prune archived reports by report date so the retention window applies ([85c5601](https://github.com/kleinnconrad/ml.prediction.stocks/commit/85c5601a371f335ad7775329bd8c81b7d15e4925))
* **macro:** build momentum and YoY acceleration features for FRED indicators ([e72fca4](https://github.com/kleinnconrad/ml.prediction.stocks/commit/e72fca4c592975241eeaf3312736dd4c39b230fc))
* **macro:** forward-fill macro series before aligning them to business days ([c861fc8](https://github.com/kleinnconrad/ml.prediction.stocks/commit/c861fc83f7835a84dde6db160f9d4ee55495468c))
* **macro:** lag FRED indicators by their publication delay to remove look-ahead bias ([8a0a61b](https://github.com/kleinnconrad/ml.prediction.stocks/commit/8a0a61b46cd8497153d79911e46eeff32d0deef5))
* **macro:** replace discontinued FRED series and add copper, gold and crude oil futures ([35d845f](https://github.com/kleinnconrad/ml.prediction.stocks/commit/35d845fa5a352cce1562fe9aa44a0b4e4cb30ced))
* **pipeline:** share step runners between local and CI runs and keep per-ticker diagnostics ([2746985](https://github.com/kleinnconrad/ml.prediction.stocks/commit/27469858d913f19a0fa8041ef9ffd20c1f5793ba))
* **profiles:** persist the company profile cache in CI to avoid nightly Gemini calls ([f79511a](https://github.com/kleinnconrad/ml.prediction.stocks/commit/f79511abbf970b3a08c7472b5019c17153202c1c))
* rm penny stocks from uplift chart ([3fea5d7](https://github.com/kleinnconrad/ml.prediction.stocks/commit/3fea5d72324fb06552acc027599188ca10ae6a73))
* **scripts:** log the T7 download through the configured logger ([f8b6d14](https://github.com/kleinnconrad/ml.prediction.stocks/commit/f8b6d14f2956570926b6ad719b478162d33f23da))
* **step1:** classify every model with the KS cutoff learned on its own training rows ([3f6f96d](https://github.com/kleinnconrad/ml.prediction.stocks/commit/3f6f96d0f7b7d90691882fe01135ec4ad38914af))
* **step1:** keep feature names aligned when a macro feature has no observations ([a2b7b60](https://github.com/kleinnconrad/ml.prediction.stocks/commit/a2b7b6016b022a9864ba6967c241e76bc2ee6122))
* **step1:** purge overlapping labels between CV folds and drop the in-sample accuracy fallback ([7f65157](https://github.com/kleinnconrad/ml.prediction.stocks/commit/7f651570737620359532940d2ea8e9377113339d))
* **step1:** report a failing final model fit as a rejection instead of dropping the ticker ([f24052d](https://github.com/kleinnconrad/ml.prediction.stocks/commit/f24052d1f84e35fd026c99920e43eb9eeb13258a))
* **step1:** select the KS cutoff walk-forward and reject tickers with degenerate cutoffs ([c512c22](https://github.com/kleinnconrad/ml.prediction.stocks/commit/c512c2203b555e44911d47bafa13e8262695d5d3))
* **uplift:** measure returns from each report's own date and mark empty cohorts as missing ([57282bc](https://github.com/kleinnconrad/ml.prediction.stocks/commit/57282bc94b6d2da6d79a5438986ec5b98ba91928))

## [1.5.0](https://github.com/kleinnconrad/ml.prediction.stocks/compare/v1.4.0...v1.5.0) (2026-09-22)


### Features

* added rolling (1m, 3m, 6m) uplift diagnostic ([81f7cf3](https://github.com/kleinnconrad/ml.prediction.stocks/commit/81f7cf35ce7fb978d7631ba9f861d15b7675c0ec))


### Bug Fixes

* add historic data ([e3c8dc9](https://github.com/kleinnconrad/ml.prediction.stocks/commit/e3c8dc9e47c3e3b8f6c394f3c66d908a3304c061))
* artefact constraint for pages deployment ([692a7ef](https://github.com/kleinnconrad/ml.prediction.stocks/commit/692a7ef414333baca46b9a78944641ca5488585d))
* include actual legacy results for the uplift chart ([6e365a5](https://github.com/kleinnconrad/ml.prediction.stocks/commit/6e365a53c52624dca452f6e18df2915064579f40))
* uplift visual in dashboard ([c247cfb](https://github.com/kleinnconrad/ml.prediction.stocks/commit/c247cfb85370f26b8925c2f53892d83af7d70279))

## [1.4.0](https://github.com/kleinnconrad/ml.prediction.stocks/compare/v1.3.1...v1.4.0) (2026-09-16)


### Features

* switching to pyproject.toml ([1118e23](https://github.com/kleinnconrad/ml.prediction.stocks/commit/1118e23663dbe58fcb96b8a2cd9f398e989761a1))

## [1.3.1](https://github.com/kleinnconrad/ml.prediction.stocks/compare/v1.3.0...v1.3.1) (2026-07-23)


### Bug Fixes

* leakage of absolute target ticker values (Open, High, Low, Close, Volume) into the predictor set. Included those vars into the feature engineering turning them stationary. ([9ce4fbd](https://github.com/kleinnconrad/ml.prediction.stocks/commit/9ce4fbd2a007459f7fe1446018be66f00b2dcf40))

## [1.3.0](https://github.com/kleinnconrad/ml.prediction.stocks/compare/v1.2.1...v1.3.0) (2026-07-16)


### Features

* save 20260716_final_batch_results.json for model validation ([291d069](https://github.com/kleinnconrad/ml.prediction.stocks/commit/291d069e9499e3b3771dba5133954106b64d6a79))
