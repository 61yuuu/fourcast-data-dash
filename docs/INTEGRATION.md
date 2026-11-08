# Final integration and reproduction

This document supersedes the earlier integration notes, retained in docs/archive/previous_INTEGRATION.md.

## Final run confirmed on 2026-11-07

The team completed pip check, pytest, validate, report, evaluate, predict, check and verify_delivery on its Mac. The supplied test screenshot recorded 49 passed / 1 skipped (optional CatBoost). The final verification JSON records submission_check=passed and zero validation predictions differing from the member reference for BOTH tracks.

| Track | Homes | MdAPE (%) | MAPE (%) | Median signed error (%) |
|---|---:|---:|---:|---:|
| off_market / history_text_boost | 10,069 | 9.474159 | 12.107426 | +0.064972 |
| list_aware / current_v3 | 10,069 | 4.321280 | 5.398640 | +0.731284 |

Validation and final prediction metadata agree: Python 3.11.5, pandas 2.0.3, numpy 1.24.3, scikit-learn 1.3.0. The full package snapshot is requirements-final.txt. requirements.txt now includes that file; no longer use the earlier newer-version requirements as the default.

## Reproduce

Follow README.md with Python 3.11.5 and requirements-final.txt. Official synthetic files download automatically, or set DATA_DIR to a directory containing listings, test_off_market and test_list_aware as parquet or CSV. Run validation, report, evaluation, prediction, checks in that order.

The shipped submissions contain 3,974 off-market and 4,064 list-aware predictions. The current outputs and docs/run_verification.json came from the team run. During this packaging review, saved validation errors were independently recalculated, both tracks' prediction values were compared with the member reference by ID, and validation/prediction environment metadata were checked for agreement. No full-data training was repeated during packaging.

## Scope and interpretation

The model factory, feature engineering, experiment implementations and evaluation script are unchanged. Original Member 1 code stays isolated in reference/member1_original; starter features.py remains active for Member 3's selected model. Optional experiments may change selected_off_market.json and require regenerating all dependent outputs.

Monthly sale-month walk-forward is not a listing-day backtest. Validation months were reused for development and selection. Aggregate bootstrap intervals in reports/validation_report.json resample homes as independent; they do not account for shared monthly effects or model-selection uncertainty. Subgroup tables have no confidence intervals. Synthetic results do not establish accuracy on real Toronto sales.

## Archived material

Earlier logs, requirements and integration notes were moved/copied to docs/archive with previous_ prefixes. Their 9.53% off-market score is an earlier run, not the final result. Earlier output hashes were replaced by a fresh integration_manifest.json for the current package (the manifest excludes itself).

## Still to complete

report/report.tex remains the starter template. Assemble the final four-page report, full team names, video and GitHub links. Keep all report/video values consistent with current outputs. No repository publishing or external submission was performed during packaging.
