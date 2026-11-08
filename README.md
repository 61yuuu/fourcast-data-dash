# Forecast — SSU Data Dash 2026

Forecast's integrated code for the Toronto Home-Price Prediction Challenge. The pipeline trains separate off-market and list-aware models, performs monthly walk-forward validation, generates error diagnostics, and writes both submission files.

## Final verified results

The team completed the full pipeline on November 7, 2026, using Python 3.11.5 on macOS. Dependency checks passed; 49 tests passed and one optional CatBoost test was skipped. All 10,069 validation predictions per track matched the corresponding member reference, and both submission checks passed.

| Track / selected model | Validation MdAPE | Validation MAPE | Median signed error | Test homes |
|---|---:|---:|---:|---:|
| Off-market / `history_text_boost` | 9.47% | 12.11% | +0.06% | 3,974 |
| List-aware / `current_v3` | 4.32% | 5.40% | +0.73% | 4,064 |

These are development-validation results for December 2025 through May 2026, not hidden-test scores. Positive signed errors indicate overprediction.

`requirements-final.txt` records the full package versions from the verified run. `requirements.txt` includes that file. `requirements-tested.txt` is an earlier minimal environment record, not the final installation entry point.

## Quick start

Use **Python 3.11.5**. Run the following commands from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-final.txt
python -m pytest -q
python run.py validate
python run.py report
python run.py evaluate
python run.py predict
python run.py check
python verify_delivery.py
```

On Windows, activate the environment with `.venv\Scripts\activate`. If `python3.11` is unavailable, use the command pointing to your Python 3.11.5 installation.

The first run downloads the official synthetic dataset from [Hugging Face](https://huggingface.co/datasets/Baliyang/SSU_data_dash). Alternatively, set `DATA_DIR` to a directory containing `listings.parquet`, `test_off_market.parquet`, and `test_list_aware.parquet`; CSV inputs are also supported. Raw data are not included in this repository package.

## Selected models and team contributions

| Contribution | Main files | Final use |
|---|---|---|
| Member 1 | `notebooks/eda.ipynb`, `member1_features.py`, `check_member1_features.py` | EDA, data cleaning, property/area features, historical sales summaries |
| Member 2 | `off_market_model.py`, `experiment_off_market.py` | `history_text_boost`: 70 features; histogram gradient boosting on log sale price |
| Member 3 | `list_aware_feature_models.py`, `list_aware_member1_model.py`, `list_aware_models.py`, experiment scripts | `current_v3`: 25 features; histogram gradient boosting on the sale-to-list price ratio |
| Member 4 | `evaluation.py` | Prediction checks and error analysis by price, property type, month, and community |
| Shared pipeline | `models.py`, `run.py`, `data.py`, `check_submission.py` | Validation, reporting, prediction, and submission checks |

`features.py` retains the starter interface used by the selected list-aware model. Member 1's enhanced features are isolated in `member1_features.py`; do not overwrite the shared `features.py` with the original Member 1 version. `selected_off_market.json` fixes the selected off-market candidate, so reproducing the final pipeline does not require rerunning every experiment.

## Outputs

- `outputs/validation_predictions.csv`: individual validation predictions for both tracks, including model labels and community identifiers.
- `outputs/validation_metadata.json` and `outputs/prediction_metadata.json`: Python/package versions and model information for the recorded runs.
- `reports/validation_report.md`, `reports/validation_report.json`, `reports/results_table.tex`, and `figures/error_by_price_range.png`: aggregate validation results and report materials.
- `reports/diagnostics/`: subgroup tables, three diagnostic figures, community summaries, and interpretation notes.
- `submissions/submission_off_market.csv`: 3,974 off-market predictions.
- `submissions/submission_list_aware.csv`: 4,064 list-aware predictions.
- `reference_results/member2/`, `reference_results/member3/`, and `reference_results/member4/`: original member result snapshots, retained for comparison.
- `docs/`: member report materials, integration notes, and run-verification records.
- `reference/member1_original/features.py`: the original Member 1 reference implementation, containing a known historical-feature indentation issue. This reference file is not imported by the final models.

The pipeline regenerates files under `reports/` and `submissions/`. Original member outputs remain under `reference_results/`. See `docs/INTEGRATION.md` and `docs/run_verification.json` for the final reproduction record.

## Optional candidate experiments

These commands are for investigating model alternatives, not for reproducing the fixed final models:

```bash
python experiment_off_market.py
python build_report.py
python experiment_list_aware_features.py
python experiment_list_aware_member1.py
```

Outputs are written under `experiments/off_market/`, `experiments/list_aware_v3/`, and `experiments/list_aware_v4/`. The off-market experiment updates `selected_off_market.json`; rerun validation, reporting, evaluation, and prediction after changing the selected candidate. This experiment requires an explicit `DATA_DIR` and uses the audited baseline predictions in `reference/starter_validation_predictions.csv`.

The earliest algorithm comparison includes optional CatBoost:

```bash
python -m pip install -r requirements-experiments.txt
python experiment_list_aware.py
```

Its outputs are stored under `experiments/list_aware_v2/`. Neither final model requires CatBoost. Without it, the corresponding optional test is skipped. `compare_list_aware.py` is an earlier FeatureModel comparison and should not be used with the currently selected model. The R baseline and quickstart notebook are starter teaching materials rather than final model entry points.

## Exploratory data analysis

Start Jupyter from the repository root and open `notebooks/eda.ipynb`, keeping the working directory at the root so that `data.py` can be imported. The notebook preserves the analysis and saved outputs; it was not rerun during final integration and does not automatically retrain the selected models.

## Validation scope and limitations

Validation uses monthly sale-month walk-forward evaluation from December 2025 through May 2026, not a strict listing-day backtest. These months were reused for development and model selection, so the reported scores are not independent test estimates and do not guarantee hidden-test performance.

The aggregate bootstrap intervals resample individual homes as independent observations. They do not account for shared monthly effects or model-selection uncertainty. Subgroup tables do not provide confidence intervals. All data and community locations are synthetic; validation results do not establish accuracy on real Toronto transactions. No external real-property records or third-party AVMs were added to the prediction inputs.

## Report and submission materials

The competition submission consists of a PDF report linking to a video and a GitHub repository containing the code and both prediction CSVs. `report/report.tex` is the starter report template; the material under `docs/` supports the final write-up. The submitted PDF should include all team members' full names and the final video and repository links.

## Final artifacts and archives

Use `outputs/`, `reports/`, `figures/`, `submissions/`, and `docs/run_verification.json` as the current results. `docs/archive/` retains older environment notes, run logs, and checksums. In particular, the archived **9.53%** off-market score belongs to an earlier run; the final verified score is **9.47%**. `reference_results/` preserves member experiment records.

The final packaging preserved the model/evaluation source, prediction CSVs, and diagnostic figures. Pinned versions improve reproducibility, but different operating systems and numerical libraries may still produce small differences. Environment records and saved outputs provide the reference for comparison.

## Starter code and data

This project builds on the [official SSU Data Dash starter repository](https://github.com/erwinzhang7/SSU-Data-Dash) and its supplied synthetic data. See `LICENSE` for the code license.
