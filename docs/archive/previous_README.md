# Data Dash — integrated team code

四位成员代码的统一运行版本。原附件未修改；所有变更仅在本目录。

## 最快开始

推荐 Python 3.11 或 3.12，在本目录打开终端：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python run.py validate
python run.py report
python run.py evaluate
python run.py predict
python run.py check
python verify_delivery.py
```

Windows 激活方式为 `.venv\Scripts\activate`。首次运行默认由 Hugging Face 下载官方 synthetic 数据；已有文件时设置 `DATA_DIR` 为包含 `listings.parquet`、`test_off_market.parquet`、`test_list_aware.parquet` 的目录（也支持 CSV），然后运行相同命令。原始数据不包含在 ZIP 中。

## 选中模型与分工

| 成员 | 代码 | 最终用途 |
|---|---|---|
| Member 1 | `notebooks/eda.ipynb`、`member1_features.py`、`check_member1_features.py` | EDA、清洗、面积/房屋/历史成交特征 |
| Member 2 | `off_market_model.py`、`experiment_off_market.py` | off-market 最终 `history_text_boost`，70 项特征，log price HGB |
| Member 3 | `list_aware_feature_models.py`、`list_aware_member1_model.py`、`list_aware_models.py`、实验脚本 | list-aware 最终 `current_v3`，25 项特征，sold/list ratio HGB |
| Member 4 | `evaluation.py` | 两轨预测校验、价格/房型/月份/社区误差分析 |
| 团队入口 | `models.py`、`run.py`、`data.py`、`check_submission.py` | 统一验证、报告、预测、检查 |

`features.py` 保持 starter 接口，用于复现 Member 3 已选中的输入；Member 1 增强特征在独立 `member1_features.py`。不要直接用原始 Member 1 features.py 覆盖它。`selected_off_market.json` 已固定选中方案，日常运行不需要先重跑所有实验。

## 输出

- `outputs/validation_predictions.csv`：两轨逐条验证预测，含 model、community，可直接给 Member 4 分析。
- `outputs/validation_metadata.json`、`prediction_metadata.json`：实际 Python/包版本和模型信息。
- `reports/validation_report.md/json`、`results_table.tex`、`figures/error_by_price_range.png`：团队整体评估。
- `reports/diagnostics/`：Member 4 六张结果表、三张图和文字分析。
- `submissions/submission_off_market.csv`、`submission_list_aware.csv`：最终两轨预测。
- `reference_results/member2/`、`member3/`、`member4/`：原交付的结果快照，供核对，重跑不会覆盖它们。
- `docs/`：成员报告素材、整合说明和本次执行结果。
- `reference/member1_original/features.py`：只读参考用途的原版备份，有已知历史缩进问题，不参与模型导入。

本目录中的 reports/submissions 会被相应命令重新生成；原始团队交付结果保留在 reference_results。代码、预测与实际环境是否复现原始成绩，见 `docs/INTEGRATION.md`。

## 重跑候选实验（可选）

```bash
python experiment_off_market.py
python build_report.py
python experiment_list_aware_features.py
python experiment_list_aware_member1.py
```

以上输出分别到 `experiments/off_market/`、`experiments/list_aware_v3/`、`experiments/list_aware_v4/`；off-market 实验会更新 `selected_off_market.json`，之后须重新 validate/predict/report/evaluate。off-market 实验要求显式设置 DATA_DIR，并使用 `reference/starter_validation_predictions.csv` 作为已审计 baseline。

最早的全算法比较包含可选 CatBoost：

```bash
python -m pip install -r requirements-experiments.txt
python experiment_list_aware.py
```

结果到 `experiments/list_aware_v2/`。未安装 CatBoost 时只跳过对应可选测试；两个最终模型不依赖 CatBoost。`compare_list_aware.py` 仅供最早 FeatureModel 比较参考，当前选中模型下不应运行。R baseline 和 quickstart notebook 是 starter 教学材料，不是最终模型入口。

## EDA

从本目录启动 Jupyter，再打开 `notebooks/eda.ipynb`，使 working directory 能找到 data.py。Notebook 保留原分析及保存输出，整合时未重新执行；它不会自动重训练最终模型。

## 解释边界与提交

验证采用 December 2025–May 2026 的 monthly sale-month walk-forward，而非严格 listing-day backtest。同一期间参与过开发与选模，因此是 development validation；不能当作 untouched test 或保证 hidden-test 表现。没有加入外部真实房产记录或 AVM。

`report/report.tex` 仍是待团队填写的模板；`docs/` 为素材。最终 PDF、全员姓名、视频、GitHub 链接及其权限需要团队另行完成。本次交付范围是代码整合和运行验证，没有发布 repository 或发送任何材料。
