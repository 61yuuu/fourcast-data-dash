# Code integration / 代码整合说明

## 来源与目录

本目录由实际上传的 Member 2 source 作为集成基础，保留其两轨 models.py 和选模配置；补入 Member 3 的全部 Python 实验和测试、starter handout/rubric、R baseline、quickstart notebook；补入 Member 1 原 EDA notebook、修正特征及更完整历史特征测试；补入 Member 4 evaluation.py 和文字报告素材。

原文件、ZIP、sources 均未修改。本目录 reference_results 保留三位成员原交付的实验结果；reference/source_manifest.json 记录整合前 Python 来源的 SHA256。原始有缺陷的 Member 1 features.py 单独隔离，不参与运行。

## 本次代码调整

1. 保留 off-market history_text_boost 和 list-aware current_v3；不重新根据本次成绩选择模型。
2. 保持共享 starter features.py，避免无意改变 Member 3 的 25 项输入；Member 1 增强特征独立导入。
3. Member 1 面积解析返回显式 float/np.nan，避免 nullable/object array 导致旧 pandas/sklearn 下的 ExtraTrees 缺失面积失败。
4. run.py validate 输出增加 community 和明确 model 名称，并保存实际环境 metadata。
5. 新增统一 `python run.py evaluate` 调用 Member 4；evaluate 不需要手工拼接另一个成员的 metadata。
6. 实验输出统一到 experiments/；原结果快照保持在 reference_results，不覆盖。
7. 根 requirements.txt 补齐 pyarrow、下载器、threadpoolctl；CatBoost 为 requirements-experiments.txt 中可选实验依赖，其缺失测试明确 skip。
8. 加入两轨 factory、缺失面积数值边界和评估冲突检查，并将 Member 4 加入已有小样本全流程测试。
9. 默认限制 OMP/BLAS 线程为 2，报告默认使用无窗口绘图后端。

## 测试与依赖

整合主测试：43 passed / 1 skipped；新增 Member 1 完整特征测试：6 passed。合计 49 passed / 1 skipped。跳过项仅为可选 CatBoost，非最终模型。

本机完整测试栈写在 requirements-tested.txt；Python 3.11、pandas 2.0.3、numpy 1.24.3、scikit-learn 1.3.0。根 requirements.txt 保留团队模型交付的主要版本 pandas 2.3.3 / numpy 2.2.6 / scikit-learn 1.7.2，并补齐运行必需依赖。尝试在独立临时环境安装时，当前可用 package index 未提供所需 numpy 版本，故未能运行 exact-version 环境，也没有改动用户全局软件。

requirements-tested.txt 对应这次实际执行的离线 DATA_DIR 路径；若从网络自动下载，还需安装 requirements.txt 中的 huggingface_hub。建议优先使用根 requirements.txt 的团队版本完成最终赛前复现；不要将本机旧版环境下重跑的成绩与原始交付成绩混写。

## 运行与证据

- test_results.log、member1_test_results.log：本机单元/集成测试日志。
- validation_run.log：完整官方数据六个月训练验证日志。
- pipeline_run.log：report/evaluate/predict/check 日志。
- outputs/*_metadata.json：实际环境和选中模型。
- docs/run_verification.json：本次验证汇总、与原结果差异及提交检查。

上面证据随本次执行生成。README 给出 fresh clone 的通用运行命令；绝对本机缓存目录不会成为代码默认值。

## 需要团队继续填写的内容

report/report.tex 仍是 starter 模板；docs 中是成员报告素材，尚未整合为完整四页 PDF。本次没有创建或发布 GitHub、视频或对外提交。月度 sale-month walk-forward、开发期重复选模、真实市场外推限制均需在最终报告保持明确。

## 本次完整官方数据执行结果

已完成 validate → report → evaluate，并全量训练 predict，最后 check 和 verify_delivery 均成功。每轨验证 10,069 条，最终提交 off-market 3,974 条、list-aware 4,064 条。

| Track | 本机重跑 MdAPE | 原交付 MdAPE |
|---|---:|---:|
| Off-market | 9.530471% | 9.474159% |
| List-aware | 4.321280% | 4.321280% |

Off-market 6,540 条验证预测与原交付不同，pooled MdAPE 差 0.056313 个百分点；list-aware 预测全部相同。当前运行依赖与原交付不同，且 Member 1 特征增加显式 float 边界处理；这里不把本次结果说成逐位复现，也未因差异重新选模或调参。新 submissions 与新 reports 均来自本次整合代码，原 CSV 仍在 reference_results 中。

若要重现本次本机结果，可用 Python 3.11 安装 requirements-tested.txt，并设置 DATA_DIR；若要核对原团队目标版本，则安装根 requirements.txt 再跑全流程。不能把原 9.474% 标签用于本次新预测。
