"""Write the Chinese handoff and an English report section from measured results."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "off_market"


def table(frame, formats=None):
    formats = formats or {}
    result = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    for _, row in frame.iterrows():
        result.append("| " + " | ".join(formats.get(c, str)(row[c]) for c in frame.columns) + " |")
    return "\n".join(result)


def main():
    summary = pd.read_csv(OUT / "summary.csv")
    selected = json.loads((OUT / "selection.json").read_text())["selected_model"]
    best = summary[summary.model == selected].iloc[0]
    baseline = summary[summary.model == "nearest_baseline"].iloc[0]
    delta = baseline.mdape-best.mdape
    relative = delta/baseline.mdape*100
    monthly = pd.read_csv(OUT / "monthly.csv")
    monthly = monthly[monthly.model.isin(["nearest_baseline", selected])].pivot(index="month", columns="model", values="mdape").reset_index()
    types = pd.read_csv(OUT / "by_property_type.csv")
    types = types[types.model == selected].sort_values("mdape", ascending=False)
    prices = pd.read_csv(OUT / "by_price.csv")
    prices = prices[prices.model == selected]
    expensive = prices[prices.price_band == "over_2m"].iloc[0]
    middle = prices[prices.price_band == "600k_to_1m"].iloc[0]
    coop = types[types.house_type_name == "Co-Op Apt"].iloc[0]
    base_score = summary[summary.model == "member1_boost"].iloc[0]
    history_score = summary[summary.model == "history_boost"].iloc[0]
    text_score = summary[summary.model == "history_text_boost"].iloc[0]
    timing = pd.read_csv(OUT / "timing.csv")
    features = int(timing[timing.model == selected].n_features.iloc[0])
    summary_table = table(summary[["model", "mdape", "mape", "median_signed_pct", "p90_ape"]],
        {c: lambda x: f"{x:.3f}" for c in ["mdape", "mape", "median_signed_pct", "p90_ape"]})
    monthly_table = table(monthly, {"nearest_baseline":lambda x:f"{x:.3f}", selected:lambda x:f"{x:.3f}"})
    type_table = table(types[["house_type_name", "n", "mdape", "mape"]], {"n":lambda x:str(int(x)), "mdape":lambda x:f"{x:.3f}", "mape":lambda x:f"{x:.3f}"})
    price_table = table(prices[["price_band", "n", "mdape", "mape", "over_20pct"]], {"n":lambda x:str(int(x)), "mdape":lambda x:f"{x:.3f}", "mape":lambda x:f"{x:.3f}", "over_20pct":lambda x:f"{x:.3f}"})
    narrative = (
        "以修正后的 Member 1 房屋特征为基础，加入房屋风格、行政区和社区的训练折内类别编码。"
        "历史方案增加挂牌日前90/365天的成交统计；文字方案再加入16个固定描述关键词指标。"
        "各方案预测成交价的自然对数，再取指数转换回加元。")
    text = f"""# Member 2：模型说明与结果

## 本次交付结论

最终选中 `{selected}`，共 {features} 项输入。官方2025-12至2026-05逐月验证，合计 {int(best.n):,} 套房屋：

- MdAPE：**{best.mdape:.3f}%**；原最近成交基线 {baseline.mdape:.3f}%。
- MdAPE 下降 **{delta:.3f} 个百分点**，相对下降 {relative:.1f}%。
- MAPE：**{best.mape:.3f}%**；原基线 {baseline.mape:.3f}%。
- 有符号百分比误差的中位数：{best.median_signed_pct:+.3f}%。
- 单套房绝对百分比误差的第90百分位：{best.p90_ape:.3f}%。这描述历史误差分布，不是未来单套房的保证。

## 模型在做什么

{narrative}

`starter_boost` 保留官方13项特征和默认梯度提升树参数；`member1_boost` 使用27项数值/指示特征加3项原生类别；`history_boost` 再加24项历史统计；`history_text_boost` 再加16项文字特征。`extra_trees` 用 Member 1 基础特征及训练折拟合的一热编码作另一类算法比较。

其中 `member1_boost`、`history_boost`、`history_text_boost` 使用相同固定参数：220轮、学习率0.08、最多31个叶节点、最小叶样本25、L2正则3、随机种子0、关闭提前停止。没有大规模调参。原官方方案的参数不同，因此与其比较衡量的是整个方案变化，而非纯粹的特征作用。

## 五个候选与基线

表内误差单位均为百分数。

{summary_table}

最终选择依据为所有验证房屋合并后的 MdAPE 最低，MAPE 打破平局，不根据测试集结果选择。

受控比较并不支持“加特征就一定更好”：仅加入历史成交统计后，MdAPE 从 {base_score.mdape:.3f}% 变为 {history_score.mdape:.3f}%；在相同历史方案中再加入固定文字指标后降至 {text_score.mdape:.3f}%。本轮没有单独测试“不含历史但含文字”的方案，不能断言历史统计是最终模型改进的原因。

## 逐月稳定性

{monthly_table}

## 按房型误差

{type_table}

这些是描述性分组结果。小样本房型的数值可能不稳定，不能只按某一组误差决定模型对所有房屋的价值。

## 按真实价格区间误差

{price_table}

以上价格分组使用验证房屋已知成交价，仅用于诊断，不是预测输入。

具体发现：超过200万加元房屋的 MdAPE 为 {expensive.mdape:.3f}%，有符号误差中位数 {expensive.median_signed_pct:+.3f}%，存在低估倾向；60万至100万加元组的 MdAPE 为 {middle.mdape:.3f}%。Co-Op Apt 的 MdAPE 为 {coop.mdape:.3f}%，但只有 {int(coop.n)} 套验证房屋，解释时应说明小样本限制。不能因总体误差较低而认为所有房屋都同样可靠。

## 验证、数据泄漏与解释边界

1. 每月只训练成交日期早于该月月初的记录，与官方时间验证一致。模型对新房屋只读取测试集所提供字段。
2. 不使用 `list_price`、`date_sold`、`days_on_market` 或外部AVM作为当前房屋输入。`sold_price` 仅作训练目标；历史特征仅取训练折中严格早于该房屋挂牌日的成交，且排除同日和自身。
3. 类别编码、历史库和 ExtraTrees 的缺失值填补均在每折内拟合。原始训练数据不因高价格而删行。
4. 官方协议按成交月构造训练/验证。对于挂牌较早的房屋，这不等同于完整模拟挂牌当日能训练到的模型；报告应明确这是官方月度协议。
5. 同一验证区间比较了五个候选，成绩是用于选择的验证成绩，不是独立最终测试。不能保证隐藏测试误差相同，也不把本轮结果当成比赛最终成绩。
6. `p90_ape` 和误差超过20%的比例揭示尾部风险，点预测不提供可靠的逐套房保证或经过独立校准的预测区间。

## 已完成的验证

- 自动测试：禁止字段污染不影响预测；未知类别和缺失面积可预测；预测不随批次顺序变化；历史统计排除同日、未来和自身；面积逗号和非法区间处理。
- 每折训练截止与预测输入检查；输出价格有限且为正。
- 基线参考文件的房屋ID、真实成交价和月份与当前训练集交叉核对。
- 最终 off-market 文件覆盖全部3,974套测试房屋，并通过官方格式检查。

## 与队友衔接

Member 3 的 list-aware 最佳模型在 `source/models.py` 中保留。共享 `features.py` 保持官方版本，Member 1 修正版独立放在 `member1_features.py`，避免影响 Member 3 的已有结果。

Member 4 应使用逐条验证预测做诊断，并将本节与 Member 1、Member 3 的结果整合。最终报告最多4页，因此本文件是素材，不应原样全部放入报告。
"""
    (ROOT / "docs" / "member2_results.md").write_text(text)
    english = f"""# Off-market model: report-ready draft

We compared five prespecified candidates using the official expanding-window evaluation for December 2025 through May 2026 ({int(best.n):,} homes). Each monthly model was fitted only on transactions completed before the first day of that month. Our selected candidate, `{selected}`, used {features} features. We compared the official gradient-boosted feature model, corrected Member 1 property/location features, additions of trailing sales statistics and fixed description keywords, and an ExtraTrees alternative. The enhanced boosting candidates shared fixed hyperparameters. We modeled log sold price and exponentiated predictions. We did not use list price, days on market, third-party AVMs or the current home's actual sale date as predictors. Categories and any imputation were learned separately within each training fold; historical features used only that fold's sales strictly before the query home's listing date, excluding same-day and self transactions.

The selected model achieved pooled validation MdAPE of **{best.mdape:.3f}%** and MAPE of **{best.mape:.3f}%**, versus **{baseline.mdape:.3f}%** and **{baseline.mape:.3f}%** for the supplied nearest-sales baseline. This is a MdAPE reduction of **{delta:.3f} percentage points ({relative:.1f}% relative)**. Median signed percentage error was **{best.median_signed_pct:+.3f}%**. The 90th percentile of absolute percentage error was **{best.p90_ape:.3f}%**, indicating that point estimates can still have substantial individual errors. See the accompanying monthly, property-type and price-band diagnostics; price bands are diagnostic only and are never inputs.

Error was larger for homes above CAD 2 million (MdAPE {expensive.mdape:.3f}%, median signed error {expensive.median_signed_pct:+.3f}%) than for CAD 600,000-1 million homes (MdAPE {middle.mdape:.3f}%). Co-Op apartments had MdAPE {coop.mdape:.3f}% but only {int(coop.n)} observations. Historical statistics alone did not improve the enhanced property model ({base_score.mdape:.3f}% to {history_score.mdape:.3f}%); adding fixed text indicators to the history candidate reduced error to {text_score.mdape:.3f}%. We did not separately test text without history, so we cannot attribute the final gain to history itself.

We selected the lowest pooled validation MdAPE, with MAPE as a tie-breaker. These same months were used for model comparison, so this is selection-set performance, not an independent final test; hidden-test performance may deteriorate. The supplied baseline predictions were audited and rescored rather than retrained. The monthly protocol follows the competition runner and is not a full listing-day backtest for long-running listings. Small property subgroups and shared monthly market shocks limit precision. We do not claim calibrated prediction intervals or that validation accuracy guarantees an individual home's value. No external data or hidden test outcomes were used.
"""
    (ROOT / "docs" / "member2_report_section.md").write_text(english)
    print(f"Wrote Chinese handoff and English report section for {selected}")


if __name__ == "__main__":
    main()
