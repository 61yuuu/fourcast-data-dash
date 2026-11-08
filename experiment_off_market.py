"""Fixed five-candidate comparison on official six monthly folds.

Run from any directory with DATA_DIR pointing at the three supplied CSV files.
The imported baseline is an audited reference, not a newly trained candidate.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import data
from off_market_model import CANDIDATES, OffMarketModel, TEXT_PATTERNS
from check_submission import check_file

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE
OUT = ROOT / "experiments" / "off_market"
MONTHS = pd.period_range("2025-12", "2026-05", freq="M")


def score(g):
    signed = 100 * (g.predicted_price - g.sold_price) / g.sold_price
    absolute = signed.abs()
    return dict(n=len(g), mdape=float(absolute.median()), mape=float(absolute.mean()),
                median_signed_pct=float(signed.median()), mae_cad=float((g.predicted_price-g.sold_price).abs().mean()),
                p90_ape=float(absolute.quantile(.9)), over_20pct=float((absolute > 20).mean()*100))


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_diagnostics(predictions):
    for keys, filename in [(["model"], "summary"), (["model", "month"], "monthly"),
                           (["model", "house_type_name"], "by_property_type"),
                           (["model", "community"], "by_community")]:
        reports = []
        for labels, g in predictions.groupby(keys, sort=True):
            if not isinstance(labels, tuple):
                labels = (labels,)
            reports.append(dict(zip(keys, labels), **score(g)))
        table = pd.DataFrame(reports)
        if filename == "summary":
            table = table.sort_values(["mdape", "mape"])
        table.to_csv(OUT / f"{filename}.csv", index=False)
    prices = predictions.assign(price_band=pd.cut(predictions.sold_price,
        [0, 600000, 1000000, 2000000, np.inf], labels=["under_600k", "600k_to_1m", "1m_to_2m", "over_2m"]))
    rows = [dict(model=k, price_band=str(b), **score(g)) for (k,b),g in
            prices.groupby(["model", "price_band"], observed=True)]
    pd.DataFrame(rows).to_csv(OUT / "by_price.csv", index=False)


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = pd.read_csv(OUT / "summary.csv").sort_values("mdape", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(summary.model, summary.mdape, color=["#b7c4d4" if k == "nearest_baseline" else "#276c86" for k in summary.model])
    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    ax.set(xlabel="Median absolute percentage error (%)", title="Off-market: official walk-forward validation", xlim=(0, summary.mdape.max()*1.18))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "comparison.png", dpi=180); plt.close(fig)
    monthly = pd.read_csv(OUT / "monthly.csv")
    selected = json.loads((OUT / "selection.json").read_text())["selected_model"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for kind in ("nearest_baseline", "starter_boost", selected):
        g = monthly[monthly.model == kind]
        axes[0].plot(g.month, g.mdape, marker="o", label=kind)
    axes[0].set(title="Error by sale month", ylabel="MdAPE (%)")
    axes[0].tick_params(axis="x", rotation=35); axes[0].legend(fontsize=8)
    types = pd.read_csv(OUT / "by_property_type.csv")
    types = types[types.model == selected].sort_values("mdape")
    bars = axes[1].barh(types.house_type_name, types.mdape, color="#276c86")
    axes[1].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[1].set(title="Selected model: error by property type", xlabel="MdAPE (%)", xlim=(0,types.mdape.max()*1.25))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "diagnostics.png", dpi=180); plt.close(fig)


def main():
    if "DATA_DIR" not in os.environ:
        default = ROOT.parent
        if not (default / "listings.csv").exists():
            raise RuntimeError("Set DATA_DIR to the folder containing the three official data files")
        os.environ["DATA_DIR"] = str(default)
    OUT.mkdir(parents=True, exist_ok=True)
    listings = data.load("listings")
    baseline = pd.read_csv(ROOT / "reference" / "starter_validation_predictions.csv")
    baseline = baseline[baseline.track == "off_market"].copy()
    truth = listings[listings.date_sold.dt.to_period("M").isin(MONTHS)]
    assert len(baseline) == len(truth) and baseline.ml_num.is_unique
    check = baseline.merge(truth[["ml_num", "sold_price", "date_sold", "community"]], on="ml_num", validate="one_to_one")
    assert len(check) == len(truth)
    assert np.array_equal(check.sold_price_x, check.sold_price_y)
    assert np.array_equal(check.month, check.date_sold.dt.to_period("M").astype(str))
    baseline = baseline.merge(truth[["ml_num", "community"]], on="ml_num", validate="one_to_one")
    baseline["model"] = "nearest_baseline"
    columns = ["ml_num", "month", "house_type_name", "community", "sold_price", "predicted_price", "model"]
    all_results = [baseline[columns]]
    timing = []
    manifest = {"validation_months": [str(m) for m in MONTHS], "candidates": list(CANDIDATES),
        "python": platform.python_version(), "random_seed": 0,
        "data_sha256": {p.name: file_hash(p) for p in [data.find_file("listings"), data.find_file("off_market")]},
        "source_sha256": {p.name: file_hash(p) for p in sorted(SOURCE.glob("*.py"))},
        "text_patterns": TEXT_PATTERNS, "baseline": "Imported and audited starter predictions; not rerun",
        "protocol": "Train on date_sold before validation sale month; history windows end strictly before each date_listed"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for month in MONTHS:
        train = listings[listings.date_sold < month.start_time]
        homes = listings[listings.date_sold.dt.to_period("M") == month]
        assert train.date_sold.max() < month.start_time
        inputs = data.hide_answers(homes, "off_market")
        assert "list_price" not in inputs and "sold_price" not in inputs
        for kind in CANDIDATES:
            start = time.perf_counter()
            with threadpool_limits(limits=2):
                model = OffMarketModel(kind).fit(train)
                fit_seconds = time.perf_counter()-start
                start = time.perf_counter()
                prices = np.round(model.predict(inputs))
            result = homes[["ml_num", "house_type_name", "community", "sold_price"]].copy()
            result["month"] = str(month); result["model"] = kind; result["predicted_price"] = prices
            result = result[columns]
            result.to_csv(OUT / f"{kind}_{month}.csv", index=False)
            all_results.append(result)
            timing.append(dict(month=str(month), model=kind, n_train=len(train), n_features=len(model.feature_names_),
                               fit_seconds=fit_seconds, predict_seconds=time.perf_counter()-start))
            pd.DataFrame(timing).to_csv(OUT / "timing.csv", index=False)
            print(f"{month} {kind}: MdAPE={score(result)['mdape']:.3f}%, features={len(model.feature_names_)}, fit={fit_seconds:.1f}s", flush=True)
    predictions = pd.concat(all_results, ignore_index=True)
    predictions.to_csv(OUT / "validation_predictions.csv", index=False)
    write_diagnostics(predictions)
    summary = pd.read_csv(OUT / "summary.csv")
    candidates = summary[summary.model.isin(CANDIDATES)]
    winner = str(candidates.iloc[0].model)
    selection = {"selected_model": winner, "rule": "Lowest pooled MdAPE; MAPE breaks ties among the five predefined candidates",
                 "warning": "Same validation months used for selection. No independent final holdout; no guarantee on hidden test."}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2))
    (SOURCE / "selected_off_market.json").write_text(json.dumps({"variant": winner}, indent=2))
    with threadpool_limits(limits=2):
        final = OffMarketModel(winner).fit(listings)
        test = data.load("off_market")
        submission = pd.DataFrame({"ml_num": test.ml_num, "predicted_price": np.round(final.predict(test))})
    submission.to_csv(OUT / "submission_off_market.csv", index=False)
    errors = check_file(OUT / "submission_off_market.csv", test.ml_num)
    assert not errors, errors
    plots()
    print(summary.to_string(index=False), flush=True)
    print(f"Selected {winner}; checked {len(test)} test predictions.", flush=True)


if __name__ == "__main__":
    main()
