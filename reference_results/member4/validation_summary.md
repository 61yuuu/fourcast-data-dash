# Validation summary

Source: team_predictions.csv
Month range: 2025-12 to 2026-05
Rows: 20,138; unique homes: 10,069

## Overall results

- current_v3 / list_aware: n=10,069, MdAPE=4.32%, MAPE=5.40%, median signed error=+0.73%.
- history_text_boost / off_market: n=10,069, MdAPE=9.47%, MAPE=12.11%, median signed error=+0.06%.

## Highest median absolute error by house_type_name

- current_v3 / list_aware: Freehold Townhouse, n=405, MdAPE=5.81%, median signed error=+1.27%.
- history_text_boost / off_market: Co-Op Apt, n=48, MdAPE=14.59%, median signed error=+7.43%.

## Highest median absolute error by price_decile

- current_v3 / list_aware: 9, n=1,007, MdAPE=5.55%, median signed error=-0.50%.
- history_text_boost / off_market: 10, n=1,007, MdAPE=14.27%, median signed error=-9.03%.

## Highest median absolute error by month

- current_v3 / list_aware: 2026-05, n=2,291, MdAPE=4.68%, median signed error=+1.09%.
- history_text_boost / off_market: 2025-12, n=1,238, MdAPE=9.94%, median signed error=+3.90%.

## Community diagnostics

Ranking includes communities with at least 30 homes per model/track.
Rows missing community (excluded from this analysis): 0.
Community metadata source: member3_selected.csv (joined by ml_num).

### list_aware / current_v3

145 communities; 132 meet the sample-size threshold.

- Sycamorehaven: n=31, MdAPE=9.22%, MAPE=8.47%, median signed error=+2.09%.
- Sumacworth: n=45, MdAPE=6.09%, MAPE=6.83%, median signed error=+0.20%.
- Hickoryvale: n=52, MdAPE=6.07%, MAPE=6.07%, median signed error=-1.41%.
- Magnoliabank: n=43, MdAPE=6.06%, MAPE=6.85%, median signed error=-0.02%.
- Maplegrove: n=54, MdAPE=6.04%, MAPE=6.69%, median signed error=+0.91%.
- Elmbourne: n=36, MdAPE=5.98%, MAPE=6.68%, median signed error=-1.08%.
- Maplevale: n=93, MdAPE=5.98%, MAPE=6.94%, median signed error=+2.10%.
- Sycamorewood: n=34, MdAPE=5.80%, MAPE=6.51%, median signed error=+3.67%.
- Lindencliffe: n=37, MdAPE=5.70%, MAPE=7.01%, median signed error=+1.09%.
- Mapleview: n=76, MdAPE=5.63%, MAPE=6.84%, median signed error=+2.25%.

### off_market / history_text_boost

145 communities; 132 meet the sample-size threshold.

- Lindenglen: n=62, MdAPE=17.50%, MAPE=19.61%, median signed error=+7.43%.
- Cypressglen: n=54, MdAPE=15.51%, MAPE=19.69%, median signed error=+3.96%.
- Maplepark: n=36, MdAPE=15.46%, MAPE=21.12%, median signed error=-2.77%.
- Hazelfield: n=68, MdAPE=15.39%, MAPE=17.42%, median signed error=-6.45%.
- Sumacfield: n=52, MdAPE=15.32%, MAPE=16.54%, median signed error=+3.89%.
- Lindencliffe: n=37, MdAPE=14.95%, MAPE=18.05%, median signed error=+3.56%.
- Elmview: n=184, MdAPE=14.86%, MAPE=18.92%, median signed error=+0.25%.
- Juniperheights: n=93, MdAPE=14.55%, MAPE=17.91%, median signed error=+0.20%.
- Sycamorewood: n=34, MdAPE=14.40%, MAPE=14.31%, median signed error=+2.21%.
- Sumacbank: n=54, MdAPE=12.49%, MAPE=14.78%, median signed error=-1.08%.

## Interpretation and limits

- Positive signed error means overprediction; negative means underprediction.
- Q25 and Q75 describe the middle 50% of absolute errors, not confidence intervals.
- Price groups use observed outcomes for diagnosis only. Never use them as input features.
- Check subgroup sample sizes. Small groups can have unstable error summaries.
- Community names are used as supplied; confirm they uniquely identify your intended geographic groups.
- The CSV cannot prove temporal separation or absence of leakage. Review training and feature code.
- Matching homes do not prove matching training cutoffs. Confirm the same validation protocol with teammates.
- Repeated tuning on these months makes them development results, not an untouched final test.
- Synthetic-data performance does not establish accuracy on real Toronto transactions.
- No confidence intervals are produced by this script. Transactions may share monthly market effects.
