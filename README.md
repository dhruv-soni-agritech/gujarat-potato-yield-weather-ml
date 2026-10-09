# Gujarat Potato Yield Modelling Using Weather and Machine Learning

District-wise AI/ML framework to model potato yield in two major North Gujarat potato districts (**Sabarkantha and Mehsana**) from week-wise rabi-season weather. The project integrates 29 rabi seasons of **NASA POWER** daily weather with district-level crop statistics from the Directorate of Horticulture, Govt. of Gujarat, engineers 95 weekly agro-meteorological features plus a **lag-1 yield anomaly** (96 candidate predictors), applies feature selection, and benchmarks 6 machine learning models using **nested Leave-One-Out Cross-Validation (LOOCV)**.

Academic project submitted to Dr. V. B. Vaidya.

---

## Why this project

- Gujarat produced about **48.59 lakh MT** of potato in 2024-25 and contributes about **6.62%** of India's potato production.
- Production is concentrated in the northern districts, so district-specific yield estimation matters for cold-storage planning, market stability and agricultural resource management.
- Yield depends not only on how much weather occurs but **when** it occurs. A seasonal average can hide a short cold or heat spell at a sensitive stage, so this project uses week-wise features aligned to crop growth stages instead of seasonal means.

---

## Study area and data

### Yield data (ground truth)

- **Source:** Directorate of Horticulture, Govt. of Gujarat, "Area & Production" yearly reports, 1994-95 to 2024-25 (31 years).
- **File:** `Data/Yield_Data/Gujarat_Potato_Data.xlsx` (with a flat CSV version, `Gujarat_Potato_Yield_Data.csv`).
- District-wise **area (ha), production (MT) and productivity (MT/ha)** for all Gujarat districts.
- Tables were extracted programmatically from the source PDFs; the 2024-25 report was a scanned image and was read manually and re-verified.
- District totals for every year were summed and **cross-checked against each report's printed total** (all 31 years matched).
- Where productivity was not printed, it was calculated as **Production / Area** and flagged in the `Productivity_Source` column.
- Zone grouping (South, Middle, North Gujarat, Saurashtra-Kutch) is applied consistently across all years, and district-name typos from the scans were standardised.
- Known anomalies in the government source (for example Kheda in 1996-97 and 2007-08) were kept as printed and documented in the workbook's Read Me sheet.

### Weather data

- **Source:** NASA POWER (Prediction Of Worldwide Energy Resources), daily data from **1 Jan 1996 to 31 Dec 2025** (10,958 daily records per district).
- **Files:** `Data/Weather_Data/` (one NASA POWER workbook per district).
- **Parameters used (5):**
  - **MAXT:** maximum temperature
  - **MINT:** minimum temperature
  - **MEAN_RH:** mean relative humidity
  - **WS:** wind speed at 10 m
  - **BSS:** bright sunshine hours
- **BSS derivation:** NASA POWER provides solar radiation, not sunshine hours. BSS was derived with the **Angstrom-Prescott relation**, `Rs = Ra [a + b (n/N)]`, rearranged as `n = N * ((Rs/Ra) - a) / b` and clipped to 0..N, using extraterrestrial radiation (Ra) and maximum possible sunshine hours (N) from standard 24 deg N latitude tables and coefficients `a = 0.25`, `b = 0.50`.
- **Rainfall is intentionally excluded.** The framework isolates atmospheric micro-climate drivers (temperature, humidity, wind, sunshine).

### Rabi-season climate of the two districts (SMW 42-8 mean)

| District | MAXT (deg C) | MINT (deg C) | RH (%) | WS (m/s) | BSS (h/day) |
|---|---|---|---|---|---|
| Sabarkantha | 30.40 | 14.38 | 34.81 | 3.07 | 8.64 |
| Mehsana | 31.38 | 14.83 | 35.82 | 2.92 | 9.24 |

Interpolated district weather maps (inverse-distance weighting, power 2, between district centroids, clipped to district boundaries) are included in the project presentation for visualisation.

---

## Methodology

The pipeline below is implemented in the Sabarkantha notebook (`Code_Script/`). Differences for Mehsana are noted where they apply.

### 1. Week-wise temporal design (19 Standard Meteorological Weeks)

The rabi potato crop spans two calendar years, so each season is built across the year boundary: **SMW 42 to 52 of year Y and SMW 1 to 8 of year Y+1**, giving **19 weeks**. Only seasons with all 19 weeks and a complete daily record are kept, which leaves **29 seasons (1996-97 to 2024-25)**. Weeks are mapped to five crop stages for the growth-stage analysis:

| Stage | SMW |
|---|---|
| Establishment and early vegetative growth | 42-45 |
| Vegetative growth and stolon formation | 46-48 |
| Tuber initiation | 49-51 |
| Tuber bulking | 52, 1-4 |
| Late bulking and maturation | 5-8 |

### 2. Detrending: isolating the climate signal

District yield series carry a long-term upward trend from improved varieties, irrigation and management. A **linear time trend** is fitted to productivity and removed:

`Observed Yield = Long-Term Trend + Yield Anomaly`

The **detrended yield anomaly** is the machine learning target, so the models are asked to learn weather-driven year-to-year variation only. For Sabarkantha the fitted trend is `Productivity = 0.5133 x time + 21.717` MT/ha with **R2 = 0.741**, meaning the trend alone accounts for about three quarters of the variance in yield (see the interpretation note under Results).

### 3. Feature engineering

- **5 weather parameters x 19 SMW = 95 weekly features** (weekly means of the daily values)
- **+ 1 lag-1 yield anomaly** (`Lag1_Yield_Anomaly`)
- **= 96 candidate predictors** per crop year

**Lag-1 yield anomaly.** For season Y this is the detrended yield anomaly of season Y-1, a proxy for carry-over (biological memory) effects such as seed quality and soil condition. It uses the anomaly rather than raw yield so the trend is not counted twice. For the first modelled season, the previous year's yield is taken from the yield record (which starts before the weather record) and detrended against the same trend line extended back one year, so no season is lost. The lag feature competes with the weather features during feature selection; it is not forced into any model.

### 4. Feature selection (nested inside every LOOCV fold)

Three selection techniques are evaluated separately for every model, each with a candidate feature count **K = 3 to 8**:

- **Mutual Information** (filter)
- **RFE** (wrapper, with an estimator matched to the model family where possible)
- **SelectKBest with the F-test** (filter)

For every LOOCV fold, **median imputation, feature selection and model fitting use only that fold's training seasons**, so the held-out season never influences which weeks are chosen. For each model, the (method, K) pair with the best nested-LOOCV R2 is retained. A full-data selection is computed once, only to display the final feature list and to refit the final model.

### 5. Models benchmarked

Each district is modelled independently with all six algorithms, with no assumed winner. In the Sabarkantha notebook the hyperparameters are fixed:

- **Linear Regression** (reported as stepwise linear regression: linear model on the selected features)
- **Random Forest (RF)**: 300 trees, depth 4
- **XGBoost**: 200 trees, depth 3, learning rate 0.05
- **Support Vector Regression (SVR)**: RBF kernel, C = 10, epsilon = 0.3
- **K-Nearest Neighbours (KNN)**: k = 5
- **Artificial Neural Network (MLP)**: one hidden layer of 8 units

### 6. Validation

- **Nested LOOCV** for out-of-sample evaluation, because crop-year data is limited and a conventional train/test split wastes history.
- Models are reported on two scales: **trend-restored** (anomaly prediction plus fitted trend, comparable to real yield) and **detrended residual** (weather skill alone).
- Metrics: **R2, RMSE and MAE**, plus a pseudo-AIC for exploratory comparison.

### 7. Reconstructing real yield

The selected model is retrained on the full dataset and its predicted anomaly is added back to the fitted long-term trend:

`Predicted Yield (MT/ha) = Predicted Detrended Anomaly + Fitted Long-Term Trend`

---

## Results

### Sabarkantha (current notebook with lag-1)

LOOCV results with feature selection nested inside each fold. R2, RMSE and MAE are on **trend-restored yield (MT/ha)**; the last column is the R2 on the **detrended residual**, which isolates what weather and lag-1 explain beyond the trend.

| Model | Feature selection | K | R2 | RMSE | MAE | Residual R2 |
|---|---|---|---|---|---|---|
| Stepwise Linear Regression | RFE | 4 | 0.674 | 2.848 | 2.297 | -0.259 |
| Random Forest | Mutual Information | 6 | 0.704 | 2.715 | 1.985 | -0.143 |
| XGBoost | Mutual Information | 8 | 0.681 | 2.820 | 2.125 | -0.234 |
| SVR | Mutual Information | 5 | 0.734 | 2.571 | **1.744** | -0.025 |
| **KNN** | Mutual Information | 5 | **0.749** | **2.502** | 1.827 | **0.029** |
| ANN (MLP) | SelectKBest | 8 | 0.273 | 4.254 | 3.343 | -1.807 |

**Best model: KNN with Mutual Information (5 features), R2 = 0.749, RMSE = 2.502 MT/ha, MAE = 1.827 MT/ha.** SVR is a close second and has the lowest MAE.

**How to read these numbers.** The long-term trend explains 74% of the yield variance by itself, so the trend-restored R2 values are driven largely by the trend. On the detrended residual, which is the part weather is supposed to explain, the best model reaches only R2 = 0.03 and most models are at or below zero. In other words, weather and lag-1 together add little predictive skill beyond the trend for Sabarkantha under leakage-free validation.

### Selected features (parameter - SMW)

| Model | Selected features |
|---|---|
| Stepwise Linear Regression | MAXT-50, MAXT-1, MINT-3, MINT-4 |
| Random Forest | Lag-1 anomaly, MAXT-48, MEAN_RH-49, MINT-8, MINT-48, BSS-44 |
| XGBoost | Lag-1 anomaly, MAXT-48, MEAN_RH-49, MINT-8, MINT-48, BSS-44, MAXT-43, MINT-45 |
| SVR | Lag-1 anomaly, MAXT-48, MEAN_RH-49, MINT-8, MINT-48 |
| **KNN** | Lag-1 anomaly, MAXT-48, MEAN_RH-49, MINT-8, MINT-48 |
| ANN (MLP) | MAXT-3, MAXT-4, BSS-43, WS-42, BSS-44, MEAN_RH-44, MINT-5, MEAN_RH-46 |

**Selection stability (share of LOOCV folds that independently chose each feature, KNN model):** Lag-1 anomaly 100%, MAXT-48 100%, MEAN_RH-49 90%, MINT-8 59%, MINT-48 31%. The lag-1 anomaly was chosen in every fold by Random Forest, XGBoost, SVR and KNN, making it the most consistently selected predictor. By contrast, the linear model's RFE-selected weeks were rarely re-selected inside the folds (3% to 34%), which signals an unstable choice.

### Effect of adding the lag-1 feature (Sabarkantha)

- The lag-1 anomaly is only weakly related to the current anomaly: **Pearson r = +0.27 (p = 0.16, n = 29)**, not statistically significant.
- It was frequently selected, but it did not raise trend-restored accuracy in this run. For example, SVR R2 was 0.793 in the earlier run without the lag feature and 0.734 with it; KNN was 0.758 and 0.749, and the linear model rose from 0.645 to 0.674.
- It is therefore reported as a tested, mentor-suggested predictor with no clear gain for Sabarkantha.

### Weather-yield relationships (Pearson correlation with detrended yield)

**Sabarkantha (current notebook).** No seasonal, weekly or growth-stage association is statistically significant after Benjamini-Hochberg FDR correction (smallest raw p = 0.068).

| Level | Strongest associations |
|---|---|
| Seasonal means | MINT r = -0.28, MAXT r = -0.24, MEAN_RH r = +0.14, BSS r = -0.13, WS r = +0.05 (all not significant) |
| Weekly, negative | MAXT SMW 3 (r = -0.34), MAXT SMW 4 (r = -0.33), BSS SMW 43 (r = -0.33), WS SMW 42 (r = -0.31) |
| Weekly, positive | MEAN_RH SMW 44 (r = +0.30), MEAN_RH SMW 46 (r = +0.29) |

Weeks with the highest average association strength are SMW 44, 3, 43 and 4 (early establishment and mid-January).

**Mehsana (from the project presentation).**

| Strongest positive | Strongest negative |
|---|---|
| Wind speed, SMW 49 (r = +0.34) | BSS at SMW 49 and MINT at SMW 4 (r = -0.33 each) |

Week-wise correlation heatmaps for each district are in `Outputs/`.

### Growth-stage sensitivity

**Sabarkantha (current notebook).** Mean absolute correlation per stage: establishment and early vegetative **0.198**, tuber bulking 0.144, late bulking and maturation 0.134, vegetative and stolon formation 0.082, tuber initiation 0.016. The early establishment weeks (SMW 42-45) and the bulking and maturation weeks (January to February) are the most weather-sensitive, while tuber initiation (SMW 49-51) shows almost no association. The strongest single stage-level associations are MINT during late bulking and maturation (r = -0.29), MAXT during tuber bulking (r = -0.29) and MEAN_RH during establishment (r = +0.25). None are significant after FDR correction, so these are exploratory patterns.

**Mehsana (from the project presentation).** Tuber initiation is the sensitive stage, with wind speed at SMW 49 the strongest positive association.

### Mehsana model results (from the project presentation)

| Model | R2 | RMSE | MAE | Selection |
|---|---|---|---|---|
| **Stepwise Linear Regression** | **0.590** | **1.231** | **0.948** | SelectKBest (F-test) |
| Random Forest | 0.390 | 1.506 | 1.120 | SelectKBest (F-test) |
| XGBoost | 0.460 | 1.421 | 1.094 | SelectKBest (F-test) |
| SVR | 0.410 | 1.483 | 1.258 | Forward Selection |
| KNN | 0.450 | 1.434 | 1.265 | Forward Selection |
| ANN (MLP) | 0.360 | 1.539 | 1.254 | RFE |

Best Mehsana model: Multiple Linear Regression with 5 features (WS-49, WS-52, WS-1, Tmin-43, Lag-1 yield anomaly).

### Best model per district

| District | Best model | Feature selection | Features | R2 | RMSE (MT/ha) | MAE (MT/ha) |
|---|---|---|---|---|---|---|
| Sabarkantha | KNN | Mutual Information | 5 | **0.749** | 2.502 | 1.827 |
| Mehsana | Multiple Linear Regression | SelectKBest (F-test) | 5 | **0.590** | 1.231 | 0.948 |

### Key takeaways

- **District-specific modelling is necessary.** The best algorithm and feature-selection setup differ between the two districts.
- **The trend dominates.** For Sabarkantha the linear time trend explains 74% of yield variance, and under leakage-free nested LOOCV the weather-plus-lag signal on the detrended residual is close to zero (best residual R2 = 0.03).
- **The lag-1 anomaly is the most consistently selected predictor** for four of the six Sabarkantha models, but it is weakly correlated with the current anomaly (r = +0.27, not significant) and did not improve trend-restored accuracy.
- **Parsimonious models work best.** The top Sabarkantha models use 5 of the 96 candidate predictors.
- **Tree ensembles and the neural network did not outperform KNN, SVR or linear models** on these small district series.
- **Week-wise features show where sensitivity concentrates** (early establishment and January to February for Sabarkantha), although no single weather-week association is statistically significant after multiple-testing correction.

---

## How to run (Sabarkantha notebook)

1. Install the dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `xgboost`, `statsmodels`, `openpyxl`.
2. Place `Gujarat_Potato_Yield_Data.csv` and `Sabarkantha_Nasa_power.xlsx` in the same folder as the notebook (the notebook also searches `/content` for Colab).
3. Run all cells. The nested LOOCV step runs 29 folds x 6 models x 3 methods x 6 K values and takes roughly 15 minutes.
4. Results are written to `Final_Outputs/Charts` and `Final_Outputs/Tables` (model comparison, selected features, correlation tables, growth-stage summary, predicted-versus-actual and residual plots).

---

## Repository structure

```
.
├── Code_Script/                          # District-wise modelling notebooks
├── Data/
│   ├── Weather_Data/                     # NASA POWER daily weather (1996-2025)
│   └── Yield_Data/                       # District-wise potato area, production, productivity
├── Outputs/                              # Correlation matrices and heatmaps
├── AI,ML_based_potato_yield_modelling.pptx   # Full project presentation
├── LICENSE
└── README.md
```

---

## Limitations

- **Small sample:** 29 rabi seasons per district against 96 candidate predictors. Results are indicative of weather sensitivity patterns, and the models need validation on additional future seasons before operational use.
- **Weak weather skill beyond the trend:** the headline R2 is largely the long-term trend. Detrended-residual R2 is near zero for Sabarkantha, and no weather-week correlation survives FDR correction.
- **Mild optimism from model search:** feature selection is nested, but choosing the best (method, K) per model from the nested scores still gives a slightly optimistic R2.
- **Unstable linear-model selection:** the weeks chosen by RFE for the linear model were rarely re-selected inside the folds.
- **Lag-1 construction:** the first season's lag uses the trend extended back one year, and lag features in LOOCV are not strictly causal because neighbouring seasons are shared between training and test folds.
- **District-level aggregation:** yield is a district average, so within-district variation (variety, planting date, irrigation, soil) is not captured.
- **Gridded weather:** NASA POWER is satellite and reanalysis-derived data rather than ground-station observation, and **BSS is derived** from solar radiation through the Angstrom-Prescott relation rather than measured.
- **Rainfall excluded** by design, so the framework captures atmospheric drivers only.
- **Mehsana results** are taken from the project presentation and have not been re-run with the nested Sabarkantha pipeline.
- **Government statistics** include some anomalous values in the original reports, which were retained as published.

---

## Data sources and references

- NASA POWER, Prediction Of Worldwide Energy Resources (power.larc.nasa.gov)
- Directorate of Horticulture, Govt. of Gujarat; Horticultural Statistics at a Glance, MoA&FW, Govt. of India
- India Meteorological Department (IMD), Standard Meteorological Week calendar
- Guyon and Elisseeff (2003), *An Introduction to Variable and Feature Selection*, JMLR 3:1157-1182
- Breiman (2001), Random Forests; Chen and Guestrin (2016), XGBoost; Drucker et al. (1997), Support Vector Regression
- Hastie, Tibshirani and Friedman (2009), *The Elements of Statistical Learning*
- Allen et al. (1998), FAO Irrigation and Drainage Paper 56; Haverkort (1990)

---

## Team and acknowledgements

Project team: Sibbala Yoshitha, Vodnala Likitha, Swathi Patil, Yash Patel, Tushar Vadodariya and Dhruv Soni.

Submitted to Dr. V. B. Vaidya. Sincere thanks to Dr. V. B. Vaidya and Dr. Manoj Lunagaria for their guidance and support throughout the project.

## License

Released under the MIT License. See `LICENSE`.
