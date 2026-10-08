# Gujarat Potato Yield Modelling Using Weather and Machine Learning

District-wise AI/ML framework to model potato yield in the three main North Gujarat potato districts (**Banaskantha, Sabarkantha, Mehsana**) from week-wise rabi-season weather. The project integrates 29 rabi seasons of **NASA POWER** daily weather with district-level crop statistics from the Directorate of Horticulture, Govt. of Gujarat, engineers 95 weekly agro-meteorological features (plus one lag-1 yield anomaly), applies feature selection, and benchmarks 6 machine learning models using **Leave-One-Out Cross-Validation (LOOCV)**.

Academic project submitted to Dr. V. B. Vaidya.

---

## Why this project

- Gujarat produced about **48.59 lakh MT** of potato in 2024-25 and contributes about **6.62%** of India's potato production.
- Production is concentrated in three northern districts, so district-specific yield estimation matters for cold-storage planning, market stability and agricultural resource management.
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

- **Source:** NASA POWER (Prediction Of Worldwide Energy Resources), daily data from **1 Jan 1996 to 31 Dec 2025** (10,958 daily records per district, no missing values in the Mehsana file).
- **Files:** `Data/Weather_Data/` (one NASA POWER workbook per district).
- **Parameters used (5):**
  - **MAXT:** maximum temperature
  - **MINT:** minimum temperature
  - **MEAN_RH:** mean relative humidity
  - **WS:** wind speed at 10 m
  - **BSS:** bright sunshine hours
- **BSS derivation:** NASA POWER provides solar radiation, not sunshine hours. BSS was derived with the **Angstrom-Prescott relation**, `Rs = Ra [a + b (n/N)]`, rearranged as `n = N * ((Rs/Ra) - a) / b` and clipped to 0..N, using extraterrestrial radiation (Ra) and maximum possible sunshine hours (N) from standard 24 deg N latitude tables and coefficients `a = 0.25`, `b = 0.50`.
- **Rainfall is intentionally excluded.** The framework isolates atmospheric micro-climate drivers (temperature, humidity, wind, sunshine).

### Rabi-season climate of the three districts (SMW 42-8 mean)

| District | MAXT (deg C) | MINT (deg C) | RH (%) | WS (m/s) | BSS (h/day) |
|---|---|---|---|---|---|
| Banaskantha | 30.04 | 14.06 | 34.12 | 2.77 | 8.79 |
| Sabarkantha | 30.40 | 14.38 | 34.81 | 3.07 | 8.64 |
| Mehsana | 31.38 | 14.83 | 35.82 | 2.92 | 9.24 |

Interpolated district weather maps (inverse-distance weighting, power 2, between district centroids, clipped to district boundaries) are included in the project presentation for visualisation.

---

## Methodology

### 1. Week-wise temporal design (19 Standard Meteorological Weeks)

The rabi potato crop spans two calendar years, so the season is built across the year boundary: **SMW 42 to 52 of year 1 and SMW 1 to 8 of year 2**, giving **19 weeks**. Weeks are mapped to crop stages using IMD's SMW calendar and SDAU potato growth-stage guidance:

| SMW | Approx. period | Crop stage |
|---|---|---|
| 42-45 | Mid-Oct to early Nov | Planting and early vegetative growth |
| 46-52 | Nov to Dec | Canopy development and tuber initiation |
| 1-5 | January | Tuber bulking |
| 6-8 | February | Maturation and harvest |

### 2. Detrending: isolating the climate signal

District yield series carry a long-term upward trend from improved varieties, irrigation and management. The model is not asked to learn this trend. Instead:

`Observed Yield = Long-Term Trend + Yield Anomaly`

The **detrended yield anomaly** is the machine learning target, so the models learn weather-driven year-to-year variation only.

### 3. Feature engineering

- **5 weather parameters x 19 SMW = 95 weekly features**
- **+ 1 lag-1 yield anomaly** (previous crop year's anomaly, a proxy for carry-over effects)
- **= 96 candidate predictors** per crop year

### 4. Feature selection

With 96 candidates and only about 29 seasons per district, dimensionality reduction is essential. The following methods were applied and compared, reducing the set to roughly **3 to 10 predictors** per model:

- **Forward Selection**
- **Backward Elimination** (stepwise)
- **Recursive Feature Elimination (RFE)**
- **SelectKBest (F-test)**
- **Mutual Information**

### 5. Models benchmarked

Each district is modelled independently with all six algorithms, with no assumed winner:

- **Multiple Linear Regression** (stepwise)
- **Random Forest (RF)**
- **XGBoost**
- **Support Vector Regression (SVR)**
- **K-Nearest Neighbours (KNN)**
- **Artificial Neural Network (MLP)**

### 6. Validation

- **5-fold cross-validation** for hyperparameter tuning.
- **LOOCV** for out-of-sample evaluation, because crop-year data is limited and a conventional train/test split wastes history.
- Models ranked by out-of-sample **R2, RMSE and MAE**.

### 7. Reconstructing real yield

The selected model is retrained on the full dataset and its predicted anomaly is added back to the fitted long-term trend:

`Predicted Yield (MT/ha) = Predicted Detrended Anomaly + Fitted Long-Term Trend`

---

## Results

### Best model per district

| District | Best model | Feature selection | Features | R2 | RMSE (MT/ha) | MAE (MT/ha) |
|---|---|---|---|---|---|---|
| Sabarkantha | SVR | Mutual Information | 3 | **0.871** | 1.789 | 1.250 |
| Banaskantha | SVR | RFE | 10 | **0.732** | 1.083 | 0.937 |
| Mehsana | Multiple Linear Regression | SelectKBest (F-test) | 5 | **0.590** | 1.231 | 0.948 |

**No single algorithm or selection method wins everywhere.** Each district needs its own model and feature selection logic.

### Full LOOCV results

**Banaskantha**

| Model | R2 | RMSE | MAE | Selection |
|---|---|---|---|---|
| Stepwise Linear Regression | 0.714 | 1.119 | 0.922 | RFE |
| Random Forest | -0.007 | 2.099 | 1.644 | SelectKBest |
| XGBoost | 0.013 | 2.078 | 1.629 | SelectKBest |
| **SVR** | **0.732** | **1.083** | **0.937** | RFE |
| KNN | 0.230 | 1.836 | 1.461 | SelectKBest |
| ANN (MLP) | 0.490 | 1.494 | 1.280 | RFE |

**Sabarkantha**

| Model | R2 | RMSE | MAE | Selection |
|---|---|---|---|---|
| Stepwise Linear Regression | 0.714 | 2.669 | 2.109 | RFE |
| Random Forest | 0.785 | 2.314 | 1.641 | RFE |
| XGBoost | 0.784 | 2.318 | 1.869 | RFE |
| **SVR** | **0.871** | **1.789** | **1.250** | Mutual Information |
| KNN | 0.824 | 2.094 | 1.472 | Mutual Information |
| ANN (MLP) | 0.541 | 3.378 | 2.526 | Mutual Information |

**Mehsana**

| Model | R2 | RMSE | MAE | Selection |
|---|---|---|---|---|
| **Stepwise Linear Regression** | **0.590** | **1.231** | **0.948** | SelectKBest (F-test) |
| Random Forest | 0.390 | 1.506 | 1.120 | SelectKBest (F-test) |
| XGBoost | 0.460 | 1.421 | 1.094 | SelectKBest (F-test) |
| SVR | 0.410 | 1.483 | 1.258 | Forward Selection |
| KNN | 0.450 | 1.434 | 1.265 | Forward Selection |
| ANN (MLP) | 0.360 | 1.539 | 1.254 | RFE |

### Selected features of the best models (parameter - SMW)

- **Sabarkantha (SVR):** Tmax-48, Mean RH-49, Tmin-8
- **Banaskantha (SVR):** BSS-44, WS-44, WS-45, BSS-49, Tmin-50, Tmin-52, Tmax-5, BSS-5, Tmin-6, WS-8
- **Mehsana (MLR):** WS-49, WS-52, WS-1, Tmin-43, Lag-1 yield anomaly

### Weather-yield relationships (Pearson correlation with detrended yield)

Week-wise correlation heatmaps for each district are in `Outputs/`.

| District | Strongest positive | Strongest negative |
|---|---|---|
| Banaskantha | Wind speed, SMW 6 (r = +0.47) | MAXT, SMW 3 (r = -0.36) |
| Sabarkantha | MEAN_RH, SMW 44-46 (r = +0.30, +0.29) | MAXT, SMW 3 and 4 (r = -0.34, -0.33) |
| Mehsana | Wind speed, SMW 49 (r = +0.34) | BSS at SMW 49 and MINT at SMW 4 (r = -0.33 each) |

### Growth-stage sensitivity

Mapping the strongest weather associations back to crop stages shows a different dominant vulnerability in each district:

- **Sabarkantha: establishment stage.** Mean temperature around SMW 44 is the strongest negative association; humidity at SMW 43 is the strongest positive one.
- **Mehsana: tuber initiation.** Wind speed at SMW 49 is the strongest positive association.
- **Banaskantha: maturation.** Wind speed at SMW 6 is the strongest positive association, alongside negative associations with temperature in the same weeks.

### Key takeaways

- **District-specific modelling is necessary.** The best algorithm, feature-selection method and feature count all differ across districts.
- **Parsimonious models work best.** The top Sabarkantha model uses only 3 predictors out of 96.
- **Tree ensembles did not outperform SVR or linear models** on these small district series. Random Forest and XGBoost were near zero R2 for Banaskantha.
- **Week-wise features carry signal that seasonal averages would hide**, and different growth stages are sensitive to different parameters in each district.

---

## Repository structure

```
.
├── Code_Script/                          # District-wise modelling code
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

- **Small sample:** about 29 rabi seasons per district against 96 candidate predictors. Reported scores should be read as indicative of weather sensitivity patterns, and the models need validation on additional future seasons before operational use.
- **District-level aggregation:** yield is a district average, so within-district variation (variety, planting date, irrigation, soil) is not captured.
- **Gridded weather:** NASA POWER is satellite and reanalysis-derived data rather than ground-station observation, and **BSS is derived** from solar radiation through the Angstrom-Prescott relation rather than measured.
- **Rainfall excluded** by design, so the framework captures atmospheric drivers only.
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
