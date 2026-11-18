# Customer Churn Analysis – Data Quality Audit and Exploratory Data Analysis

> **Dataset substitution notice:** This project uses the public IBM Telco Customer Churn sample as an alternative because EdVyro resource files were unavailable. It is **not** the official EdVyro-provided dataset, and compliance with any unavailable EdVyro-specific instructions is not claimed.

## Dataset overview

- Source: IBM's `telco-customer-churn-on-icp4d` public GitHub repository and its `data/Telco-Customer-Churn.csv` file.
- Dataset URL: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Repository URL: https://github.com/IBM/telco-customer-churn-on-icp4d
- Downloaded file SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- Source dimensions: 7,043 rows × 21 columns.
- Cleaned dimensions: 7,043 rows × 21 columns.
- Row grain: one record per customer (`customerID`).
- Target field: `Churn` (`Yes`/`No`).
- Pull date: 2026-11-18.

The raw source file is retained unchanged under `data/raw/`; the cleaned file is `data/processed/customer_churn_cleaned.csv`.

## Data quality audit

### Missing values

Whitespace-only `TotalCharges` entries were treated as missing after stripping. Raw missing-value counts:

| Column | Missing rows |
| --- | ---: |
| TotalCharges | 11 |

There were 11 blank `TotalCharges` values; 11 occurred with `tenure = 0`, and 0 with positive tenure. These are imputed to `0.0` because the field is cumulative charges and no months of service had accrued. This is an explicit domain-based assumption, not a recovered source value.

### Duplicates and key checks

- Exact duplicate rows in the source: 0.
- Duplicate `customerID` values in the source: 0.
- `customerID` is treated as the row key; the cleaned ID is unique: True.

No duplicate rows were removed because none were present.

### Invalid and inconsistent values

- String fields were whitespace-trimmed; blank strings were converted to missing values.
- `TotalCharges` was stored as text and converted to numeric.
- Invalid values outside the expected categorical domains: 0.
- Invalid numeric-domain values (senior citizen outside 0/1, tenure outside 0–72, or negative charges): 0.
- `SeniorCitizen` and `tenure` are integer fields; `MonthlyCharges` and `TotalCharges` are floating-point numeric fields after cleaning.

### Unique-value review

There are 21 analyzed source columns. Category cardinalities are documented in `analysis_evidence.json`; small-domain values are listed there. Notable counts: `Contract` has 3 levels, `PaymentMethod` has 4 levels, and `Churn` has 2 levels.

### Outliers

The 1.5×IQR rule was used as a screening flag on tenure and charge measures. It is not an error detector. Flagged rows were retained because long tenure and cumulative charges are plausible observations; no row was removed solely for being statistically extreme.

| Measure | Q1 | Q3 | IQR lower fence | IQR upper fence | Rows flagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| tenure | 9.00 | 55.00 | -60.00 | 124.00 | 0 (0.0%) |
| MonthlyCharges | 35.50 | 89.85 | -46.02 | 171.38 | 0 (0.0%) |
| TotalCharges | 398.55 | 3786.60 | -4683.52 | 8868.67 | 0 (0.0%) |

## Cleaning decisions

1. Kept the downloaded raw CSV unchanged and wrote a separate processed CSV.
2. Trimmed surrounding whitespace in text fields.
3. Converted whitespace-only strings to missing values.
4. Converted `TotalCharges` from numeric text to a floating-point number.
5. Imputed the 11 blank `TotalCharges` values to `0.0` only after verifying that all corresponded to `tenure = 0`.
6. Kept every customer record; no duplicate or outlier records were removed.
7. Preserved the source's categorical labels and encoded `SeniorCitizen`/`tenure` as integers.

## Validation results

All validation checks passed:

| Check | Result |
| --- | --- |
| row count preserved except duplicates | PASS |
| unique customerID | PASS |
| no exact duplicate rows | PASS |
| no missing values | PASS |
| categorical values in expected domain | PASS |
| numeric domain checks | PASS |
| churn labels are binary | PASS |
| charges and tenure nonnegative | PASS |
| headline churn independent check | PASS |

## Exploratory data analysis

### Churn overview

- Customers: 7,043.
- Churned: 1,869 (26.54%).
- Retained: 5,174 (73.46%).

### Churn by contract

| Contract | Customers | Churned | Churn rate |
| --- | ---: | ---: | ---: |
| Month-to-month | 3,875 | 1,655 | 42.7% |
| One year | 1,473 | 166 | 11.3% |
| Two year | 1,695 | 48 | 2.8% |

### Churn by payment method

| Payment method | Customers | Churned | Churn rate |
| --- | ---: | ---: | ---: |
| Electronic check | 2,365 | 1,071 | 45.3% |
| Mailed check | 1,612 | 308 | 19.1% |
| Bank transfer (automatic) | 1,544 | 258 | 16.7% |
| Credit card (automatic) | 1,522 | 232 | 15.2% |

### Churn by tenure band

| Tenure (months) | Customers | Churned | Churn rate |
| --- | ---: | ---: | ---: |
| 0–12 | 2,186 | 1,037 | 47.4% |
| 13–24 | 1,024 | 294 | 28.7% |
| 25–36 | 832 | 180 | 21.6% |
| 37–48 | 762 | 145 | 19.0% |
| 49–60 | 832 | 120 | 14.4% |
| 61–72 | 1,407 | 93 | 6.6% |

### Numeric measures and monthly charges

| Measure | Count | Mean | Std. dev. | Min | Q1 | Median | Q3 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SeniorCitizen | 7,043 | 0.16 | 0.37 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| tenure | 7,043 | 32.37 | 24.56 | 0.00 | 9.00 | 29.00 | 55.00 | 72.00 |
| MonthlyCharges | 7,043 | 64.76 | 30.09 | 18.25 | 35.50 | 70.35 | 89.85 | 118.75 |
| TotalCharges | 7,043 | 2279.73 | 2266.79 | 0.00 | 398.55 | 1394.55 | 3786.60 | 8684.80 |

`SeniorCitizen` is binary (0/1); its numeric summary should not be read as a continuous measurement. For `MonthlyCharges`, the overall mean is 64.76 and median is 70.35 reported charge units.

| Churn status | Customers | Mean MonthlyCharges | Median | Q1 | Q3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained | 5,174 | 61.27 | 64.43 | 25.10 | 88.40 |
| Churned | 1,869 | 74.44 | 79.65 | 56.15 | 94.20 |

The source does not explicitly define a currency symbol in the file, so values are described as reported charge units.

### Correlation

Pearson correlations for tenure, MonthlyCharges, and TotalCharges:

| Measure | tenure | MonthlyCharges | TotalCharges |
| --- | ---: | ---: | ---: |
| tenure | 1.000 | 0.248 | 0.826 |
| MonthlyCharges | 0.248 | 1.000 | 0.651 |
| TotalCharges | 0.826 | 0.651 | 1.000 |

TotalCharges is cumulative and mechanically related to tenure and monthly charges; correlations are descriptive associations, not independent drivers or evidence of cause.

## Initial patterns and key findings

- The overall observed churn rate is 26.54% (1,869 of 7,043 customers).
- The highest observed contract churn rate is for **Month-to-month** at 42.71% (1,655 of 3,875); the lowest is **Two year** at 2.83% (48 of 1,695).
- The highest observed payment-method churn rate is for **Electronic check** at 45.29% (1,071 of 2,365).
- In this sample, the churned cohort has a higher mean MonthlyCharges (74.44) than the retained cohort (61.27); this association does not establish that charges caused churn.
- Churn rates vary across the displayed tenure bands; interpret band rates together with their customer counts in the table and dashboard.

## Limitations

- This is an alternative public sample, not the unavailable EdVyro dataset; fields, instructions, and outcomes may differ from the internship task's official resources.
- The source is a static sample with no observation period or extraction date encoded in the CSV. It does not support a time-trend or current-business performance claim.
- Results are descriptive. No significance testing, causal attribution, or predictive model is included.
- The 11 blank cumulative-charge values were set to zero based on zero tenure. Review this assumption if official billing semantics become available.
- IQR flags identify statistical extremes, not data errors.
- Charge units are retained as supplied; no currency is inferred.

## Conclusion

The cleaned sample passes the stated structural and domain checks after one documented, narrowly scoped `TotalCharges` imputation. Descriptive analysis shows variation in churn across contract, payment, and tenure groups, and higher mean monthly charges among customers who churned in this sample. These patterns can inform follow-up questions, but they should not be treated as causal conclusions or as EdVyro-specific results.
