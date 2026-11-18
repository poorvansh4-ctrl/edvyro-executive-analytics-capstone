"""Reproducible quality audit and descriptive EDA for the IBM Telco sample."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.name == "analysis" and (SCRIPT_DIR.parent / "customer-churn-analysis").exists():
    ROOT = SCRIPT_DIR.parent / "customer-churn-analysis"
elif SCRIPT_DIR.name == "scripts":
    ROOT = SCRIPT_DIR.parent
else:
    ROOT = SCRIPT_DIR
RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
OUT = (
    SCRIPT_DIR
    if SCRIPT_DIR.name == "analysis" and (SCRIPT_DIR.parent / "customer-churn-analysis").exists()
    else ROOT / "analysis"
)
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
VISUALS = ROOT / "visuals"
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
REPOSITORY_URL = "https://github.com/IBM/telco-customer-churn-on-icp4d"

for folder in (OUT, PROCESSED, REPORTS, VISUALS):
    folder.mkdir(parents=True, exist_ok=True)

raw_bytes = RAW.read_bytes()
raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
raw = pd.read_csv(RAW, dtype={"customerID": "string"}, keep_default_na=False)
source_rows, source_columns = raw.shape

# Profile the source without modifying it. Whitespace-only strings count as missing.
raw_text_cols = [
    col for col in raw.columns if pd.api.types.is_string_dtype(raw[col].dtype)
]
raw_blank_counts = {
    col: int(raw[col].astype(str).str.strip().eq("").sum()) for col in raw_text_cols
}
raw_missing_counts = {
    col: int(raw[col].isna().sum()) + raw_blank_counts.get(col, 0)
    for col in raw.columns
}
duplicate_rows = int(raw.duplicated().sum())
duplicate_customer_ids = int(raw["customerID"].duplicated().sum())

expected_categories = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}

# Build a cleaned copy; the source CSV is kept byte-for-byte unchanged.
clean = raw.copy()
for col in clean.columns:
    if not pd.api.types.is_string_dtype(clean[col].dtype):
        continue
    clean[col] = clean[col].astype("string").str.strip()
    clean[col] = clean[col].replace("", pd.NA)

clean = clean.drop_duplicates().copy()
clean["SeniorCitizen"] = pd.to_numeric(clean["SeniorCitizen"], errors="coerce")
clean["tenure"] = pd.to_numeric(clean["tenure"], errors="coerce")
clean["MonthlyCharges"] = pd.to_numeric(clean["MonthlyCharges"], errors="coerce")
clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce")

# The only blank TotalCharges entries are zero-tenure customers. No billing months
# have accrued, so a value of 0 is consistent with the dataset's cumulative field.
blank_total_charges = int(clean["TotalCharges"].isna().sum())
blank_total_with_zero_tenure = int(
    (clean["TotalCharges"].isna() & clean["tenure"].eq(0)).sum()
)
blank_total_with_positive_tenure = int(
    (clean["TotalCharges"].isna() & clean["tenure"].gt(0)).sum()
)
if blank_total_with_positive_tenure:
    raise ValueError("Blank TotalCharges found for positive-tenure customers.")
clean.loc[clean["TotalCharges"].isna() & clean["tenure"].eq(0), "TotalCharges"] = 0.0

clean["SeniorCitizen"] = clean["SeniorCitizen"].astype("int64")
clean["tenure"] = clean["tenure"].astype("int64")
clean["MonthlyCharges"] = clean["MonthlyCharges"].astype("float64")
clean["TotalCharges"] = clean["TotalCharges"].astype("float64")

invalid_categories = {}
for col, allowed in expected_categories.items():
    found = set(clean[col].dropna().astype(str).unique())
    invalid_categories[col] = sorted(found - allowed)
invalid_numeric = {
    "SeniorCitizen_not_0_or_1": int((~clean["SeniorCitizen"].isin([0, 1])).sum()),
    "tenure_outside_0_to_72": int((~clean["tenure"].between(0, 72)).sum()),
    "negative_MonthlyCharges": int((clean["MonthlyCharges"] < 0).sum()),
    "negative_TotalCharges": int((clean["TotalCharges"] < 0).sum()),
}

unique_values = {}
for col in clean.columns:
    values = clean[col].dropna().unique().tolist()
    unique_values[col] = {
        "count": int(len(values)),
        "values": sorted([str(v) for v in values]) if len(values) <= 20 else None,
    }

def iqr_outlier_summary(series: pd.Series) -> dict:
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    count = int(((series < low) | (series > high)).sum())
    return {
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "lower_fence": float(low),
        "upper_fence": float(high),
        "flagged_rows": count,
        "flagged_pct": float(count / len(series) * 100),
        "handling": "Flagged for review only; retained because values are plausible.",
    }

outliers = {
    col: iqr_outlier_summary(clean[col])
    for col in ("tenure", "MonthlyCharges", "TotalCharges")
}

def grouped_churn_rate(column: str, order: list[str] | None = None) -> list[dict]:
    grouped = (
        clean.groupby(column, dropna=False, observed=False)
        .agg(customers=("customerID", "size"), churned=("Churn", lambda s: int(s.eq("Yes").sum())))
        .reset_index()
    )
    grouped["churn_rate_pct"] = grouped["churned"] / grouped["customers"] * 100
    if order:
        grouped[column] = pd.Categorical(grouped[column], categories=order, ordered=True)
        grouped = grouped.sort_values(column)
    else:
        grouped = grouped.sort_values(column)
    return [
        {
            "label": str(row[column]),
            "customers": int(row["customers"]),
            "churned": int(row["churned"]),
            "churn_rate_pct": float(row["churn_rate_pct"]),
        }
        for _, row in grouped.iterrows()
    ]

churn_counts = clean["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
total_customers = int(len(clean))
churned_customers = int((clean["Churn"] == "Yes").sum())
retained_customers = int((clean["Churn"] == "No").sum())
churn_rate = churned_customers / total_customers * 100

# Independently recompute the headline churn count from direct filtering and
# compare it to the aggregation used for the report and visuals.
direct_churn_count = int(clean.loc[clean["Churn"].eq("Yes"), "customerID"].count())
independent_churn_check = {
    "grouped_churn_count": churned_customers,
    "direct_filter_churn_count": direct_churn_count,
    "match": bool(churned_customers == direct_churn_count),
}

contract_order = ["Month-to-month", "One year", "Two year"]
payment_order = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
tenure_order = ["0–12", "13–24", "25–36", "37–48", "49–60", "61–72"]
clean["tenure_band"] = pd.cut(
    clean["tenure"],
    bins=[-1, 12, 24, 36, 48, 60, 72],
    labels=tenure_order,
)
contract_rates = grouped_churn_rate("Contract", contract_order)
payment_rates = grouped_churn_rate("PaymentMethod", payment_order)
tenure_rates = grouped_churn_rate("tenure_band", tenure_order)

numeric_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
descriptive = {}
for col in numeric_columns:
    s = clean[col]
    descriptive[col] = {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "q1": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q3": float(s.quantile(0.75)),
        "max": float(s.max()),
    }

monthly_by_churn = {}
for label in ["No", "Yes"]:
    s = clean.loc[clean["Churn"].eq(label), "MonthlyCharges"]
    monthly_by_churn[label] = {
        "customers": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
    }

correlation = (
    clean[["tenure", "MonthlyCharges", "TotalCharges"]]
    .corr(method="pearson")
    .round(6)
    .to_dict()
)

# Distinct-value and schema audit, recorded in the evidence object.
unique_values_for_report = {
    col: {
        "count": details["count"],
        **({"values": details["values"]} if details["values"] is not None else {}),
    }
    for col, details in unique_values.items()
}
schema = [
    {
        "column": col,
        "dtype": str(clean[col].dtype),
        "non_missing": int(clean[col].notna().sum()),
        "unique_values": int(clean[col].nunique(dropna=True)),
    }
    for col in clean.columns
    if col != "tenure_band"
]

validation = {
    "row_count_preserved_except_duplicates": bool(
        len(clean) == source_rows - duplicate_rows
    ),
    "unique_customerID": bool(clean["customerID"].is_unique),
    "no_exact_duplicate_rows": bool(not clean.duplicated().any()),
    "no_missing_values": bool(not clean.isna().any().any()),
    "categorical_values_in_expected_domain": bool(
        all(not vals for vals in invalid_categories.values())
    ),
    "numeric_domain_checks": bool(all(v == 0 for v in invalid_numeric.values())),
    "churn_labels_are_binary": bool(set(clean["Churn"].unique()) == {"Yes", "No"}),
    "charges_and_tenure_nonnegative": bool(
        clean["MonthlyCharges"].ge(0).all()
        and clean["TotalCharges"].ge(0).all()
        and clean["tenure"].ge(0).all()
    ),
    "headline_churn_independent_check": independent_churn_check["match"],
}
if not all(validation.values()):
    raise AssertionError(f"Validation failure: {validation}")

clean = clean.drop(columns=["tenure_band"])
clean.to_csv(PROCESSED / "customer_churn_cleaned.csv", index=False)

def row_to_markdown(row: dict, rate_key: str = "churn_rate_pct") -> str:
    return (
        f"| {row['label']} | {row['customers']:,} | {row['churned']:,} | "
        f"{row[rate_key]:.1f}% |"
    )

contract_rows_md = "\n".join(row_to_markdown(row) for row in contract_rates)
payment_rows_md = "\n".join(row_to_markdown(row) for row in payment_rates)
tenure_rows_md = "\n".join(row_to_markdown(row) for row in tenure_rates)
descriptive_rows_md = "\n".join(
    (
        f"| {col} | {vals['count']:,} | {vals['mean']:.2f} | {vals['std']:.2f} | "
        f"{vals['min']:.2f} | {vals['q1']:.2f} | {vals['median']:.2f} | "
        f"{vals['q3']:.2f} | {vals['max']:.2f} |"
    )
    for col, vals in descriptive.items()
)
monthly_rows_md = "\n".join(
    f"| {'Churned' if label == 'Yes' else 'Retained'} | {vals['customers']:,} | "
    f"{vals['mean']:.2f} | {vals['median']:.2f} | {vals['q1']:.2f} | {vals['q3']:.2f} |"
    for label, vals in monthly_by_churn.items()
)
correlation_columns = list(correlation.keys())
correlation_rows_md = "\n".join(
    "| " + " | ".join(
        [col] + [f"{correlation[col][other]:.3f}" for other in correlation_columns]
    ) + " |"
    for col in correlation_columns
)

report = f"""# Customer Churn Analysis – Data Quality Audit and Exploratory Data Analysis

> **Dataset substitution notice:** This project uses the public IBM Telco Customer Churn sample as an alternative because EdVyro resource files were unavailable. It is **not** the official EdVyro-provided dataset, and compliance with any unavailable EdVyro-specific instructions is not claimed.

## Dataset overview

- Source: IBM's `telco-customer-churn-on-icp4d` public GitHub repository and its `data/Telco-Customer-Churn.csv` file.
- Dataset URL: {SOURCE_URL}
- Repository URL: {REPOSITORY_URL}
- Downloaded file SHA-256: `{raw_sha256}`
- Source dimensions: {source_rows:,} rows × {source_columns} columns.
- Cleaned dimensions: {len(clean):,} rows × {len(clean.columns)} columns.
- Row grain: one record per customer (`customerID`).
- Target field: `Churn` (`Yes`/`No`).
- Pull date: 2026-11-18.

The raw source file is retained unchanged under `data/raw/`; the cleaned file is `data/processed/customer_churn_cleaned.csv`.

## Data quality audit

### Missing values

Whitespace-only `TotalCharges` entries were treated as missing after stripping. Raw missing-value counts:

| Column | Missing rows |
| --- | ---: |
{chr(10).join(f"| {col} | {count:,} |" for col, count in raw_missing_counts.items() if count)}

There were {blank_total_charges:,} blank `TotalCharges` values; {blank_total_with_zero_tenure:,} occurred with `tenure = 0`, and {blank_total_with_positive_tenure:,} with positive tenure. These are imputed to `0.0` because the field is cumulative charges and no months of service had accrued. This is an explicit domain-based assumption, not a recovered source value.

### Duplicates and key checks

- Exact duplicate rows in the source: {duplicate_rows:,}.
- Duplicate `customerID` values in the source: {duplicate_customer_ids:,}.
- `customerID` is treated as the row key; the cleaned ID is unique: {validation['unique_customerID']}.

No duplicate rows were removed because none were present.

### Invalid and inconsistent values

- String fields were whitespace-trimmed; blank strings were converted to missing values.
- `TotalCharges` was stored as text and converted to numeric.
- Invalid values outside the expected categorical domains: {sum(len(v) for v in invalid_categories.values()):,}.
- Invalid numeric-domain values (senior citizen outside 0/1, tenure outside 0–72, or negative charges): {sum(invalid_numeric.values()):,}.
- `SeniorCitizen` and `tenure` are integer fields; `MonthlyCharges` and `TotalCharges` are floating-point numeric fields after cleaning.

### Unique-value review

There are {len(unique_values_for_report)} analyzed source columns. Category cardinalities are documented in `analysis_evidence.json`; small-domain values are listed there. Notable counts: `Contract` has {unique_values['Contract']['count']} levels, `PaymentMethod` has {unique_values['PaymentMethod']['count']} levels, and `Churn` has {unique_values['Churn']['count']} levels.

### Outliers

The 1.5×IQR rule was used as a screening flag on tenure and charge measures. It is not an error detector. Flagged rows were retained because long tenure and cumulative charges are plausible observations; no row was removed solely for being statistically extreme.

| Measure | Q1 | Q3 | IQR lower fence | IQR upper fence | Rows flagged |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(f"| {col} | {vals['q1']:.2f} | {vals['q3']:.2f} | {vals['lower_fence']:.2f} | {vals['upper_fence']:.2f} | {vals['flagged_rows']:,} ({vals['flagged_pct']:.1f}%) |" for col, vals in outliers.items())}

## Cleaning decisions

1. Kept the downloaded raw CSV unchanged and wrote a separate processed CSV.
2. Trimmed surrounding whitespace in text fields.
3. Converted whitespace-only strings to missing values.
4. Converted `TotalCharges` from numeric text to a floating-point number.
5. Imputed the {blank_total_charges} blank `TotalCharges` values to `0.0` only after verifying that all corresponded to `tenure = 0`.
6. Kept every customer record; no duplicate or outlier records were removed.
7. Preserved the source's categorical labels and encoded `SeniorCitizen`/`tenure` as integers.

## Validation results

All validation checks passed:

| Check | Result |
| --- | --- |
{chr(10).join(f"| {key.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |" for key, value in validation.items())}

## Exploratory data analysis

### Churn overview

- Customers: {total_customers:,}.
- Churned: {churned_customers:,} ({churn_rate:.2f}%).
- Retained: {retained_customers:,} ({retained_customers / total_customers * 100:.2f}%).

### Churn by contract

| Contract | Customers | Churned | Churn rate |
| --- | ---: | ---: | ---: |
{contract_rows_md}

### Churn by payment method

| Payment method | Customers | Churned | Churn rate |
| --- | ---: | ---: | ---: |
{payment_rows_md}

### Churn by tenure band

| Tenure (months) | Customers | Churned | Churn rate |
| --- | ---: | ---: | ---: |
{tenure_rows_md}

### Numeric measures and monthly charges

| Measure | Count | Mean | Std. dev. | Min | Q1 | Median | Q3 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{descriptive_rows_md}

`SeniorCitizen` is binary (0/1); its numeric summary should not be read as a continuous measurement. For `MonthlyCharges`, the overall mean is {descriptive['MonthlyCharges']['mean']:.2f} and median is {descriptive['MonthlyCharges']['median']:.2f} reported charge units.

| Churn status | Customers | Mean MonthlyCharges | Median | Q1 | Q3 |
| --- | ---: | ---: | ---: | ---: | ---: |
{monthly_rows_md}

The source does not explicitly define a currency symbol in the file, so values are described as reported charge units.

### Correlation

Pearson correlations for tenure, MonthlyCharges, and TotalCharges:

| Measure | tenure | MonthlyCharges | TotalCharges |
| --- | ---: | ---: | ---: |
{correlation_rows_md}

TotalCharges is cumulative and mechanically related to tenure and monthly charges; correlations are descriptive associations, not independent drivers or evidence of cause.

## Initial patterns and key findings

- The overall observed churn rate is {churn_rate:.2f}% ({churned_customers:,} of {total_customers:,} customers).
- The highest observed contract churn rate is for **{max(contract_rates, key=lambda r: r['churn_rate_pct'])['label']}** at {max(contract_rates, key=lambda r: r['churn_rate_pct'])['churn_rate_pct']:.2f}% ({max(contract_rates, key=lambda r: r['churn_rate_pct'])['churned']:,} of {max(contract_rates, key=lambda r: r['churn_rate_pct'])['customers']:,}); the lowest is **{min(contract_rates, key=lambda r: r['churn_rate_pct'])['label']}** at {min(contract_rates, key=lambda r: r['churn_rate_pct'])['churn_rate_pct']:.2f}% ({min(contract_rates, key=lambda r: r['churn_rate_pct'])['churned']:,} of {min(contract_rates, key=lambda r: r['churn_rate_pct'])['customers']:,}).
- The highest observed payment-method churn rate is for **{max(payment_rates, key=lambda r: r['churn_rate_pct'])['label']}** at {max(payment_rates, key=lambda r: r['churn_rate_pct'])['churn_rate_pct']:.2f}% ({max(payment_rates, key=lambda r: r['churn_rate_pct'])['churned']:,} of {max(payment_rates, key=lambda r: r['churn_rate_pct'])['customers']:,}).
- In this sample, the churned cohort has a higher mean MonthlyCharges ({monthly_by_churn['Yes']['mean']:.2f}) than the retained cohort ({monthly_by_churn['No']['mean']:.2f}); this association does not establish that charges caused churn.
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
"""
(REPORTS / "data_quality_report.md").write_text(report, encoding="utf-8")

def bars_spec(title: str, aside: str, labels: list[str], label: str, values: list[float],
              value_kind: str, orientation: str = "horizontal", subject_index: int | None = None) -> dict:
    spec = {
        "version": 1,
        "form": "categorical-bars",
        "title": title,
        "aside": aside,
        "mode": "grouped",
        "orientation": orientation,
        "labels": labels,
        "axis": {"zero": True},
        "series": [{
            "label": label,
            "valueKind": value_kind,
            "role": "subject",
            "data": values,
        }],
    }
    if subject_index is not None:
        spec["subjectIndex"] = subject_index
    return spec

chart_specs = {
    "churn_distribution": bars_spec(
        "Most customers were retained",
        f"Customer records in the IBM sample · {total_customers:,} total",
        ["Retained", "Churned"],
        "Customers",
        [retained_customers, churned_customers],
        "count",
        orientation="vertical",
    ),
    "contract_churn": bars_spec(
        "Month-to-month had the highest churn rate",
        "Observed churn rate by contract · rates use each contract's customers",
        [r["label"] for r in contract_rates],
        "Churn rate",
        [round(r["churn_rate_pct"], 4) for r in contract_rates],
        "percent",
        subject_index=max(range(len(contract_rates)), key=lambda i: contract_rates[i]["churn_rate_pct"]),
    ),
}
switcher = {
    "version": 1,
    "form": "switcher",
    "title": "Churn and charges differ across customer segments",
    "aside": "Observed rates and cohort means · full sample",
    "views": [
        {
            "label": "Payment method",
            "form": "categorical-bars",
            "title": "Electronic check had the highest churn rate",
            "aside": "Observed churn rate by payment method",
            "mode": "grouped",
            "orientation": "horizontal",
            "labels": [r["label"] for r in payment_rates],
            "axis": {"zero": True},
            "series": [{
                "label": "Churn rate",
                "valueKind": "percent",
                "role": "subject",
                "data": [round(r["churn_rate_pct"], 4) for r in payment_rates],
            }],
            "subjectIndex": max(range(len(payment_rates)), key=lambda i: payment_rates[i]["churn_rate_pct"]),
        },
        {
            "label": "Tenure band",
            "form": "categorical-bars",
            "title": "Churn rate by tenure band",
            "aside": "Observed churn rate · each band's customers",
            "mode": "grouped",
            "orientation": "horizontal",
            "labels": [r["label"] for r in tenure_rates],
            "axis": {"zero": True},
            "series": [{
                "label": "Churn rate",
                "valueKind": "percent",
                "role": "subject",
                "data": [round(r["churn_rate_pct"], 4) for r in tenure_rates],
            }],
        },
        {
            "label": "Monthly charges",
            "form": "categorical-bars",
            "title": "Mean MonthlyCharges were higher in the churned cohort",
            "aside": "Cohort mean · reported charge units",
            "mode": "grouped",
            "orientation": "horizontal",
            "labels": ["Retained", "Churned"],
            "axis": {"zero": True},
            "series": [{
                "label": "Mean MonthlyCharges",
                "valueKind": "number",
                "role": "subject",
                "data": [
                    round(monthly_by_churn["No"]["mean"], 4),
                    round(monthly_by_churn["Yes"]["mean"], 4),
                ],
            }],
        },
    ],
}

for name, spec in chart_specs.items():
    (VISUALS / f"{name}.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
(VISUALS / "segment_analysis.json").write_text(json.dumps(switcher, indent=2), encoding="utf-8")

evidence = {
    "source": {
        "repository_url": REPOSITORY_URL,
        "dataset_url": SOURCE_URL,
        "alternative_not_edvyro": True,
        "retrieved_on": "2026-11-18",
        "sha256": raw_sha256,
    },
    "population": {
        "grain": "one row per customer",
        "source_rows": source_rows,
        "source_columns": source_columns,
        "clean_rows": len(clean),
        "clean_columns": len(clean.columns),
        "total_customers": total_customers,
        "churned_customers": churned_customers,
        "retained_customers": retained_customers,
        "churn_rate_pct": churn_rate,
        "direct_churn_count_check": independent_churn_check,
    },
    "quality": {
        "raw_missing_counts": raw_missing_counts,
        "raw_whitespace_blank_counts": raw_blank_counts,
        "exact_duplicate_rows": duplicate_rows,
        "duplicate_customer_ids": duplicate_customer_ids,
        "blank_total_charges": blank_total_charges,
        "blank_total_with_zero_tenure": blank_total_with_zero_tenure,
        "blank_total_with_positive_tenure": blank_total_with_positive_tenure,
        "invalid_categories": invalid_categories,
        "invalid_numeric": invalid_numeric,
        "unique_values": unique_values_for_report,
        "schema": schema,
        "outliers": outliers,
        "validation": validation,
    },
    "descriptive_statistics": descriptive,
    "contract_churn": contract_rates,
    "payment_method_churn": payment_rates,
    "tenure_band_churn": tenure_rates,
    "monthly_charges_by_churn": monthly_by_churn,
    "pearson_correlation": correlation,
    "comparison_plan": [
        {
            "question": "How many sampled customers churned versus stayed?",
            "grain": "customer",
            "population": f"{total_customers} cleaned customer records",
            "unit": "customers and percent of customers",
            "derivation": "Direct customer count by Churn label; rate = churned / total",
            "finding": f"{churned_customers} churned of {total_customers} ({churn_rate:.4f}%)",
            "display": "chart",
        },
        {
            "question": "How does churn rate vary by contract and payment method?",
            "grain": "customer grouped by one contract or payment category",
            "population": "all customers in each mutually exclusive category",
            "unit": "percent",
            "derivation": "category churned customers / category customer count",
            "finding": "Observed rates differ across both supplied categorical cuts.",
            "display": "charts",
        },
        {
            "question": "How do tenure bands and monthly charges relate descriptively to churn?",
            "grain": "customer grouped by tenure band or churn status",
            "population": "all customers with valid values in the respective measure",
            "unit": "percent and reported charge units",
            "derivation": "within-band churn count / band total; cohort mean MonthlyCharges",
            "finding": "Rates and cohort means are descriptive; no causal interpretation.",
            "display": "chart and report prose",
        },
    ],
}
(OUT / "analysis_evidence.json").write_text(
    json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
)
(ROOT / "analysis_evidence.json").write_text(
    json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
)

print(json.dumps({
    "source_rows": source_rows,
    "clean_rows": len(clean),
    "source_columns": source_columns,
    "churned_customers": churned_customers,
    "churn_rate_pct": round(churn_rate, 4),
    "missing_total_charges": blank_total_charges,
    "duplicates": duplicate_rows,
    "validation_passed": all(validation.values()),
    "evidence": str(OUT / "analysis_evidence.json"),
    "cleaned_csv": str(PROCESSED / "customer_churn_cleaned.csv"),
    "report": str(REPORTS / "data_quality_report.md"),
    "raw_sha256": raw_sha256,
}, indent=2))