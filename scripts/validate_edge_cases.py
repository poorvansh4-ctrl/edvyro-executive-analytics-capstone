"""Edge-case validation for the Task 04 segmentation and risk-scoring pipeline.

Each check feeds deliberately awkward records through the same functions the
notebook uses and compares the real result with the expected behaviour.
Run from the project root to regenerate ``reports/edge_case_validation.md``:

    python scripts/validate_edge_cases.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import segmentation_model as sm  # noqa: E402

CLEAN_PATH = ROOT / "data" / "processed" / "customer_churn_cleaned.csv"
REPORT_PATH = ROOT / "reports" / "edge_case_validation.md"


def _template(clean: pd.DataFrame, n: int = 1, **overrides) -> pd.DataFrame:
    """Copy real customer rows, give them test IDs and apply field overrides."""
    rows = clean.head(n).copy().reset_index(drop=True)
    rows["customerID"] = [f"EDGE-{i + 1:02d}" for i in range(n)]
    for column, value in overrides.items():
        rows[column] = value
    return rows


def _row(scored: pd.DataFrame, customer_id: str) -> pd.Series:
    return scored.loc[scored["customerID"].eq(customer_id)].iloc[0]


def _all_flags_customer(clean: pd.DataFrame) -> pd.DataFrame:
    return _template(
        clean, Contract="Month-to-month", tenure=1, PaymentMethod="Electronic check",
        InternetService="Fiber optic", OnlineSecurity="No", TechSupport="No",
        MonthlyCharges=100.0, PaperlessBilling="Yes",
    )


def _no_flags_customer(clean: pd.DataFrame) -> pd.DataFrame:
    return _template(
        clean, Contract="Two year", tenure=60, PaymentMethod="Credit card (automatic)",
        InternetService="No", OnlineSecurity="No internet service",
        TechSupport="No internet service", MonthlyCharges=20.0, PaperlessBilling="No",
    )


def run_edge_case_checks(clean: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Run every edge case and return a results table (never raises)."""
    checks = []

    def check(case_id: str, name: str, scenario: str, expected: str):
        def decorator(test):
            try:
                actual, passed = test()
            except Exception as error:  # a crash is itself a failed check
                actual, passed = f"Raised {type(error).__name__}: {error}", False
            checks.append({
                "id": case_id, "test_case": name, "scenario": scenario,
                "expected": expected, "actual": actual, "status": "PASS" if passed else "FAIL",
            })
            return test
        return decorator

    @check("EC01", "Missing customer IDs",
           "Three records: one valid ID, one missing ID, one ID of only spaces.",
           "Both records without an ID are dropped and logged; the valid customer is scored.")
    def _():
        frame = _template(clean, 3)
        frame.loc[1, "customerID"] = np.nan
        frame.loc[2, "customerID"] = "   "
        scored, issues = sm.run_pipeline(frame, weights)
        dropped = issues["handling"].str.contains("customer ID missing").sum()
        return (f"{len(scored)} customer kept, {dropped} records dropped and logged",
                len(scored) == 1 and dropped == 2 and scored["risk_score"].notna().all())

    @check("EC02", "Duplicate customer records",
           "Customer EDGE-01 appears three times (one exact copy, one with a different tenure) plus EDGE-02.",
           "First EDGE-01 record kept, the two repeats dropped and logged; 2 customers scored.")
    def _():
        frame = _template(clean, 2)
        exact_copy = frame.iloc[[0]].copy()
        changed_copy = frame.iloc[[0]].copy()
        changed_copy["tenure"] = 70
        frame = pd.concat([frame.iloc[[0]], exact_copy, changed_copy, frame.iloc[[1]]], ignore_index=True)
        scored, issues = sm.run_pipeline(frame, weights)
        kept_tenure = _row(scored, "EDGE-01")["tenure"]
        dropped = issues["handling"].str.contains("repeated customer ID").sum()
        return (f"{len(scored)} customers kept, {dropped} repeats dropped, EDGE-01 tenure kept = {kept_tenure:.0f}",
                len(scored) == 2 and dropped == 2 and kept_tenure == frame.loc[0, "tenure"])

    @check("EC03", "Missing numerical values",
           "EDGE-01 has no MonthlyCharges; EDGE-02 has no tenure.",
           "EDGE-01 is still segmented and scored with the high-charge flag off; EDGE-02 goes to "
           "'Needs Data Review' and is 'Not Scored'.")
    def _():
        frame = _template(clean, 2, tenure=24)
        frame.loc[0, "MonthlyCharges"] = np.nan
        frame.loc[1, "tenure"] = np.nan
        scored, _ = sm.run_pipeline(frame, weights)
        first, second = _row(scored, "EDGE-01"), _row(scored, "EDGE-02")
        return (f"EDGE-01: {first['segment']}, score {first['risk_score']}, high-charge flag "
                f"{first['high_monthly_charges']}; EDGE-02: {second['segment']}, {second['risk_category']}",
                first["segment"] in sm.SEGMENT_ORDER and pd.notna(first["risk_score"])
                and first["high_monthly_charges"] == 0
                and second["segment"] == sm.REVIEW_SEGMENT and second["risk_category"] == sm.NOT_SCORED)

    @check("EC04", "Missing categorical values",
           "PaymentMethod and InternetService are blank for one customer.",
           "Customer is still scored; flags that depend on the blank fields stay 0; "
           "missing_risk_inputs = 2.")
    def _():
        frame = _template(clean, 1, PaymentMethod=np.nan, InternetService="")
        scored, _ = sm.run_pipeline(frame, weights)
        row = scored.iloc[0]
        dependent = ["electronic_check_payment", "fiber_optic_service", "no_online_security", "no_tech_support"]
        flags = {name: int(row[name]) for name in dependent}
        return (f"{row['segment']}, score {row['risk_score']}, dependent flags {flags}, "
                f"missing_risk_inputs = {row['missing_risk_inputs']}",
                pd.notna(row["risk_score"]) and sum(flags.values()) == 0 and row["missing_risk_inputs"] == 2)

    @check("EC05", "Zero tenure",
           "A customer who joined this month: tenure = 0 and blank TotalCharges.",
           "Segment 'New Customers', early-tenure flag on, valid score; no division by zero.")
    def _():
        frame = _template(clean, 1, tenure=0, TotalCharges=" ")
        scored, _ = sm.run_pipeline(frame, weights)
        row = scored.iloc[0]
        return (f"{row['segment']}, early_tenure = {row['early_tenure']}, score {row['risk_score']}",
                row["segment"] == "New Customers" and row["early_tenure"] == 1
                and 0 <= row["risk_score"] <= 100)

    @check("EC06", "Zero monthly charges",
           "An established customer with MonthlyCharges = 0.",
           "Record kept and scored, high-charge flag off, data-quality note asks to verify billing.")
    def _():
        frame = _template(clean, 1, tenure=30, MonthlyCharges=0.0)
        scored, _ = sm.run_pipeline(frame, weights)
        row = scored.iloc[0]
        return (f"{row['segment']}, score {row['risk_score']}, note: '{row['data_quality_note']}'",
                pd.notna(row["risk_score"]) and row["high_monthly_charges"] == 0
                and "zero monthly charge" in row["data_quality_note"])

    @check("EC07", "Very high monthly charges",
           "An established customer with MonthlyCharges = 5,000 (about 42x the observed maximum).",
           "Record kept, high-charge flag on, score still capped at 100, note flags the value for review.")
    def _():
        frame = _all_flags_customer(clean)
        frame["tenure"] = 30
        frame["MonthlyCharges"] = 5000.0
        scored, _ = sm.run_pipeline(frame, weights)
        row = scored.iloc[0]
        return (f"{row['segment']}, score {row['risk_score']}, note: '{row['data_quality_note']}'",
                row["high_monthly_charges"] == 1 and 0 <= row["risk_score"] <= 100
                and "above plausible range" in row["data_quality_note"])

    @check("EC08", "Unexpected categorical values",
           "Contract written as '  month-to-month ' (case/spaces), Contract = 'Three year', "
           "PaymentMethod = 'Bitcoin'.",
           "Case/space variants are standardised; unknown contract -> 'Needs Data Review'; "
           "unknown payment method -> missing (flag off) but customer still scored.")
    def _():
        frame = _template(clean, 3, tenure=24)
        frame.loc[0, "Contract"] = "  month-to-month "
        frame.loc[1, "Contract"] = "Three year"
        frame.loc[2, "PaymentMethod"] = "Bitcoin"
        scored, issues = sm.run_pipeline(frame, weights)
        a, b, c = (_row(scored, f"EDGE-0{i}") for i in (1, 2, 3))
        return (f"EDGE-01 contract -> '{a['Contract']}'; EDGE-02 -> {b['segment']}; "
                f"EDGE-03 payment -> {'missing' if pd.isna(c['PaymentMethod']) else c['PaymentMethod']} "
                f"(scored: {pd.notna(c['risk_score'])}); "
                f"{len(issues)} values logged",
                a["Contract"] == "Month-to-month" and b["segment"] == sm.REVIEW_SEGMENT
                and pd.isna(c["PaymentMethod"]) and pd.notna(c["risk_score"]) and len(issues) == 2)

    @check("EC09", "Empty datasets",
           "Three inputs: a zero-row table with the usual columns, a table with no columns, and None.",
           "Each returns zero scored customers and empty summary / priority tables without raising.")
    def _():
        results = []
        for frame in (clean.head(0), pd.DataFrame(), None):
            scored, _ = sm.run_pipeline(frame, weights)
            summary = sm.build_segment_summary(scored)
            priority = sm.build_priority_list(scored)
            results.append(len(scored) + len(summary) + len(priority))
        return (f"Rows returned (scored + summary + priority) per input: {results}", results == [0, 0, 0])

    @check("EC10", "Customers with insufficient information",
           "EDGE-01 has only a customer ID; EDGE-02 has tenure and contract but 4 of 8 scoring inputs missing.",
           "Both are routed to 'Needs Data Review', 'Not Scored', with a data-correction action.")
    def _():
        only_id = pd.DataFrame({"customerID": ["EDGE-01"]})
        sparse = _template(clean, 1)
        sparse["customerID"] = "EDGE-02"
        sparse[["PaymentMethod", "InternetService", "MonthlyCharges", "PaperlessBilling"]] = np.nan
        scored, _ = sm.run_pipeline(pd.concat([only_id, sparse], ignore_index=True), weights)
        outcome = scored[["segment", "risk_category", "retention_priority"]].drop_duplicates()
        return (f"{len(scored)} customers -> {outcome.to_dict('records')}",
                len(scored) == 2 and scored["segment"].eq(sm.REVIEW_SEGMENT).all()
                and scored["risk_category"].eq(sm.NOT_SCORED).all()
                and scored["retention_priority"].eq("Review").all())

    @check("EC11", "Invalid numeric entries",
           "tenure = 'abc' for EDGE-01; MonthlyCharges = -10 for EDGE-02.",
           "Both values set to missing and logged; EDGE-01 goes to data review; EDGE-02 is scored with "
           "the high-charge flag off.")
    def _():
        frame = _template(clean, 2, tenure=24)
        frame["tenure"] = frame["tenure"].astype("object")
        frame.loc[0, "tenure"] = "abc"
        frame.loc[1, "MonthlyCharges"] = -10
        scored, issues = sm.run_pipeline(frame, weights)
        a, b = _row(scored, "EDGE-01"), _row(scored, "EDGE-02")
        return (f"{len(issues)} values logged; EDGE-01: {a['segment']}; EDGE-02: score {b['risk_score']}, "
                f"MonthlyCharges = {b['MonthlyCharges']}",
                len(issues) == 2 and a["segment"] == sm.REVIEW_SEGMENT
                and pd.notna(b["risk_score"]) and pd.isna(b["MonthlyCharges"]))

    @check("EC12", "Score bounds (maximum and minimum risk profiles)",
           "One customer with every risk indicator present; one with none present.",
           "Scores are exactly 100 (Critical) and 0 (Low).")
    def _():
        frame = pd.concat([_all_flags_customer(clean), _no_flags_customer(clean)], ignore_index=True)
        frame["customerID"] = ["EDGE-MAX", "EDGE-MIN"]
        scored, _ = sm.run_pipeline(frame, weights)
        high, low = _row(scored, "EDGE-MAX"), _row(scored, "EDGE-MIN")
        return (f"All indicators: {high['risk_score']} ({high['risk_category']}); "
                f"no indicators: {low['risk_score']} ({low['risk_category']})",
                high["risk_score"] == 100 and high["risk_category"] == "Critical"
                and low["risk_score"] == 0 and low["risk_category"] == "Low")

    @check("EC13", "Risk-category boundaries",
           "Scores 0, 29.9, 30, 59.9, 60, 79.9, 80, 100 and a missing score.",
           "Low, Low, Medium, Medium, High, High, Critical, Critical, Not Scored.")
    def _():
        scores = pd.Series([0, 29.9, 30, 59.9, 60, 79.9, 80, 100, np.nan])
        expected = ["Low", "Low", "Medium", "Medium", "High", "High", "Critical", "Critical", sm.NOT_SCORED]
        actual = sm.categorize_risk(scores).tolist()
        return ", ".join(actual), actual == expected

    @check("EC14", "Historical churn status unavailable",
           "Five customers scored from a table without the Churn column.",
           "Customers are still segmented and scored; summary churn rate is blank (NaN), not 0 and not an error.")
    def _():
        frame = _template(clean, 5).drop(columns=["Churn"])
        scored, _ = sm.run_pipeline(frame, weights)
        summary = sm.build_segment_summary(scored)
        blank_rates = summary["churn_rate_pct"].isna().all()
        return (f"{scored['risk_score'].notna().sum()} of 5 scored; churn rates all blank: {blank_rates}",
                scored["risk_score"].notna().all() and summary["churn_rate_pct"].isna().all())

    @check("EC15", "Segments with zero customers",
           "Segment summary for a single customer, so four of five segments are empty.",
           "Empty segments show a count of 0 and blank averages; no division-by-zero error.")
    def _():
        scored, _ = sm.run_pipeline(_template(clean, 1), weights)
        summary = sm.build_segment_summary(scored)
        empty = summary.loc[summary["customer_count"].eq(0)]
        return (f"{len(summary)} summary rows, {len(empty)} empty segments with blank averages: "
                f"{empty['avg_tenure_months'].isna().all()}",
                len(summary) == 5 and len(empty) == 4 and empty["avg_tenure_months"].isna().all()
                and summary["customer_pct"].sum() == 100)

    @check("EC16", "Unusable weights",
           "Fit indicator weights on data where every customer has the same churn outcome.",
           "A clear ValueError explains that no indicator reaches the minimum lift (no silent, meaningless score).")
    def _():
        customers, _ = sm.prepare_customers(clean.head(200))
        try:
            sm.fit_risk_weights(sm.compute_risk_indicators(customers), pd.Series(False, index=customers.index))
        except ValueError as error:
            return f"ValueError: {error}", True
        return "No error raised", False

    @check("EC17", "Full dataset invariants",
           f"All {len(clean):,} cleaned customers, scored twice.",
           "Every customer gets exactly one of the 5 business segments; scores are within 0-100; "
           "categories match the thresholds; both runs are identical.")
    def _():
        first, _ = sm.run_pipeline(clean, weights)
        second, _ = sm.run_pipeline(clean, weights)
        in_range = first["risk_score"].between(0, 100).all()
        segmented = first["segment"].isin(sm.SEGMENT_ORDER).all()
        consistent = sm.categorize_risk(first["risk_score"]).equals(first["risk_category"])
        identical = first.equals(second)
        return (f"{len(first):,} customers; all segmented: {segmented}; scores in range: {in_range} "
                f"(min {first['risk_score'].min()}, max {first['risk_score'].max()}); "
                f"categories consistent: {consistent}; runs identical: {identical}",
                len(first) == len(clean) and in_range and segmented and consistent and identical)

    return pd.DataFrame(checks)


HANDLING_RULES = [
    ("Missing or blank customer ID", "Record dropped and logged; a customer cannot be contacted without an ID."),
    ("Repeated customer ID", "First record kept, later records dropped and logged."),
    ("Non-numeric or negative tenure / charges", "Value set to missing and logged."),
    ("Zero monthly charge", "Kept as valid, with a note to verify billing."),
    (f"Monthly charge above {sm.MAX_PLAUSIBLE_MONTHLY_CHARGE:.0f}", "Kept (still counts as a high charge), with a note to verify billing."),
    ("Category label with different case or spaces", "Standardised to the known label (e.g. 'month-to-month' -> 'Month-to-month')."),
    ("Unknown category label", "Set to missing and logged; never mapped to a guessed category."),
    ("Missing scoring input", "The related indicator scores 0 points; `missing_risk_inputs` records how many inputs were missing."),
    (f"Missing tenure or contract, or more than {sm.MAX_MISSING_RISK_INPUTS} of 8 scoring inputs missing",
     f"Segment '{sm.REVIEW_SEGMENT}', risk category '{sm.NOT_SCORED}', action: correct the record."),
    ("Missing Churn status", "Customer still scored; churn rates are calculated only over known statuses (blank if none)."),
    ("Empty input", "Empty outputs with the expected columns; no exception."),
]


def write_report(results: pd.DataFrame, path: Path = REPORT_PATH) -> Path:
    passed = int(results["status"].eq("PASS").sum())

    def cell(text: object) -> str:
        return str(text).replace("|", "\\|").replace("\n", " ")

    rows = "\n".join(
        f"| {r.id} | {cell(r.test_case)} | {cell(r.scenario)} | {cell(r.expected)} | {cell(r.actual)} | **{r.status}** |"
        for r in results.itertuples()
    )
    rules = "\n".join(f"| {cell(case)} | {cell(rule)} |" for case, rule in HANDLING_RULES)
    report = f"""# Edge Case Validation – Segmentation and Churn-Risk Scoring (Task 04)

Generated by `scripts/validate_edge_cases.py` on {date.today().isoformat()}. Every "Actual result" below is
produced by running the same functions the notebook uses (`scripts/segmentation_model.py`); nothing is
typed in by hand.

**Result: {passed} of {len(results)} checks passed.**

## How invalid records are handled

| Situation | Handling |
| --- | --- |
{rules}

Risk indicators are yes/no flags, so an extreme or missing value can switch at most one flag and the
normalised score can never leave the 0–100 range.

## Test cases

| ID | Test case | Scenario | Expected behaviour | Actual result | Status |
| --- | --- | --- | --- | --- | --- |
{rows}

## How to re-run

```bash
python scripts/validate_edge_cases.py
```

The script exits with a non-zero status if any check fails. Test records are built from real rows of
`data/processed/customer_churn_cleaned.csv` with test IDs (`EDGE-xx`) and targeted changes, so no
customer data is altered.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path


def main() -> int:
    clean = pd.read_csv(CLEAN_PATH)
    customers, _ = sm.prepare_customers(clean)
    weights, _, _ = sm.fit_scorecard(customers)
    results = run_edge_case_checks(clean, weights)
    write_report(results)
    print(results[["id", "test_case", "status"]].to_string(index=False))
    print(f"\n{results['status'].eq('PASS').sum()} of {len(results)} checks passed -> {REPORT_PATH.relative_to(ROOT)}")
    return 0 if results["status"].eq("PASS").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
