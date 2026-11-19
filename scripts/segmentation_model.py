"""Explainable customer segmentation, churn-risk scoring and retention
recommendations for the IBM Telco churn sample (EdVyro Task 04).

All business rules live in this module. It does no plotting and writes no
files, so the notebook, the edge-case validation script and any later
dashboard share exactly the same logic.

Terminology used throughout:
- historical churn status: the ``Churn`` column recorded in the dataset;
- churn-risk score: a transparent, rule-based 0-100 points score built from
  account and service attributes. It is NOT a trained predictive model and it
  never uses ``Churn`` as an input. ``Churn`` is used only to set indicator
  weights on a training split and to check the score on a held-out test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TEST_SHARE = 0.30

# Thresholds come from the feature distributions or the contract calendar,
# never from churn outcomes.
EARLY_TENURE_MONTHS = 12               # customers still in their first year
HIGH_MONTHLY_CHARGE = 90.0             # about the 75th percentile (89.85) of MonthlyCharges
MAX_PLAUSIBLE_MONTHLY_CHARGE = 250.0   # about 2x the observed maximum (118.75)
MIN_INDICATOR_LIFT_PP = 5.0            # weaker indicators earn no points
MAX_MISSING_RISK_INPUTS = 3            # more missing scoring inputs -> not scored

RISK_THRESHOLDS = [0, 30, 60, 80]      # lower bound of each risk category
RISK_LABELS = ["Low", "Medium", "High", "Critical"]
NOT_SCORED = "Not Scored"
REVIEW_SEGMENT = "Needs Data Review"

NUMERIC_FIELDS = ("tenure", "MonthlyCharges", "TotalCharges")
CATEGORY_DOMAINS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "PaperlessBilling": ["Yes", "No"],
    "Churn": ["Yes", "No"],
}
ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
# Without both of these a customer cannot be placed in a lifecycle or contract
# segment, so the record is routed to data review instead of being guessed.
REQUIRED_FIELDS = ("tenure", "Contract")

# Each indicator is a yes/no flag built only from account and service fields.
# Demographic fields (gender, SeniorCitizen, Partner, Dependents) are excluded
# on purpose: they are not actionable and could lead to unfair treatment.
RISK_INDICATORS = {
    "month_to_month_contract": "Contract is month-to-month (customer can leave in any month)",
    "early_tenure": f"Tenure of {EARLY_TENURE_MONTHS} months or less (first contract year)",
    "electronic_check_payment": "Pays by electronic check (manual, non-automatic payment)",
    "fiber_optic_service": "Has fiber optic internet service",
    "no_online_security": "Internet customer without the Online Security add-on",
    "no_tech_support": "Internet customer without the Tech Support add-on",
    "high_monthly_charges": f"Monthly charges of {HIGH_MONTHLY_CHARGE:.0f} or more",
    "paperless_billing": "Uses paperless billing",
}
RISK_INPUT_FIELDS = (
    "Contract",
    "tenure",
    "PaymentMethod",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "MonthlyCharges",
    "PaperlessBilling",
)

# Segments are mutually exclusive and assigned in this priority order: the
# first rule a customer matches wins. Churn is never used to assign a segment.
SEGMENTS = [
    {
        "name": "New Customers",
        "rule": f"Tenure of {EARLY_TENURE_MONTHS} months or less",
        "why": "The first year is the onboarding period; lifecycle stage matters more than spend here.",
    },
    {
        "name": "High-Value Customers",
        "rule": f"Tenure above {EARLY_TENURE_MONTHS} months and monthly charges of {HIGH_MONTHLY_CHARGE:.0f} or more",
        "why": "The top quarter of monthly spend; each lost customer removes the most recurring revenue.",
    },
    {
        "name": "Phone-Only Basic Customers",
        "rule": f"Tenure above {EARLY_TENURE_MONTHS} months and no internet service",
        "why": "Buy only the low-cost phone plan; very different needs from internet customers.",
    },
    {
        "name": "Loyal Contract Customers",
        "rule": f"Tenure above {EARLY_TENURE_MONTHS} months on a one- or two-year contract",
        "why": "Established customers who have already committed to a fixed term.",
    },
    {
        "name": "Flexible Month-to-Month Customers",
        "rule": f"Tenure above {EARLY_TENURE_MONTHS} months on a month-to-month contract",
        "why": "Established customers with no fixed term, so they can leave in any month.",
    },
]
SEGMENT_ORDER = [segment["name"] for segment in SEGMENTS]

# Recommendation engine, part 1: what the conversation should be about.
# Interpretations describe the rule-defined group; measured statistics are
# added from the data by build_segment_summary().
SEGMENT_PLAYBOOK = {
    "New Customers": {
        "interpretation": (
            "Customers still forming their opinion of the service. Most are on month-to-month "
            "terms, so a poor first-year experience can end the relationship quickly."
        ),
        "action": (
            "Structured onboarding: setup help and check-ins during the first months, "
            "introduce Online Security / Tech Support trials, and present one- or two-year "
            "contract options once the customer is settled."
        ),
        "focus": "onboarding check-in, setup help and an add-on trial",
    },
    "High-Value Customers": {
        "interpretation": (
            "The highest-spending established customers, typically on fiber optic with several "
            "add-ons. Their risk depends heavily on contract type, so the segment mixes very "
            "secure and very exposed accounts."
        ),
        "action": (
            "Relationship management: premium or priority support, a periodic plan-fit review "
            "so customers are not paying for services they do not use, and loyalty benefits "
            "tied to a longer contract for those on month-to-month terms."
        ),
        "focus": "priority support and a plan-fit review",
    },
    "Phone-Only Basic Customers": {
        "interpretation": (
            "Long-standing customers who buy only the low-cost phone plan. They are the most "
            "stable group and need little retention effort."
        ),
        "action": (
            "Low-cost maintenance: keep service reliable, use light-touch satisfaction surveys, "
            "and test relevant internet bundle offers only where customers show interest."
        ),
        "focus": "light-touch satisfaction check",
    },
    "Loyal Contract Customers": {
        "interpretation": (
            "Established customers on fixed-term contracts with long tenure. Their main exposure "
            "is the contract renewal date rather than month-to-month switching."
        ),
        "action": (
            "Renewal management: contact customers ahead of contract end, recognise loyalty, "
            "and invite them into a referral programme."
        ),
        "focus": "renewal reminder and loyalty recognition",
    },
    "Flexible Month-to-Month Customers": {
        "interpretation": (
            "Customers who have stayed beyond their first year without committing to a contract. "
            "They have shown some loyalty but can still leave in any month."
        ),
        "action": (
            "Commitment and value building: offer contract options with clear benefits, promote "
            "automatic payment, and bundle protective add-ons such as Online Security and Tech Support."
        ),
        "focus": "contract options and add-on bundle",
    },
    REVIEW_SEGMENT: {
        "interpretation": "Key account fields (tenure or contract) are missing or invalid, so the customer cannot be classified safely.",
        "action": "Correct the source record, then re-run the scoring.",
        "focus": "data correction",
    },
}

# Recommendation engine, part 2: how urgently and how intensively to act.
RISK_TIER_PLAYBOOK = {
    "Critical": {
        "priority": "P1 - Immediate",
        "action": "Personal retention call; review recent service issues and discuss suitable contract options",
    },
    "High": {
        "priority": "P2 - Targeted",
        "action": "Proactive outreach with a personalised retention offer matched to the customer's services",
    },
    "Medium": {
        "priority": "P3 - Engage",
        "action": "Engagement campaign with service-usage tips and a relevant add-on or plan offer",
    },
    "Low": {
        "priority": "P4 - Nurture",
        "action": "Loyalty and referral programme with a routine satisfaction survey",
    },
    NOT_SCORED: {
        "priority": "Review",
        "action": "Complete the missing account data before contacting the customer",
    },
}

CUSTOMER_FEATURE_COLUMNS = [
    "tenure",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "OnlineSecurity",
    "TechSupport",
    "addon_services",
]


# ---------------------------------------------------------------------------
# 1. Input validation and cleaning
# ---------------------------------------------------------------------------

def _issue(record: str, field: str, value: object, handling: str) -> dict:
    return {"record": record, "field": field, "value": "" if pd.isna(value) else str(value), "handling": handling}


def prepare_customers(raw: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and standardise customer records before segmentation.

    Returns the usable customers and a log of every issue found. Records
    without a customer ID and repeated customer IDs are dropped; invalid
    values are set to missing so they can never add risk points.
    """
    issues: list[dict] = []
    frame = pd.DataFrame() if raw is None else raw.copy()

    expected = ["customerID", *NUMERIC_FIELDS, *CATEGORY_DOMAINS]
    for column in expected:
        if column not in frame.columns:
            frame[column] = pd.Series(np.nan, index=frame.index, dtype="object")
            if len(frame):
                issues.append(_issue("all rows", column, None, "Column absent; treated as missing"))

    # Customer ID: needed to act on a recommendation, so rows without one are dropped.
    ids = frame["customerID"].astype("string").str.strip()
    frame["customerID"] = ids.mask(ids.eq(""))
    missing_id = frame["customerID"].isna()
    for position in np.flatnonzero(missing_id.to_numpy()):
        issues.append(_issue(f"row {position}", "customerID", None, "Dropped: customer ID missing"))
    frame = frame.loc[~missing_id]

    duplicate_id = frame["customerID"].duplicated(keep="first")
    for customer_id in frame.loc[duplicate_id, "customerID"]:
        issues.append(_issue(customer_id, "customerID", customer_id, "Dropped: repeated customer ID (first record kept)"))
    frame = frame.loc[~duplicate_id].reset_index(drop=True)

    notes: dict[int, list[str]] = {}

    def note(index: int, text: str) -> None:
        notes.setdefault(index, []).append(text)

    # Numeric fields: text that is not a number, and negative values, become missing.
    for column in NUMERIC_FIELDS:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        blank = original.astype("string").str.strip().fillna("").eq("")
        unparseable = numeric.isna() & ~blank
        negative = numeric.lt(0)
        for index in frame.index[unparseable | negative]:
            reason = "not a number" if unparseable[index] else "negative value"
            issues.append(_issue(frame.at[index, "customerID"], column, original[index], f"Set to missing ({reason})"))
            note(index, f"{column} invalid")
        frame[column] = numeric.mask(negative).astype("float64")

    monthly = frame["MonthlyCharges"]
    for index in frame.index[monthly.eq(0)]:
        issues.append(_issue(frame.at[index, "customerID"], "MonthlyCharges", 0, "Kept: zero monthly charge, verify billing"))
        note(index, "zero monthly charge")
    for index in frame.index[monthly.gt(MAX_PLAUSIBLE_MONTHLY_CHARGE)]:
        issues.append(_issue(
            frame.at[index, "customerID"], "MonthlyCharges", monthly[index],
            f"Kept: above plausible maximum of {MAX_PLAUSIBLE_MONTHLY_CHARGE:.0f}, verify billing",
        ))
        note(index, "monthly charge above plausible range")

    # Categorical fields: match known labels ignoring case and surrounding
    # spaces; anything else becomes missing rather than a guessed category.
    for column, allowed in CATEGORY_DOMAINS.items():
        lookup = {label.casefold(): label for label in allowed}
        text = frame[column].astype("string").str.strip()
        text = text.mask(text.eq(""))
        canonical = text.str.casefold().map(lookup)
        unexpected = text.notna() & canonical.isna()
        for index in frame.index[unexpected]:
            issues.append(_issue(frame.at[index, "customerID"], column, text[index], "Set to missing (unexpected category)"))
            note(index, f"{column} unexpected value")
        frame[column] = canonical.astype("object")

    frame["data_quality_note"] = [
        "; ".join(notes.get(index, [])) for index in frame.index
    ]
    issue_log = pd.DataFrame(issues, columns=["record", "field", "value", "handling"])
    return frame, issue_log


def has_sufficient_data(customers: pd.DataFrame) -> pd.Series:
    """True when a customer can be segmented and scored without guessing."""
    required_present = customers[list(REQUIRED_FIELDS)].notna().all(axis=1)
    missing_inputs = customers[list(RISK_INPUT_FIELDS)].isna().sum(axis=1)
    return required_present & missing_inputs.le(MAX_MISSING_RISK_INPUTS)


# ---------------------------------------------------------------------------
# 2. Segmentation
# ---------------------------------------------------------------------------

def count_addon_services(customers: pd.DataFrame) -> pd.Series:
    """Number of the six optional add-on services the customer has (0-6)."""
    return customers[ADDON_SERVICES].eq("Yes").sum(axis=1).astype("int64")


def assign_segments(customers: pd.DataFrame) -> pd.Series:
    """Assign each customer to exactly one segment using ordered business rules."""
    sufficient = has_sufficient_data(customers)
    tenure = customers["tenure"]
    established = tenure.gt(EARLY_TENURE_MONTHS)
    conditions = [
        ~sufficient,
        tenure.le(EARLY_TENURE_MONTHS),
        established & customers["MonthlyCharges"].ge(HIGH_MONTHLY_CHARGE),
        established & customers["InternetService"].eq("No"),
        established & customers["Contract"].isin(["One year", "Two year"]),
        established & customers["Contract"].eq("Month-to-month"),
    ]
    choices = [REVIEW_SEGMENT, *SEGMENT_ORDER]
    masks = [pd.Series(c, index=customers.index).fillna(False).to_numpy(dtype=bool) for c in conditions]
    segments = np.select(masks, choices, default=REVIEW_SEGMENT)
    return pd.Series(segments, index=customers.index, name="segment", dtype="object")


# ---------------------------------------------------------------------------
# 3. Churn-risk scoring
# ---------------------------------------------------------------------------

def compute_risk_indicators(customers: pd.DataFrame) -> pd.DataFrame:
    """Return one 0/1 column per risk indicator. Unknown inputs never add points."""
    internet = customers["InternetService"].isin(["DSL", "Fiber optic"])
    flags = pd.DataFrame(
        {
            "month_to_month_contract": customers["Contract"].eq("Month-to-month"),
            "early_tenure": customers["tenure"].le(EARLY_TENURE_MONTHS),
            "electronic_check_payment": customers["PaymentMethod"].eq("Electronic check"),
            "fiber_optic_service": customers["InternetService"].eq("Fiber optic"),
            "no_online_security": internet & customers["OnlineSecurity"].eq("No"),
            "no_tech_support": internet & customers["TechSupport"].eq("No"),
            "high_monthly_charges": customers["MonthlyCharges"].ge(HIGH_MONTHLY_CHARGE),
            "paperless_billing": customers["PaperlessBilling"].eq("Yes"),
        },
        index=customers.index,
    )
    return flags.fillna(False).astype("int64")[list(RISK_INDICATORS)]


def split_train_test(customers: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """Stratified, seeded split of customers with a known churn status."""
    labelled = customers.loc[customers["Churn"].notna() & has_sufficient_data(customers)]
    train, test = train_test_split(
        labelled.index,
        test_size=TEST_SHARE,
        random_state=RANDOM_SEED,
        stratify=labelled["Churn"],
    )
    return pd.Index(sorted(train)), pd.Index(sorted(test))


def fit_risk_weights(indicators: pd.DataFrame, churned: pd.Series) -> pd.DataFrame:
    """Set indicator points from churn-rate lift measured on TRAINING data only.

    lift = churn rate when the flag is present - churn rate when it is absent
    (percentage points). Indicators below MIN_INDICATOR_LIFT_PP earn 0 points;
    the rest share 100 points in proportion to their lift.
    """
    churned = churned.astype(float)
    rows = []
    for name in indicators.columns:
        flagged = indicators[name].eq(1)
        rate_on = churned[flagged].mean() * 100 if flagged.any() else np.nan
        rate_off = churned[~flagged].mean() * 100 if (~flagged).any() else np.nan
        rows.append({
            "indicator": name,
            "definition": RISK_INDICATORS[name],
            "train_customers_flagged": int(flagged.sum()),
            "train_churn_rate_flagged_pct": round(rate_on, 2),
            "train_churn_rate_not_flagged_pct": round(rate_off, 2),
            "lift_pp": round(rate_on - rate_off, 2),
        })
    weights = pd.DataFrame(rows)
    eligible = weights["lift_pp"].ge(MIN_INDICATOR_LIFT_PP)
    if not eligible.any():
        raise ValueError("No indicator reaches the minimum lift; cannot build a score.")
    share = weights["lift_pp"].where(eligible, 0) / weights.loc[eligible, "lift_pp"].sum()
    weights["points"] = (share * 100).round().astype("int64")
    return weights


def fit_scorecard(customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index, pd.Index]:
    """Split prepared customers, then fit indicator points on the training part only."""
    train_index, test_index = split_train_test(customers)
    train = customers.loc[train_index]
    weights = fit_risk_weights(compute_risk_indicators(train), train["Churn"].eq("Yes"))
    return weights, train_index, test_index


def categorize_risk(scores: pd.Series) -> pd.Series:
    """Map 0-100 scores to Low / Medium / High / Critical; missing -> Not Scored."""
    bins = [*RISK_THRESHOLDS, np.inf]
    categories = pd.cut(scores, bins=bins, labels=RISK_LABELS, right=False)
    return categories.astype("object").where(categories.notna(), NOT_SCORED).rename("risk_category")


def score_customers(indicators: pd.DataFrame, weights: pd.DataFrame, sufficient: pd.Series) -> pd.DataFrame:
    """Add up indicator points and normalise to a 0-100 risk score."""
    points = weights.set_index("indicator")["points"]
    max_points = int(points.sum())
    if max_points <= 0:
        raise ValueError("Risk weights must contain at least one positive point value.")
    raw_points = indicators[points.index].mul(points, axis=1).sum(axis=1)
    score = (raw_points / max_points * 100).round(1).clip(0, 100)
    score = score.where(sufficient)
    return pd.DataFrame({
        "raw_risk_points": raw_points.where(sufficient),
        "risk_score": score,
        "risk_category": categorize_risk(score),
    })


# ---------------------------------------------------------------------------
# 4. Recommendation engine
# ---------------------------------------------------------------------------

def recommend(segment: str, risk_category: str) -> tuple[str, str]:
    """Combine urgency (risk tier) with content (segment) into one action."""
    tier = RISK_TIER_PLAYBOOK.get(risk_category, RISK_TIER_PLAYBOOK[NOT_SCORED])
    playbook = SEGMENT_PLAYBOOK.get(segment, SEGMENT_PLAYBOOK[REVIEW_SEGMENT])
    if segment == REVIEW_SEGMENT or risk_category == NOT_SCORED:
        return RISK_TIER_PLAYBOOK[NOT_SCORED]["priority"], SEGMENT_PLAYBOOK[REVIEW_SEGMENT]["action"]
    return tier["priority"], f"{tier['action']}. Focus: {playbook['focus']}."


# ---------------------------------------------------------------------------
# 5. End-to-end pipeline and summaries
# ---------------------------------------------------------------------------

def run_pipeline(raw: pd.DataFrame | None, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate, segment, score and recommend for every usable customer."""
    customers, issues = prepare_customers(raw)
    customers["addon_services"] = count_addon_services(customers)
    customers["segment"] = assign_segments(customers)
    customers["missing_risk_inputs"] = customers[list(RISK_INPUT_FIELDS)].isna().sum(axis=1).astype("int64")

    indicators = compute_risk_indicators(customers)
    scores = score_customers(indicators, weights, has_sufficient_data(customers))
    scored = pd.concat([customers, indicators, scores], axis=1)

    recommendations = [
        recommend(segment, category)
        for segment, category in zip(scored["segment"], scored["risk_category"])
    ]
    scored["retention_priority"] = [priority for priority, _ in recommendations]
    scored["recommended_action"] = [action for _, action in recommendations]
    return scored, issues


def churn_rate_pct(churn: pd.Series) -> float:
    """Historical churn rate among customers with a known status (NaN if none)."""
    known = churn.dropna()
    return float(known.eq("Yes").mean() * 100) if len(known) else float("nan")


def describe_segment(group: pd.DataFrame) -> str:
    """Short, data-driven profile used as a segment's main characteristics."""
    if group.empty:
        return "No customers"

    def share(mask: pd.Series) -> str:
        return f"{mask.mean() * 100:.0f}%"

    return (
        f"Median tenure {group['tenure'].median():.0f} months; "
        f"{share(group['Contract'].eq('Month-to-month'))} month-to-month; "
        f"{share(group['InternetService'].eq('Fiber optic'))} fiber optic; "
        f"{share(group['PaymentMethod'].eq('Electronic check'))} electronic check; "
        f"average {group['addon_services'].mean():.1f} add-on services"
    )


def build_segment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per segment with size, value, churn, risk and recommended action."""
    columns = [
        "segment", "customer_count", "customer_pct", "avg_tenure_months",
        "avg_monthly_charges", "churn_rate_pct", "avg_risk_score",
        "high_or_critical_pct", "main_characteristics", "business_interpretation",
        "recommended_action",
    ]
    total = len(scored)
    if total == 0:
        return pd.DataFrame(columns=columns)

    order = SEGMENT_ORDER + ([REVIEW_SEGMENT] if scored["segment"].eq(REVIEW_SEGMENT).any() else [])
    rows = []
    for name in order:
        group = scored.loc[scored["segment"].eq(name)]
        count = len(group)
        scored_group = group["risk_score"].dropna()
        rows.append({
            "segment": name,
            "customer_count": count,
            "customer_pct": round(count / total * 100, 2),
            "avg_tenure_months": round(group["tenure"].mean(), 1) if count else np.nan,
            "avg_monthly_charges": round(group["MonthlyCharges"].mean(), 2) if count else np.nan,
            "churn_rate_pct": round(churn_rate_pct(group["Churn"]), 2),
            "avg_risk_score": round(scored_group.mean(), 1) if len(scored_group) else np.nan,
            "high_or_critical_pct": (
                round(group["risk_category"].isin(["High", "Critical"]).mean() * 100, 2) if count else np.nan
            ),
            "main_characteristics": describe_segment(group),
            "business_interpretation": SEGMENT_PLAYBOOK[name]["interpretation"],
            "recommended_action": SEGMENT_PLAYBOOK[name]["action"],
        })
    return pd.DataFrame(rows, columns=columns)


def build_priority_list(scored: pd.DataFrame) -> pd.DataFrame:
    """Active (not churned) High/Critical customers, highest risk and spend first."""
    active = scored.loc[scored["Churn"].eq("No") & scored["risk_category"].isin(["Critical", "High"])]
    ranked = active.sort_values(
        ["risk_score", "MonthlyCharges", "customerID"], ascending=[False, False, True]
    ).copy()
    ranked.insert(0, "priority_rank", np.arange(1, len(ranked) + 1))
    return ranked[[
        "priority_rank", "customerID", "segment", "risk_score", "risk_category",
        "retention_priority", "tenure", "Contract", "MonthlyCharges", "recommended_action",
    ]].reset_index(drop=True)
