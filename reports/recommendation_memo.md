# Customer Segmentation and Churn Risk Recommendation Memo

**To:** Customer Retention and Marketing Leads
**From:** Poorvansh, Data Analytics Intern (EdVyro Task 04)
**Date:** 19 November 2026
**Supporting analysis:** [`notebooks/04_segmentation_recommendation.ipynb`](../notebooks/04_segmentation_recommendation.ipynb) · outputs in [`outputs/`](../outputs/)

> **Dataset notice:** All figures come from the public IBM Telco Customer Churn sample (7,043 customers), used because the EdVyro resource files were unavailable. They describe that sample, not EdVyro or any current business. Monetary values are the dataset's reported charge units; no currency is assumed.

---

## 1. Executive Summary

- The customer base splits into **five explainable segments**. **New Customers** (first 12 months) are the largest segment, at 31.0% of customers, and they had the highest historical churn rate (47.4%).
- A transparent **0–100 churn-risk score** built from eight account and service indicators ranks customers well on data it had not seen. Held-out ROC AUC was **0.826**, against 0.831 for a fitted logistic-regression benchmark. Historical churn rises at every step from Low (3.8%) to Critical (66.5%).
- On the held-out test customers, the **High and Critical tiers contain 34.5% of customers but captured 70.8% of churners**.
- **1,090 active customers** (not churned) are in the High or Critical tiers. They account for **83,087.30 in monthly charges**, which is 26.2% of all active customers' monthly charges. They form the recommended retention call list (`outputs/priority_retention_list.csv`).
- **Recommendation:** focus retention effort on three groups:
  1. structured onboarding for first-year customers;
  2. contract-option conversations for high-spend customers on month-to-month terms;
  3. commitment and add-on offers for established month-to-month customers.

  Protect the stable Loyal Contract and Phone-Only groups with low-cost programmes. Test every offer against a control group before claiming results.

## 2. Business Objective

Move from describing churn to **prioritising customers**. The retention team needs to know who is most likely to leave, why, and what kind of action fits each customer. The model must be explainable to non-technical colleagues, reproducible on every run and robust to bad data.

## 3. Dataset Overview

| Item | Value |
| --- | --- |
| Customers (one row each) | 7,043 |
| Source columns | 21 (account, services, demographics, churn status) |
| Historical churn | 1,869 customers (26.54%) |
| Data quality | No missing values, duplicates or out-of-range categories after the documented cleaning step. The 11 zero-tenure customers keep `TotalCharges = 0`. |
| Time dimension | None. The file is a single snapshot with no dates. |

## 4. Segmentation Methodology

Segments describe **who the customer is**. They are assigned with five ordered business rules, and the first rule that matches decides the segment. Churn status is **never** used to assign a segment. Thresholds come from the contract calendar (12 months) and the charge distribution (90 is about the 75th percentile of monthly charges, which is 89.85).

| Priority | Segment | Rule |
| --- | --- | --- |
| 1 | New Customers | Tenure of 12 months or less |
| 2 | High-Value Customers | Tenure above 12 months and monthly charges of 90 or more |
| 3 | Phone-Only Basic Customers | Tenure above 12 months and no internet service |
| 4 | Loyal Contract Customers | Tenure above 12 months on a one- or two-year contract |
| 5 | Flexible Month-to-Month Customers | Tenure above 12 months on a month-to-month contract |

Customers with missing tenure or contract information would go to a separate *Needs Data Review* group. None exist in this dataset.

**Clustering was evaluated and not adopted.** K-Means on standardised tenure, charges, add-on count and contract length reached a best silhouette score of only 0.403 (k = 3), which means the groups overlap. Its clusters largely reproduced the tenure, spend and contract split already in the rules, but they cannot be explained to a customer-facing team and they move whenever the data is refreshed.

"At-Risk" and "High-Risk" are deliberately expressed as **risk tiers** rather than segments, so that customer type and churn likelihood are not mixed together.

## 5. Customer Segment Findings

| Segment | Customers | Share | Avg tenure (months) | Avg monthly charges | Historical churn | Avg risk score | High/Critical share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| New Customers | 2,186 | 31.0% | 4.7 | 56.10 | 47.4% | 65.8 | 64.5% |
| High-Value Customers | 1,525 | 21.7% | 50.3 | 101.62 | 26.7% | 52.1 | 36.2% |
| Phone-Only Basic Customers | 1,013 | 14.4% | 43.7 | 21.44 | 2.1% | 5.9 | 0.0% |
| Loyal Contract Customers | 1,221 | 17.3% | 51.8 | 67.59 | 5.7% | 18.2 | 0.7% |
| Flexible Month-to-Month Customers | 1,098 | 15.6% | 30.4 | 67.64 | 30.5% | 56.9 | 42.4% |
| **All customers** | **7,043** | **100%** | | **64.76** | **26.5%** | | |

- **New Customers:** 91% are on month-to-month terms and 45% pay by electronic check. They average 1.0 add-on service. The churn rate is close to half.
- **High-Value Customers:** 97% have fiber optic and they average 3.9 add-ons. Their churn rate (26.7%) matches the overall average, but the segment contains two very different groups (see Section 8).
- **Phone-Only Basic Customers:** the lowest spend and the lowest churn (2.1%).
- **Loyal Contract Customers:** long tenure (median 56 months), all on one- or two-year contracts, and 5.7% churn.
- **Flexible Month-to-Month Customers:** past their first year but uncommitted. Their churn (30.5%) is above the overall average despite a median tenure of 27 months.

## 6. Churn Risk Scoring Methodology

The score shows **how likely a customer is to leave, relative to other customers**. It is a **rule-based points scorecard, not a trained predictive model**, and it never takes churn status as an input.

1. Customers were split 70/30 into training (4,930) and test (2,113) sets, stratified by churn with a fixed seed (42).
2. For each indicator, *lift* was measured on the **training set only**: the churn rate with the indicator minus the churn rate without it.
3. Indicators with at least 5 points of lift share 100 points in proportion to their lift.
4. `risk_score = points earned ÷ maximum points × 100`, which always falls between 0 and 100.
5. Risk categories: **Low 0–29 · Medium 30–59 · High 60–79 · Critical 80–100**.

| Indicator | Training lift (pp) | Points |
| --- | ---: | ---: |
| Month-to-month contract | 36.17 | 17 |
| Tenure of 12 months or less | 30.74 | 15 |
| Electronic check payment | 28.77 | 14 |
| No Online Security (internet customers) | 30.66 | 14 |
| No Tech Support (internet customers) | 30.14 | 14 |
| Fiber optic internet | 27.43 | 13 |
| Paperless billing | 18.34 | 9 |
| Monthly charges of 90 or more | 9.42 | 4 |
| **Total** | | **100** |

Demographic fields (gender, senior-citizen status, partner, dependents) are excluded on purpose. They cannot be acted on, and using them could lead to unfair treatment of customer groups.

**Validation on the held-out test set**

| Risk category | Test customers | Historical churn rate |
| --- | ---: | ---: |
| Low | 731 | 3.8% |
| Medium | 653 | 20.8% |
| High | 362 | 42.3% |
| Critical | 367 | 66.5% |

The ROC AUC was 0.836 on training and 0.826 on test, so the weights are not over-fitted. A logistic regression on the same eight indicators reached 0.831 on test, so the simpler scorecard gives up very little ranking quality.

## 7. High-Risk Customer Analysis

Across all 7,043 customers, 1,167 (16.6%) are High and 1,269 (18.0%) are Critical. The actionable group is the **1,090 customers who have not churned** and score High (664) or Critical (426):

| Segment | Active High/Critical customers | Monthly charges |
| --- | ---: | ---: |
| New Customers | 539 | 34,605.25 |
| High-Value Customers | 300 | 29,465.35 |
| Flexible Month-to-Month Customers | 245 | 18,505.70 |
| Loyal Contract Customers | 6 | 511.00 |
| Phone-Only Basic Customers | 0 | 0.00 |
| **Total** | **1,090** | **83,087.30** |

Common traits of this list:
- 94.4% are on month-to-month contracts.
- 90.1% have internet without Online Security, and 90.0% have internet without Tech Support.
- 71.7% are fiber optic customers.
- 66.0% pay by electronic check.
- 49.4% are in their first year.
- Their average monthly charge is 76.23.

## 8. Key Business Insights

1. **Risk is concentrated and identifiable.** About a third of customers (34.6%) fall in High or Critical tiers, and on unseen customers those tiers contained 70.8% of churners.
2. **The first year is the most exposed period.** New Customers make up 49.4% of the active priority list, and 64.5% of all New Customers score High or Critical.
3. **Contract type splits the High-Value segment.** High-Value customers on month-to-month terms (647) churned at 44.0%, versus 13.9% for those on one- or two-year contracts (878). This is an association: customers who choose contracts may differ in other ways.
4. **Missing protective add-ons travel with risk.** Nine in ten priority customers have internet without Online Security or Tech Support. These are concrete, low-cost things to offer.
5. **High charges alone are a weak signal.** Monthly charges of 90 or more add only 4 of 100 points. Contract, tenure and payment behaviour matter far more.
6. **Large stable groups need little spend.** Phone-Only Basic (2.1% churn) and Loyal Contract (5.7%) customers together make up 31.7% of the base and almost none of the risk.

## 9. Segment-wise Recommendations

Risk tier sets the **urgency**: P1 Critical → personal retention call; P2 High → proactive personalised offer; P3 Medium → engagement campaign; P4 Low → loyalty and referral. Segment sets the **content**:

| Segment | Recommended retention action |
| --- | --- |
| New Customers | **Structured onboarding.** Help with setup and check in during the first months. Offer Online Security and Tech Support trials. Present one- or two-year contract options once the customer is settled. |
| High-Value Customers | **Relationship management.** Provide priority support and run periodic plan-fit reviews so customers do not pay for services they do not use. For month-to-month customers, offer loyalty benefits tied to a longer contract. |
| Flexible Month-to-Month Customers | **Commitment and value building.** Offer contract options with clear benefits, promote automatic payment, and bundle protective add-ons. |
| Loyal Contract Customers | **Renewal management.** Contact customers before their contract ends, recognise their loyalty and invite referrals. |
| Phone-Only Basic Customers | **Low-cost maintenance.** Keep the service reliable, run light-touch satisfaction surveys, and offer internet bundles only where customers show interest. |

Every customer's combined action is in `outputs/customer_segments.csv` (`retention_priority`, `recommended_action`).

## 10. Business Impact Opportunities

These are opportunities to **test**, not forecasts. The data cannot show how many customers an action would retain.

- **Revenue exposure:** the 1,090 active High/Critical customers represent 83,087.30 in monthly charges (26.2% of active monthly charges). Every customer retained from this list protects their monthly charge.
- **Onboarding pilot:** 539 first-year customers are on the priority list (34,605.25 in monthly charges). This is the largest single group to test an onboarding programme on.
- **Contract-conversion pilot:** 362 active High-Value customers are on month-to-month terms (35,823.05 in monthly charges). Of these, 251 are already High or Critical.
- **Add-on bundle test:** offer Online Security / Tech Support trials to priority customers who lack them, and measure the effect against a control group.
- **Efficiency:** routing Low-tier customers (34.8% of the base) to automated loyalty programmes frees agent time for P1/P2 customers.

## 11. Limitations

- **Not EdVyro data.** The IBM sample is a substitute, and its patterns may not transfer to another business.
- **Snapshot without dates.** The score measures how closely a customer resembles past churners. It is not a time-bound forecast, and it has not been tested on future churn.
- **Rule-based, not causal.** The indicators are associations. A customer flagged for electronic-check payment is not leaving *because* of the payment method, so the recommended actions are hypotheses to test.
- **Overlapping indicators.** Weights come from single-indicator lift, so related indicators (fiber service and missing add-ons, for example) partly double-count. The logistic-regression benchmark shows that the ranking cost of this is small.
- **Not a probability.** A score of 80 does not mean an 80% chance of churn. Use the tier churn rates in Section 6 for orientation only.
- **Fixed thresholds.** The 12-month and 90-charge thresholds and the 30/60/80 tier cut-offs are documented business choices. Review them if the customer base or pricing changes.
- **Unusual values are flagged, not corrected.** For new data, zero or implausibly high charges are kept and flagged for review, and records missing key fields are routed to data review (see `reports/edge_case_validation.md`).

## 12. Final Recommendations

1. **Adopt the scorecard for prioritisation.** Work the 1,090-customer priority list in rank order, starting with the 426 Critical customers.
2. **Launch an onboarding programme for first-year customers.** It is the largest at-risk group.
3. **Pilot contract-option and loyalty offers for high-spend month-to-month customers,** using a randomised control group.
4. **Bundle protective add-ons (Online Security, Tech Support) into retention offers,** and test their effect before scaling.
5. **Keep stable segments on low-cost programmes** (renewal reminders, referrals, satisfaction surveys).
6. **Re-run the notebook whenever the data is refreshed.** Re-check the tier churn rates, and revisit the weights if the AUC or the tier ordering degrades.
7. **Collect dated churn events** so a future version can be validated on genuinely future outcomes.
