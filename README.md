# Customer Churn Analysis – Data Quality Audit and Exploratory Data Analysis

> **Dataset substitution notice:** EdVyro resource files were unavailable during this project. The analysis therefore uses the public IBM Telco Customer Churn sample as an alternative. It is **not** the official EdVyro-provided dataset, and no claim of compliance with unavailable EdVyro-specific instructions is made.

An end-to-end, beginner-friendly customer churn audit and EDA project. It includes the unmodified raw data, a cleaned CSV, an executed analysis notebook, a generated data-quality report, chart artifacts, and an interactive Streamlit dashboard.

## Dataset source and verification

- IBM repository: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)
- Direct CSV: [Telco-Customer-Churn.csv](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- The GitHub repository describes its Telco customer churn dataset and includes the CSV at `data/Telco-Customer-Churn.csv`.
- Downloaded source SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

The repository is archived/read-only. Check the source terms and data provenance before reusing it beyond this educational project.

## Objectives

- Inspect the dataset's dimensions, schema, missingness, duplicate records, unique values, invalid values, and potential statistical outliers.
- Preserve the raw download and document every cleaning decision.
- Validate the cleaned data and export a reproducible CSV.
- Describe churn overall and across contract, payment, and tenure segments.
- Examine monthly charges and numerical relationships without making causal claims.
- Provide a filterable dashboard for customer segments and an auditable written report.

## Project structure

```text
customer-churn-analysis/
├── data/
│   ├── raw/Telco-Customer-Churn.csv
│   └── processed/customer_churn_cleaned.csv
├── notebooks/customer_churn_analysis.ipynb
├── reports/data_quality_report.md
├── dashboard/app.py
├── visuals/
│   ├── churn_distribution_notion_agent_chart.html
│   ├── contract_churn_notion_agent_chart.html
│   ├── segment_analysis_notion_agent_chart.html
│   └── chart JSON specifications
├── scripts/analyze_customer_churn.py
├── analysis_evidence.json
├── requirements.txt
└── .gitignore
```

## Results from the downloaded sample

These values are calculated from the included source CSV and recorded in `analysis_evidence.json`:

- **7,043** customer records across **21** source columns.
- **1,869** churned customers; observed churn rate **26.54%**.
- **0** exact duplicate records and **0** repeated customer IDs.
- **11** blank `TotalCharges` values; all 11 belong to zero-tenure rows and are set to `0.0` under the documented cumulative-charge assumption.
- Month-to-month contracts have the highest observed churn rate (**42.71%**, 1,655 of 3,875); two-year contracts have the lowest (**2.83%**, 48 of 1,695).
- Electronic-check payment has the highest observed payment-method churn rate (**45.29%**, 1,071 of 2,365).
- All implemented data validation checks passed.

These are descriptive results for the alternative sample—not estimates for EdVyro or a current telecom provider.

## Methodology

1. Preserve the source CSV in `data/raw/`.
2. Trim string whitespace and interpret blank strings as missing.
3. Convert `TotalCharges` from numeric text to a numeric type.
4. Impute the 11 blank `TotalCharges` entries to zero only after checking that each row has zero tenure. This is a transparent domain assumption, not a value recovered from the source.
5. Do not delete any customer: the source has no exact duplicate rows or duplicate customer IDs, and IQR outliers are plausible customer observations.
6. Check field domains, missing values, key uniqueness, nonnegative charges/tenure, churn labels, and a direct-filter cross-check of the churn count.
7. Use descriptive summaries and within-segment churn rates; avoid causal interpretations.

`scripts/analyze_customer_churn.py` regenerates the cleaned CSV, evidence JSON, report, and chart JSON specifications from the included raw CSV:

```bash
python scripts/analyze_customer_churn.py
```

The generated HTML chart artifacts are checked in under `visuals/`. They were built from the included specifications; regenerating those particular HTML files requires the chart-builder environment used to create them.

## Technologies

- Python, pandas, NumPy
- Jupyter Notebook
- Matplotlib and Seaborn (listed for learning and extension; the included chart artifacts are delivered as HTML)
- Streamlit
- Git and GitHub

## Installation and local run

Use Python 3.10 or later. From the project root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and run the dashboard:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

To execute the notebook:

```bash
jupyter notebook notebooks/customer_churn_analysis.ipynb
```

The dashboard's sidebar filters update the customer KPIs, segment tables, customer preview, and descriptive business insights. The embedded HTML charts intentionally remain a full-sample reference so the chart context is stable while exploring filtered cohorts.

## GitHub publication

No GitHub repository has been created or pushed from this environment. To publish:

1. Create a **public** repository on GitHub, for example `customer-churn-analysis`.
2. From this folder, initialize and commit the project:

   ```bash
   git init
   git add .
   git commit -m "Add customer churn data quality and EDA project"
   git branch -M main
   git remote add origin https://github.com/<your-username>/customer-churn-analysis.git
   git push -u origin main
   ```

3. Replace the repository placeholder below after verifying that the public repository is accessible.

**GitHub repository:** Not created yet. Add the verified URL here after publication.

## Streamlit Community Cloud deployment

The dashboard is prepared for Streamlit Community Cloud, but it has **not** been deployed and no live URL is available yet.

1. Push the project to a public GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **Create app**, choose the repository and branch (`main`), then set the main file path to `dashboard/app.py`.
4. Deploy, wait for the app to build, and test the public URL in a private/incognito browser window.
5. Add the live URL here only after it loads successfully.

**Live dashboard:** Not deployed yet.

## Screenshots

No deployed screenshot is included because the dashboard has not been published. After starting the app locally or deploying it, capture the full dashboard and add an image such as `visuals/dashboard_screenshot.png` here. The three source-backed HTML charts are already included in `visuals/` and are rendered by the dashboard.

## Limitations

- This is a substitute IBM dataset, not the unavailable EdVyro-provided resource.
- The CSV is a static sample with no extraction date or observation window.
- Churn-rate differences are descriptive and do not establish causation.
- The 11 zero-tenure `TotalCharges` blanks are imputed using a documented business assumption.
- IQR outliers are retained; statistical extremeness does not establish an error.
- Charge units are used as provided; this project does not infer a currency.