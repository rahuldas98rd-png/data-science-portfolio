# Project 01 — Telco Customer Churn (EDA + Supervised Classification)

## Why this project first

Telco churn is the canonical "first real ML project." It hits every fundamental skill you'll use for the rest of your career:
- Mixed data types (numeric + categorical)
- A real missing-value trap (a numeric column stored as object)
- Class imbalance (~26% churn)
- A clear business question ("who will leave us?")
- Multiple classifiers can be benchmarked
- Threshold tuning matters (precision vs recall trade-off)

You'll do **EDA first** and submit your answers. Only after that do we move to feature engineering and modeling.

---

## 1. Dataset Source

**Name:** Telco Customer Churn (IBM Sample Data Set)
**Primary source:** Kaggle — `blastchar/telco-customer-churn`
**URL:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
**Size:** ~7,043 rows × 21 columns (~1 MB) — fits in memory easily

**How to download:**
- Option A (manual): Sign in to Kaggle → Download CSV → place in `data/raw/`.
- Option B (CLI): `pip install kaggle`, set up `~/.kaggle/kaggle.json` API token, then:
  ```
  kaggle datasets download -d blastchar/telco-customer-churn -p data/raw/ --unzip
  ```

---

## 2. Environment Setup

### Install (one-time)
1. **Miniconda** — https://docs.conda.io/en/latest/miniconda.html (lighter than full Anaconda)
2. **Git** — https://git-scm.com/download/win
3. **VS Code** — https://code.visualstudio.com/
4. **VS Code extensions:** Python, Pylance, Jupyter, GitLens, Black Formatter, Rainbow CSV, Data Wrangler

### Create the project environment
```bash
conda create -n ds-portfolio python=3.11 -y
conda activate ds-portfolio
pip install -r requirements.txt
```

### `requirements.txt` (starter)
```
numpy
pandas
matplotlib
seaborn
plotly
scikit-learn
scipy
statsmodels
jupyter
jupyterlab
ipykernel
missingno
phik
xgboost
lightgbm
shap
joblib
black
ruff
```

Register the kernel so Jupyter inside VS Code sees it:
```bash
python -m ipykernel install --user --name ds-portfolio --display-name "Python (ds-portfolio)"
```

---

## 3. Repository Structure

Create a **single public repo** called `data-science-portfolio` on GitHub. Each project lives in its own folder.

```
data-science-portfolio/
├── README.md                   # Portfolio landing page — index of all projects
├── LICENSE                     # MIT is a safe default
├── .gitignore                  # Python + Jupyter + data ignores
├── requirements.txt            # Shared deps across projects
├── environment.yml             # Optional: conda env spec
│
├── projects/
│   └── 01-telco-churn/
│       ├── README.md           # Problem, approach, results, key findings
│       ├── data/
│       │   ├── raw/            # Original CSV (gitignored if large)
│       │   ├── interim/        # Cleaned, not yet feature-engineered
│       │   └── processed/      # Final model-ready data
│       ├── notebooks/
│       │   ├── 01_eda.ipynb
│       │   ├── 02_preprocessing.ipynb
│       │   ├── 03_modeling.ipynb
│       │   └── 04_evaluation.ipynb
│       ├── src/
│       │   ├── __init__.py
│       │   ├── data.py         # Load/clean functions
│       │   ├── features.py     # Feature engineering
│       │   ├── models.py       # Train/predict
│       │   └── evaluate.py     # Metrics, plots
│       ├── reports/
│       │   ├── figures/        # Saved PNG/SVG plots
│       │   └── findings.md     # Your written EDA answers
│       └── models/             # Saved .pkl/.joblib (gitignored if heavy)
│
├── notebooks_scratch/          # Throwaway experiments (gitignored)
└── docs/
    └── learning_roadmap.md     # Full project sequence (below)
```

### `.gitignore` essentials
```
# Python
__pycache__/
*.py[cod]
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# Data & models (commit small samples only)
data/raw/*
data/interim/*
data/processed/*
!data/**/.gitkeep
models/*.pkl
models/*.joblib

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Secrets
.env
*.key
.kaggle/
```

Add empty `.gitkeep` files in each `data/` subfolder so the structure is preserved on clone.

---

## 4. Full Learning Roadmap (so you see where this is heading)

| # | Topic Area | Project | Dataset |
|---|---|---|---|
| 01 | Supervised — Classification | **Telco Churn** | IBM Telco |
| 02 | Supervised — Regression | House Price Prediction | Ames Housing (Kaggle) |
| 03 | Unsupervised — Clustering & RFM | Customer Segmentation | Online Retail II (UCI) |
| 04 | Time Series Forecasting | Retail Sales / Energy Demand | Rossmann or PJM Hourly Energy |
| 05 | NLP — Text Classification | Disaster Tweets / IMDB Sentiment | Kaggle NLP Disaster Tweets |
| 06 | NLP — Transformers + Embeddings | Semantic Search / Q&A | arXiv or news corpus |
| 07 | Deep Learning — CV | Image Classification | Intel Image Classification / CIFAR-10 |
| 08 | Deep Learning — Tabular | Credit Default | Lending Club / Give Me Some Credit |
| 09 | Recommender Systems | Movie / Book Recommender | MovieLens 25M |
| 10 | Reinforcement Learning | CartPole → LunarLander → Custom env | OpenAI Gymnasium |
| 11 | Data Engineering | ETL pipeline + DuckDB + dbt | NYC Taxi (public S3) |
| 12 | MLOps Capstone | Re-deploy Project 01 with FastAPI + Docker + MLflow | Telco |

---

## 5. EDA Questions — Project 01

**Submit your answers** as `projects/01-telco-churn/reports/findings.md` plus the notebook `notebooks/01_eda.ipynb`. Show code + output + a 1–3 sentence written interpretation for each.

### Part A — Data Understanding
1. What is the shape of the dataset? How many features, how many records?
2. What are the data types of each column? Identify **at least one column whose dtype is wrong** and explain how you know.
3. Are there missing values? Where, and how many? (Hint: missing values may be hiding as whitespace strings, not NaN.)
4. Are there duplicate rows? Duplicate `customerID`s?

### Part B — Target Variable
5. What is the overall churn rate? Is the dataset balanced or imbalanced?
6. Plot the churn distribution. Will accuracy be a meaningful metric here? Why or why not?

### Part C — Numerical Features
7. Compute summary statistics (mean, median, std, min, max, quartiles) for `tenure`, `MonthlyCharges`, `TotalCharges`.
8. Plot the distributions (histogram + KDE). Comment on shape (skew, modality).
9. Plot box plots of each numeric feature **split by churn**. What stands out?
10. Is there a relationship between `tenure` and `TotalCharges`? Plot it. Why does this make business sense?

### Part D — Categorical Features
11. List every categorical feature and its unique values.
12. Compute churn rate **per category** for: `Contract`, `PaymentMethod`, `InternetService`, `OnlineSecurity`, `TechSupport`, `PaperlessBilling`.
13. Which 3 categorical features show the **strongest visual association** with churn? Justify with plots (stacked bars or grouped bars).

### Part E — Relationships & Multicollinearity
14. Correlation matrix of numeric features — any strong correlations? Implications for linear models?
15. Use Cramér's V or the `phik` library to measure association strength between **all features and the target**. Rank top 10.

### Part F — Business Questions
16. Average tenure of churned vs retained customers — what's the gap?
17. Among month-to-month contract customers, what % churn? Compare to 1-year and 2-year contracts.
18. Senior citizens — do they churn more than non-seniors? Quantify.
19. Define a "high-risk segment" using 2–3 features and report its size + churn rate. (Example: month-to-month + fiber optic + electronic check.)
20. If the company could only intervene on **1,000 customers**, which segment would you target and why?

### Part G — Synthesis (write as prose, not code)
21. List 5 hypotheses about what drives churn, ordered by your confidence.
22. List 5+ features you'd **engineer** before modeling and explain the intuition (e.g., `tenure_group`, `services_count`, `avg_monthly_spend`, `is_new_customer`, interaction features).
23. Two data quality issues you'd flag to the business team.

---

## 6. Submission Format

When done, commit and push:
```
projects/01-telco-churn/
├── notebooks/01_eda.ipynb          ← clean, run top-to-bottom, all outputs saved
├── reports/findings.md             ← prose answers to all 23 questions
└── reports/figures/                ← saved plots referenced in findings.md
```

Then tell me **"Project 01 EDA done — please review"** and paste the GitHub link. I will:
- Audit the notebook (code quality, plot choice, missed insights)
- Quiz you on 3–4 follow-up questions
- Then we move to **Phase 2: Preprocessing + Modeling** (feature engineering, encoding, pipelines, baseline → tree-based → boosted models, threshold tuning, SHAP interpretation).

---

## 7. Ground Rules

- **No copy-paste from Kaggle notebooks** until *after* you submit. Read others' work afterwards as a cross-check.
- **Comment your code** — future-you and recruiters will read it.
- **Markdown cells in notebooks** — every section needs a header and a one-line "why this step."
- **Commit often** with meaningful messages: `feat(eda): add tenure distribution analysis`, not `update`.
- If you get stuck for >30 minutes on one question, ask. Don't grind silently.

Good luck. Ping me when you're done with EDA.
