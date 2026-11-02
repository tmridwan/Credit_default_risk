# 🏦 Home Credit Default Risk — Data Analytics Project

> **Predicting loan repayment difficulty for underserved borrowers using multi-table feature engineering and gradient boosting ensembles.**

[![Tableau Public](https://img.shields.io/badge/Tableau-Dashboard-blue?logo=tableau)](https://public.tableau.com)
[![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Home%20Credit-20BEFF?logo=kaggle)](https://www.kaggle.com/c/home-credit-default-risk)

---

## 📌 Project Overview

Many people struggle to get loans because they lack sufficient credit history. Home Credit aims to broaden financial inclusion by using alternative data to predict whether an applicant will have repayment difficulty.

This project builds a complete end-to-end data analytics pipeline on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset, from raw multi-table ingestion through feature engineering, exploratory data analysis, model training, and interactive Tableau dashboard visualization.

**Best model:** 3-model blend (LightGBM + CatBoost + XGBoost) — **ROC-AUC 0.7669**

---

## 📊 Tableau Dashboard

Three-page interactive dashboard built in Tableau Public:

| Page | Focus | Charts |
|------|-------|--------|
| **Portfolio Overview** | Loan volume, credit distribution, income segments | Credit Amount by Contract Type · Loan Volume by Purpose · Income Group Distribution · Credit-Income Ratio by Risk |
| **Risk Trends** | Default rate patterns across demographics and time | Default Rate by Gender × Contract Type · Age Group Risk Profile · Default Rate by Bureau Credit History · Quarterly Default Trends |
| **Model Performance** | ML evaluation metrics | ROC Curve · Precision-Recall Curve · Calibration Chart · Feature Importance · Confusion Matrix |

> 🔗 **[View Dashboard on Tableau Public]([#](https://public.tableau.com/views/DAproject_17903669641940/RiskTrends?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link))**

---

## 🗂️ Dataset

Source: [Kaggle — Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)

| File | Rows | Columns | Role |
|------|------|---------|------|
| `application_train.csv` | 307,511 | 122 | Primary labeled applicant table (TARGET present) |
| `application_test.csv` | 48,744 | 121 | Kaggle test set (TARGET absent) |
| `bureau.csv` | 1,716,428 | 17 | Previous external credit records |
| `bureau_balance.csv` | 27,299,925 | 3 | Monthly bureau status history |
| `previous_application.csv` | 1,670,214 | 37 | Prior Home Credit applications |
| `installments_payments.csv` | 13,605,401 | 8 | Installment repayment history |

**Target variable:** `TARGET` — 1 = repayment difficulty (8.07%), 0 = no difficulty (91.93%)

---

## 🔧 Feature Engineering

Starting from 122 raw columns, the pipeline adds **+40 engineered features** through multi-table aggregation:

### Application-Level (5 features)
| Feature | Description |
|---------|-------------|
| `INCOME_GROUP` | Income quintile (pd.qcut) |
| `CREDIT_GROUP` | Credit amount quintile |
| `CREDIT_INCOME_RATIO` | Loan size relative to income |
| `AGE_YEARS` | Applicant age in years |
| `AGE_GROUP` | Age bucket [18–25, 25–35, …, 65+] |

### Bureau Aggregates (13 features)
Credit count, active ratio, overdue count/ratio, max overdue days, debt-income ratio, severity buckets, and history flags.

### Bureau Balance (5 features)
Monthly history length, overdue months count, severe overdue months, overdue ratio per applicant.

### Previous Applications (6 features)
Application count, refusal count, cancellation count, average previous credit, approval/refusal rates.

### Installment Payments (4 features)
Late payment count, late payment rate, average payment delay, underpayment rate.

**Final dataset shape:** 307,511 × 162 (train) after merge; reduced to 107 numeric features for gradient boosting.

---

## 🧹 Data Preprocessing

1. **Missing values:** Drop 38 numerical columns with >50% missingness; median-impute remaining numerics using training statistics only
2. **Categorical encoding:** Binary/ordinal mappings → frequency encoding → association-based nominal reduction → one-hot encoding
3. **Imbalance handling:** `class_weight="balanced"` for tree models; `scale_pos_weight ≈ 11.39` for XGBoost
4. **Scaling:** StandardScaler applied only for Logistic Regression
5. **Train/validation split:** Stratified 80/20, `random_state=42`

---

## 🤖 Models Trained

| Model | Features | Key Config | ROC-AUC | PR-AUC |
|-------|----------|-----------|---------|--------|
| Logistic Regression (top 40) | 40 selected | max_iter=1000, balanced | 0.7418 | — |
| Logistic Regression (all 107) | 107 numeric | max_iter=1000, balanced | 0.7538 | — |
| Random Forest | 40 selected | n_estimators=200, balanced | 0.7332 | — |
| LightGBM | 107 numeric | 1500 trees, lr=0.03, early stop | 0.7647 | 0.2544 |
| CatBoost | 103 native-cat | 3000 iter, depth=8, GPU | 0.7654 | 0.2579 |
| XGBoost | 107 numeric | 2000 trees, depth=8, GPU hist | 0.7609 | 0.2495 |
| **3-Model Blend** | — | LightGBM + CatBoost + XGBoost | **0.7669** | **0.2593** |

> CatBoost is the best individual model. The blended ensemble marginally improves over each single model.

---

## 📁 Repository Structure

```
home-credit-default-risk/
│
├── data/                          # Raw CSVs (not included — download from Kaggle)
│
├── notebooks/
│   ├── mainul.ipynb               # EDA, feature engineering, preprocessing
│   └── mainul_2nd.ipynb           # Model training, evaluation, submission
│
├── outputs/
│   ├── applications.csv           # Decoded portfolio data (Tableau source 1)
│   ├── model_predictions.csv      # Validation predictions with probabilities
│   └── model_performance_reference.xlsx  # Model metrics for Tableau (source 2)
│       ├── Model_Summary          # AUC, PR-AUC per model
│       ├── ROC_Curve              # FPR / TPR points
│       ├── PR_Curve               # Precision / Recall points
│       ├── Calibration            # Predicted vs actual probability bins
│       ├── Confusion_Matrix       # TP / TN / FP / FN at 0.5 threshold
│       ├── Feature_Importance     # Permutation importance scores
│       └── README                 # Sheet descriptions
│
├── tableau/
│   └── home_credit_dashboard.twbx # Packaged Tableau workbook
│
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/home-credit-default-risk.git
cd home-credit-default-risk
```

### 2. Download the Data
Download all CSV files from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place them in `data/`.

### 3. Install Dependencies
```bash
pip install pandas numpy scikit-learn lightgbm catboost xgboost openpyxl matplotlib seaborn
```

### 4. Run the Notebooks
Execute `mainul.ipynb` first (EDA + feature engineering), then `mainul_2nd.ipynb` (model training + submission).

### 5. Open the Tableau Dashboard
Open `tableau/home_credit_dashboard.twbx` in **Tableau Public** (free) or **Tableau Desktop**.

---

## 📈 Key Findings

- **Gender gap:** Male applicants default at 10.1% vs female 7.0%
- **Income non-linearity:** The 4th income quintile (432k–604k) has the *highest* default rate (10.05%), not the lowest
- **Bureau history doubles risk:** Applicants with any previous overdue bureau credit default at 16.2% vs 8.0%
- **Working applicants highest risk:** 9.6% default rate vs 5.4% for pensioners
- **Top predictors:** EXT_SOURCE_2, EXT_SOURCE_3, BUREAU_DEBT_INCOME_RATIO, BUREAU_MAX_OVERDUE_DAYS, CREDIT_INCOME_RATIO

---

## 👤 Author

**Ridwan**  
BSc in Data Science — United International University (UIU), Bangladesh  
📧 tridwan.ds25@gmail.com

---

## 📄 License

This project is for academic purposes. Dataset © Home Credit Group, made available via Kaggle.
