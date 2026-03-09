# Diabetes Risk Prediction — End-to-End ML Pipeline with SHAP Interpretability

A complete, production-style machine learning pipeline that predicts diabetes risk from patient clinical and demographic data. Built to reflect real-world ML practice — from messy raw data to explainable model predictions.

---

## Problem Statement

Diabetes affects hundreds of millions of people worldwide, and a large share of cases go undiagnosed until serious complications appear. This project builds a binary classifier that flags high-risk individuals from routine clinical measurements, enabling earlier screening without expensive tests.

The core challenge: the dataset is **severely imbalanced** (~8.5% diabetic prevalence), meaning a naive model that always predicts "no diabetes" achieves 91.5% accuracy while being completely useless. The entire pipeline is designed around this reality.

---

## Dataset

**[Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)** — Kaggle (~100,000 rows)

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Male / Female |
| `age` | Numeric | Age in years |
| `hypertension` | Binary | 0 = no, 1 = yes |
| `heart_disease` | Binary | 0 = no, 1 = yes |
| `smoking_history` | Categorical | never / former / current / No Info |
| `bmi` | Numeric | Body Mass Index |
| `HbA1c_level` | Numeric | Average blood sugar over ~3 months |
| `blood_glucose_level` | Numeric | Blood glucose at time of measurement |
| `diabetes` | Binary (target) | 0 = no diabetes, 1 = diabetes |

---

## Pipeline Overview

```
Raw Data
   │
   ├── 1. EDA & Data Understanding
   ├── 2. Data Cleaning
   │       ├── Drop 3,854 duplicate rows
   │       ├── Cap BMI outliers > 60
   │       └── Drop 18 rows with gender = 'Other'
   │
   ├── 3. Stratified Train/Test Split (80/20)
   │
   ├── 4. Preprocessing (fit on train only — no leakage)
   │       ├── StandardScaler  →  numeric features
   │       └── OneHotEncoder   →  categorical features
   │
   ├── 5. Class Imbalance Handling
   │       └── class_weight="balanced" on all classifiers
   │
   ├── 6. Model Training & Comparison
   │       ├── Logistic Regression
   │       ├── Decision Tree
   │       ├── Random Forest  ← best
   │       └── KNN
   │
   ├── 7. Hyperparameter Tuning
   │       └── GridSearchCV + StratifiedKFold (5-fold), scored on F1
   │
   ├── 8. Final Evaluation
   │       ├── Confusion Matrix
   │       ├── Classification Report
   │       ├── ROC Curve (AUC)
   │       └── Precision-Recall Curve
   │
   └── 9. SHAP Interpretability
           ├── Global feature importance (summary plot)
           └── Per-patient waterfall plots
```

---

## Key Design Decisions

**Why F1 and not accuracy?**
With only 8.5% positive cases, accuracy is meaningless. A model predicting "no diabetes" for everyone scores 91.5% accuracy. F1-score balances precision and recall on the minority class — the class that actually matters.

**Why Recall over Precision?**
A false negative (missing a diabetic patient) leads to delayed treatment and serious long-term complications. A false positive leads to a cheap follow-up blood test. The pipeline prioritizes recall while keeping precision at an acceptable level to avoid clinical alert fatigue.

**Why class weighting over SMOTE?**
`class_weight="balanced"` adjusts the loss function directly without synthesizing artificial patient records, which can produce biologically implausible clinical profiles. It is computationally cleaner and integrates natively into all sklearn classifiers.

**Why leakage-free preprocessing matters?**
All scalers and encoders are fitted exclusively on the training set inside a `Pipeline`, then applied to the test set. Fitting on the full dataset before splitting is data leakage — it inflates test metrics by leaking test-set distribution into training.

---

## Results

| Metric | Score |
|---|---|
| F1-Score (diabetic class) | **0.777** |
| ROC-AUC | **0.972** |
| Precision | 86.5% |
| Recall | 70.6% |
| Overall Accuracy | 96.4% |

---

## SHAP Interpretability

SHAP (`TreeExplainer`) was used to explain model predictions at both the global and individual patient level.

**Global findings:** `HbA1c_level` and `blood_glucose_level` are the two strongest predictors — consistent with clinical diagnostic thresholds (HbA1c ≥ 6.5 is the standard diabetes diagnosis criterion). `age` and `bmi` are secondary risk factors.

**False-negative analysis:** Misclassified patients (false negatives) cluster at borderline clinical values — mean HbA1c of **6.11** vs. **7.23** for correctly identified diabetic patients. The model fails logically: borderline blood sugar values are offset by younger age and lower BMI, pushing predictions below the classification threshold. These are pre-diabetic grey-zone cases, not arbitrary failures.

---

## Project Structure

```
diabetes-risk-prediction/
│
├── diabetes_prediction_starter.ipynb   # Full pipeline notebook
├── diabetes_prediction_dataset.csv     # Dataset (download from Kaggle)
└── README.md
```

---

## Setup & Usage

**1. Clone the repo**
```bash
git clone https://github.com/bharat710/diabetes-risk-prediction.git
cd diabetes-risk-prediction
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn shap
```

**3. Download the dataset**

Download `diabetes_prediction_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) and place it in the project root directory.

**4. Run the notebook**
```bash
jupyter notebook diabetes_prediction_starter.ipynb
```

Run cells in order — each section builds on the previous one.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Pipeline | Scikit-learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`) |
| Models | Logistic Regression, Decision Tree, Random Forest, KNN |
| Explainability | SHAP (`TreeExplainer`) |

---

## Limitations

- **Missing predictors:** Family history, physical activity, lipid panels (HDL/LDL/triglycerides), and gestational history are absent — all clinically significant for diabetes risk.
- **Smoking data quality:** ~37% of `smoking_history` values are `'No Info'`, limiting the model's ability to learn robust smoking-diabetes relationships.
- **Population generalizability:** Performance on external cohorts (different ethnicities, geographies, or clinical settings) has not been validated.


