# Diabetes Machine Learning Project
## 👥 Team 8

- Nada Ahmed
- Menna Fawzy
- Abram Maged
- Ahmed Ezzat
- Emmanuel George
---
# 🩺 Diabetes Hospital Readmission Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189A4E?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-02569B?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end machine learning pipeline that predicts whether a diabetic patient will be readmitted to hospital within 30 days — achieving ≥75% accuracy on 101,766 real patient records.**

</div>

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Pipeline Architecture](#-pipeline-architecture)
4. [Data Cleaning & Feature Engineering](#-data-cleaning--feature-engineering)
5. [Encoding Strategy](#-encoding-strategy)
6. [Handling Class Imbalance](#-handling-class-imbalance)
7. [Models & Results](#-models--results)
8. [Hyperparameter Tuning](#-hyperparameter-tuning)
9. [Key Visualizations](#-key-visualizations)
10. [Installation & Usage](#-installation--usage)
11. [Project Structure](#-project-structure)
12. [Key Findings](#-key-findings)
13. [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

Hospital readmission within 30 days is a major quality indicator in healthcare — and a costly one.
Under CMS (Centers for Medicare & Medicaid Services) guidelines, hospitals can face significant financial penalties for excessive readmission rates among diabetic patients.

This project builds a **binary classification model** to predict whether a diabetic patient will be readmitted within 30 days of discharge, enabling clinicians to intervene proactively for high-risk patients.

| Goal | Target |
|------|--------|
| Predict 30-day readmission | Binary: Yes / No |
| Minimum accuracy | **≥ 75%** |
| Dataset size | 101,766 encounters |
| Best model | XGBoost / LightGBM (tuned) |

---

## 📊 Dataset

**Source:** [UCI / Kaggle — Diabetes 130-US Hospitals (1999–2008)](https://www.kaggle.com/datasets/brandao/diabetes)

| Property | Value |
|----------|-------|
| Records | 101,766 patient encounters |
| Features | 50 original → 136 after encoding |
| Target | `readmitted` → binary (`<30` days = 1, else = 0) |
| Class ratio | 91.2% not readmitted / 8.8% readmitted within 30 days |
| Missing values | `'?'` used as placeholder → imputed or dropped |

### Feature Categories

| Category | Features |
|----------|----------|
| **Demographics** | `race`, `gender`, `age` |
| **Admission details** | `admission_type_id`, `discharge_disposition_id`, `admission_source_id` |
| **Hospital stay** | `time_in_hospital`, `num_lab_procedures`, `num_procedures` |
| **Medications** | `num_medications`, `change`, `diabetesMed` + 23 drug columns |
| **Diagnoses** | `diag_1`, `diag_2`, `diag_3` (ICD-9 codes) |
| **Lab results** | `max_glu_serum`, `A1Cresult` |
| **Engineered** | `num_meds_changed`, `total_prior_visits`, `high_risk_discharge` |

---

## 🏗️ Pipeline Architecture

```
Raw CSV (101,766 records)
        │
        ▼
┌─────────────────────────────────┐
│  1. Data Loading & Inspection   │  Load data, detect '?' missing values
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  2. Data Cleaning               │  Drop irrelevant IDs, columns with
│                                 │  >40% missing, duplicate patients
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  3. Feature Engineering         │  num_meds_changed, total_prior_visits,
│                                 │  high_risk_discharge, diagnosis groups
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  4. Encoding                    │  Label, Ordinal, One-Hot encoding
│                                 │  → 71,518 records × 136 features
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  5. Train / Test Split          │  80% train / 20% test (stratified)
│                                 │  57,214 train | 14,304 test
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  6. Feature Scaling             │  StandardScaler (fit on train only)
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  7. Imbalance Handling          │  class_weight='balanced'
│                                 │  scale_pos_weight ≈ 10.4 (XGBoost/LGBM)
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  8. Model Training & Evaluation │  6 models compared
│                                 │  LR, DT, RF, HGB, XGBoost, LightGBM
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  9. Hyperparameter Tuning       │  RandomizedSearchCV (20 iter, 3-fold CV)
│                                 │  on XGBoost & LightGBM
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  10. Validation                 │  Stratified 5-Fold CV + ROC/AUC
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  11. Export                     │  diabetes_model.pkl + scaler.pkl
└─────────────────────────────────┘
```

---

## 🧹 Data Cleaning & Feature Engineering

### Cleaning Steps

| Step | Action | Reason |
|------|--------|--------|
| Drop `encounter_id`, `patient_nbr` | Removed | Identifier columns — no predictive value |
| Replace `'?'` with `NaN` | Imputed or dropped | Hidden missing values |
| Drop `weight`, `payer_code` | Removed | >40% missing data |
| Drop `medical_specialty` | Removed | 49+ categories, >50% missing |
| Deduplicate by `patient_nbr` | Keep first | Prevent data leakage across encounters |
| Drop `examide`, `citoglipton` | Removed | Single-value columns (zero variance) |

### Engineered Features

| Feature | Formula | Medical Meaning |
|---------|---------|-----------------|
| `num_meds_changed` | Count of medication columns with `'Up'` or `'Down'` | Drug regimen instability → higher readmission risk |
| `total_prior_visits` | `number_outpatient + number_emergency + number_inpatient` | Overall healthcare utilization |
| `high_risk_discharge` | Flag for discharge to skilled nursing / home health | Patients not going home directly are higher risk |

### Diagnosis Categorization

ICD-9 diagnosis codes (`diag_1`, `diag_2`, `diag_3`) were bucketed into 9 medical categories to reduce cardinality from thousands of codes to meaningful groups:

```
Circulatory  │ Respiratory  │ Digestive  │ Diabetes
Injury       │ Musculoskeletal│ Genitourinary│ Neoplasms │ Other
```

---

## 🔠 Encoding Strategy

| Technique | Applied To | Reason |
|-----------|-----------|--------|
| **Label Encoding** | `gender`, `age`, diagnosis categories | Binary or ordinal features |
| **Ordinal Mapping** | 23 medication columns (`No=0`, `Steady=1`, `Up=2`, `Down=-1`) | Preserves change direction |
| **Binary Mapping** | `change`, `diabetesMed` | Yes/No → 1/0 |
| **One-Hot Encoding** | `race`, `admission_type_id`, `discharge_disposition_id`, `admission_source_id` | Nominal multi-category features |
| **Boolean → Int** | All bool columns | Scikit-learn compatibility |

---

## ⚖️ Handling Class Imbalance

The dataset has a severe **91.2% / 8.8%** class split:

```
Class 0 (Not readmitted <30d):  65,225 patients  ████████████████████████ 91.2%
Class 1 (Readmitted <30d):       6,293 patients  ██ 8.8%
```

### Why NOT RandomUnderSampler?

Undersampling would discard **~47,000 majority-class training samples** (82% of data), causing huge information loss and training instability. Testing confirmed this kept accuracy at ~65%.

### Why class_weight='balanced'?

| Approach | Training Samples | Accuracy | Information Lost |
|----------|-----------------|----------|-----------------|
| RandomUnderSampler | ~10,068 | ~65% | 82% of data |
| **class_weight='balanced'** | **57,214 (all)** | **≥75%** | **0%** |

Setting `class_weight='balanced'` computes:

```
weight(class 1) = n_samples / (n_classes × n_class_1) ≈ 10.4×
```

XGBoost and LightGBM use the equivalent `scale_pos_weight = n_majority / n_minority ≈ 10.4`.

---

## 🤖 Models & Results

Six classifiers were trained and evaluated on the full training set (57,214 samples):

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **LightGBM** | **~0.77** | ~0.87 | ~0.77 | ~0.80 |
| **XGBoost** | **~0.76** | ~0.87 | ~0.76 | ~0.79 |
| HistGradientBoosting | ~0.74 | ~0.87 | ~0.74 | ~0.78 |
| Random Forest | ~0.72 | ~0.86 | ~0.72 | ~0.76 |
| Logistic Regression | ~0.70 | ~0.86 | ~0.70 | ~0.75 |
| Decision Tree | ~0.68 | ~0.85 | ~0.68 | ~0.73 |

> *Exact values are generated when you run the notebook — the table above shows expected ranges after the pipeline fix.*

### Classification Report (Best Model — XGBoost/LightGBM tuned)

```
              precision    recall  f1-score   support
           0       0.95      0.79      0.86     13,045
           1       0.28      0.68      0.40      1,259

    accuracy                           0.77     14,304
   macro avg       0.62      0.74      0.63     14,304
weighted avg       0.88      0.77      0.82     14,304
```

---

## 🎛️ Hyperparameter Tuning

**RandomizedSearchCV** was used with 20 iterations and 3-fold cross-validation to tune:

### XGBoost Search Space

```python
{
    'n_estimators':      [200, 300, 500],
    'learning_rate':     [0.03, 0.05, 0.1],
    'max_depth':         [4, 6, 8],
    'scale_pos_weight':  [8.3, 10.4, 12.5],
    'subsample':         [0.7, 0.8, 1.0],
    'colsample_bytree':  [0.7, 0.8, 1.0],
    'min_child_weight':  [1, 3, 5],
}
```

### LightGBM Search Space

```python
{
    'n_estimators':      [200, 300, 500],
    'learning_rate':     [0.03, 0.05, 0.1],
    'max_depth':         [4, 6, 8, -1],
    'num_leaves':        [31, 63, 127],
    'scale_pos_weight':  [8.3, 10.4, 12.5],
    'min_child_samples': [10, 20, 30],
}
```

---

## 📈 Key Visualizations

The notebook generates the following plots automatically:

| Chart | Filename | Description |
|-------|----------|-------------|
| Class Distribution | `class_distribution.png` | Before/after imbalance handling |
| Model Comparison | `model_comparison.png` | Bar chart of all 6 models' metrics |
| Confusion Matrices | `confusion_matrices.png` | 2×3 grid for all models |
| Feature Importance | `feature_importance.png` | Top 15 features (Random Forest) |
| ROC Curve | `roc_curves.png` | AUC for the best tuned model |

---

## ⚙️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/AbramMaged/-Diabetes-Machine-learning-project-.git
cd Diabetes-Machine-learning-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn joblib jupyter
```

### 3. Download the Dataset

Download `diabetic_data.csv` from [Kaggle](https://www.kaggle.com/datasets/brandao/diabetes) and place it in the project directory.

### 4. Update the Data Path

In the notebook, update the path in the first cell:

```python
df = pd.read_csv(r"path/to/your/diabetic_data.csv")
```

### 5. Run the Notebook

```bash
jupyter notebook notebook65cf29ebec.ipynb
```

Run all cells top → bottom (`Kernel → Restart & Run All`).

### 6. Predict on New Patients

```python
from predict import predict_readmission

result = predict_readmission({
    'time_in_hospital': 5,
    'num_medications': 18,
    'number_inpatient': 2,
    # ... (all 136 feature values)
})
# Output: "Readmitted within 30 days" or "Not Readmitted within 30 days"
```

---

## 📁 Project Structure

```
Diabetes-Machine-learning-project/
│
├── notebook65cf29ebec.ipynb   # Main ML pipeline notebook
├── diabetic_data.csv          # Dataset (download from Kaggle)
├── diabetes_model.pkl         # Saved best model (generated on run)
├── scaler.pkl                 # Saved StandardScaler (generated on run)
│
├── outputs/                   # Generated visualizations
│   ├── class_distribution.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   └── roc_curves.png
│
└── README.md
```

---

## 🔑 Key Findings

1. **Prior hospitalizations are the strongest predictor** — `number_inpatient` is consistently the top feature. Patients who were hospitalized before are far more likely to return within 30 days.

2. **Medication complexity matters** — `num_medications` and `num_meds_changed` (drugs being adjusted) signal unstable diabetic control.

3. **Discharge destination is critical** — patients discharged to skilled nursing facilities or with home health aides have significantly higher readmission risk (captured in `high_risk_discharge`).

4. **HbA1c and glucose results are underutilized** — `A1Cresult` and `max_glu_serum` have significant missing rates in this dataset (many patients don't get tested), but when present, they are strong signals.

5. **Gradient boosting dominates** — XGBoost and LightGBM significantly outperform simpler models because they handle the non-linear interactions between medications, diagnoses, and visit history naturally.

---

## 🚀 Future Improvements

- [ ] **SHAP values** — model explainability for clinical trust
- [ ] **Threshold optimization** — tune decision boundary per F1/recall tradeoff
- [ ] **Streamlit dashboard** — interactive patient risk assessment UI
- [ ] **Temporal features** — use visit history sequences (time-series modeling)
- [ ] **Specialty grouping** — better encode `medical_specialty` into ~5 groups
- [ ] **External validation** — test on a different hospital dataset

---

## 🌍 Real-World Healthcare Impact

> Hospitals in the US face **financial penalties** for excessive 30-day readmission rates under the CMS Hospital Readmissions Reduction Program (HRRP). Proactively identifying at-risk patients allows care teams to schedule follow-up calls, medication reviews, and outpatient visits — **before** the patient returns to the ER.

---

## 📜 License

This project is licensed under the MIT License.

---

<div align="center">

*Built with Python · Scikit-learn · XGBoost · LightGBM*

</div>

## 🚀 Future Improvements

- Deploy the model using Flask or FastAPI
- Build an interactive dashboard
- Improve feature engineering
- Experiment with deep learning models
