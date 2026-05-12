# Diabetes Machine Learning Project
## 👥 Team 8

- Nada Ahmed
- Menna Fawzy
- Abram Maged
- Ahmed Ezzat
- Emmanuel George
---
## Installation and Execution

### Prerequisites
- Python 3.8+
- Required Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `joblib`

### Running the Pipeline
1. Clone the repository and ensure `diabetic_data.csv` is in the root directory alongside the script.
2. Install the required dependencies:
  ```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```
3. Execute the pipeline script:
   ```bash
   python "diabetes_pipeline.py"
   ```
4. The script will output dataset health checks, model evaluation metrics, and generate `.png` visualizations (Confusion Matrices, ROC Curves, and Feature Importance) directly in the working directory.

## Technologies Used
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `imbalanced-learn`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Model Serialization:** `joblib`

## 📂 Project Structure

```bash
├── diabetic_data.csv
├── diabetes_pipeline.py
├── README.md
├── confusion_matrix.png
├── roc_curve.png
└── feature_importance.png

# 📌 Project Overview

Hospital readmission is a critical metric in healthcare, serving as a primary indicator of treatment efficacy and patient outcomes. High readmission rates, particularly within a 30-day window, often lead to severe financial penalties for hospitals under standard CMS (Centers for Medicare & Medicaid Services) guidelines.

This project implements a complete Machine Learning pipeline designed to predict hospital readmissions for diabetic patients. Using a comprehensive healthcare dataset of over 100,000 records, the system processes raw clinical data, mitigates class imbalance, and trains robust classification models to identify high-risk patients.

---
## ⚙️ ML Pipeline

1. Data Cleaning
2. Feature Engineering
3. Encoding
4. Data Balancing
5. Model Training
6. Hyperparameter Tuning
7. Evaluation
8. Model Saving


# 📊 Problem Statement

Hospital readmission of diabetic patients is a critical healthcare issue.  
Predicting whether a patient will be readmitted can help improve treatment quality and reduce healthcare costs.

---

# 📁 Dataset Description
The dataset utilized is based on the **Diabetes 130-US hospitals for years 1999-2008** dataset. It includes:
- **Demographics:** Age, Gender, Race
- **Hospital Admission Details:** Admission type, discharge disposition, time in hospital
- **Clinical Results:** Number of lab procedures, medications administered, A1C test results
- **Diagnoses:** Primary, secondary, and tertiary ICD-9 diagnosis codes
- **Target Variable:** `readmitted` (Formulated as a Binary Classification: `<30 days` vs. `Other`)

- Total Records: 101,766 patients  
- Total Features: 50 columns  
- Data Type: Real-world hospital clinical dataset  

### Key Features:
- Patient demographics (age, gender, race)  
- Hospital admission details  
- Laboratory test results  
- Medication information  
- Diagnosis codes  
- Readmission status (target variable)

---

# 🧹 Data Cleaning

The dataset was cleaned through the following steps:

- Handling missing values in multiple columns  
- Removing irrelevant features such as:
  - encounter_id  
  - patient_nbr  
  - weight  
  - payer_code  
- Standardizing inconsistent data entries  
- Preparing data for analysis and modeling  

---

# 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand data distribution and relationships:

- Analysis of target variable (readmission)
- Age distribution of patients
- Gender distribution
- Hospital stay duration
- Medication usage patterns
- Correlation between numerical features

---

# 🔄 Feature Encoding

Since most machine learning models require numerical input, categorical features were converted into numerical format using different encoding techniques:

## 🔹 1. Label Encoding
Applied to:
- gender

## 🔹 2. One-Hot Encoding
Applied to:
- race  
- admission_type_id  
- discharge_disposition_id  
- admission_source_id  

## 🔹 3. Ordinal Encoding (Medical Features)
Applied to:
- age  
- medical_specialty  
- diag_1  
- diag_2  
- diag_3  
- max_glu_serum  
- A1Cresult  

## 🔹 4. Medication Encoding
Medication-related features were mapped as:
- No → 0  
- Steady → 1  
- Up → 2  
- Down → -1  

## 🔹 5. Binary Features
- change  
- diabetesMed  

Converted into binary format for modeling.

---
### 3. Machine Learning Modeling
Five distinct classifiers were trained and evaluated using stratified data splits:
1. Logistic Regression
2. Decision Tree
3. Random Forest (Ensemble)
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

Models were evaluated based on **F1 Score (Weighted)**, **Precision/Recall**, and **Macro-average AUC** to account for true predictive power.

### 4. Hyperparameter Tuning
The top-performing tree-based model (Random Forest) underwent hyperparameter optimization using `RandomizedSearchCV` over 3-fold cross-validation. Tuned parameters included tree depth, estimator count, and minimum sample splits.


## 📈 Results

The Random Forest model achieved the best performance:

- Accuracy: XX%
- F1 Score: XX%
- ROC-AUC: XX%

## 💾 Model Serialization

The best trained model was saved using joblib for future deployment and inference.

ولو عندكم:

## Key Insights
Feature importance analysis from the tuned Random Forest model revealed the strongest predictors of hospital readmission:
1. **Number of Inpatient Visits (`number_inpatient`):** Patients with a history of prior admissions are at a significantly higher risk of returning.
2. **Number of Medications & Diagnoses:** High medication counts and multiple concurrent diagnoses strongly correlate with clinical instability and readmission risk.
3. **Discharge Disposition (`discharge_disposition_id`):** The facility or care setting a patient is discharged to heavily influences their likelihood of readmission.

## 🚀 Future Improvements

- Deploy the model using Flask or FastAPI
- Build an interactive dashboard
- Improve feature engineering
- Experiment with deep learning models
