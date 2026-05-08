# -Diabetes-Machine-learning-project-
## 👥 Team 8

- Nada Ahmed  
- Menna Fawzy  
- Abram Maged  
- Ahmed Ezzat  
- Emmanuel George  

---

# 📌 Project Overview

This project focuses on analyzing hospital data for diabetic patients and building a Machine Learning pipeline to understand factors affecting patient readmission.

The main goal is to explore the dataset, perform data preprocessing, and prepare the data for predictive modeling.

---

# 📊 Problem Statement

Hospital readmission of diabetic patients is a critical healthcare issue.  
Predicting whether a patient will be readmitted can help improve treatment quality and reduce healthcare costs.

---

# 📁 Dataset Description

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

