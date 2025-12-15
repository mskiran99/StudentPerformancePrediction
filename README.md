## Student Performance Prediction Using AWS Cloud

A cloud‑native, **serverless machine learning pipeline** to predict student academic performance using the **UCI Student Performance Dataset** and visualize at‑risk students through a **Streamlit dashboard**.

The system combines:

-  Machine Learning (Random Forest, Gradient Boosting)  
-  AWS (S3, Lambda, API Gateway, IAM, CloudWatch)  
-  Streamlit for UI, synthetic data generation & analytics  

##  Project Overview

This project predicts the **final grade (G3)** of students in Portuguese secondary schools using:

- Past grades (G1, G2)  
- Demographic details  
- Family & social background  
- Behavioral and school‑related attributes  

Instead of running everything locally, we deploy an **end‑to‑end prediction pipeline on AWS**, using **serverless** components and an interactive web dashboard.

## Objectives

- Build an **end‑to‑end ML pipeline** for student performance prediction  
- Host trained models in **Amazon S3**  
- Use **Streamlit** for:
  - Generating synthetic student data  
  - Loading models from S3  
  - Running predictions and evaluating models (RMSE)  
  - Visualizing at‑risk students  
- Use **AWS Lambda + API Gateway** to store predictions in S3 via a **single API call**  
- Demonstrate a scalable, low‑maintenance, **serverless cloud architecture**

## Dataset Description

**Dataset:** Student Performance Dataset  
**Source:** Kaggle 
Link: `https://www.kaggle.com/datasets/whenamancodes/student-performance/versions/2?resource=download`

The dataset describes student achievement in **two Portuguese secondary schools**, for two subjects:

- Mathematics
- Portuguese language

Data was collected using **school reports** and **student questionnaires**, and includes:

- **Demographic features:**  
  - `age`, `sex`, `address` (urban/rural)

- **Family & social features:**  
  - `famsize`, `Pstatus`, `Medu`, `Fedu`, `guardian`, `famrel`, `freetime`, `goout`

- **Behavioral & lifestyle features:**  
  - `Dalc`, `Walc`, `health`, `absences`, `romantic`

- **Academic features:**  
  - `studytime`, `failures`, `schoolsup`, `famsup`, `paid`, `activities`, `higher`, `internet`

- **Grades:**  
  - `G1` – 1st period grade  
  - `G2` – 2nd period grade  
  - `G3` – **final grade (target)**  

> Note: G3 is strongly correlated with G1 and G2.  
> G3 = final year grade (3rd period), while G1 and G2 are 1st and 2nd period grades.

## Machine Learning Approach

We trained **two regression models** on the dataset:

- **Random Forest Regressor**
- **Gradient Boosting Regressor**

### Training (done offline / locally)

1. Load the UCI dataset (Math / Portuguese).  
2. Perform preprocessing & feature selection.  
3. Train **Random Forest** and **Gradient Boosting** models.  
4. Evaluate performance using **RMSE / R²**.  
5. Save both trained models as `.pkl` using `joblib`.

These `.pkl` files are then uploaded to **Amazon S3** and used by the Streamlit app.

## Cloud Architecture

<img width="1090" height="727" alt="image" src="https://github.com/user-attachments/assets/0eeed672-b84b-4735-99c2-0b0b422283a2" />

