# 🎓 Student Performance Tracker & Analysis System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### An end-to-end Machine Learning system that analyses student learning behaviour, identifies key performance factors, and predicts exam scores — all served through a beautiful interactive dashboard.

[What It Does](#-what-this-project-does) •
[Features](#-features) •
[Installation](#-installation) •
[Usage](#-usage) •
[ML Models](#-machine-learning-models) •
[Dashboard](#-dashboard-walkthrough) •
[Results](#-model-results)

</div>

---

## 🧠 What This Project Does

Every student learns differently. Some study for hours but still underperform. Others score well with less effort. Why? The answer lies in a combination of factors — attendance, sleep, motivation, parental support, access to resources, peer influence, and more.

This project uses **real student data** and **machine learning** to:

1. **Understand** which factors most strongly affect exam performance
2. **Visualise** patterns and correlations in student behaviour through rich charts
3. **Predict** a student's expected exam score based on their personal learning profile
4. **Recommend** specific, actionable improvements personalised to each student

Whether you are a **teacher** trying to identify at-risk students early, a **school administrator** looking at systemic patterns, or a **student** wanting to understand what to improve — this system gives you clear, data-driven answers.

---

## 🎯 Problem Statement

> *"Given 19 features about a student's learning environment and habits, can we accurately predict their exam score and identify the most influential factors?"*

This is a **supervised regression problem**. The target variable is `Exam_Score` (a continuous numeric value). We train multiple machine learning models, compare their accuracy, and deploy the best one inside an interactive web application.

---

## ✨ Features

### 📊 Data Analysis
- Full exploratory data analysis (EDA) with 10+ interactive charts
- Correlation heatmap across all 19 features
- Score distributions, box plots, scatter plots with trend lines
- Automatic insight generation from the data

### 🤖 Machine Learning
- Trains **5–6 models** simultaneously: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, MLP Neural Network, and XGBoost (optional)
- Auto-selects the **best model** by R² score
- Evaluates every model with **MAE, RMSE, and R²**
- Saves trained model to disk with `joblib` for instant reloading

### 🔮 Predictions
- **Single prediction** — enter one student's details, get an instant score + grade
- **Batch prediction** — upload a CSV of multiple students, download a results file
- **What-If simulator** — see how changing study hours shifts the predicted score
- **Personalised recommendations** — tips tailored to the student's specific weak areas

### 🖥️ Interactive Dashboard
- 3-page Streamlit web app with a dark professional UI
- Sliders, dropdowns, charts, gauge meter, model comparison radar chart
- No coding needed once deployed — fully point-and-click

---

## 📁 Project Structure

```
student-performance-ml/
│
├── 📂 data/
│   └── StudentPerformanceFactors.csv    ← Dataset: 6,607 students, 20 columns
│
├── 📂 src/
│   ├── train.py                         ← Full ML training pipeline
│   └── predict.py                       ← Prediction module (single + batch)
│
├── 📂 app/
│   └── app.py                           ← Streamlit 3-page dashboard
│
├── 📂 model/
│   └── model.pkl                        ← Saved best model + encoders + metadata
│
├── 📂 notebooks/
│   ├── analysis.ipynb                   ← Jupyter EDA notebook
│   └── figures/                         ← Saved chart images
│       ├── correlation_heatmap.png
│       ├── score_distribution.png
│       ├── feature_importance.png
│       └── pairplot.png
│
├── requirements.txt                     ← All Python dependencies
└── README.md                            ← You are here
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `StudentPerformanceFactors.csv` |
| Students | 6,607 |
| Features | 19 input features + 1 target |
| Target | `Exam_Score` (numeric, 55–101) |
| Source | [Kaggle — Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) |

### All 19 Features Explained

| # | Feature | Type | Range / Values | What It Captures |
|---|---------|------|----------------|-----------------|
| 1 | `Hours_Studied` | Numeric | 1 – 44 hrs/week | Time dedicated to studying |
| 2 | `Attendance` | Numeric | 60 – 100 % | Class attendance rate |
| 3 | `Parental_Involvement` | Categorical | Low / Medium / High | How involved parents are in education |
| 4 | `Access_to_Resources` | Categorical | Low / Medium / High | Books, internet, study materials |
| 5 | `Extracurricular_Activities` | Categorical | Yes / No | Sports, clubs, arts participation |
| 6 | `Sleep_Hours` | Numeric | 4 – 10 hrs/night | Nightly sleep duration |
| 7 | `Previous_Scores` | Numeric | 50 – 100 | Past academic performance |
| 8 | `Motivation_Level` | Categorical | Low / Medium / High | Internal drive to study |
| 9 | `Internet_Access` | Categorical | Yes / No | Home internet availability |
| 10 | `Tutoring_Sessions` | Numeric | 0 – 8 per month | Extra academic support received |
| 11 | `Family_Income` | Categorical | Low / Medium / High | Household income level |
| 12 | `Teacher_Quality` | Categorical | Low / Medium / High | Quality of instruction received |
| 13 | `School_Type` | Categorical | Public / Private | Type of school attended |
| 14 | `Peer_Influence` | Categorical | Negative / Neutral / Positive | Effect of friends on study habits |
| 15 | `Physical_Activity` | Numeric | 0 – 6 hrs/week | Exercise and physical wellness |
| 16 | `Learning_Disabilities` | Categorical | Yes / No | Diagnosed learning challenges |
| 17 | `Parental_Education_Level` | Categorical | High School / College / Postgraduate | Highest education of parent |
| 18 | `Distance_from_Home` | Categorical | Near / Moderate / Far | Travel time to school |
| 19 | `Gender` | Categorical | Male / Female | Student gender |
| 20 ⭐ | `Exam_Score` | **TARGET** | 55 – 101 | **Final exam score to predict** |

---

## ⚙️ Installation

### Requirements
- Python 3.9 or higher
- pip

### Step 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-ml.git
cd student-performance-ml
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — (Optional) Install XGBoost for an extra model

```bash
pip install xgboost
```

---

## 🚀 Usage

### Option A — Launch the full dashboard (recommended)

```bash
streamlit run app/app.py
```

This opens the interactive web app at **http://localhost:8501**
The pre-trained model is already included — no training needed.

---

### Option B — Train models from scratch

```bash
python src/train.py
```

This will:
- Load and preprocess the CSV
- Label-encode all categorical columns
- Apply Standard Scaling for Linear Regression and MLP
- Train all 5–6 models with an 80/20 train-test split
- Print R², MAE, and RMSE for each model
- Auto-select the best model and save it to `model/model.pkl`

Sample output:
```
============================================================
  Student Performance ML — Training Pipeline
============================================================

[1/4] Loading & preprocessing data...
  Features : ['Hours_Studied', 'Attendance', ...]
  Train    : (5285, 19)  |  Test : (1322, 19)

[2/4] Training models...
  Training Linear Regression...   R²=0.525  MAE=4.187  RMSE=5.382
  Training Decision Tree...       R²=0.458  MAE=3.681  RMSE=5.747
  Training Random Forest...       R²=0.684  MAE=3.042  RMSE=4.385
  Training Gradient Boosting...   R²=0.771  MAE=2.754  RMSE=3.731
  Training MLP Neural Network...  R²=0.623  MAE=3.681  RMSE=4.794

[3/4] Selecting best model (by R²)...
  Best model : Gradient Boosting  R²=0.771

[4/4] Saving artifacts...
  Saved → model/model.pkl
============================================================
```

---

### Option C — Predict directly from Python

**Single student:**

```python
import sys
sys.path.insert(0, 'src')
from predict import predict_single

score = predict_single(
    Hours_Studied        = 25,
    Attendance           = 88,
    Sleep_Hours          = 7,
    Previous_Scores      = 78,
    Tutoring_Sessions    = 3,
    Physical_Activity    = 4,
    Motivation_Level     = 'High',
    Parental_Involvement = 'High',
    Access_to_Resources  = 'High',
    Family_Income        = 'Medium',
    Teacher_Quality      = 'High',
    Peer_Influence       = 'Positive',
    Internet_Access      = 'Yes',
    Extracurricular_Activities = 'Yes',
    Learning_Disabilities      = 'No',
    School_Type                = 'Private',
    Parental_Education_Level   = 'College',
    Distance_from_Home         = 'Near',
    Gender                     = 'Male',
)

print(f"Predicted Exam Score: {score}")
# Output → Predicted Exam Score: 97.43
```

**Multiple students from a CSV:**

```python
from predict import predict_batch
import pandas as pd

df = pd.read_csv("my_students.csv")      # same feature columns, no Exam_Score
df["Predicted_Score"] = predict_batch(df)
df.to_csv("results_with_predictions.csv", index=False)
print(df[["Hours_Studied", "Attendance", "Predicted_Score"]].head())
```

---

## 🤖 Machine Learning Models

### Models Trained

| Model | Algorithm Type | Key Hyperparameters |
|---|---|---|
| Linear Regression | Linear | Default (with Standard Scaling) |
| Decision Tree | Tree-based | `max_depth=8`, `random_state=42` |
| Random Forest | Bagging Ensemble | `n_estimators=150`, `random_state=42` |
| **Gradient Boosting** ✅ | Boosting Ensemble | `n_estimators=150`, `learning_rate=0.1` |
| MLP Neural Network | Deep Learning | `layers=(128,64,32)`, `max_iter=500` |
| XGBoost *(optional)* | Boosting Ensemble | `n_estimators=150`, `learning_rate=0.1` |

### Preprocessing Pipeline

```
Raw CSV
   ↓
Drop rows with missing Exam_Score
   ↓
Label Encode all categorical columns (LabelEncoder per column)
   ↓
80 / 20 train-test split (random_state=42)
   ↓
Standard Scale features for Linear Regression & MLP only
   ↓
Train all models → Evaluate → Pick best by R²
   ↓
Save: model + scaler + label encoders + feature names → model.pkl
```

### Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **R²** | 1 - SS_res/SS_tot | % of score variance explained. 1.0 = perfect. |
| **MAE** | mean(\|y - ŷ\|) | Average prediction error in exam score points |
| **RMSE** | √mean((y - ŷ)²) | Penalises large errors more than MAE |

---

## 📈 Model Results

Trained on **6,607 students** — 80% train, 20% test:

| Model | R² Score | MAE | RMSE | Rank |
|---|---|---|---|---|
| 🥇 **Gradient Boosting** | **0.7715** | **2.754** | **3.731** | **Best** |
| 🥈 Random Forest | 0.6844 | 3.042 | 4.385 | 2nd |
| 🥉 MLP Neural Network | 0.6228 | 3.681 | 4.794 | 3rd |
| Linear Regression | 0.5247 | 4.187 | 5.382 | 4th |
| Decision Tree | 0.4579 | 3.681 | 5.747 | 5th |

> **Gradient Boosting** achieves an R² of **0.77**, meaning it explains 77% of the variation in exam scores. Its average prediction error (MAE) is only **2.75 exam score points** — highly accurate for a real-world educational dataset.

---

## 🖥️ Dashboard Walkthrough

### Page 1 — 📊 Data Analysis

This page gives you a full picture of the dataset before any modelling.

**What you see:**
- **5 KPI cards** at the top — total students, feature count, average score, max score, min score
- **Exam Score Distribution** — histogram showing how scores are spread across all students
- **Score by Motivation Level** — box plot comparing Low / Medium / High motivation groups
- **Study Hours vs Score** — scatter plot with regression line showing the positive correlation
- **Attendance vs Score** — scatter plot proving attendance matters
- **Sleep Hours vs Score** — shows the optimal sleep range for performance
- **Parental Involvement** — bar chart showing average score by involvement level
- **Full Correlation Heatmap** — colour-coded matrix of all 19 features against each other
- **Previous Scores vs Exam Score** — scatter coloured by motivation level
- **6 Data Insights** — automatically generated plain-English observations from the data

---

### Page 2 — 🤖 Model Performance

This page shows how well each machine learning model performed and which features matter most.

**What you see:**
- **Best Model Badge** — highlights the winning model with its R², MAE, and RMSE
- **Full Metrics Table** — all models compared side by side in a sortable table
- **3 Comparison Bar Charts** — one each for R², MAE, and RMSE (best model highlighted in green)
- **Feature Importance Chart** — horizontal bar chart ranking all 19 features by their influence on predictions
- **Radar Chart** — spider/radar plot overlaying all models on 3 normalised axes for quick visual comparison

---

### Page 3 — 🔮 Score Predictor

This is the core user-facing tool — enter student details and get an instant prediction.

**What you see:**
- **19 input controls** — sliders for numeric features, dropdowns for categorical ones
- **Predict button** — triggers the trained model to compute a score
- **Score gauge** — animated arc gauge showing the predicted score from 0 to 100
- **Grade badge** — A / B / C / D label with colour coding
- **Personalised tips** — specific recommendations based on what the student scored low on
- **What-If Simulator** — line chart showing how the predicted score changes as study hours increase from 1 to 44, with all other factors held fixed
- **Batch Prediction section** — upload a CSV file, click predict, download a results CSV with predicted scores for every student

---

## 🛠️ Technologies Used

| Tool | Purpose | Version |
|---|---|---|
| **Python** | Core programming language | 3.12 |
| **Pandas** | Data loading, manipulation, preprocessing | ≥ 3.0 |
| **NumPy** | Numerical operations and array handling | ≥ 2.4 |
| **Scikit-learn** | All ML models, preprocessing, metrics | ≥ 1.8 |
| **XGBoost** | Optional extra boosting model | ≥ 2.0 |
| **Matplotlib** | All chart rendering in the dashboard | ≥ 3.10 |
| **Seaborn** | Heatmaps and statistical plots | ≥ 0.13 |
| **Joblib** | Saving and loading trained model artifacts | ≥ 1.5 |
| **Streamlit** | Interactive web dashboard framework | ≥ 1.40 |
| **Jupyter** | EDA notebook environment | any |

---

## 📦 Requirements

Full contents of `requirements.txt`:

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
streamlit>=1.28.0
```

---

## 🔑 Key Findings from the Data

After analysing 6,607 students, the most important factors affecting exam scores are:

| Rank | Factor | Finding |
|---|---|---|
| 🥇 1 | **Hours Studied** | Strongest predictor. Students studying 30+ hrs/week score significantly higher. |
| 🥈 2 | **Previous Scores** | Past performance is a strong indicator of future results. |
| 🥉 3 | **Attendance** | Students with ≥90% attendance consistently outperform those below 75%. |
| 4 | **Motivation Level** | High motivation students score ~8–10 points more than low motivation peers. |
| 5 | **Parental Involvement** | High involvement correlates with noticeably better outcomes. |
| 6 | **Access to Resources** | Better access = better scores, especially for self-study. |
| 7 | **Tutoring Sessions** | Even 2–3 sessions/month produce measurable improvement. |
| 8 | **Peer Influence** | Positive peer environment lifts performance; negative peers drag it down. |
| 9 | **Sleep Hours** | 7–9 hours is the optimal range; both under and over-sleeping hurts scores. |
| 10 | **Teacher Quality** | High quality teaching has a clear positive effect on outcomes. |

---

## 🐛 Known Issues & Fixes

### ❌ `KeyError: 'Parental_Involvement'`
**Cause:** Old `predict.py` used a hardcoded feature dict that didn't include all real dataset columns.  
**Fix:** `predict.py` now uses `**kwargs` and reads feature names dynamically from `model.pkl` — works with any dataset version.

### ❌ `use_container_width` deprecation warning
**Cause:** Streamlit removed `use_container_width=` after version 1.40.  
**Fix:** All instances replaced with `width='stretch'` throughout `app.py`.

---

## 🤝 Contributing

Contributions are welcome! Here is how to get started:

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/student-performance-ml.git

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make your changes and commit
git add .
git commit -m "Add: your feature description"

# 5. Push to GitHub
git push origin feature/your-feature-name

# 6. Open a Pull Request on GitHub
```

### Ideas for contributions
- Add XGBoost and LightGBM tuning with Optuna
- Add SHAP values for explainable AI
- Deploy to Streamlit Cloud or Hugging Face Spaces
- Add student comparison mode (compare two profiles side by side)
- Export full PDF report from the dashboard

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with attribution.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👨‍💻 Author

Built with ❤️ using Python, Scikit-learn, and Streamlit.

If this project helped you, please ⭐ star the repository on GitHub!

---

<div align="center">

**📊 Analyse → 🤖 Train → 🔮 Predict → 💡 Improve**

</div>
