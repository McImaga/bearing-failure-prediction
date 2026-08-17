# BearingGuard AI 🚀
### Bearing Fault Diagnosis using Machine Learning - 94.57% Accuracy

## 🔗 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bearingguard.streamlit.app)


# ⚙️ Bearing Fault Diagnosis Using Machine Learning
### Predictive Maintenance App for Industrial Rotating Machinery

**94.57% Test Accuracy | Random Forest | Streamlit | CWRU Dataset**


## 📌 About
This app predicts bearing faults using vibration data and a Random Forest model...


Predict bearing failures *before* they cause costly downtime. 

This intelligent diagnostic system classifies the health state of rolling element bearings using statistical vibration features. Built for engineers, plant managers, and IoT maintenance teams.


## 🎯 The Problem
Rolling element bearing failure = unplanned downtime, equipment damage, lost revenue. 
Traditional maintenance only catches faults after damage is done. 
We need predictive maintenance.

## 💡 Our Solution
A production-ready ML app that takes 9 vibration features and classifies bearings into 4 states:
- **Normal State**
- **Inner Race Fault** 
- **Ball Fault**
- **Outer Race Fault**

**Key Features:**
1.  **Dual Input**: Manual feature input OR Batch CSV upload for 1000s of readings
2.  **5 Models Compared**: Logistic Regression, SVM, KNN, Decision Tree, **Random Forest**
3.  **94.57% Accuracy**: Random Forest outperformed all models on Precision, Recall, and F1
4.  **EDA Dashboard**: Built-in data insights, Mutual Information scores, Train/Test split viz
5.  **Deployable**: Full preprocessing pipeline with StandardScaler included

## 📊 Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| **Random Forest** | **94.57%** | **94.65%** | **94.35%** | **94.38%** |
| Decision Tree | 93.26% | 93.54% | 93.26% | 93.26% |
| SVM | 91.30% | 92.24% | 91.30% | 91.15% |
| KNN | 90.87% | 91.73% | 90.87% | 90.90% |
| Logistic Regression | 90.65% | 90.87% | 90.65% | 90.46% |

## 🛠 Tech Stack
`Python` `Scikit-learn` `Pandas` `NumPy` `Streamlit` `Plotly` `Matplotlib`
**Dataset**: Case Western Reserve University Bearing Data Center - 2,300 samples

## ⚙️ How To Run Locally
```bash
git clone https://github.com/your-username/bearing-fault-diagnosis
cd bearing-fault-diagnosis
pip install -r requirements.txt
streamlit run app.py