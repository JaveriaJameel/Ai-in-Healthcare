# 🧠 Stroke Prediction using Machine Learning  

A complete **end-to-end machine learning project** that predicts whether a patient is at risk of **stroke** based on health parameters.  

The project covers:  
- Exploratory Data Analysis (EDA)  
- Data preprocessing & feature engineering  
- Model training & evaluation (Logistic Regression, Random Forest, SVM)  
- Deployment with **Streamlit app** for real-time predictions  

---

## 📑 Table of Contents  
1. [Project Overview](#project-overview)  
2. [Dataset & Context](#dataset--context)  
3. [Attribute Information](#attribute-information)  
4. [Setup (Requirements)](#setup-requirements)  
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
6. [Model Training & Evaluation](#model-training--evaluation)  
7. [Model Comparison](#model-comparison)  
8. [Streamlit App](#streamlit-app)  
9. [Results](#results)  
10. [Conclusion & Future Work](#conclusion--future-work)  
11. [About the Author](#about-the-author)  

---

## 📌 Project Overview  

Stroke is the **second leading cause of death globally** according to the **World Health Organization (WHO)**, responsible for nearly **11% of deaths worldwide**.  

This project builds a **machine learning classifier** that predicts stroke risk using patient health records, including:  
- Age  
- Hypertension & Heart Disease history  
- Glucose levels & BMI  
- Lifestyle & Demographics  

---

## 📊 Dataset & Context  

- **Dataset:** Healthcare Stroke Prediction Dataset (5,110 entries, 12 features)  
- **Source:** Confidential (used for educational purposes only)  
- **Target Variable:**  
  - `stroke` = `1` → Patient had a stroke  
  - `stroke` = `0` → Patient did not have a stroke  

---

## 🧾 Attribute Information  

| Feature             | Description |
|----------------------|-------------|
| id                  | Unique identifier |
| gender              | Male, Female, Other |
| age                 | Patient’s age |
| hypertension        | 0 → No, 1 → Yes |
| heart_disease       | 0 → No, 1 → Yes |
| ever_married        | No / Yes |
| work_type           | Children, Govt_job, Private, Self-employed, Never_worked |
| Residence_type      | Rural / Urban |
| avg_glucose_level   | Average glucose level |
| bmi                 | Body Mass Index |
| smoking_status      | Formerly smoked, Never smoked, Smokes, Unknown |
| stroke              | 1 → Stroke, 0 → No Stroke |

---

## ⚙️ Setup (Requirements)  

Clone the repository and install dependencies:  

```bash
git clonehttps://github.com/JaveriaJameel/stroke-prediction.git
cd stroke-prediction
---
requirements.txt
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
pickle-mixin
---
📈 Exploratory Data Analysis (EDA)

EDA included:

- Stroke distribution (imbalanced dataset ~5% strokes)

- Gender & Age distribution

- Hypertension / Heart Disease vs Stroke correlation

- Smoking status & stroke occurrence

- Correlation heatmap
---
# 📊 Example Visualizations:

- Countplots of categorical variables

- Histograms for numerical features (Age, Glucose, BMI)

- Heatmap of feature correlations
---
# 🤖 Model Training & Evaluation

Trained three ML models:

- Logistic Regression

- Random Forest Classifier

- Support Vector Machine (SVM)

Evaluation metrics: Accuracy, Confusion Matrix, Classification Report
---
📊 Model Comparison
| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | \~95.1%  |
| Random Forest       | \~95.0%  |
| SVM                 | \~95.1%  |

✅ Best Choice: Random Forest (chosen for deployment)
---
# 🌐 Streamlit App

An interactive Streamlit App was built for real-time stroke prediction.

Features:

-  Dataset exploration (shape, preview, summary, missing values)

- Interactive EDA visualizations

- Model training & comparison

- Real-time prediction form with patient data input

- Risk classification:

 - ⚠️ High Risk of Stroke

 -  ✅ Low Risk of Stroke

Run the app locally:
streamlit run stroke_streamlit_app.py
---
📊 Results

- Achieved ~95% accuracy across models

- Dataset imbalance (few stroke cases) made recall lower for minority class

 - Random Forest chosen as final model for deployment

 # 🚀 Conclusion & Future Work

- This project demonstrates an end-to-end ML pipeline for health risk prediction.

- Challenges: Imbalanced dataset → poor recall for stroke cases.

- Improvements:

  -  Apply SMOTE or class weighting to handle imbalance

  - Try advanced ML (XGBoost, LightGBM) or deep learning models

  - Deploy on cloud (Heroku, AWS, etc.) for public access
---
# 👩‍💻 About the Author

Javeria Jameel —
Data Scientist (Python • SQL • ML & Analytics) based in Pakistan

I build portfolio-ready projects that transform raw data into deployable ML applications.

Focus Areas:

= 📝 NLP (Text Classification)

- 🎯 Recommender Systems

- 📊 Predictive Modeling

📫 Connect with me:

- GitHub: https://github.com/JaveriaJameel

- LinkedIn: https://www.linkedin.com/in/javeria-jameel-88a95a380/

⭐ If you found this project helpful, don’t forget to star the repo!

---

👉 Do you want me to also generate a **`requirements.txt`** file for you (based on both your notebook + Streamlit code), so you can directly push both files and run them on GitHub?




psername/stroke-prediction.git
cd stroke-prediction
pip install -r requirements.txt
