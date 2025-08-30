# stroke_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ==============================
# Streamlit App Configuration
# ==============================
st.set_page_config(page_title="🧠 Stroke Prediction App", layout="wide")



st.title("🧠 Stroke Prediction App")
st.markdown("Built with **Streamlit** | Dataset: Healthcare Stroke Prediction")

# Sidebar navigation
menu = ["About Author", "Dataset Overview", "EDA", "Model Training", "Prediction"]
choice = st.sidebar.selectbox("Navigate", menu)

# ==============================
# Load Dataset
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    return df

df = load_data()

# ==============================
# About Author
# ==============================
if choice == "About Author":
    st.markdown("""
    # 👩‍💻 About the Author  

    **Javeria Jameel**  
    *Data Scientist (Python • SQL • ML & Analytics) based in Pakistan.*  

    Focus areas include:
    - 🟢 NLP  
    - 🟠 Recommender Systems  
    - 🔴 Predictive Modeling  

    ---
    """)
    st.success("Open to internships and junior data roles; happy to collaborate on open-source DS/ML projects.")

# ==============================
# Dataset Overview
# ==============================
elif choice == "Dataset Overview":
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    st.markdown("**Shape of dataset:**")
    st.write(df.shape)

    st.markdown("**Missing Values:**")
    st.write(df.isnull().sum())

    st.markdown("**Statistical Summary:**")
    st.write(df.describe())

# ==============================
# Exploratory Data Analysis (EDA)
# ==============================
elif choice == "EDA":
    st.subheader("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="stroke", palette="Set2", ax=ax)
        ax.set_title("Stroke Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="gender", palette="Set1", ax=ax)
        ax.set_title("Gender Distribution")
        st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.histplot(df["age"], bins=30, kde=True, color="blue", ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# ==============================
# Model Training
# ==============================
elif choice == "Model Training":
    st.subheader("🤖 Model Training & Comparison")

    # Preprocessing
    data = df.copy()
    data["bmi"].fillna(data["bmi"].mean(), inplace=True)
    data = pd.get_dummies(data, columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"], drop_first=True)

    scaler = StandardScaler()
    data[["age", "avg_glucose_level", "bmi"]] = scaler.fit_transform(data[["age", "avg_glucose_level", "bmi"]])

    X = data.drop("stroke", axis=1)
    y = data["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

    results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
    st.write(results_df)

    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="Accuracy", data=results_df, palette="viridis", ax=ax)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    best_model_name = max(results, key=results.get)
    st.success(f"✅ Best Model: **{best_model_name}** with Accuracy = {results[best_model_name]:.2f}")

    # Save best model
    best_model = models[best_model_name]
    with open("stroke_prediction_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

# ==============================
# Prediction Page
# ==============================
elif choice == "Prediction":
    st.subheader("🔮 Make a Prediction")

    with open("stroke_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)

    # User input form
    age = st.slider("Age", 0, 100, 30)
    avg_glucose = st.number_input("Average Glucose Level", min_value=40.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["children", "Govt_job", "Never_worked", "Private", "Self-employed"])
    residence = st.selectbox("Residence Type", ["Rural", "Urban"])
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    if st.button("Predict Stroke Risk"):
        # Preprocess input
        user_data = pd.DataFrame({
            "id": [0],  # dummy
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "ever_married": [ever_married],
            "work_type": [work_type],
            "Residence_type": [residence],
            "avg_glucose_level": [avg_glucose],
            "bmi": [bmi],
            "smoking_status": [smoking_status],
            "gender": [gender]
        })

        # Match preprocessing
        user_data = pd.get_dummies(user_data, columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"], drop_first=True)
        all_cols = model.feature_names_in_
        for col in all_cols:
            if col not in user_data.columns:
                user_data[col] = 0
        user_data = user_data[all_cols]

        # Scale numeric values (assuming model already trained on scaled values)
        scaler = StandardScaler()
        user_data[["age", "avg_glucose_level", "bmi"]] = scaler.fit_transform(user_data[["age", "avg_glucose_level", "bmi"]])

        prediction = model.predict(user_data)[0]

        if prediction == 1:
            st.error("⚠️ High risk of Stroke")
        else:
            st.success("✅ Low risk of Stroke")
