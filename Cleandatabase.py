import pandas as pd

# Load the dataset
df = pd.read_csv("C:/Users/sansk/Documents/Project 2/heart_disease_uci.csv")

# Check missing values (optional)
print("Missing values before cleaning:\n", df.isnull().sum())

# Drop all rows containing any missing values
df_clean = df.dropna()

# Check shape after cleaning (optional)
print("Before:", df.shape)
print("After :", df_clean.shape)

# Save the cleaned dataset
df_clean.to_csv("heart_disease_uci_cleaned.csv", index=False)
output_path = 'C:/Users/sansk/Desktop/heart_disease_uci_cleaned.csv'
df_clean.to_csv(output_path, index=False)


print("Cleaned file saved at:", output_path)
print("Cleaned file saved as: heart_disease_uci_cleaned.csv")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

file_path = r"C:/Users/sansk/Documents/Project 2/heart_disease_uci_cleaned.csv"
df = pd.read_csv(file_path)

# 2. Separate Features and Target
# ---------------------------------------------
# Change the label name if different in your file
target_column = "num"
X = df.drop(columns=[target_column])
y = df[target_column]

# 3. Identify numeric & categorical columns
# ---------------------------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# ---------------------------------------------
# 4. Preprocessing
# ---------------------------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 5. Build the Model Pipeline
# ---------------------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=10
    ))
])

# 6. Train-Test Split
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------
# 7. Train the Model
# ---------------------------------------------
model.fit(X_train, y_train)

# ---------------------------------------------
# 8. Evaluate Model
# ---------------------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Save Model
# ---------------------------------------------
model_path = r"C:\Users\sansk\Documents\Project 2\heart_disease_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("Model saved at:", model_path)

import streamlit as st
import pandas as pd
import pickle

#Load the trained model
# -----------------------------
model_path = r"C:/Users/sansk/Documents/Project 2/heart_disease_model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict the risk of heart disease.")

# Define Input Fields
# (Use names exactly matching dataset)
# --------------------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["male", "female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal", [0, 1, 2, 3])

# --------------------------------------
# Prediction
# --------------------------------------
if st.button("Predict"):
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
        
