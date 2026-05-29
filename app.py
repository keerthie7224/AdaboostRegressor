# Import libraries
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Page configuration
st.set_page_config(
    page_title="AdaBoost Regression",
    layout="wide"
)

# Custom styling
st.markdown(
    """
    <style>
    h1{
        text-align:center;
    }

    .stButton>button{
        width:100%;
        height:50px;
        background-color:#4CAF50;
        color:white;
        border:none;
        border-radius:10px;
        font-size:18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title(
    "Student Performance Prediction Using AdaBoost Regressor"
)

# Base directory
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Dataset path
DATA_PATH = os.path.join(
    BASE_DIR,
    "student-por.csv"
)

# Models directory
MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

# Create models directory
os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

# Read dataset
df = pd.read_csv(DATA_PATH)

# Show dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Dataset shape
st.write(
    "Dataset Shape :",
    df.shape
)

# Remove duplicates
df = df.drop_duplicates()

# Numerical columns
num_cols = df.select_dtypes(
    include=np.number
).columns

# Categorical columns
cat_cols = df.select_dtypes(
    include="object"
).columns

# Fill numerical missing values
num_imputer = SimpleImputer(
    strategy="mean"
)

df[num_cols] = num_imputer.fit_transform(
    df[num_cols]
)

# Fill categorical missing values
if len(cat_cols) > 0:

    # Create imputer
    cat_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    # Transform columns
    df[cat_cols] = cat_imputer.fit_transform(
        df[cat_cols]
    )

# Store encoders
encoders = {}

# Encode categorical columns
if len(cat_cols) > 0:

    for col in cat_cols:

        # Create encoder
        le = LabelEncoder()

        # Encode values
        df[col] = le.fit_transform(
            df[col]
        )

        # Store encoder
        encoders[col] = le

# Correlation matrix
st.subheader("Correlation Matrix")

# Create correlation matrix
corr = df.corr()

# Create figure
fig1, ax1 = plt.subplots(
    figsize=(12, 8)
)

# Plot heatmap
heatmap = ax1.imshow(corr)

# Add labels
ax1.set_xticks(
    range(len(corr.columns))
)

ax1.set_yticks(
    range(len(corr.columns))
)

# Column names
ax1.set_xticklabels(
    corr.columns,
    rotation=90
)

ax1.set_yticklabels(
    corr.columns
)

# Add colorbar
plt.colorbar(heatmap)

# Show graph
st.pyplot(fig1)

# Features
X = df.drop(
    "G3",
    axis=1
)

# Target
y = df["G3"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale dataset
scaler = StandardScaler()

# Fit train data
X_train_scaled = scaler.fit_transform(
    X_train
)

# Transform test data
X_test_scaled = scaler.transform(
    X_test
)

# Hyperparameter tuning
st.subheader("Hyperparameter Tuning")

# Base estimator
base_model = DecisionTreeRegressor(
    max_depth=4
)

# Parameter grid
param_grid = {
    "n_estimators": [50, 100],
    "learning_rate": [0.01, 0.1, 1]
}

# Create model
ada = AdaBoostRegressor(
    estimator=base_model,
    random_state=42
)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=ada,
    param_grid=param_grid,
    cv=3,
    scoring="r2"
)

# Train model
grid_search.fit(
    X_train_scaled,
    y_train
)

# Best model
model = grid_search.best_estimator_

# Model path
MODEL_PATH = os.path.join(
    MODELS_DIR,
    "adaboost_regressor.pkl"
)

# Scaler path
SCALER_PATH = os.path.join(
    MODELS_DIR,
    "scaler.pkl"
)

# Save model
joblib.dump(
    model,
    MODEL_PATH
)

# Save scaler
joblib.dump(
    scaler,
    SCALER_PATH
)

# Best parameters
st.write(
    "Best Parameters :",
    grid_search.best_params_
)

# Predict output
y_pred = model.predict(
    X_test_scaled
)

# Metrics
mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

# Performance section
st.subheader("Model Performance")

# Create columns
c1, c2, c3, c4 = st.columns(4)

# Show metrics
c1.metric(
    "MAE",
    round(mae, 2)
)

c2.metric(
    "MSE",
    round(mse, 2)
)

c3.metric(
    "RMSE",
    round(rmse, 2)
)

c4.metric(
    "R2 Score",
    round(r2, 2)
)

# Actual vs predicted graph
st.subheader(
    "Actual vs Predicted"
)

# Create figure
fig2, ax2 = plt.subplots(
    figsize=(7, 5)
)

# Scatter plot
ax2.scatter(
    y_test,
    y_pred
)

# Labels
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")

# Title
ax2.set_title(
    "AdaBoost Regression"
)

# Show graph
st.pyplot(fig2)

# Prediction section
st.subheader(
    "Predict Student Final Grade"
)

# Store user input
user_input = {}

# Create columns
col1, col2 = st.columns(2)

# Feature columns
columns = X.columns.tolist()

# Split columns
first_half = columns[:len(columns)//2]
second_half = columns[len(columns)//2:]

# First column inputs
with col1:

    for col in first_half:

        user_input[col] = st.number_input(
            f"{col}",
            value=float(df[col].mean())
        )

# Second column inputs
with col2:

    for col in second_half:

        user_input[col] = st.number_input(
            f"{col}",
            value=float(df[col].mean())
        )

# Predict button
if st.button("Predict Final Grade"):

    # Convert dataframe
    input_df = pd.DataFrame(
        [user_input]
    )

    # Arrange columns
    input_df = input_df[X.columns]

    # Scale input
    input_scaled = scaler.transform(
        input_df
    )

    # Predict
    prediction = model.predict(
        input_scaled
    )

    # Show result
    st.success(
        f"Predicted Final Grade : {round(prediction[0], 2)}"
    )