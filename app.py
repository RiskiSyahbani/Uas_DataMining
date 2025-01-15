import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Logistic Regression Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS for background
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://www.harianbatakpos.com/wp-content/uploads/2024/04/addfg.jpg');
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("📊 Logistic Regression Classification Dashboard")
st.write("An interactive dashboard for visualizing and classifying data using Logistic Regression.")

# File uploader
uploaded_file = "Regression.csv"

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Handle categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        st.sidebar.write("### Encoding Categorical Columns")
        for col in categorical_cols:
            st.sidebar.write(f"Encoding column: {col}")
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])

    # Dataset overview
    st.sidebar.header("Dataset Overview")
    st.sidebar.write("### Preview:")
    st.sidebar.write(data.head())

    st.sidebar.write("### Dataset Info:")
    buffer = StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    st.sidebar.text(info_str)

    # Select target and features
    st.sidebar.header("Model Configuration")
    target_column = st.sidebar.selectbox("Select Target Column:", data.columns)
    feature_columns = st.sidebar.multiselect("Select Feature Columns:", [col for col in data.columns if col != target_column])

    if target_column and feature_columns:
        X = data[feature_columns]
        y = data[target_column]

        # Ensure target is categorical
        if y.dtype in [np.float64, np.int64] and y.nunique() > 2:
            st.sidebar.write("### Target Conversion")
            threshold = st.sidebar.slider("Set Threshold for Binary Classification:", float(y.min()), float(y.max()), float(y.mean()))
            y = (y > threshold).astype(int)

        # Train-test split
        test_size = st.sidebar.slider("Test Size (%):", 10, 50, 20, step=5) / 100
        random_state = st.sidebar.number_input("Random State:", value=42, step=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Logistic Regression
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        st.write("## Model Performance")
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")

        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.write("### Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))  # Reduced size
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Visualizations
        st.write("## Data Visualizations")
        plot_type = st.selectbox("Select Plot Type:", ["Scatter Plot", "Histogram", "Box Plot"])

        if plot_type == "Scatter Plot":
            x_axis = st.selectbox("X-axis:", feature_columns)
            y_axis = st.selectbox("Y-axis:", feature_columns)
            hue = st.selectbox("Color By:", [None] + feature_columns)
            fig, ax = plt.subplots(figsize=(5, 4))  # Reduced size
            sns.scatterplot(data=data, x=x_axis, y=y_axis, hue=hue, ax=ax)
            st.pyplot(fig)

        elif plot_type == "Histogram":
            column = st.selectbox("Select Column:", feature_columns)
            bins = st.slider("Number of Bins:", 5, 50, 20)
            fig, ax = plt.subplots(figsize=(5, 4))  # Reduced size
            sns.histplot(data[column], bins=bins, kde=True, ax=ax)
            st.pyplot(fig)

        elif plot_type == "Box Plot":
            column = st.selectbox("Select Column:", feature_columns)
            fig, ax = plt.subplots(figsize=(5, 4))  # Reduced size
            sns.boxplot(data=data, y=column, ax=ax)
            st.pyplot(fig)

        # Prediction for new data
        st.write("## Predict New Data")
        input_data = {col: st.number_input(f"Input {col}:", value=0.0) for col in feature_columns}
        input_df = pd.DataFrame([input_data])

        if st.button("Predict"):
            prediction = model.predict(input_df)[0]
            st.write("### Prediction:", prediction)
    else:
        st.warning("Please select both target and feature columns to proceed.")
else:
    st.info("Regression.csv")
