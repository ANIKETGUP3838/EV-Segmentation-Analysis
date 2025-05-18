import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Streamlit settings
st.set_page_config(page_title="EV Sales Analysis", layout="wide")
st.title("Electric Vehicle Sales Data Analysis & Clustering")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your EV Sales CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Preprocessing
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')
        data['Year'] = data['Date'].dt.year

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # K-Means Clustering
    st.subheader("K-Means Clustering")
    try:
        categorical_features = ['State', 'Vehicle_Class', 'Vehicle_Category']
        numerical_features = ['EV_Sales_Quantity']

        X = data[categorical_features + numerical_features].copy()
        X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

        scaler = StandardScaler()
        X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])

        # Elbow method
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X_encoded)
            wcss.append(kmeans.inertia_)

        st.write("### Elbow Method Plot")
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker='o')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)

        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        data['Cluster'] = kmeans.fit_predict(X_encoded)

        st.write("### Sample Cluster Assignments")
        st.dataframe(data[["State", "Vehicle_Class", "Vehicle_Category", "EV_Sales_Quantity", "Cluster"]].head())

        st.write("### Cluster Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=data['Cluster'], palette='viridis', ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Clustering failed: {e}")

    # Sales Trend
    st.subheader("EV Sales Over Years")
    if 'Year' in data.columns:
        yearly_sales = data.groupby('Year')['EV_Sales_Quantity'].sum()
        fig, ax = plt.subplots()
        sns.lineplot(x=yearly_sales.index, y=yearly_sales.values, marker='o', ax=ax)
        ax.set_title("Yearly EV Sales")
        st.pyplot(fig)

    # Top states
    st.subheader("Top 10 States by EV Sales")
    top_states = data.groupby("State")["EV_Sales_Quantity"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_states.values, y=top_states.index, palette="viridis", ax=ax)
    st.pyplot(fig)

    # Forecasting
    st.subheader("State-wise Forecast using Exponential Smoothing")
    try:
        df_trend = data.groupby("State")["EV_Sales_Quantity"].sum().reset_index()
        df_trend = df_trend.sort_values(by="EV_Sales_Quantity")

        model = ExponentialSmoothing(df_trend["EV_Sales_Quantity"], trend="add", seasonal=None)
        fit_model = model.fit()
        forecast = fit_model.forecast(5)

        fig, ax = plt.subplots()
        ax.plot(df_trend["State"], df_trend["EV_Sales_Quantity"], marker="o", label="Actual")
        ax.plot(range(len(df_trend), len(df_trend) + 5), forecast, marker="o", linestyle="--", color="red", label="Forecast")
        ax.set_xticks(range(len(df_trend) + 5))
        ax.set_xticklabels(list(df_trend["State"]) + [f"F{i+1}" for i in range(5)], rotation=45)
        ax.set_title("Forecast of EV Sales per State")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Forecasting failed: {e}")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    try:
        numeric_df = data.select_dtypes(include=[np.number])
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    except:
        st.warning("Correlation analysis not possible — ensure numeric columns exist.")
else:
    st.info("Please upload a CSV file to begin analysis.")
