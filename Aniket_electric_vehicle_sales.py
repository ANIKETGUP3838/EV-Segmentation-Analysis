import io
import sys
from PIL import Image
import requests
from io import BytesIO
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
st.title("Electric Vehicle Sales Data Analysis")

url = "https://github.com/ANIKETGUP3838/EV-Segmentation-Analysis/raw/main/what-is-an-ev-scaled.jpg"  # Use raw GitHub link
response = requests.get(url)
image = Image.open(BytesIO(response.content))

st.image(image, caption='Electric Vehicle Charging', use_column_width=True)

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

    st.subheader("Data Types: ")
    st.write(data.dtypes)

    st.subheader("Dataset Info")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

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

        # Cluster-wise Analysis of Categorical Features
    st.subheader("Cluster-wise Breakdown of Key Features")

    # 1. EV Sales Quantity by Cluster
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Cluster", y="EV_Sales_Quantity", data=data, palette="viridis", ax=ax)
    ax.set_title("EV Sales Quantity by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("EV Sales Quantity")
    st.pyplot(fig)

    # 2. Vehicle Category Count by Cluster
    fig, ax = plt.subplots(figsize=(10, 6))
    category_counts = data.groupby(['Cluster', 'Vehicle_Category']).size().reset_index(name='Count')
    sns.barplot(x='Cluster', y='Count', hue='Vehicle_Category', data=category_counts, palette="viridis", ax=ax)
    ax.set_title("Vehicle Category Distribution by Cluster")
    st.pyplot(fig)

    # 3. Vehicle Type Count by Cluster
    fig, ax = plt.subplots(figsize=(12, 6))
    type_counts = data.groupby(['Cluster', 'Vehicle_Type']).size().reset_index(name='Count')
    sns.barplot(x='Cluster', y='Count', hue='Vehicle_Type', data=type_counts, palette="viridis", ax=ax)
    ax.set_title("Vehicle Type Distribution by Cluster")
    st.pyplot(fig)

    # 4. Vehicle Class Count by Cluster
    fig, ax = plt.subplots(figsize=(9, 6))
    class_counts = data.groupby(['Cluster', 'Vehicle_Class']).size().reset_index(name='Count')
    sns.barplot(x='Cluster', y='Count', hue='Vehicle_Class', data=class_counts, palette="viridis", ax=ax)
    ax.set_title("Vehicle Class Distribution by Cluster")
    st.pyplot(fig)

    st.header("Yearly Sales Trends by Vehicle Type")
    fig6, ax6 = plt.subplots(figsize=(10, 8))
    class_trends = data.groupby(['Year', 'Vehicle_Type'])['EV_Sales_Quantity'].sum().unstack()
    class_trends.plot(kind='line', marker='o', ax=ax6)
    ax6.set_title("Yearly EV Sales Trends by Vehicle Type")
    ax6.set_xlabel("Year")
    ax6.set_ylabel("Total EV Sales Quantity")
    ax6.set_yticks([100000,200000,300000,400000,500000,600000,700000,800000])
    ax6.legend(title="Vehicle_Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid()
    plt.tight_layout()
    st.pyplot(fig6)

    st.header("🗺️ State-wise Line Trends")

    df_sorted = data.sort_values("State")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Line 1: EV Sales Quantity
    sns.lineplot(x="State", y="EV_Sales_Quantity", data=df_sorted, marker='o', ax=axes[0, 0], color='b')
    axes[0, 0].set_title("EV Sales Quantity by State")
    axes[0, 0].tick_params(axis='x', rotation=90)

    # Line 2: Vehicle Type
    sns.lineplot(x="State", y="Vehicle_Type", data=df_sorted, marker='o', ax=axes[0, 1], color='r')
    axes[0, 1].set_title("Vehicle Type by State")
    axes[0, 1].tick_params(axis='x', rotation=90)

    # Line 3: Vehicle Category
    sns.lineplot(x="State", y="Vehicle_Category", data=df_sorted, marker='o', ax=axes[1, 0], color='g')
    axes[1, 0].set_title("Vehicle Category by State")
    axes[1, 0].tick_params(axis='x', rotation=90)

    # Line 4: Vehicle Class
    sns.lineplot(x="State", y="Vehicle_Class", data=df_sorted, marker='o', ax=axes[1, 1], color='purple')
    axes[1, 1].set_title("Vehicle Class by State")
    axes[1, 1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
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
