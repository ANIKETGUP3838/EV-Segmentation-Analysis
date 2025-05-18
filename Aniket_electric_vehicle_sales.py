#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries:-

import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation


# In[2]:


# Load the dataset:-

data=pd.read_csv("C:\\Users\\DELL\\Downloads\\EV_Dataset.csv")


# Data Overview
# 
# Let's take a look at the first few rows of the dataset to understand its structure.

# In[3]:


data.head(10)


# In[4]:


data.dtypes


# In[5]:


data.columns


# In[6]:


data.shape


# In[9]:


data.info()


# Data Preprocessing
# 
# Before diving into analysis, we need to ensure our data is clean and ready for exploration. This includes parsing dates and checking for missing values.

# In[10]:


# Convert 'Date' column to datetime format:-

data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')


# In[11]:


# Check for missing values:-

data.isnull().sum()


# In[20]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Extract relevant features for clustering (e.g., EV_Sales_Quantity)
X = data[['EV_Sales_Quantity']]

# Determine the maximum number of clusters to test
max_clusters = 10

# Calculate Within-Cluster Sum of Squared Errors (WCSS) for different numbers of clusters
wcss = []
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.title('Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[23]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

categorical_features = ['State', 'Vehicle_Class', 'Vehicle_Category']
numerical_features = ['EV_Sales_Quantity']
X = data[categorical_features + numerical_features].copy()
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

scaler = StandardScaler()
X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])


# Applying K-Means clustering
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)

# Fit K-Means on the preprocessed (encoded and scaled) data
data["Cluster"] = kmeans.fit_predict(X_encoded)

# Displaying cluster assignments with original non-encoded columns for readability
# We use the original 'data' DataFrame which now includes the 'Cluster' column
# and the original categorical columns for easier interpretation.
print(data[["State", "Vehicle_Class", "Vehicle_Category", "EV_Sales_Quantity", "Cluster"]].head(10))


# In[24]:


# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=data["EV_Sales_Quantity"],
    y=[0] * len(data),  # Use a constant y-value for a simple visualization
    hue=data["Cluster"],
    palette="viridis",
    s=100,
    alpha=0.7
)
plt.xlabel("EV Sales Quantity")
plt.ylabel(" ")  # Remove y-axis label as it doesn't have a meaningful value
plt.title("K-Means Clustering of EV Sales")
plt.legend(title="Cluster")
plt.show()

# Visualize the distribution of data points across clusters
plt.figure(figsize=(8, 6))
data['Cluster'].value_counts().plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Distribution of Data Points Across Clusters (K=3)')
plt.xlabel('Cluster')
plt.ylabel('Number of Data Points')
plt.show()


# In[33]:


# Creating bar charts for different numerical features grouped by clusters
fig, axes = plt.subplots(1, 1, figsize=(10, 6))  # Reduced to 1x1 grid

# EV Sales Quantity by Clusters
sns.barplot(x=data["Cluster"], y=data["EV_Sales_Quantity"], palette="viridis", ax=axes) # changed axes[0,0] to axes
axes.set_title("EV Sales Quantity by Cluster")
axes.set_xlabel("Cluster")
axes.set_ylabel("EV Sales Quantity")

# Adjust layout and display
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(10, 6))  # Reduced to 1x1 grid

# EV Sales Quantity by Clusters
sns.barplot(x=data["Cluster"], y=data["Vehicle_Category"], palette="viridis", ax=axes) # changed axes[0,0] to axes
axes.set_title("Vehicle Category by Cluster")
axes.set_xlabel("Cluster")
axes.set_ylabel("Vehicle Category")

# Adjust layout and display
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(20, 10))  # Reduced to 1x1 grid

# EV Sales Quantity by Clusters
sns.barplot(x=data["Cluster"], y=data["Vehicle_Type"], palette="viridis", ax=axes) # changed axes[0,0] to axes
axes.set_title("Vehicle Type by Cluster")
axes.set_xlabel("Cluster")
axes.set_ylabel("Vehicle Type")

# Adjust layout and display
plt.tight_layout()
plt.show()
#Vehicle_Class

fig, axes = plt.subplots(1, 1, figsize=(9, 6))  # Reduced to 1x1 grid

# EV Sales Quantity by Clusters
sns.barplot(x=data["Cluster"], y=data["Vehicle_Class"], palette="viridis", ax=axes) # changed axes[0,0] to axes
axes.set_title("Vehicle Class by Cluster")
axes.set_xlabel("Cluster")
axes.set_ylabel("Vehicle Class")

# Adjust layout and display
plt.tight_layout()
plt.show()


# Exploratory Data Analysis
# 
# Let's explore the data to uncover trends and patterns in EV sales across different states and vehicle categories.

# In[12]:


# Plot EV sales over year:-

plt.figure(figsize=(10, 6))
yearly_sales = data.groupby('Year')['EV_Sales_Quantity'].sum()
sns.lineplot(x=yearly_sales.index, y=yearly_sales.values, marker='o')
plt.title("EV Sales Over Years")
plt.xlabel("Year")
plt.ylabel("Total EV Sales Quantity")
plt.grid()
plt.show()


# Inference
# 
# EV sales grew quickly in recent years, showing that more people are interested in them.
# The sudden drop in 2024 suggests there might be new challenges that need attention to keep EV sales going up.

# In[34]:


# Plot sales by vehicle category
plt.figure(figsize=(10, 6))
sns.barplot(x='Vehicle_Category', y='EV_Sales_Quantity',
data=data, ci=None)
plt.title('EV Sales by Vehicle Category')
plt.show()


# Inference
# 
# The chart shows the total electric vehicle (EV) sales by category. It highlights that 2-wheelers have the highest sales, followed by 3-wheelers, while 4-wheelers, buses, and other vehicle categories have significantly lower sales. This indicates that EV adoption is primarily driven by 2-wheelers and 3-wheelers.

# Correlation Analysis
# 
# Let's examine the correlation between numeric variables to understand potential relationships.

# In[35]:


# Select only numeric columns for correlation analysis:-

numeric_df = data.select_dtypes(include=[np.number])

# Plot the correlation heatmap:-

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Inference
# 
# The heatmap shows the correlation between the year and EV sales quantity. The correlation value is 0.095, which is very low. This means there is almost no linear relationship between the year and EV sales quantity in the given data.

# Top 10 States by EV Sales
# 
# A horizontal bar chart highlighting the states with the highest EV sales.

# In[36]:


#EV sales by state (top 10 states):-

plt.figure(figsize=(12, 8))
state_sales = data.groupby('State')['EV_Sales_Quantity'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=state_sales.values, y=state_sales.index, palette='viridis')
plt.title("Top 10 States by EV Sales")
plt.xlabel("Total EV Sales Quantity")
plt.ylabel("State")
plt.grid(axis='x')
plt.show()


# In[16]:


#Distribution of EV sales quantities:-

plt.figure(figsize=(10, 6))
sns.histplot(data['EV_Sales_Quantity'], bins=30, kde=True, color='green')
plt.title("Distribution of EV Sales Quantities")
plt.xlabel("EV Sales Quantity")
plt.ylabel("Frequency")
plt.grid(axis='y')
plt.show()


# In[37]:


# Set figure size
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Sorting data by Place for better visualization
df_sorted = data.sort_values("State")

# Line plots for different metrics
sns.lineplot(x="State", y="EV_Sales_Quantity", data=df_sorted, marker='o', ax=axes[0, 0], color='b')
axes[0, 0].set_title("EV Sales Quantity by Place")
axes[0, 0].tick_params(axis='x', rotation=90)

sns.lineplot(x="State", y="Vehicle_Type", data=df_sorted, marker='o', ax=axes[0, 1], color='r')
axes[0, 1].set_title("Vehicle Type by Place")
axes[0, 1].tick_params(axis='x', rotation=90)

sns.lineplot(x="State", y="Vehicle_Category", data=df_sorted, marker='o', ax=axes[1, 0], color='g')
axes[1, 0].set_title("Vehicle Category by Place")
axes[1, 0].tick_params(axis='x', rotation=90)

sns.lineplot(x="State", y="Vehicle_Class", data=df_sorted, marker='o', ax=axes[1, 1], color='purple')
axes[1, 1].set_title("Vehicle Class by Place")
axes[1, 1].tick_params(axis='x', rotation=90)

# Adjust layout for clarity
plt.tight_layout()
plt.show()


# Inference
# 
# The graph shows the distribution of EV sales quantities, with a clear peak indicating a high frequency of sales in the range of around 100,000 units. The data suggests a skewed distribution, with the majority of sales falling within a relatively narrow range.

# Yearly Sales Trends for Different Vehicle Classes
# 
# Compare trends of EV sales over years for different vehicle classes.

# In[17]:


#Yearly sales trends for different vehicle classes:-

plt.figure(figsize=(10, 8))
class_trends = data.groupby(['Year', 'Vehicle_Type'])['EV_Sales_Quantity'].sum().unstack()
class_trends.plot(kind='line', marker='o', figsize=(10, 8))
plt.title("Yearly EV Sales Trends by Vehicle Class")
plt.xlabel("Year")
plt.ylabel("Total EV Sales Quantity")
plt.yticks([100000,200000,300000,400000,500000,600000,700000,800000])
plt.legend(title="Vehicle_Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()


# Inference
# 
# The graph shows the yearly EV sales trends across various vehicle classes, with rapid growth particularly in the 2-wheeler and 3-wheeler personal and shared vehicle segments. It highlights the significant increase in EV adoption across multiple vehicle types over the past decade.

# EV Sales by Vehicle Category
# 
# Visualize the distribution of sales across different vehicle categories.

# In[18]:


#EV sales by vehicle category:-

plt.figure(figsize=(10, 6))
vehicle_category_sales = data.groupby('Vehicle_Category')['EV_Sales_Quantity'].sum().sort_values(ascending=False)
sns.barplot(x=vehicle_category_sales.values, y=vehicle_category_sales.index, palette='spring')
plt.title("EV Sales by Vehicle Category")
plt.xlabel("Total EV Sales Quantity")
plt.ylabel("Vehicle Category")
plt.show()


# In[50]:


import pandas as pd
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

print(data.head())

# Print the column names and their data types to verify how columns are represented
print(data.info())

# Aggregate the data by 'State' and 'Vehicle_Type'
df_trend = data.groupby("State")["Vehicle_Type"].sum().reset_index()
df_trend = df_trend.sort_values(by="Vehicle_Type")
# Print the first few rows of the aggregated data to discern its structure
print(df_trend.head())

# Print the column names and their data types to verify how columns are represented
print(df_trend.info())

# Aggregate sales data by state
df_trend = data.groupby("State")["EV_Sales_Quantity"].sum().reset_index()

# Sorting by total sales to identify growth trends
df_trend = df_trend.sort_values(by="EV_Sales_Quantity")

# Fitting an Exponential Smoothing model
model = ExponentialSmoothing(df_trend["EV_Sales_Quantity"], trend="add", seasonal=None)
fit_model = model.fit()

# Forecasting the next 5 years
future_steps = 5
future_sales = fit_model.forecast(future_steps)

# Plotting trend
plt.figure(figsize=(10, 5))
plt.plot(df_trend["State"], df_trend["EV_Sales_Quantity"], marker="o", label="Actual Sales")
plt.plot(range(len(df_trend), len(df_trend) + future_steps), future_sales, marker="o", linestyle="dashed", label="Forecasted Sales", color="red")
plt.xticks(rotation=45)
plt.xlabel("State")
plt.ylabel("EV Sales Quantity")  # Corrected y-label
plt.title("EV Market Sales Trend and Forecast")  # Corrected title
plt.legend()
plt.show()

