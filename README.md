# 🔋 Electric Vehicle (EV) Sales Segmentation and Forecasting in India

An interactive data science project to analyze, segment, forecast, and simulate electric vehicle (EV) sales trends in India. Built with Python and Streamlit, this tool helps startups and stakeholders identify the best regions for EV business entry, understand key sales drivers, and simulate policy impacts.

---

## 🚀 Live Demo

👉 **Try the App Now:**  
🔗 [EV Segmentation Streamlit App](https://ev-segmentation-analysis-ntfyv4dkbs3cvvybqnkkpd.streamlit.app/)

---

## 📌 Problem Statement

**Identify high-potential regions and strategies for launching an electric vehicle (EV) business in India using data-driven market segmentation and sales forecasting.**

---

## 🧠 Background

India’s EV market is rapidly growing due to environmental concerns, rising fuel prices, and government incentives. However, regional disparities in EV infrastructure, policy, and demand make it challenging for startups to decide where and how to launch. This project uses machine learning and visualization to uncover actionable insights for EV adoption strategy.

---

## ✅ Project Outcomes

- 🔹 **Segmented the Indian EV market** into high-growth, mid-tier, and low-adoption clusters using K-Means.
- 🔹 **Forecasted future sales trends** with Exponential Smoothing, highlighting states with the most growth potential.
- 🔹 **Identified key sales drivers** (region, vehicle type, etc.) using a Random Forest Regressor.
- 🔹 **Detected anomalies in EV sales** using Isolation Forest to uncover unusual demand patterns.
- 🔹 **Built a Policy Impact Simulator** to visualize the effect of subsidies, petrol prices, and infra growth on projected EV sales.
- 🔹 **Developed a full-featured Streamlit dashboard** for interactive EDA, modeling, and simulation.

---

## 🛠️ Features

- 📥 Upload EV sales CSV files for analysis  
- 📊 Dynamic visualizations (line charts, bar plots, heatmaps, cluster plots)  
- 🧪 Machine Learning models (K-Means, Random Forest, Isolation Forest, Holt-Winters Forecasting)  
- 📈 Interactive time-series forecasting by state  
- ⚙️ Policy impact simulation using sliders  
- 📤 Download cleaned datasets  
- 🚧 Preview of future features (LSTM, NLP, Recommendation engine)

---

## 📊 Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Data Analysis & ML |
| Streamlit | Web App UI |
| scikit-learn | Clustering, Regression, Anomaly Detection |
| statsmodels | Time Series Forecasting |
| Plotly & Seaborn | Interactive Visualizations |
| PyCaret | AutoML experiments (optional) |

---
## 📂 Dataset

To use this application, make sure your uploaded CSV file includes the following columns:

| Column Name         | Description                                                  | Example                         |
|---------------------|--------------------------------------------------------------|----------------------------------|
| `State`             | The name of the Indian state                                 | Maharashtra, Gujarat             |
| `Vehicle_Class`     | Class of vehicle (e.g., 2-wheeler, 3-wheeler, 4-wheeler)     | 2W, 3W, 4W                       |
| `Vehicle_Category`  | Vehicle usage category                                       | Passenger, Commercial            |
| `Vehicle_Type`      | Specific type within class/category                          | E-Scooter, E-Rickshaw, EV Car    |
| `EV_Sales_Quantity` | Number of EV units sold                                      | 12345                            |
| `Date`              | Date of sale (used to extract year for trend analysis)       | 15/08/2022                       |

💡 The `Date` column should be in `DD/MM/YYYY` format. The app will automatically convert it to extract the `Year` for time series analysis and forecasting.

---

## 📌 Simulation Sliders

The app includes interactive sliders to simulate the impact of different policies and external factors on EV adoption:

- **EV Subsidy per Vehicle (₹)**
- **Petrol Price per Litre (₹)**
- **Annual Growth in Charging Infrastructure (%)**

💡 These parameters dynamically affect the projected annual increase in EV sales shown within the dashboard.

---

## 🔮 Future Enhancements

Planned advanced features for future development:

- 📉 **LSTM-based deep learning forecasting**  
  More robust time-series modeling using Recurrent Neural Networks.

- 📃 **NLP on EV policy documents and consumer sentiment**  
  Understand public opinion and policy impacts through natural language processing.

- 🤖 **EV recommendation system for user preferences**  
  Recommend EV models to users based on use-case, budget, and features.

---
## 📬 Contact

For questions or collaborations, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/aniket-gupta-90b49725a/) or [Email](aniket25287@gmail.com).
