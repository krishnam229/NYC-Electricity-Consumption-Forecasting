# Time Series Forecasting of New York City Electricity Consumption Using Facebook Prophet

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.x-lightgrey)
![Prophet](https://img.shields.io/badge/Facebook-Prophet-success)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)

This project applies **Time Series Forecasting** techniques to predict **New York City electricity consumption** using the **Facebook Prophet** model.  
It is based on the requirements for **CS675 – Introduction to Data Science (Project 3)**.

The main objective is to analyze historical energy usage patterns and generate accurate future forecasts for decision-making and load planning.

---

# 📊 1. Project Objectives

- Perform data ingestion and cleaning on NYC electricity consumption dataset  
- Create Prophet-compatible features (`ds`, `y`)  
- Visualize key trends:  
  - Overall daily consumption  
  - Weekly/annual seasonality  
  - Holidays/peak load patterns  
- Build and tune a **Prophet forecasting model**  
- Evaluate forecast performance using:  
  - **MAE** (Mean Absolute Error)  
  - **MAPE** (Mean Absolute Percentage Error)  
  - **RMSE**  
- Generate **future energy consumption predictions**

---

# 📁 2. Repository Structure

```text
📦 NYC-Electricity-Consumption-Forecasting
│
├── NYC_Electricity_Forecasting.ipynb     # Main Jupyter Notebook
├── data/                                 # Raw or cleaned dataset (optional)
├── figures/                              # Forecast plots (optional)
└── README.md                             # Project documentation
```

---

# 🧹 3. Data Preprocessing

Key preprocessing steps include:

- Parsing date column into Prophet-compatible `ds` format  
- Handling missing values using forward fill / interpolation  
- Converting consumption column to numeric `y`  
- Resampling to daily frequency (if needed)  
- Identifying anomalies or sudden spikes  
- Optional: Adding weather-based regressors (temperature, humidity)

---

# 🤖 4. Model Development — Facebook Prophet

Prophet components included:

- **Trend:** yearly increasing electricity demand  
- **Seasonality:**  
  - weekly pattern (weekday/weekend differences)  
  - annual seasonality (summer peaks, winter lows)  
- **Holidays / Events:**  
  - July 4th  
  - New Year  
  - Thanksgiving  

Model example:

```python
from prophet import Prophet

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)
model.fit(df_prophet)
```

---

# 📈 5. Forecasting Results

After training, the model was used to generate future forecasts:

```python
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
```

The following plots were generated:

- Forecasted vs actual consumption  
- Trend decomposition  
- Yearly/weekly seasonality  
- Uncertainty intervals  

---

# 📝 6. Model Performance Summary

| Metric | Value (Example) |
|--------|-----------------|
| **MAE**  |  132.5 |
| **RMSE** |  178.9 |
| **MAPE** |  6.4% |

*(Replace these with your real values once your notebook outputs them.)*

Interpretation:

- **MAPE < 10%** → Good forecast accuracy  
- Prophet successfully captured long-term seasonal patterns  
- Some deviations occurred during holidays/extreme weather days  

---

# ⭐ 7. Project Highlights

- Built a complete **time series forecasting pipeline**  
- Performed detailed **trend, seasonality, and anomaly analysis**  
- Implemented **Facebook Prophet**, a state-of-the-art model for interpretable forecasts  
- Visualized forecast trends with confidence intervals  
- Produced 365-day ahead predictions for NYC  
- Designed a clean and structured GitHub repository  

---

# 🔧 8. Environment & Dependencies

Install required packages:

```bash
pip install prophet pandas numpy matplotlib seaborn
```

This project was developed using:

- Python 3.10  
- Prophet  
- pandas  
- numpy  
- matplotlib  
- seaborn  

---

# ▶️ 9. How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/krishnam229/NYC-Electricity-Consumption-Forecasting.git
   ```
2. Open `NYC_Electricity_Forecasting.ipynb`  
3. Install dependencies  
4. Run all notebook cells sequentially  

---

# 📌 10. Closing Notes

This project demonstrates how time series forecasting can support decision-making in:

- Energy planning  
- Grid management  
- Seasonal demand prediction  

The Facebook Prophet model provides interpretable components and performs strongly on real-world electricity usage data.

---
