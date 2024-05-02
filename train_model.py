import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import pickle
import joblib

# Load data
cpi_data = pd.read_csv("cpi.csv")
exchange_data = pd.read_csv("exchange_rate.csv")

# Preprocess data
cpi_data["Date"] = pd.to_datetime(cpi_data["Date"])
exchange_data["Date"] = pd.to_datetime(exchange_data["Date"])

# Convert 'Date' to datetime format
exchange_data.set_index("Date", inplace=True)
print(exchange_data)
# Merge data
data = pd.merge(cpi_data, exchange_data, on="Date", how="inner")

# Split data into features and targets
X_cpi = data["Exchange"].values.reshape(-1, 1)
y_cpi = data["Annula_Inflation_Rate"].values

X_exchange = data["Date"].values.reshape(-1, 1)
y_exchange = data["Exchange"].values

# Train linear regression model for CPI prediction
linear_reg = LinearRegression()
linear_reg.fit(X_cpi, y_cpi)

print(y_exchange)
exchange_data = exchange_data.asfreq("D")
# Train ARIMA model for exchange rate prediction
arima_model = ARIMA(exchange_data["Exchange"], order=(3, 2, 3))
arima_model_fit = arima_model.fit()

# Predict CPI
new_exchange_rate = 5000.0
predicted_cpi = linear_reg.predict([[new_exchange_rate]])
print(f"Predicted CPI for exchange rate {new_exchange_rate}: {predicted_cpi[0]}")

filename = "cpi_model.pkl"
pickle.dump(linear_reg, open(filename, "wb"))

# Save the ARIMA model
filename = "exchange_rate_model.joblib"
joblib.dump(arima_model_fit, filename)

# Predict exchange rate
import datetime

new_date = datetime.datetime(2024, 5, 1)
predicted_exchange_rate = arima_model_fit.forecast(steps=1, exog=new_date.toordinal())
print(f"Predicted exchange rate for {new_date}: {predicted_exchange_rate[0]}")
