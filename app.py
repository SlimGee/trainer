from flask import Flask, request, jsonify
import pickle
import joblib
import pandas as pd
import datetime
from datetime import date

app = Flask(__name__)

# Load the saved models
cpi_model = pickle.load(open("cpi_model.pkl", "rb"))
exchange_rate_model = joblib.load("exchange_rate_model.joblib")


@app.route("/predict_cpi", methods=["POST"])
def predict_cpi():
    data = request.get_json()

    new_exchange_rate = float(data["value"])

    predicted_cpi = cpi_model.predict([[new_exchange_rate]])[0]
    return jsonify({"predicted": predicted_cpi})


@app.route("/predict_exchange_rate", methods=["POST"])
def predict_exchange_rate():
    data = request.get_json()
    new_date = pd.to_datetime(data["value"])

    new_date_ordinal = new_date.toordinal()
    provided_date = date.fromordinal(new_date_ordinal)

    # Base date - Dec 31, 2023
    base_date = date(year=2023, month=12, day=31)

    # Calculate the number of days to forecast
    num_forecasts = (provided_date - base_date).days

    predicted_exchange_rate = exchange_rate_model.forecast(
        steps=num_forecasts, exog=new_date_ordinal
    )[num_forecasts - 1]
    return jsonify({"predicted": predicted_exchange_rate})


if __name__ == "__main__":
    app.run(debug=True)
