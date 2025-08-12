
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model


app = FastAPI(
    title="EV Energy Forecasting & Smart Charging API",
    description="API for predicting energy consumption and charging duration using GRU-XGBoost model.",
    version="1.0.0"
)

# Load models and scalers at startup
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")
gru_model = load_model("gru_model.h5")
xgb_energy = joblib.load("xgb_energy.pkl")
xgb_duration = joblib.load("xgb_duration.pkl")

# Example input schema (adjust fields as needed)
class PredictionInput(BaseModel):
    Vehicle_Model: int
    Battery_Capacity_kWh: float
    Charging_Station_ID: int
    Charging_Station_Location: int
    Charging_End_Time: int
    Charging_Rate_kW: float
    Charging_Cost_USD: float
    Time_of_Day: int
    Day_of_Week: int
    State_of_Charge_Start: float
    State_of_Charge_End: float
    Distance_Driven_since_last_charge_km: float
    Temperature_C: float
    Vehicle_Age_years: float
    Charger_Type: int
    User_Type: int
    Hour: int
    Weekday: int
    IsWeekend: int
    Power_x_Duration: float
    Model_x_Weekday: int
    Lag1_Energy: float
    Lag2_Energy: float
    Lag1_Duration: float
    Rolling_Mean_Energy: float
    Rolling_Std_Energy: float

@app.get("/health", summary="Health Check", description="Check if the API is running.")
def health():
    """
    Returns API health status.
    """
    return {"status": "ok"}


@app.post("/predict", summary="Predict Energy & Charging Duration", description="Predicts energy consumed and charging duration based on input features.")
def predict(input: PredictionInput):
    """
    Returns predicted energy consumption and charging duration for given input features.
    """
    # Convert input to numpy array (order must match training)
    X_input = np.array([
        input.Vehicle_Model,
        input.Battery_Capacity_kWh,
        input.Charging_Station_ID,
        input.Charging_Station_Location,
        input.Charging_End_Time,
        input.Charging_Rate_kW,
        input.Charging_Cost_USD,
        input.Time_of_Day,
        input.Day_of_Week,
        input.State_of_Charge_Start,
        input.State_of_Charge_End,
        input.Distance_Driven_since_last_charge_km,
        input.Temperature_C,
        input.Vehicle_Age_years,
        input.Charger_Type,
        input.User_Type,
        input.Hour,
        input.Weekday,
        input.IsWeekend,
        input.Power_x_Duration,
        input.Model_x_Weekday,
        input.Lag1_Energy,
        input.Lag2_Energy,
        input.Lag1_Duration,
        input.Rolling_Mean_Energy,
        input.Rolling_Std_Energy
    ]).reshape(1, -1)

    # Scale input
    X_input_scaled = scaler_X.transform(X_input)
    X_input_gru = X_input_scaled.reshape((1, 1, X_input_scaled.shape[1]))

    # GRU prediction (scaled output)
    gru_pred_scaled = gru_model.predict(X_input_gru)

    # Inverse transform GRU output to original scale
    gru_pred = scaler_y.inverse_transform(gru_pred_scaled)

    # Merge GRU output with scaled input for XGBoost
    X_input_xgb = np.hstack([X_input_scaled, gru_pred])

    # XGBoost predictions
    energy_pred = xgb_energy.predict(X_input_xgb)
    duration_pred = xgb_duration.predict(X_input_xgb)

    # Inverse transform predictions to original scale
    preds = scaler_y.inverse_transform(np.array([[energy_pred[0], duration_pred[0]]]))
    energy_final = float(preds[0][0])
    duration_final = float(preds[0][1])

    return {
        "energy_consumed": energy_final,
        "charging_duration": duration_final
    }

