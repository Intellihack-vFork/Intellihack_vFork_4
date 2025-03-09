from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

# Load model and data
model = load_model('../models/lstm_model.h5')
df = pd.read_csv('../data/trading_summary.csv', parse_dates=['Date'], index_col='Date')

# Initialize scaler
scaler = MinMaxScaler()
features = ['Close', '5D_MA', '10D_MA', 'Price_Change', 'Volatility', 
           'Lag_1', 'Lag_3', 'Lag_5', 'Volume']
scaler.fit(df[features])

class PredictionRequest(BaseModel):
    last_n_days: int = 30

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Prepare input data
    input_data = df[features].tail(request.last_n_days).values
    scaled_data = scaler.transform(input_data)
    
    # Reshape for model
    sequence = scaled_data.reshape(1, request.last_n_days, len(features))
    
    # Make prediction
    prediction = model.predict(sequence)
    prediction = scaler.inverse_transform(
        np.hstack([prediction, np.zeros((prediction.shape[0], len(features)-1))])
    )[:,0]
    
    return {
        "prediction_dates": pd.date_range(
            start=df.index[-1] + pd.Timedelta(days=1),
            periods=5
        ).strftime('%Y-%m-%d').tolist(),
        "predicted_prices": prediction.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)