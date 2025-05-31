from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from keras.models import load_model
from datetime import datetime 
from dateutil.relativedelta import relativedelta
import pandas as pd
import joblib
import os
from pydantic import BaseModel
from typing import Optional, List
import pickle
from app.utils import (
    load_production_model,
    standardize_input_columns,
    multi_step_forecasting
)
import boto3
from io import StringIO

class ForecastRequest(BaseModel):
    forecast_horizon: Optional[int] = 48  # default if not provided

class PredictionResponse(BaseModel):
    predicted_value: float

class ForecastResponse(BaseModel):
    predictions: List[PredictionResponse]

app = FastAPI()

timelabel = datetime.today() - relativedelta(days=1)
timelabel = timelabel.strftime('%Y_%m_%d')

model_source = 'app/lstm_model_3.h5' #'src/baq/libs/models/lstm_model_3.h5'
raw_data_source = f's3://soccer-storage/webapp-storage/data/raw/raw_data_{timelabel}.csv'
processed_data_source = f's3://soccer-storage/webapp-storage/data/processed/processed_data_{timelabel}.csv'  #save path
seasonal_medians_source = "s3://soccer-storage/webapp-storage/data/raw/seasonal_medians.csv"

# Add S3 client
s3_client = boto3.client('s3')



@app.get("/")
def read_root():
    return {"message": "Welcome to the Weather Forecasting API"}



@app.post("/predict/onetime")
async def predict_onetime():
    '''
    This endpoint performs a one-time prediction using the production model.
    It loads the model from W&B, processed the data from S3, and returns the predictions.
    '''
    try:
        model = load_production_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    try:
        df = pd.read_csv(processed_data_source)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        df_processed = standardize_input_columns(df)
        predictions = multi_step_forecasting(model, df_processed, forecast_horizon=1)
        prediction_records = [{"predicted_value": float(val)} for val in predictions.values]
        return {"predictions": prediction_records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading or preprocessing data: {e}")

from fastapi import Body



@app.post("/predict/next", response_model=ForecastResponse)
async def predict_next(request: ForecastRequest):
    '''
    This endpoint performs multi-step forecasting using the production model.
    It loads the model from W&B, processes the data from S3, and returns the predictions.
    '''
    try:
        # Load the production model
        model = load_production_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    try:
        # Load and preprocess the data
        df = pd.read_csv(processed_data_source)
        
        # Convert time column to datetime and set as index if it exists
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        
        # Standardize input columns
        df_processed = standardize_input_columns(df)
        
        # Perform multi-step forecasting
        predictions = multi_step_forecasting(model, df_processed, forecast_horizon=request.forecast_horizon)
        
        # Convert predictions to the required format
        prediction_records = [
            {"predicted_value": float(val)} 
            for val in predictions.values
        ]
        
        return {"predictions": prediction_records}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")



@app.post("/predict/cache")
async def predict_and_cache():
    '''
    This endpoint performs multi-step forecasting using the production model.
    It is the same as the /predict/next endpoint, but it caches the predictions csv in S3.
    '''
    try:
        # Load the production model
        model = load_production_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    try:
        # Load and preprocess the data
        df = pd.read_csv(processed_data_source)
        
        # Convert time column to datetime and set as index if it exists
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        
        # Standardize input columns with error handling
        try:
            df_processed = standardize_input_columns(df)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error standardizing input columns: {str(e)}")
        
        # Perform multi-step forecasting for 96 steps
        predictions = multi_step_forecasting(model, df_processed, forecast_horizon=96)
        
        # Safely get the last timestamp
        try:
            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
                last_time = df.index[-1]
            else:
                last_time = pd.Timestamp.now()
        except Exception:
            last_time = pd.Timestamp.now()
            
        # Generate future timestamps
        prediction_times = pd.date_range(start=last_time, periods=len(predictions)+1, freq='H')[1:]
        
        # Ensure predictions and timestamps match in length
        predictions_df = pd.DataFrame({
            'time': prediction_times[:len(predictions)],
            'predicted_value': predictions.values[:len(prediction_times)]
        })
        
        # Save to S3
        current_time = datetime.now().strftime('%Y_%m_%d_%H')
        s3_path = f'webapp-storage/prediction-cache/predictions_{current_time}.csv'
        
        # Convert DataFrame to CSV string
        csv_buffer = StringIO()
        predictions_df.to_csv(csv_buffer, index=False)
        
        try:
            # Upload to S3
            s3_client.put_object(
                Bucket='soccer-storage',
                Key=s3_path,
                Body=csv_buffer.getvalue()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error uploading to S3: {str(e)}")
        
        return {
            "message": "Predictions cached successfully",
            "file_path": f"s3://soccer-storage/{s3_path}",
            "predictions_count": len(predictions_df),
            "time_range": {
                "start": predictions_df['time'].min().strftime('%Y-%m-%d %H:%M:%S'),
                "end": predictions_df['time'].max().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction caching: {str(e)}")



if __name__ == "__main__":
    # debugger
    df = pd.read_csv(processed_data_source)
    print(df.head())
    lstm_model = load_model(model_source, compile=False)
    y_pred = rolling_forecast(lstm_model, df, target_col='pm2_5_(μg/m³)', sequence_length=24, forecast_horizon=48)
    print(y_pred)

