import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from dateutil.relativedelta import relativedelta
from datetime import datetime
import wandb
from pathlib import Path
import os 
import boto3
import io

timelabel = datetime.today() - relativedelta(days=1)
timelabel = timelabel.strftime('%Y_%m_%d')

bucket_name = "soccer-storage"
feature_scaler_source = "webapp-storage/encoder/feature_scaler.pkl"
target_scaler_source = 'app/target_scaler.pkl' # webapp-storage/encoder/target_scaler.pkl"
label_encoder_source = "webapp-storage/encoder/label_encoder.pkl"
model_source = 'app/lstm_model_3.h5' #'src/baq/libs/models/lstm_model_3.h5'
raw_data_source = f's3://soccer-storage/webapp-storage/data/raw/raw_data_{timelabel}.csv'
processed_data_source = f's3://soccer-storage/webapp-storage/data/processed/processed_data_{timelabel}.csv'  #save path
seasonal_medians_source = "s3://soccer-storage/webapp-storage/data/raw/seasonal_medians.csv"

os.environ['WANDB_API_KEY'] = 'SORRY ITS A SECRET JING'

def create_sequences(data, target_column, sequence_length):
    X = []
    feature_data = data.drop(columns=[target_column])
    feature_data = feature_data.select_dtypes(include=[np.number]).values.astype(np.float32)

    for i in range(len(data) - sequence_length + 1):
        X.append(feature_data[i:i + sequence_length])

    return np.array(X)

def predict(model, input_data):
    y_pred = model.predict(input_data)
    return y_pred

def rolling_forecast(model, data, target_col, sequence_length, forecast_horizon):
    rolling_data = data.copy()
    rolling_forecast_df = pd.DataFrame(columns=['time', 'predicted_value'])

    for _ in range(forecast_horizon):
        input_sequence = create_sequences(rolling_data.tail(sequence_length), target_col, sequence_length)
        predicted_value = predict(model, input_sequence)

        # Create new timestamp
        new_row = rolling_data.tail(1)
        new_timestamp = pd.to_datetime(new_row.index[0]) + pd.Timedelta(hours=1)
        new_row.index = pd.DatetimeIndex([new_timestamp])

        # Append new row to rolling data
        rolling_data = pd.concat([rolling_data, new_row])

        # Extract datetime directly (not as a DatetimeIndex)
        rolling_forecast_df = pd.concat(
            [rolling_forecast_df, pd.DataFrame([[new_timestamp, predicted_value]], columns=['time', 'predicted_value'])],
            ignore_index=True
        )
        

    return rolling_forecast_df

def load_production_model():
    '''
    Load the latest production model artifact from W&B without opening a session.
    '''
    try:
        # Get latest production model artifact without opening a session
        api = wandb.Api()
        artifact = api.artifact('chogerlate/wandb-registry-model/baq-forecastors:production', type='model')
        
        # Download and load model
        artifact_dir = artifact.download()
        model_path = Path(artifact_dir) / "model"
        model = joblib.load(model_path)
        
        print(f"Successfully loaded production model from {model_path}")
        return model
        
    except Exception as e:
        raise Exception(f"Failed to load production model: {str(e)}")

def single_step_forecasting(
    model: object,
    X_test: pd.DataFrame,
) -> pd.Series:
    """
    Generate predictions for a single step forecasting model.
    
    This function makes predictions using a trained model on test data
    for a single time step ahead forecasting task.
    
    Args:
        model: Trained model object with a predict method
        X_test: Test features dataframe
        
    Returns:
        pd.Series: Series containing the predicted values
    """
    predictions = model.predict(X_test)
    return pd.Series(predictions, index=X_test.index)

def multi_step_forecasting(
    model: object,
    X_test: pd.DataFrame,
    forecast_horizon: int,
) -> pd.Series:
    """
    Generate predictions for a multi-step forecasting model.
    
    This function iteratively predicts multiple steps ahead by using each prediction
    as an input for the next step prediction. It updates the feature matrix after
    each prediction to simulate real forecasting conditions.
    
    Args:
        model: Trained model object
        X_test: Test features
        forecast_horizon: Number of time steps to forecast ahead
        
    Returns:
        pd.Series: Series containing multi-step predictions
    """
    # Make a copy of the test data to avoid modifying the original
    X_forecast = X_test.copy()
    
    # Initialize predictions array
    predictions = []
    
    # Iteratively predict each step
    for step in range(forecast_horizon):
        # Generate prediction for current step
        step_pred = model.predict(X_forecast.iloc[[step]])
        predictions.append(step_pred[0])
        
        # If not the last step, update features for next prediction
        if step < forecast_horizon - 1:
            # Update lag features if they exist in the dataset
            for lag_col in [col for col in X_forecast.columns if col.endswith(f'_lag_{step+1}')]:
                target_col = lag_col.split('_lag_')[0]
                # Find the next lag column to update
                next_lag_col = f"{target_col}_lag_{step+2}"
                if next_lag_col in X_forecast.columns:
                    X_forecast.loc[X_forecast.index[step+1], next_lag_col] = step_pred[0]
    
    return pd.Series(predictions, index=X_test.index[:forecast_horizon])

def standardize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize input DataFrame by ensuring all required columns are present and in correct order.
    
    Args:
        df (pd.DataFrame): Input DataFrame to standardize
        
    Returns:
        pd.DataFrame: Standardized DataFrame with all required columns
    """
    required_columns = [
        'temperature_2m_(°C)', 'relative_humidity_2m_(%)', 'dew_point_2m_(°C)', 
        'apparent_temperature_(°C)', 'precipitation_(mm)', 'rain_(mm)', 'snowfall_(cm)', 
        'snow_depth_(m)', 'pressure_msl_(hPa)', 'surface_pressure_(hPa)', 'cloud_cover_(%)', 
        'cloud_cover_low_(%)', 'cloud_cover_mid_(%)', 'cloud_cover_high_(%)', 
        'et0_fao_evapotranspiration_(mm)', 'vapour_pressure_deficit_(kPa)', 
        'wind_speed_10m_(km/h)', 'wind_speed_100m_(km/h)', 'wind_direction_10m_(°)', 
        'wind_direction_100m_(°)', 'wind_gusts_10m_(km/h)', 'soil_temperature_0_to_7cm_(°C)', 
        'soil_temperature_7_to_28cm_(°C)', 'soil_temperature_28_to_100cm_(°C)', 
        'soil_temperature_100_to_255cm_(°C)', 'soil_moisture_0_to_7cm_(m³/m³)', 
        'soil_moisture_7_to_28cm_(m³/m³)', 'soil_moisture_28_to_100cm_(m³/m³)', 
        'soil_moisture_100_to_255cm_(m³/m³)', 'pm10_(μg/m³)', 'carbon_monoxide_(μg/m³)', 
        'carbon_dioxide_(ppm)', 'nitrogen_dioxide_(μg/m³)', 'sulphur_dioxide_(μg/m³)', 
        'ozone_(μg/m³)', 'methane_(μg/m³)', 'uv_index_clear_sky', 'uv_index', 'dust_(μg/m³)', 
        'aerosol_optical_depth', 'weather_code', 'hour', 'dayofweek', 'month', 'is_weekend', 
        'is_night', 'sin_hour', 'cos_hour', 'pm2_5_(μg/m³)_lag1', 'pm2_5_(μg/m³)_lag3', 
        'pm2_5_(μg/m³)_lag6', 'pm2_5_(μg/m³)_lag12', 'pm2_5_(μg/m³)_lag24', 
        'pm2_5_(μg/m³)_rollmean3', 'pm2_5_(μg/m³)_rollmean6', 'pm2_5_(μg/m³)_rollmean12', 
        'pm2_5_tier'
    ]
    
    # Create a copy of the input DataFrame
    df_processed = df.copy()
    
    # Create missing columns with zeros
    missing_columns = set(required_columns) - set(df_processed.columns)
    if missing_columns:
        print("Adding missing columns:")
        for col in missing_columns:
            print(f"  - {col}")
            df_processed[col] = 0
    
    # Select only the required columns in the correct order
    df_processed = df_processed[required_columns]
    
    return df_processed

if __name__ == "__main__":
    
    model = load_production_model()
    print("Model loaded successfully.")
    
    # Load and preprocess the data
    df = pd.read_csv(processed_data_source)
    print("Original data head:")
    print(df.head())
    
    # Convert time column to datetime and set as index if it exists
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    
    # Standardize input columns
    df_processed = standardize_input_columns(df)
    
    print("\nProcessed data head:")
    print(df_processed.head())
    
    # Perform multi-step forecasting
    predictions = multi_step_forecasting(model, df_processed, forecast_horizon=5)
    print("\nMulti-step forecasting completed.")
    print("Predictions:")
    print(predictions)

    # model = joblib.load(target_scaler_source)
    # print("Target scaler loaded successfully.")