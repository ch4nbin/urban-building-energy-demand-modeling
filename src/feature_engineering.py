"""
Feature Engineering Module

Creates time-series features for energy consumption prediction:
- Cyclical encoding for hour of day and day of week
- Lagged features (previous hour, previous day)
- Rolling statistics
"""

import pandas as pd
import numpy as np


def create_cyclical_features(df):
    """
    Create cyclical encodings for temporal features.
    
    Cyclical encoding (sin/cos) is preferred over one-hot encoding for
    temporal features because it preserves the cyclical nature (e.g., 
    hour 23 is close to hour 0).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with datetime index
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added cyclical features
    """
    df = df.copy()
    
    # Hour of day (0-23) -> sin/cos encoding
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Day of week (0=Monday, 6=Sunday) -> sin/cos encoding
    df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    # Day of year (1-365/366) -> sin/cos encoding for seasonal patterns
    df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    
    # Month (1-12) -> sin/cos encoding
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    return df


def create_lag_features(df, target_col='energy_consumption', lags=[1, 24, 168]):
    """
    Create lagged features from the target variable.
    
    Lag features capture temporal dependencies:
    - Lag 1: Previous hour (strong short-term correlation)
    - Lag 24: Same hour previous day (daily patterns)
    - Lag 168: Same hour previous week (weekly patterns)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with datetime index
    target_col : str
        Name of the target column to create lags from
    lags : list of int
        List of lag periods (in hours) to create
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added lag features
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df


def create_rolling_features(df, target_col='energy_consumption', windows=[3, 6, 24]):
    """
    Create rolling statistics features.
    
    Rolling features capture short-term trends and patterns:
    - Mean: Average consumption over recent period
    - Std: Variability in recent consumption
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with datetime index
    target_col : str
        Name of the target column
    windows : list of int
        Rolling window sizes (in hours)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added rolling features
    """
    df = df.copy()
    
    for window in windows:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}'] = (
            df[target_col].rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling standard deviation
        df[f'{target_col}_rolling_std_{window}'] = (
            df[target_col].rolling(window=window, min_periods=1).std()
        )
    
    return df


def create_weather_features(df):
    """
    Create additional features from weather data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with weather columns (temperature, humidity, etc.)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added weather-derived features
    """
    df = df.copy()
    
    if 'temperature' in df.columns:
        # Temperature deviation from comfort zone (assumed 20-24°C)
        df['temp_deviation'] = np.abs(df['temperature'] - 22)
        
        # Heating degree hours (when temp < 20°C)
        df['heating_degree_hours'] = np.maximum(20 - df['temperature'], 0)
        
        # Cooling degree hours (when temp > 24°C)
        df['cooling_degree_hours'] = np.maximum(df['temperature'] - 24, 0)
    
    if 'humidity' in df.columns:
        # Humidity deviation from comfort zone (assumed 40-60%)
        df['humidity_deviation'] = np.abs(df['humidity'] - 50)
    
    return df


def engineer_features(df, target_col='energy_consumption'):
    """
    Complete feature engineering pipeline.
    
    Creates all engineered features for the model:
    1. Cyclical temporal features
    2. Lag features
    3. Rolling statistics
    4. Weather-derived features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe with datetime index
    target_col : str
        Name of the target variable column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with all engineered features
    tuple
        (feature_names, target_name) for model training
    """
    print("Engineering features...")
    
    # Apply all feature engineering steps
    df = create_cyclical_features(df)
    df = create_lag_features(df, target_col=target_col)
    df = create_rolling_features(df, target_col=target_col)
    df = create_weather_features(df)
    
    # Remove rows with NaN (from lag features)
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} rows after feature engineering")
    
    # Identify feature columns (exclude target and original temporal columns)
    exclude_cols = [
        target_col,
        'hour',  # Original hour (we use cyclical encoding)
        'day_of_week',  # Original day (we use cyclical encoding)
        'day_of_year'  # Original day of year (we use cyclical encoding)
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Created {len(feature_cols)} features")
    print(f"Final dataset: {len(df)} records")
    
    return df, feature_cols, target_col

