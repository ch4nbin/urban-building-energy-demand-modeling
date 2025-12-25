"""
Data Processing Module

Handles loading, cleaning, and basic preprocessing of building energy and weather data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(file_path=None):
    """
    Load building energy and weather data from CSV file.
    
    If no file is provided, generates synthetic data for demonstration purposes.
    In production, this would load from ASHRAE or similar public datasets.
    
    Parameters:
    -----------
    file_path : str or Path, optional
        Path to CSV file containing energy and weather data.
        Expected columns: timestamp, energy_consumption, temperature, humidity, etc.
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with datetime index and required columns
    """
    if file_path is None or not Path(file_path).exists():
        print("No data file provided or file not found. Generating synthetic data...")
        return generate_synthetic_data()
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Falling back to synthetic data generation...")
        return generate_synthetic_data()


def generate_synthetic_data(n_days=365):
    """
    Generate synthetic building energy and weather data for demonstration.
    
    Creates realistic patterns:
    - Daily cycles (higher during business hours)
    - Weekly cycles (lower on weekends)
    - Seasonal patterns (higher in summer/winter)
    - Weather-dependent variations
    
    Parameters:
    -----------
    n_days : int
        Number of days of data to generate
    
    Returns:
    --------
    pd.DataFrame
        Synthetic dataset with hourly timestamps
    """
    # Create hourly timestamps
    start_date = pd.Timestamp('2023-01-01 00:00:00')
    timestamps = pd.date_range(start=start_date, periods=n_days * 24, freq='H')
    
    # Generate weather features
    np.random.seed(42)
    n = len(timestamps)
    
    # Temperature: seasonal pattern with daily variation
    day_of_year = timestamps.dayofyear
    hour_of_day = timestamps.hour
    temperature = (
        20 + 10 * np.sin(2 * np.pi * day_of_year / 365) +  # Seasonal
        5 * np.sin(2 * np.pi * hour_of_day / 24) +  # Daily
        np.random.normal(0, 2, n)  # Random variation
    )
    
    # Humidity: inverse relationship with temperature
    humidity = 60 - 0.5 * temperature + np.random.normal(0, 5, n)
    humidity = np.clip(humidity, 20, 90)
    
    # Energy consumption: complex pattern based on time and weather
    hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
    day_of_week = timestamps.dayofweek
    
    # Base consumption: higher during business hours (9-17), lower on weekends
    base_consumption = (
        100 +  # Base load
        50 * (hour_of_day >= 9) * (hour_of_day < 17) * (day_of_week < 5) +  # Business hours
        30 * hour_sin +  # Daily cycle
        -20 * (day_of_week >= 5)  # Weekend reduction
    )
    
    # Weather-dependent consumption (HVAC load)
    # Higher consumption when temperature deviates from comfort zone (20-24°C)
    hvac_load = (
        20 * np.abs(temperature - 22) +  # Heating/cooling demand
        5 * np.abs(humidity - 50) / 10  # Humidity control
    )
    
    # Add some noise
    energy_consumption = base_consumption + hvac_load + np.random.normal(0, 5, n)
    energy_consumption = np.maximum(energy_consumption, 50)  # Minimum load
    
    # Create dataframe
    df = pd.DataFrame({
        'timestamp': timestamps,
        'energy_consumption': energy_consumption,
        'temperature': temperature,
        'humidity': humidity,
        'hour': hour_of_day,
        'day_of_week': day_of_week,
        'day_of_year': day_of_year
    })
    
    print(f"Generated {len(df)} hourly records ({n_days} days)")
    return df


def clean_data(df):
    """
    Clean and preprocess the energy data.
    
    Handles missing values, outliers, and ensures proper data types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe with energy and weather data
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif df.index.dtype == 'object':
        df.index = pd.to_datetime(df.index)
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Handle missing values
    # Forward fill for short gaps, then backward fill
    df = df.fillna(method='ffill', limit=6)  # Fill up to 6 hours forward
    df = df.fillna(method='bfill', limit=6)  # Fill up to 6 hours backward
    
    # Remove any remaining NaN rows (should be minimal)
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} rows with missing values")
    
    # Remove extreme outliers (beyond 3 standard deviations)
    for col in ['energy_consumption', 'temperature', 'humidity']:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            outliers = (df[col] < mean - 3 * std) | (df[col] > mean + 3 * std)
            if outliers.sum() > 0:
                print(f"Removed {outliers.sum()} outliers from {col}")
                df = df[~outliers]
    
    # Ensure numeric columns are float
    numeric_cols = ['energy_consumption', 'temperature', 'humidity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Data cleaning complete. Final dataset: {len(df)} records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def prepare_data(file_path=None):
    """
    Complete data preparation pipeline: load and clean.
    
    Parameters:
    -----------
    file_path : str or Path, optional
        Path to data file
    
    Returns:
    --------
    pd.DataFrame
        Cleaned and prepared dataframe
    """
    df = load_data(file_path)
    df = clean_data(df)
    return df

