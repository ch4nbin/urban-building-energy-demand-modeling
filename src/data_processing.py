"""
Data Processing Module

Handles loading, cleaning, and basic preprocessing of building energy and weather data.
Supports:
- ASHRAE Great Energy Predictor III (Kaggle)
- UCI Building Energy Dataset
- Custom CSV files
- Synthetic data generation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_ashrae_data(train_path=None, weather_path=None, building_path=None):
    """
    Load ASHRAE Great Energy Predictor III dataset from Kaggle.
    
    The ASHRAE dataset consists of multiple files:
    - train.csv: Hourly energy consumption (meter readings)
    - weather_train.csv: Weather data
    - building_metadata.csv: Building characteristics
    
    Download from: https://www.kaggle.com/c/ashrae-energy-prediction/data
    
    Parameters:
    -----------
    train_path : str or Path, optional
        Path to train.csv file
    weather_path : str or Path, optional
        Path to weather_train.csv file
    building_path : str or Path, optional
        Path to building_metadata.csv file
    
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with energy, weather, and building data
    """
    data_dir = Path('data')
    
    # Try to find files if paths not provided
    if train_path is None:
        train_path = data_dir / 'train.csv'
    if weather_path is None:
        weather_path = data_dir / 'weather_train.csv'
    if building_path is None:
        building_path = data_dir / 'building_metadata.csv'
    
    train_path = Path(train_path)
    weather_path = Path(weather_path)
    building_path = Path(building_path)
    
    if not train_path.exists():
        raise FileNotFoundError(
            f"ASHRAE train.csv not found at {train_path}. "
            "Please download from https://www.kaggle.com/c/ashrae-energy-prediction/data"
        )
    
    print("Loading ASHRAE dataset...")
    
    # Load training data (energy consumption)
    train_df = pd.read_csv(train_path)
    print(f"Loaded {len(train_df):,} energy meter readings")
    
    # Load weather data
    if weather_path.exists():
        weather_df = pd.read_csv(weather_path)
        print(f"Loaded weather data for {weather_df['site_id'].nunique()} sites")
        
        # Merge weather data
        train_df = train_df.merge(weather_df, on=['site_id', 'timestamp'], how='left')
    else:
        print("Warning: weather_train.csv not found. Weather features will be missing.")
    
    # Load building metadata
    if building_path.exists():
        building_df = pd.read_csv(building_path)
        print(f"Loaded metadata for {len(building_df)} buildings")
        
        # Merge building data
        train_df = train_df.merge(building_df, on='building_id', how='left')
    else:
        print("Warning: building_metadata.csv not found. Building features will be missing.")
    
    # Convert timestamp
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    
    # Rename meter_reading to energy_consumption for consistency
    if 'meter_reading' in train_df.columns:
        train_df = train_df.rename(columns={'meter_reading': 'energy_consumption'})
    
    # For simplicity, aggregate to hourly building-level consumption
    # (ASHRAE has multiple meters per building)
    if 'building_id' in train_df.columns:
        # Group by building and timestamp, sum energy consumption
        df = train_df.groupby(['building_id', 'timestamp']).agg({
            'energy_consumption': 'sum',
            'air_temperature': 'mean',
            'dew_temperature': 'mean',
            'cloud_coverage': 'mean',
            'sea_level_pressure': 'mean',
            'wind_direction': 'mean',
            'wind_speed': 'mean',
            'precip_depth_1_hr': 'mean'
        }).reset_index()
        
        # Use air_temperature as temperature, calculate humidity from dew point
        if 'air_temperature' in df.columns:
            df['temperature'] = df['air_temperature']
        if 'dew_temperature' in df.columns and 'air_temperature' in df.columns:
            # Approximate humidity from dew point and temperature
            df['humidity'] = 100 * np.exp((17.27 * df['dew_temperature']) / (237.7 + df['dew_temperature'])) / \
                            np.exp((17.27 * df['temperature']) / (237.7 + df['temperature']))
            df['humidity'] = df['humidity'].clip(0, 100)
        
        # For single building analysis, select first building or aggregate all
        if df['building_id'].nunique() > 1:
            print(f"Dataset contains {df['building_id'].nunique()} buildings.")
            print("Aggregating across all buildings for building-level analysis...")
            df = df.groupby('timestamp').agg({
                'energy_consumption': 'sum',
                'temperature': 'mean',
                'humidity': 'mean'
            }).reset_index()
        else:
            # Single building - just use timestamp as index
            df = df.set_index('timestamp')[['energy_consumption', 'temperature', 'humidity']]
            df = df.reset_index()
    else:
        df = train_df
    
    # Set timestamp as index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    print(f"Processed ASHRAE data: {len(df)} hourly records")
    return df


def load_uci_data(file_path=None):
    """
    Load UCI Energy Efficiency Dataset.
    
    Note: UCI dataset is not time-series data (it's building characteristics),
    so this function converts it to a format compatible with time-series analysis
    by creating synthetic timestamps. For true time-series analysis, use ASHRAE data.
    
    Download from: https://archive.ics.uci.edu/dataset/242/energy+efficiency
    
    Parameters:
    -----------
    file_path : str or Path, optional
        Path to UCI dataset CSV file
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe
    """
    data_dir = Path('data')
    
    if file_path is None:
        file_path = data_dir / 'energy_efficiency.csv'
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"UCI dataset not found at {file_path}. "
            "Download from https://archive.ics.uci.edu/dataset/242/energy+efficiency"
        )
    
    print("Loading UCI Energy Efficiency dataset...")
    df = pd.read_csv(file_path)
    
    # UCI dataset has building characteristics, not time-series
    # For this project, we'll use ASHRAE instead
    print("Warning: UCI dataset is not time-series data.")
    print("For hourly energy consumption analysis, please use ASHRAE dataset.")
    
    return df


def load_data(file_path=None, dataset_type='auto'):
    """
    Load building energy and weather data from various sources.
    
    Supports:
    - ASHRAE dataset (from Kaggle)
    - Custom CSV files
    - Synthetic data generation (fallback)
    
    Parameters:
    -----------
    file_path : str or Path, optional
        Path to data file or directory containing ASHRAE files
    dataset_type : str
        Type of dataset: 'ashrae', 'uci', 'custom', or 'auto' (auto-detect)
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with datetime index and required columns
    """
    data_dir = Path('data')
    
    # Auto-detect dataset type
    if dataset_type == 'auto':
        if (data_dir / 'train.csv').exists():
            dataset_type = 'ashrae'
        elif file_path and Path(file_path).exists():
            dataset_type = 'custom'
        else:
            dataset_type = 'synthetic'
    
    # Load ASHRAE dataset
    if dataset_type == 'ashrae':
        try:
            return load_ashrae_data()
        except FileNotFoundError as e:
            print(f"ASHRAE dataset not found: {e}")
            print("Falling back to synthetic data generation...")
            return generate_synthetic_data()
    
    # Load custom CSV file
    if file_path and Path(file_path).exists():
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded data from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Falling back to synthetic data generation...")
            return generate_synthetic_data()
    
    # Generate synthetic data
    print("No data file provided. Generating synthetic data...")
    print("To use real data:")
    print("  1. ASHRAE: Download from https://www.kaggle.com/c/ashrae-energy-prediction/data")
    print("     Place train.csv, weather_train.csv, and building_metadata.csv in data/ folder")
    print("  2. Custom: Place your CSV file in data/ folder with columns: timestamp, energy_consumption, temperature, humidity")
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
    df = df.ffill(limit=6)  # Fill up to 6 hours forward
    df = df.bfill(limit=6)  # Fill up to 6 hours backward
    
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


def prepare_data(file_path=None, dataset_type='auto'):
    """
    Complete data preparation pipeline: load and clean.
    
    Parameters:
    -----------
    file_path : str or Path, optional
        Path to data file or directory
    dataset_type : str
        Type of dataset: 'ashrae', 'custom', or 'auto' (auto-detect)
    
    Returns:
    --------
    pd.DataFrame
        Cleaned and prepared dataframe
    """
    df = load_data(file_path, dataset_type=dataset_type)
    df = clean_data(df)
    return df

