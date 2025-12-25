"""
Data Processing Module

Handles loading, cleaning, and basic preprocessing of building energy and weather data.
Supports:
- UCI Building Energy Dataset (.xlsx, .xls, .csv)
- Custom CSV/Excel files
- Synthetic data generation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_uci_data(file_path=None):
    """
    Load UCI Energy Efficiency Dataset.
    
    The UCI dataset contains building characteristics and energy loads.
    Since it's not time-series data, we convert it to hourly time-series format
    by creating timestamps and using the energy loads as consumption values.
    
    Download from: https://archive.ics.uci.edu/dataset/242/energy+efficiency
    
    Parameters:
    -----------
    file_path : str or Path, optional
        Path to UCI dataset file (supports .xlsx, .xls, or .csv)
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with time-series format
    """
    data_dir = Path('data')
    
    if file_path is None:
        # Try to find UCI dataset file (check for .xlsx, .xls, or .csv)
        possible_files = [
            data_dir / 'energy_efficiency.xlsx',
            data_dir / 'energy_efficiency.xls',
            data_dir / 'energy_efficiency.csv',
            data_dir / 'ENB2012_data.xlsx',  # Common UCI filename
            data_dir / 'ENB2012_data.xls',
        ]
        
        file_path = None
        for pf in possible_files:
            if pf.exists():
                file_path = pf
                break
        
        if file_path is None:
            raise FileNotFoundError(
                f"UCI dataset not found in {data_dir}. "
                "Please place your .xlsx file in the data/ directory. "
                "Download from https://archive.ics.uci.edu/dataset/242/energy+efficiency"
            )
    else:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"UCI dataset not found at {file_path}")
    
    print(f"Loading UCI Energy Efficiency dataset from {file_path}...")
    
    # Load based on file extension
    if file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # UCI dataset typically has columns like:
    # X1-X8: Building characteristics (relative compactness, surface area, etc.)
    # Y1: Heating Load
    # Y2: Cooling Load
    
    # Find energy consumption column (could be Y1, Y2, or heating/cooling load)
    energy_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if 'heating' in col_lower and 'load' in col_lower:
            energy_col = col
            break
        elif 'cooling' in col_lower and 'load' in col_lower:
            energy_col = col
            break
        elif col in ['Y1', 'Y2', 'heating_load', 'cooling_load']:
            energy_col = col
            break
    
    # If no specific column found, use Y1 (heating load) or first numeric column
    if energy_col is None:
        if 'Y1' in df.columns:
            energy_col = 'Y1'
        elif 'Y2' in df.columns:
            energy_col = 'Y2'
        else:
            # Use first numeric column that looks like energy
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                energy_col = numeric_cols[-1]  # Usually last column is target
            else:
                raise ValueError("Could not find energy consumption column in UCI dataset")
    
    print(f"Using '{energy_col}' as energy consumption column")
    
    # Create time-series format by assigning hourly timestamps
    # Each building record becomes one hour of data
    n_records = len(df)
    start_date = pd.Timestamp('2023-01-01 00:00:00')
    timestamps = pd.date_range(start=start_date, periods=n_records, freq='H')
    
    # Create time-series dataframe
    result_df = pd.DataFrame({
        'timestamp': timestamps,
        'energy_consumption': df[energy_col].values
    })
    
    # Add temperature if available (UCI doesn't have weather, so we'll generate realistic values)
    # Or try to find temperature-related columns
    temp_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if 'temp' in col_lower or 'temperature' in col_lower:
            temp_col = col
            break
    
    if temp_col:
        result_df['temperature'] = df[temp_col].values
    else:
        # Generate realistic temperature based on seasonal patterns
        day_of_year = timestamps.dayofyear
        result_df['temperature'] = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365.25) + np.random.normal(0, 2, n_records)
    
    # Add humidity (UCI doesn't have this, generate realistic values)
    result_df['humidity'] = 60 - 0.5 * result_df['temperature'] + np.random.normal(0, 5, n_records)
    result_df['humidity'] = result_df['humidity'].clip(20, 90)
    
    # Set timestamp as index
    result_df = result_df.set_index('timestamp')
    
    print(f"Converted to time-series format: {len(result_df)} hourly records")
    print(f"Energy consumption range: {result_df['energy_consumption'].min():.2f} - {result_df['energy_consumption'].max():.2f}")
    
    return result_df


def load_data(file_path=None, dataset_type='auto'):
    """
    Load building energy and weather data from various sources.
    
    Supports:
    - UCI Building Energy Dataset (.xlsx, .xls, .csv)
    - Custom CSV files
    - Synthetic data generation (fallback)
    
    Parameters:
    -----------
    file_path : str or Path, optional
        Path to data file
    dataset_type : str
        Type of dataset: 'uci', 'custom', or 'auto' (auto-detect)
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with datetime index and required columns
    """
    data_dir = Path('data')
    
    # Auto-detect dataset type
    if dataset_type == 'auto':
        # Check for UCI dataset files
        uci_files = list(data_dir.glob('*.xlsx')) + list(data_dir.glob('*.xls')) + \
                    list(data_dir.glob('energy_efficiency.*'))
        if uci_files:
            dataset_type = 'uci'
        elif file_path and Path(file_path).exists():
            dataset_type = 'custom'
        else:
            dataset_type = 'synthetic'
    
    # Load UCI dataset
    if dataset_type == 'uci':
        try:
            return load_uci_data(file_path=file_path)
        except FileNotFoundError as e:
            print(f"UCI dataset not found: {e}")
            print("Falling back to synthetic data generation...")
            return generate_synthetic_data()
    
    # Load custom CSV file
    if file_path and Path(file_path).exists():
        try:
            # Try to detect if it's Excel file
            if Path(file_path).suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
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
    print("  1. UCI: Place your .xlsx file in data/ folder")
    print("     Download from https://archive.ics.uci.edu/dataset/242/energy+efficiency")
    print("  2. Custom: Place your CSV/Excel file in data/ folder")
    print("     Required columns: timestamp, energy_consumption, temperature, humidity")
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
        Path to data file
    dataset_type : str
        Type of dataset: 'uci', 'custom', or 'auto' (auto-detect)
    
    Returns:
    --------
    pd.DataFrame
        Cleaned and prepared dataframe
    """
    df = load_data(file_path, dataset_type=dataset_type)
    df = clean_data(df)
    return df

