"""
Model Training Module

Trains baseline and tree-based models for energy consumption prediction.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Uses time-based splitting to maintain temporal order (no shuffling).
    This is important for time-series data to avoid data leakage.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Feature matrix
    y : pd.Series or np.array
        Target variable
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # For time-series, we want to split chronologically
    # Take the last test_size% as test set
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx] if hasattr(X, 'iloc') else X[:split_idx]
    X_test = X.iloc[split_idx:] if hasattr(X, 'iloc') else X[split_idx:]
    y_train = y.iloc[:split_idx] if hasattr(y, 'iloc') else y[:split_idx]
    y_test = y.iloc[split_idx:] if hasattr(y, 'iloc') else y[split_idx:]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def train_baseline_model(X_train, y_train):
    """
    Train a simple linear regression baseline model.
    
    Linear regression serves as a baseline to compare against more
    complex models. It assumes a linear relationship between features
    and target, which may not capture all patterns but provides
    interpretable coefficients.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target
    
    Returns:
    --------
    sklearn.linear_model.LinearRegression
        Trained baseline model
    """
    print("\nTraining baseline model (Linear Regression)...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("Baseline model trained successfully")
    return model


def train_tree_model(X_train, y_train, model_type='random_forest'):
    """
    Train a tree-based model (Random Forest or Gradient Boosting).
    
    Tree-based models can capture non-linear relationships and
    feature interactions, making them well-suited for energy
    consumption prediction with complex temporal and weather patterns.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target
    model_type : str
        Either 'random_forest' or 'gradient_boosting'
    
    Returns:
    --------
    sklearn model
        Trained tree-based model
    """
    print(f"\nTraining tree-based model ({model_type})...")
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            verbose=0
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'random_forest' or 'gradient_boosting'")
    
    model.fit(X_train, y_train)
    
    print(f"{model_type} model trained successfully")
    return model


def train_models(X, y, model_types=['linear', 'random_forest']):
    """
    Train multiple models and return them in a dictionary.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    model_types : list of str
        List of model types to train
    
    Returns:
    --------
    dict
        Dictionary mapping model names to trained models
    tuple
        (X_train, X_test, y_train, y_test) for evaluation
    """
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    models = {}
    
    # Train baseline model
    if 'linear' in model_types:
        models['baseline_linear'] = train_baseline_model(X_train, y_train)
    
    # Train tree-based models
    if 'random_forest' in model_types:
        models['random_forest'] = train_tree_model(X_train, y_train, 'random_forest')
    
    if 'gradient_boosting' in model_types:
        models['gradient_boosting'] = train_tree_model(X_train, y_train, 'gradient_boosting')
    
    return models, (X_train, X_test, y_train, y_test)

