"""
Evaluation Module

Evaluates model performance using MAE and RMSE metrics,
and creates visualization plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path


def calculate_metrics(y_true, y_pred):
    """
    Calculate MAE and RMSE metrics.
    
    MAE (Mean Absolute Error): Average absolute difference between
    predicted and actual values. Interpretable in original units.
    
    RMSE (Root Mean Squared Error): Square root of average squared
    differences. Penalizes larger errors more heavily.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns:
    --------
    dict
        Dictionary with 'mae' and 'rmse' metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        'mae': mae,
        'rmse': rmse
    }


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models on test set.
    
    Parameters:
    -----------
    models : dict
        Dictionary mapping model names to trained models
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target values
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with model performance metrics
    dict
        Dictionary mapping model names to predictions
    """
    results = []
    predictions = {}
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    for model_name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        predictions[model_name] = y_pred
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        
        results.append({
            'model': model_name,
            'mae': metrics['mae'],
            'rmse': metrics['rmse']
        })
        
        print(f"\n{model_name}:")
        print(f"  MAE:  {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
    
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    
    return results_df, predictions


def plot_predictions(y_true, y_pred, model_name, timestamps=None, save_path=None):
    """
    Plot predicted vs actual energy consumption over time.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model (for title)
    timestamps : pd.DatetimeIndex, optional
        Timestamps for x-axis. If None, uses index.
    save_path : str or Path, optional
        Path to save the plot
    """
    plt.figure(figsize=(14, 6))
    
    # Use provided timestamps or create index
    if timestamps is None:
        timestamps = pd.date_range(start='2023-01-01', periods=len(y_true), freq='H')
    
    # Plot actual and predicted
    plt.plot(timestamps, y_true, label='Actual', alpha=0.7, linewidth=1.5)
    plt.plot(timestamps, y_pred, label='Predicted', alpha=0.7, linewidth=1.5)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Energy Consumption (kWh)', fontsize=12)
    plt.title(f'Energy Consumption: Actual vs Predicted ({model_name})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_all_predictions(predictions, y_test, timestamps=None, save_dir='plots'):
    """
    Plot predictions for all models.
    
    Parameters:
    -----------
    predictions : dict
        Dictionary mapping model names to predictions
    y_test : array-like
        True target values
    timestamps : pd.DatetimeIndex, optional
        Timestamps for x-axis
    save_dir : str
        Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, y_pred in predictions.items():
        plot_predictions(
            y_test, 
            y_pred, 
            model_name,
            timestamps=timestamps,
            save_path=save_dir / f'{model_name}_predictions.png'
        )


def plot_residuals(y_true, y_pred, model_name, save_path=None):
    """
    Plot residual analysis (prediction errors).
    
    Residual plots help identify:
    - Systematic bias (mean residual should be ~0)
    - Heteroscedasticity (variance should be constant)
    - Outliers
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
    save_path : str or Path, optional
        Path to save the plot
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals over time
    axes[0].scatter(range(len(residuals)), residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Sample Index', fontsize=11)
    axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=11)
    axes[0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual Value', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Residual Analysis: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual plot saved to {save_path}")
    
    plt.show()


def create_evaluation_report(models, X_test, y_test, timestamps=None, save_dir='results'):
    """
    Create comprehensive evaluation report with metrics and plots.
    
    Parameters:
    -----------
    models : dict
        Dictionary mapping model names to trained models
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target values
    timestamps : pd.DatetimeIndex, optional
        Timestamps for plotting
    save_dir : str
        Directory to save results
    
    Returns:
    --------
    pd.DataFrame
        Model performance metrics
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate models
    results_df, predictions = evaluate_models(models, X_test, y_test)
    
    # Save metrics to CSV
    metrics_path = save_dir / 'model_metrics.csv'
    results_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Create plots
    print("\nGenerating plots...")
    plot_all_predictions(predictions, y_test, timestamps=timestamps, save_dir=save_dir)
    
    # Create residual plots for best model
    best_model = results_df.loc[results_df['mae'].idxmin(), 'model']
    print(f"\nCreating residual analysis for best model: {best_model}")
    plot_residuals(
        y_test,
        predictions[best_model],
        best_model,
        save_path=save_dir / f'{best_model}_residuals.png'
    )
    
    return results_df

