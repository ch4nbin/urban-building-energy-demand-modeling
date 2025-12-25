"""
Main Script: Urban Building Energy Demand Modeling

This script orchestrates the complete pipeline:
1. Data loading and cleaning
2. Feature engineering
3. Model training
4. Model evaluation

Run this script to perform the full analysis.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_processing import prepare_data
from feature_engineering import engineer_features
from model_training import train_models
from evaluation import create_evaluation_report


def main():
    """
    Main execution function.
    """
    print("="*70)
    print("URBAN BUILDING ENERGY DEMAND MODELING")
    print("="*70)
    print("\nThis project analyzes and forecasts hourly energy consumption")
    print("for commercial buildings using historical usage and weather data.\n")
    
    # Step 1: Data Processing
    print("\n" + "="*70)
    print("STEP 1: DATA PROCESSING")
    print("="*70)
    data_file = None  # Set to path if you have a data file
    df = prepare_data(file_path=data_file)
    
    # Step 2: Feature Engineering
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    df_features, feature_cols, target_col = engineer_features(df)
    
    # Prepare data for modeling
    X = df_features[feature_cols]
    y = df_features[target_col]
    
    # Step 3: Model Training
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    models, (X_train, X_test, y_train, y_test) = train_models(
        X, y, 
        model_types=['linear', 'random_forest']
    )
    
    # Step 4: Evaluation
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    # Get timestamps for test set
    test_timestamps = df_features.index[-len(y_test):]
    
    results_df = create_evaluation_report(
        models, 
        X_test, 
        y_test,
        timestamps=test_timestamps,
        save_dir='results'
    )
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))
    print("\nResults and plots saved to 'results/' directory")
    print("="*70)


if __name__ == '__main__':
    main()

