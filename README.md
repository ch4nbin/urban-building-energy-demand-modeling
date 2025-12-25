# Urban Building Energy Demand Modeling

This project explores patterns in commercial building energy consumption by analyzing how time-of-day, day-of-week, and weather conditions influence hourly energy demand. The goal is to understand these relationships and develop predictive models that capture the underlying patterns.

## Motivation

Understanding building energy consumption patterns is important for:
- **Energy Management**: Identifying peak usage periods and opportunities for optimization
- **Cost Planning**: Forecasting energy costs based on expected consumption
- **Sustainability**: Supporting efforts to reduce energy waste and carbon footprint
- **Operational Planning**: Anticipating energy needs for maintenance and capacity planning

This analysis takes an exploratory approach, focusing on understanding the relationships between temporal patterns, weather conditions, and energy consumption rather than optimizing for production deployment.

## Approach

### Data

The project works with hourly energy consumption data combined with weather information. The dataset includes:
- **Energy consumption** (kWh): Hourly building energy usage
- **Temperature** (°C): Outdoor temperature
- **Humidity** (%): Relative humidity

If a data file is not provided, the project generates synthetic data that mimics realistic patterns:
- Daily cycles (higher consumption during business hours)
- Weekly cycles (lower consumption on weekends)
- Seasonal variations
- Weather-dependent HVAC loads

### Feature Engineering

The analysis creates several types of features to capture different aspects of energy consumption patterns:

1. **Cyclical Temporal Features**: Sin/cos encodings for:
   - Hour of day (captures daily patterns)
   - Day of week (captures weekly patterns)
   - Day of year (captures seasonal patterns)
   - Month (captures monthly variations)

2. **Lagged Features**: Previous consumption values at:
   - 1 hour ago (short-term correlation)
   - 24 hours ago (daily patterns)
   - 168 hours ago (weekly patterns)

3. **Rolling Statistics**: Moving averages and standard deviations over 3, 6, and 24-hour windows

4. **Weather-Derived Features**:
   - Temperature deviation from comfort zone
   - Heating and cooling degree hours
   - Humidity deviation

### Models

Two types of models are trained and compared:

1. **Baseline Model (Linear Regression)**: 
   - Simple linear relationship between features and consumption
   - Provides interpretable coefficients
   - Serves as a baseline for comparison

2. **Tree-Based Model (Random Forest)**:
   - Captures non-linear relationships and feature interactions
   - Handles complex temporal and weather patterns
   - Generally performs better on this type of problem

### Evaluation

Models are evaluated using:
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values (interpretable in original units)
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily

The evaluation includes:
- Performance metrics comparison
- Time series plots of predicted vs. actual consumption
- Residual analysis to identify systematic biases

## Project Structure

```
.
├── main.py                 # Main execution script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Data directory (add your CSV files here)
├── results/               # Generated results and plots
├── src/
    ├── __init__.py
    ├── data_processing.py      # Data loading and cleaning
    ├── feature_engineering.py  # Feature creation
    ├── model_training.py        # Model training
    └── evaluation.py           # Model evaluation and visualization
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ch4nbin/urban-building-energy-demand-modeling.git
cd urban-building-energy-demand-modeling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

Simply run the main script:
```bash
python main.py
```

This will:
1. Load or generate data
2. Clean and preprocess the data
3. Engineer features
4. Train models
5. Evaluate and visualize results

### Using Your Own Data

To use your own data file:
1. Place your CSV file in the `data/` directory
2. Ensure it has columns: `timestamp`, `energy_consumption`, `temperature`, `humidity`
3. Modify `main.py` to specify the file path:
   ```python
   data_file = 'data/your_data.csv'
   ```

### Data Format

Your CSV file should have the following structure:
- `timestamp`: DateTime string (e.g., "2023-01-01 00:00:00")
- `energy_consumption`: Numeric values (kWh)
- `temperature`: Numeric values (°C)
- `humidity`: Numeric values (%)

## Results

The analysis generates several outputs in the `results/` directory:

- `model_metrics.csv`: Performance metrics (MAE, RMSE) for all models
- `*_predictions.png`: Time series plots showing actual vs. predicted consumption
- `*_residuals.png`: Residual analysis plots for the best-performing model

### Typical Findings

Based on the analysis, we typically observe:

1. **Temporal Patterns**:
   - Higher consumption during business hours (9 AM - 5 PM)
   - Lower consumption on weekends
   - Daily cycles with peaks during active hours

2. **Weather Influence**:
   - Increased consumption when temperature deviates from comfort zone (20-24°C)
   - Higher HVAC loads during extreme weather conditions

3. **Model Performance**:
   - Tree-based models (Random Forest) generally outperform linear regression
   - Lag features (especially 24-hour lag) are highly predictive
   - Cyclical encodings help capture temporal patterns better than raw hour/day values

## Design Choices and Assumptions

### Cyclical Encoding
We use sin/cos encoding for temporal features instead of one-hot encoding because it preserves the cyclical nature (e.g., hour 23 is close to hour 0). This helps models learn that similar times of day have similar consumption patterns.

### Time-Based Splitting
The train/test split maintains temporal order (no shuffling) to avoid data leakage. This simulates a realistic scenario where we predict future consumption based on past data.

### Feature Selection
We include lag features at 1, 24, and 168 hours to capture short-term, daily, and weekly patterns. These are common in time-series forecasting and align with expected building energy behavior.

### Synthetic Data
When real data is unavailable, we generate synthetic data with realistic patterns. This allows the project to run end-to-end while demonstrating the approach. In practice, you would use real building energy datasets (e.g., from ASHRAE competitions or building management systems).

## Limitations

This is an exploratory analysis with several limitations:

- **Simplified Model**: Focuses on understanding patterns rather than production optimization
- **Feature Engineering**: Uses domain knowledge but doesn't exhaustively search all possible features
- **Hyperparameter Tuning**: Models use default or reasonable hyperparameters without extensive tuning
- **External Factors**: Doesn't account for holidays, special events, or building-specific factors
- **Validation**: Uses simple train/test split rather than time-series cross-validation

## Future Enhancements

Potential improvements for deeper analysis:
- Time-series cross-validation for more robust evaluation
- Additional models (LSTM, XGBoost, Prophet)
- Feature importance analysis
- Hyperparameter optimization
- Integration with real-time data sources
- Multi-building analysis and comparison

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## License

This project is provided as-is for educational and exploratory purposes.

## Contact

For questions or suggestions, please open an issue on the GitHub repository.

