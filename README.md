# Machine Learning Regression Analysis Toolkit

This project provides a comprehensive toolkit for building, evaluating, and analyzing machine learning regression models, with an emphasis on model diagnostics and feature selection. Using Python's powerful data science libraries, this toolkit enables thorough analysis of regression models, primarily focusing on k-nearest neighbors (KNN) regression and random forest models.

## Features

- **Model Training with Hyperparameter Tuning**: Build and train KNN regression models with automated hyperparameter tuning.
- **Performance Diagnostics**: Visualize model performance and residuals to assess model quality.
- **Feature Selection**: Identify and select important features based on correlation and feature importance.
- **Correlation Analysis**: Analyze relationships between features and identify missing data patterns.
- **Model Evaluation Metrics**: Calculate the Akaike Information Criterion (AIC) to evaluate model fit.
- **Influence Diagnostics**: Assess influential points in regression using theoretical curves for Cook’s distance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-regression-toolkit.git
   cd ml-regression-toolkit
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Import the module and get started in your Python environment.

## Usage

### 1. Modeling and Prediction

Use `modeling()` to train a KNN regression model with optimized hyperparameters.

```python
from your_module_name import modeling

model, y_pred = modeling(X_train, y_train)
```

### 2. Results Analysis and Visualization

Use `results_analysis()` to generate diagnostic plots that help assess model fit, residual patterns, and influential points.

```python
from your_module_name import results_analysis

results_analysis(y_pred, y_test)
# Check '_visual_report.png' for the saved visualization.
```

### 3. Principal Component Analysis and Feature Selection

Use `pc_analysis()` to select important features based on correlation and feature importance.

```python
from your_module_name import pc_analysis

selected_features = pc_analysis(data, target='your_target_column')
```

### 4. Correlation and Missing Data Analysis

Use `corr_analysis()` to visualize correlations and missing data patterns in your dataset.

```python
from your_module_name import corr_analysis

corr_analysis(data)
```

### 5. Model Evaluation with AIC

Calculate the Akaike Information Criterion (AIC) for model comparison.

```python
from your_module_name import calculate_aic

aic_value = calculate_aic(n=len(y_test), mse=mean_squared_error(y_test, y_pred), k=model_params)
print("AIC:", aic_value)
```

### 6. Influence Diagnostics

Use `calculate_theoretical_curves()` to generate theoretical curves for Cook’s distance and assist with influence diagnostics.

```python
from your_module_name import calculate_theoretical_curves

curves = calculate_theoretical_curves([0.5, 1.0, 2.0])
```

## Functions Overview

| Function                 | Description                                                                                                           |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `modeling()`             | Trains a KNN model using a pipeline and hyperparameter tuning with grid search. Returns best model and predictions.   |
| `results_analysis()`     | Generates diagnostic plots for model evaluation: actual vs. predicted, residuals, skedasticity, and influence.       |
| `pc_analysis()`          | Selects important features using correlation analysis and random forest feature importance.                          |
| `corr_analysis()`        | Displays correlation and missing data matrices for a quick overview of data relationships and quality.               |
| `calculate_aic()`        | Calculates AIC for a given model to balance fit and complexity in model comparison.                                  |
| `calculate_theoretical_curves()` | Computes theoretical Cook’s distance curves for influence diagnostics.                                       |

## Example Workflow

1. **Load and Prepare Data**.
2. **Split the data** into training and test sets.
3. **Train a Model** with `modeling()`.
4. **Analyze Results** using `results_analysis()` and inspect diagnostics.
5. **Select Features** with `pc_analysis()`.
6. **Run Correlation Analysis** using `corr_analysis()`.
7. **Evaluate Models** with `calculate_aic()` and `calculate_theoretical_curves()` for influence diagnostics.

## Requirements

- Python 3.6+
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `sklearn`


## Acknowledgments

- This project leverages popular Python libraries for machine learning and statistical analysis, including `scikit-learn`, `statsmodels`, and `seaborn`.
