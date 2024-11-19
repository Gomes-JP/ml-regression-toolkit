import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.stats.api as sms
import scipy.stats as stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

"""
This module provides functions for building, evaluating, and analyzing machine learning models with a focus on 
regression tasks. The primary components include model training using k-nearest neighbors (KNN) regression, 
visualization of results, feature importance analysis with random forests, and correlation analysis.

Modules and Libraries Used:
    - numpy: Numerical operations.
    - scipy: Scientific computations.
    - pandas: Data manipulation.
    - matplotlib, seaborn: Data visualization.
    - statsmodels, sklearn: Statistical and machine learning models.
    
Functions:
    - modeling(X, y): Builds and trains a KNN regression model, performing grid search to optimize hyperparameters. 
      Returns the best model and its predictions on the input data.
      
    - results_analysis(y_pred, y_test): Analyzes model predictions through visualizations, including actual vs. predicted values, 
      residual distribution, heteroskedasticity check, and influence plots.
      
    - pc_analysis(data, target): Performs feature selection based on correlation and feature importance using 
      random forests, returning a subset of selected features.
      
    - corr_analysis(data): Provides a correlation matrix and missing data visualization for a dataset, assisting in 
      identifying patterns and data quality issues.
      
    - calculate_aic(n, mse, k): Calculates the Akaike Information Criterion (AIC) for a given model to evaluate 
      model quality, balancing goodness of fit and model complexity.
      
    - calculate_theoretical_curves(fixed_cooks_distance_values, num_params=1): Computes theoretical curves for 
      fixed Cook's distance values to assist in diagnosing influential points in regression models.

Usage:
    - To train a model and evaluate predictions, use modeling() followed by results_analysis().
    - For feature selection based on correlation and feature importance, use pc_analysis().
    - Use corr_analysis() for preliminary data inspection and missing data assessment.
    - calculate_aic() and calculate_theoretical_curves() are supplementary functions for model evaluation 
      and influence diagnostics.

Notes:
    - The results_analysis function generates diagnostic plots saved to '_visual_report.png'.
    - All visualizations are tailored for regression analysis and model diagnostics.
"""

def modeling(X, y):
    """
    Builds and trains a k-nearest neighbors (KNN) regression model using a pipeline with standard scaling.
    Performs grid search to optimize hyperparameters for the KNN model and returns the best estimator and predictions.

    Parameters:
        X (pd.DataFrame or np.ndarray): Features for model training.
        y (pd.Series or np.ndarray): Target variable for model training.

    Returns:
        model (Pipeline): Trained pipeline model with the best hyperparameters.
        y_pred (np.ndarray): Predictions of the trained model on the input features X.
    """

    model = KNeighborsRegressor()
    param_grid = {'model__n_neighbors': [3, 5, 7, 9, 12],
                  'model__weights': ['uniform', 'distance']}
    
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    model = grid_search.best_estimator_
    y_pred = model.predict(X)
    return model, y_pred

def results_analysis(y_pred, y_test):
    """
    Performs analysis and diagnostic visualizations of model predictions, including:
    - Actual vs. Predicted values plot with R-squared annotation.
    - Residuals distribution to check normality.
    - Skedasticity plot for heteroskedasticity detection.
    - Influence plot showing leverage and studentized residuals.

    Parameters:
        y_pred (np.ndarray): Predicted target values from the model.
        y_test (np.ndarray or pd.Series): Actual target values for comparison.

    Notes:
        The function generates a multi-panel plot saved as '_visual_report.png' with various diagnostics for 
        evaluating model performance and residual patterns.
    """

    ols = sm.OLS(y_test, sm.add_constant(y_pred)).fit()
    res = ols.resid
    sth = ols.get_influence().summary_frame()
    st_res = sth['student_resid']
    lev = sth['hat_diag']
    rsqueared = ols.rsquared
    
    xyrange = np.linspace(min(y_test), max(y_test), 10*int(max(y_test)))
    xyrange = [round(y_pred, 1) for y_pred in xyrange]
    
    plt.figure(figsize=(15, 15),
               tight_layout=True)
    
    # Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, facecolor='#9370db', label='Test Prediction')
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             color='black',
             linestyle='dashed',
             lw=2, label='Ideal')
    plt.xlabel('Actual Target Values')
    plt.ylabel('Predicted Target Values')
    if rsqueared <= 0.75:
        corr = 'Weakly correlated'
    else:
        corr = 'Strongly correlated'
    plt.title(f'KNN model Based ({corr}, R$^2$={rsqueared:.2f})')
    plt.legend(loc='best')
    plt.tick_params(direction='in')
    plt.xticks(xyrange)
    plt.yticks(xyrange)
    plt.grid(True)
    
    # Residuals distribution (normality)
    plt.subplot(2, 2, 2)
    norm_p_value = stats.jarque_bera(res).pvalue
    if norm_p_value <= 0.05:
        norm = 'Not normal'
    else:
        norm = 'Normal'
    sns.histplot(res,
                 kde=True,
                 color='#9370db')
    plt.title(f'Normality of residuals ({norm}, P-value={norm_p_value:.2f})')
    plt.xlabel('Residuals')
    plt.grid(True)
    plt.tick_params(direction='in')
    
    plt.subplot(2, 2, 3)
    # Skedasticity (Residuals vs Predicted)
    line = sm.OLS(abs(res), sm.add_constant(y_pred)).fit()
    test_pred = line.predict(sm.add_constant(y_pred))
    ske_p_value = sms.het_breuschpagan(res, ols.model.exog)[1]
    if ske_p_value <= 0.05:
        ske = 'Heteroskedastic'
    else:
        ske = 'Homoskedastic'
    plt.scatter(y_pred,
                abs(res),
                facecolor='#9370db')
    plt.axhline(y=test_pred[0],
                color='k',
                linestyle='dashed',
                label='Symmetry line')
    plt.xlabel('Predicted Values of Validation Data')
    plt.ylabel('Residuals')
    plt.title(f'Skedasticity of Residuals ({ske}, P-value={ske_p_value:.2f})')
    plt.grid(True)
    plt.xticks(xyrange)
    plt.legend(loc='best')
    plt.tick_params(direction='in')
    
    plt.subplot(2, 2, 4)
    
    # add cook's distance to plot
    cutoff = 4 / (len(y_pred) - 2)
    
    d_curves = calculate_theoretical_curves(fixed_cooks_distance_values=[cutoff])
    
    d1x = d_curves[cutoff]['leverage']
    d1y = d_curves[cutoff]['studentized_residuals']
    
    plt.plot(d1x, d1y, marker='none', color='#c3121e',
             linestyle='-.', label="D$_{crit}$")
    plt.plot(d1x, -d1y, marker='none', color='#c3121e',
             linestyle='-.')
    
    plt.scatter(lev, st_res, facecolor='#9370db')
    plt.axhline(-3, color='k', linestyle=':')
    plt.axhline(3, color='k', linestyle=':', label='$\hat{\sigma}_{crit}$')
    plt.axhline(0, color='grey', linestyle='--')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Leverage ($\hat{h}$)')
    plt.ylabel('Studentinized Residuals ($\hat{\sigma}$)')
    plt.title('Influence Plot')
    plt.xlim([min(lev), max(lev)])
    plt.ylim(-3.5, 3.5)
    
    plt.savefig('_visual_report.png', dpi=300)



def pc_analysis(data, target):
    """
    Performs principal component analysis (PCA)-based feature selection and evaluates feature importance using a random forest model.
    Selects features based on correlation thresholds and returns a subset of important features.

    Parameters:
        data (pd.DataFrame): The dataset containing features and the target variable.
        target (str): Name of the target variable in the dataset.

    Returns:
        selected_features (list): List of selected feature names based on importance and correlation thresholds.

    Notes:
        Generates a bar plot showing feature importance and prints selected features based on a threshold.
    """

    corr_matrix = data.corr()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    selected_features = [column for column in upper_tri.columns if any(abs(upper_tri[column]) >= 0.75)]
    
    if target in selected_features:
        selected_features.remove(target)
    
    X = data[selected_features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)
    y_train = np.array(y_train)
    
    rfc_1 = RandomForestRegressor()
    rfc_1.fit(X_train_scaled, y_train)
    rfc_1.score(X_train_scaled, y_train)
    
    feats = {}
    
    for feature, importance in zip(X.columns,
                                   rfc_1.feature_importances_):
        feats[feature] = importance
    importances = pd.DataFrame.from_dict(feats,
                                         orient='index').rename(columns={0: 'Gini-Importance'})
    importances = importances.sort_values(by='Gini-Importance',
                                          ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={'index': 'Features'})
    sns.set(font_scale=5)
    sns.set(style="whitegrid",
            color_codes=True,
            font_scale=1.7)
    
    plt.figure(figsize=(12, 8),
               tight_layout=True)
    
    sns.barplot(x=importances['Gini-Importance'],
                y=importances['Features'],
                data=importances,
                color='#7570b3')
    plt.axvline(x=0.1, color='red')
    plt.xlabel('Importance', fontsize=25, weight='bold')
    plt.ylabel('Features', fontsize=25, weight='bold')
    plt.title('Feature Importance', fontsize=25, weight='bold')
    plt.grid(True)
    
    plt.show()
    
    print(f'{selected_features=}')
    selected_features = importances[importances['Gini-Importance'] >= 0.1]['Features'].tolist()
    print(f'{selected_features=}')
    return selected_features

def corr_analysis(data):
    """
    Analyzes the dataset for missing values and correlations between features.
    Generates two visualizations:
    - Missing data matrix, showing missingness across features.
    - Correlation matrix, showing pairwise Pearson correlations among features.

    Parameters:
        data (pd.DataFrame): The dataset for analysis.

    Notes:
        Useful for identifying patterns in missing data and exploring relationships between features.
    """

    plt.figure(figsize=(8, 12),
                       tight_layout=True)
    
    plt.subplot(2,1,1)
    sns.heatmap(data.isna().transpose(),
                cmap="Greens",
                cbar_kws={'label': 'Missing Data'})
    plt.tick_params(direction='in')
    plt.title('Missing Data Matrix')
    
    plt.subplot(2,1,2)
    data_filtered = data.dropna(axis=0, how='all')
    corr_matrix = data_filtered.corr()
    sns.heatmap(corr_matrix,
                annot=False,
                cmap='Purples',
                cbar_kws={'label': 'Pearson Correlation'},
                linewidths=1.)
    plt.tick_params(direction='in')
    plt.title('Correlation Matrix')
    plt.show()

    
def calculate_aic(n, mse, k):
    """
    Calculates the Akaike information criterion (AIC) for a given model.

    Parameters:
        n (int): The number of samples
        mse (float): The mean squared error of the model
        k (int): The number of parameters in the model

    Returns:
        float: The AIC value

    Notes:
        The AIC is a measure of the relative quality of a model for a given set of data. It takes into account both the goodness of fit and the complexity of the model.
    """
    aic = n * np.log(mse) + 2 * k
    return aic

def calculate_theoretical_curves(fixed_cooks_distance_values, num_params=1):
    """
    Calculate theoretical curves for fixed values of Cook's distance.

    Args:
        leverage (array-like): Leverage values for each observation.
        studentized_residuals (array-like): Studentized residuals for each observation.
        fixed_cooks_distance_values (array-like): Fixed values of Cook's distance.
        num_params (int): Number of parameters in the model.

    Returns:
        dict: Dictionary containing the calculated theoretical curves for each fixed Cook's distance value.
    """
    theoretical_curves = {}

    for fixed_cook_distance in fixed_cooks_distance_values:
        # Calculate leverage for each observation
        leverage_values = np.linspace(0.001, 0.999, 1000)

        # Calculate the corresponding studentized residuals using the formula

        studentized_residuals_values = fixed_cook_distance / (leverage_values * (1 - leverage_values) / num_params)
        studentized_residuals_values = np.sqrt(studentized_residuals_values)

        theoretical_curves[fixed_cook_distance] = {
            'leverage': leverage_values,
            'studentized_residuals': studentized_residuals_values
        }

    return theoretical_curves
