"""
Address module for the fynesse framework.

This module handles question addressing functionality including:
- Statistical analysis
- Predictive modeling
- Data visualization for decision-making
- Dashboard creation
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def train_location_classifier(X_train, y_train, model_type='logistic'):
    """
    Train a classifier to predict location based on geographic features.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        model_type (str): Type of classifier ('logistic' or 'random_forest')
        
    Returns:
        trained model
    """
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, n_estimators=100)
    else:
        raise ValueError("Model type must be 'logistic' or 'random_forest'")
    
    model.fit(X_train, y_train)
    return model

def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate classifier performance.
    
    Args:
        model: Trained classifier
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }

# Example usage function
def create_sample_data():
    """
    Create sample data for demonstration.
    
    Returns:
        tuple: (X, y) feature matrix and labels
    """
    # Sample data for two cities
    X_nyeri = np.array([
        [5, 10, 3, 2, 1, 21],  # Many amenities, few tourist spots
        [4, 8, 2, 1, 0, 15],
        [6, 12, 4, 3, 2, 27]
    ])
    
    X_cambridge = np.array([
        [15, 25, 8, 12, 6, 66],  # Many amenities and tourist spots
        [12, 20, 6, 10, 5, 53],
        [18, 30, 10, 15, 8, 81]
    ])
    
    X = np.vstack([X_nyeri, X_cambridge])
    y = np.array(['Nyeri'] * 3 + ['Cambridge'] * 3)
    
    return X, y
from typing import Any, Union
import pandas as pd
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Here are some of the imports we might expect
# import sklearn.model_selection  as ms
# import sklearn.linear_model as lm
# import sklearn.svm as svm
# import sklearn.naive_bayes as naive_bayes
# import sklearn.tree as tree

# import GPy
# import torch
# import tensorflow as tf

# Or if it's a statistical analysis
# import scipy.stats


def analyze_data(data: Union[pd.DataFrame, Any]) -> dict[str, Any]:
    """
    Address a particular question that arises from the data.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR ANALYSIS CODE:
       - Perform statistical analysis on the data
       - Create visualizations to explore patterns
       - Build models to answer specific questions
       - Generate insights and recommendations

    2. ADD ERROR HANDLING:
       - Check if input data is valid and sufficient
       - Handle analysis failures gracefully
       - Validate analysis results

    3. ADD BASIC LOGGING:
       - Log analysis steps and progress
       - Log key findings and insights
       - Log any issues encountered

    4. EXAMPLE IMPLEMENTATION:
       if data is None or len(data) == 0:
           print("Error: No data available for analysis")
           return {}

       print("Starting data analysis...")
       # Your analysis code here
       results = {"sample_size": len(data), "analysis_complete": True}
       return results
    """
    logger.info("Starting data analysis")

    # Validate input data
    if data is None:
        logger.error("No data provided for analysis")
        print("Error: No data available for analysis")
        return {"error": "No data provided"}

    if len(data) == 0:
        logger.error("Empty dataset provided for analysis")
        print("Error: Empty dataset provided for analysis")
        return {"error": "Empty dataset"}

    logger.info(f"Analyzing data with {len(data)} rows, {len(data.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your analysis code here

        # Example: Basic data summary
        results = {
            "sample_size": len(data),
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "analysis_complete": True,
        }

        # Example: Basic statistics (students should customize this)
        numeric_columns = data.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            results["numeric_summary"] = data[numeric_columns].describe().to_dict()

        logger.info("Data analysis completed successfully")
        print(f"Analysis completed. Sample size: {len(data)}")

        return results

    except Exception as e:
        logger.error(f"Error during data analysis: {e}")
        print(f"Error analyzing data: {e}")
        return {"error": str(e)}

"""
address.py
Model training, evaluation and utility functions.
Refactored from the notebook into callable functions.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import statsmodels.api as sm

def get_feature_target(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X (DataFrame) and y (Series) for modeling (drops NA rows)."""
    df2 = df.copy()
    X = df2[feature_cols].copy()
    y = df2[target_col].copy()
    # drop rows with NA in X or y
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X.loc[mask, :]
    y = y.loc[mask]
    return X, y

def train_linear_regression(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42, return_model=True):
    """
    Train a simple OLS LinearRegression and return model and test-split results.
    Returns: model, X_train, X_test, y_train, y_test, y_pred
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if return_model:
        return model, X_train, X_test, y_train, y_test, y_pred
    else:
        return X_train, X_test, y_train, y_test, y_pred

def evaluate_regression(y_true, y_pred) -> dict:
    """Return common regression metrics (R2, RMSE, MAE)."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return {'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae}

def cross_validate_linear(X: pd.DataFrame, y: pd.Series, cv=5, scoring='r2') -> dict:
    """Perform cross-validation and return mean and std of scores."""
    model = LinearRegression()
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    return {'scores': scores, 'mean': float(scores.mean()), 'std': float(scores.std())}

def save_model(model, path: str):
    """Save model to disk using joblib."""
    joblib.dump(model, path)

def load_model(path: str):
    """Load joblib model."""
    return joblib.load(path)

def ols_summary(X: pd.DataFrame, y: pd.Series):
    """Fit statsmodels OLS and return summary object (for diagnostics)."""
    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc, missing='drop').fit()
    return ols.summary()

# adress.py  (data cleaning)
import pandas as pd

def clean_maize_data(maize_df: pd.DataFrame) -> pd.DataFrame:
    maize_df['Harvested_Area_Ha'] = pd.to_numeric(maize_df['Harvested_Area_Ha'], errors='coerce')
    maize_df['Production_Tons'] = pd.to_numeric(maize_df['Production_Tons'], errors='coerce')
    maize_df['Yield_t_per_ha'] = pd.to_numeric(maize_df['Yield_t_per_ha'], errors='coerce')
    return maize_df.dropna(subset=['Yield_t_per_ha', 'Harvested_Area_Ha', 'Production_Tons'])


