"""
Model training, evaluation, and hyperparameter tuning module.
"""
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, RANSACRegressor, QuantileRegressor
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
import optuna

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_models_to_test() -> Dict[str, Any]:
    return {
        'linearreg': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.001, max_iter=10000),
        'elasticnet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
        'huber': HuberRegressor(epsilon=1.35, max_iter=1000),
        'ransac': RANSACRegressor(random_state=42),
        'quantile': QuantileRegressor(quantile=0.5, alpha=0.001, solver='highs')
    }


def calculate_rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))


def evaluate_pipeline(
    pipeline_name: str,
    preprocessor: ColumnTransformer,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    inverse_transform: bool = False
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:

    # Fit preprocessing and model
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model.fit(X_train_processed, y_train)
    y_train_pred = model.predict(X_train_processed)
    y_test_pred = model.predict(X_test_processed)

    # Inverse transform if target was log-transformed
    if inverse_transform:
        y_train_pred = np.expm1(y_train_pred)
        y_test_pred = np.expm1(y_test_pred)
        y_train_actual = np.expm1(y_train)
        y_test_actual = np.expm1(y_test)
    else:
        y_train_actual = y_train
        y_test_actual = y_test

    # Validation checks
    prediction_issues = []
    if np.any(np.isnan(y_test_pred)) or np.any(np.isinf(y_test_pred)):
        prediction_issues.append("contains NaN/Inf")
    if np.all(np.abs(np.diff(y_test_pred)) < 1e-10):
        prediction_issues.append("near-constant predictions")
    
    # Calculate core metrics
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    test_r2 = r2_score(y_test_actual, y_test_pred)
    
    # Add validation metrics
    pred_range = np.percentile(y_test_pred, [1, 99])
    actual_range = np.percentile(y_test_actual, [1, 99])
    range_ratio = (pred_range[1] - pred_range[0]) / (actual_range[1] - actual_range[0])
    
    metrics = {
        'Pipeline': pipeline_name,
        'Train_RMSE': np.sqrt(mean_squared_error(y_train_actual, y_train_pred)),
        'Test_RMSE': test_rmse,
        'Train_RAE': calculate_rae(y_train_actual, y_train_pred),
        'Test_RAE': calculate_rae(y_test_actual, y_test_pred),
        'Train_R2': r2_score(y_train_actual, y_train_pred),
        'Test_R2': test_r2,
        'Prediction_Issues': '; '.join(prediction_issues) if prediction_issues else 'none',
        'Prediction_Range_Ratio': range_ratio
    }

    return metrics, X_train_processed, X_test_processed


def tune_model_optuna(
    model_name: str,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_trials: int = 20,
    cv: int = 3
) -> Tuple[Dict[str, Any], float]:

    logger.info(f"Tuning {model_name} with {n_trials} trials...")
    X_train_processed = preprocessor.fit_transform(X_train)

    def objective(trial):
        model_name_lower = model_name.lower()
        
        if model_name_lower == 'ridge':
            params = {'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True)}
            model = Ridge(**params)
        elif model_name_lower == 'lasso':
            params = {'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True)}
            model = Lasso(**params, max_iter=10000)
        elif model_name_lower == 'elasticnet':
            params = {
                'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9)
            }
            model = ElasticNet(**params, max_iter=10000)
        elif model_name_lower == 'huber':
            params = {'epsilon': trial.suggest_float('epsilon', 1.1, 2.0)}
            model = HuberRegressor(**params, max_iter=1000)
        elif model_name_lower == 'ransac':
            params = {'min_samples': trial.suggest_float('min_samples', 0.5, 0.9)}
            model = RANSACRegressor(**params, random_state=42)
        elif model_name_lower == 'quantile':
            params = {'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True)}
            model = QuantileRegressor(**params, quantile=0.5, solver='highs')
        else:  # LinearRegression
            params = {}
            model = LinearRegression()

        # Cross-validation RMSE
        scores = cross_val_score(
            model, X_train_processed, y_train,
            cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
        )
        rmse = np.sqrt(-scores.mean())
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best params for {model_name}: {study.best_params}, CV RMSE: {study.best_value:.2f}")
    return study.best_params, study.best_value


def create_model_from_params(model_name: str, params: Dict[str, Any]) -> Any:

    model_name_lower = model_name.lower()
    
    if model_name_lower == 'ridge':
        return Ridge(**params)
    elif model_name_lower == 'lasso':
        return Lasso(**params, max_iter=10000)
    elif model_name_lower == 'elasticnet':
        return ElasticNet(**params, max_iter=10000)
    elif model_name_lower == 'huber':
        return HuberRegressor(**params, max_iter=1000)
    elif model_name_lower == 'ransac':
        return RANSACRegressor(**params, random_state=42)
    elif model_name_lower == 'quantile':
        return QuantileRegressor(**params, quantile=0.5, solver='highs')
    else:
        return LinearRegression()