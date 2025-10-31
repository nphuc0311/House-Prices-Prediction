import logging
from typing import Tuple, List
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

logger = logging.getLogger(__name__)


def add_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    
    logger.info("Adding engineered features...")
    X_train = X_train.copy()
    X_test = X_test.copy()
    new_features = []

    for df in [X_train, X_test]:
        # Age feature
        if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
            df['Age'] = df['YrSold'] - df['YearBuilt']
            if 'Age' not in new_features:
                new_features.append('Age')

        # Total square footage
        if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
            df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'].fillna(0)
            if 'TotalSF' not in new_features:
                new_features.append('TotalSF')

        # Total bathrooms
        bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        if all(c in df.columns for c in bath_cols):
            df['TotalBath'] = (
                df['FullBath'].fillna(0) +
                df['HalfBath'].fillna(0) * 0.5 +
                df['BsmtFullBath'].fillna(0) +
                df['BsmtHalfBath'].fillna(0) * 0.5
            )
            if 'TotalBath' not in new_features:
                new_features.append('TotalBath')

    # Update numeric columns list
    updated_num_cols = num_cols + [f for f in new_features if f in X_train.columns]
    
    logger.info(f"Added {len(new_features)} new features: {new_features}")
    return X_train, X_test, updated_num_cols


def winsorize_func(X: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(lambda x: winsorize(x, limits=[0.02, 0.05]), 0, X)


def iqr_clip_func(X: np.ndarray) -> np.ndarray:
    Q1, Q3 = np.percentile(X, [25, 75], axis=0)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return np.clip(X, lower, upper)