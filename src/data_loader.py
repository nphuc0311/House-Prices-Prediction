import os
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def download_data_if_needed(data_path: str, gdrive_id: str = '1Dh_y7gFDUa2sD72_cKIa209dhbMVoGEd') -> None:
    if not os.path.exists(data_path):
        logger.info(f"Downloading data to {data_path}...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        try:
            import gdown
            gdown.download(id=gdrive_id, output=data_path, quiet=False)
            logger.info("Download complete!")
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise
    else:
        logger.info(f"Data already exists at {data_path}")


def load_csv(data_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df


def quick_eda(df: pd.DataFrame, target_col: str = 'SalePrice') -> Tuple[pd.Series, pd.Series]:
    logger.info("Performing EDA...")
    print(f"Dataset shape: {df.shape}")
    print(f"\nTarget stats:\n{df[target_col].describe()}")

    # Distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[target_col], kde=True, ax=axes[0])
    axes[0].set_title(f'{target_col} Distribution')

    # Missing values
    missing = df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
    if len(missing) > 0:
        missing[:15].plot.barh(ax=axes[1], color='skyblue')
        axes[1].set_title('Top 15 Missing Values')
        axes[1].invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Skewness and correlations
    num_cols = df.select_dtypes(include=np.number).columns
    skew = df[num_cols].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
    print(f"\nHighly skewed columns (|skew| > 1):\n{skew[abs(skew) > 1].head(10)}")

    if target_col in num_cols:
        corr = df[num_cols].corr()[target_col].sort_values(ascending=False)
        print(f"\nTop correlations with {target_col}:\n{corr.head(8)}")

    return skew, missing


def prepare_data(
    df: pd.DataFrame,
    target_col: str = 'SalePrice',
    test_size: float = 0.25,
    random_state: int = 42,
    drop_cols: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list, list]:
    
    logger.info("Preparing data...")
    
    # Drop high-missing columns and ID
    if drop_cols is None:
        drop_cols = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
    
    df = df.drop([col for col in drop_cols if col in df.columns], axis=1)
    logger.info(f"Dropped columns: {drop_cols}")

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Separate target
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    X_train = train_df.drop(target_col, axis=1)
    X_test = test_df.drop(target_col, axis=1)

    # Identify column types
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Fill categorical NaNs with 'none' for initial processing
    X_train[cat_cols] = X_train[cat_cols].fillna('none')
    X_test[cat_cols] = X_test[cat_cols].fillna('none')

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    logger.info(f"Numerical columns: {len(num_cols)}, Categorical columns: {len(cat_cols)}")

    return X_train, X_test, y_train, y_test, num_cols, cat_cols