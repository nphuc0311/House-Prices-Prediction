import logging
from typing import Dict, List, Any
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, MinMaxScaler,
    StandardScaler, RobustScaler, PowerTransformer, FunctionTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from .feature_engineering import winsorize_func, iqr_clip_func

logger = logging.getLogger(__name__)


# Component dictionaries
IMPUTERS = {
    'simple': SimpleImputer(strategy='mean'),
    'iterative': IterativeImputer(random_state=42, max_iter=10),
    'knn': KNNImputer(n_neighbors=5)
}

OUTLIER_HANDLERS = {
    'none': 'passthrough',
    'winsor': FunctionTransformer(winsorize_func, validate=False),
    'iqr': FunctionTransformer(iqr_clip_func, validate=False)
}

SKEW_TRANSFORMERS = {
    'none': 'passthrough',
    'log1p': FunctionTransformer(np.log1p, validate=False),
    'yeojohnson': PowerTransformer(method='yeo-johnson', standardize=False)
}

SCALERS = {
    'minmax': MinMaxScaler(),
    'standard': StandardScaler(),
    'robust': RobustScaler()
}

# Ordinal columns with proper ordering
ORDINAL_COLS = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond'
]
ORDINAL_CATEGORIES = [['Po', 'Fa', 'TA', 'Gd', 'Ex']] * len(ORDINAL_COLS)


class PipelineBuilder:    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Initialized PipelineBuilder")
    
    def create_pipeline(
        self,
        num_cols: List[str],
        cat_cols: List[str]
    ) -> ColumnTransformer:
        
        preproc_cfg = self.config.get('preprocessing', {})
        
        # Get components from config
        imputer_name = preproc_cfg.get('imputer', 'iterative').lower()
        outlier_name = preproc_cfg.get('outlier', 'winsor').lower()
        skew_name = preproc_cfg.get('skew', 'log1p').lower()
        scaler_name = preproc_cfg.get('scaler', 'standard').lower()
        use_ordinal = preproc_cfg.get('use_ordinal', True)
        
        logger.info(f"Creating pipeline: imputer={imputer_name}, outlier={outlier_name}, "
                   f"skew={skew_name}, scaler={scaler_name}, ordinal={use_ordinal}")
        
        # Build numerical pipeline
        num_steps = [
            ('impute', IMPUTERS[imputer_name]),
            ('outlier', OUTLIER_HANDLERS[outlier_name]),
            ('skew', SKEW_TRANSFORMERS[skew_name]),
            ('scale', SCALERS[scaler_name])
        ]
        num_pipeline = Pipeline([s for s in num_steps if s[1] != 'passthrough'])
        
        # Build categorical transformers
        ordinal_cols = [c for c in ORDINAL_COLS if c in cat_cols]
        nominal_cols = [c for c in cat_cols if c not in ordinal_cols]
        
        if use_ordinal and len(ordinal_cols) > 0:
            cat_transformers = [
                ('ordinal', OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES[:len(ordinal_cols)],
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                ), ordinal_cols),
                ('onehot', OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ), nominal_cols)
            ]
        else:
            cat_transformers = [
                ('onehot', OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ), cat_cols)
            ]
        
        # Combine into ColumnTransformer
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols)
        ] + cat_transformers)
        
        logger.info(f"Pipeline created with {len(num_cols)} numerical and {len(cat_cols)} categorical columns")
        return preprocessor


def get_benchmark_configs() -> List[tuple]:
    return [
        # Vary imputation with fixed other components
        ('Simple_Log_Winsor_Standard_OneHot', 'Simple', 'Winsor', 'Log1p', 'Standard', False),
        ('Iterative_Log_Winsor_Standard_OneHot', 'Iterative', 'Winsor', 'Log1p', 'Standard', False),
        ('KNN_Log_Winsor_Standard_OneHot', 'KNN', 'Winsor', 'Log1p', 'Standard', False),
        
        # Vary outlier handling
        ('Iterative_Log_IQR_Standard_OneHot', 'Iterative', 'IQR', 'Log1p', 'Standard', False),
        ('Iterative_Log_None_Standard_OneHot', 'Iterative', 'None', 'Log1p', 'Standard', False),
        
        # Vary skew transformation
        ('Iterative_YeoJohnson_Winsor_Standard_OneHot', 'Iterative', 'Winsor', 'YeoJohnson', 'Standard', False),
        ('Iterative_None_Winsor_Standard_OneHot', 'Iterative', 'Winsor', 'None', 'Standard', False),
        
        # Vary scaling
        ('Iterative_Log_Winsor_MinMax_OneHot', 'Iterative', 'Winsor', 'Log1p', 'MinMax', False),
        ('Iterative_Log_Winsor_Robust_OneHot', 'Iterative', 'Winsor', 'Log1p', 'Robust', False),
        
        # Try ordinal encoding
        ('Iterative_Log_Winsor_Standard_Ordinal', 'Iterative', 'Winsor', 'Log1p', 'Standard', True),
        ('Simple_Log_Winsor_MinMax_Ordinal', 'Simple', 'Winsor', 'Log1p', 'MinMax', True),
        
        # More combinations
        ('KNN_YeoJohnson_IQR_Robust_OneHot', 'KNN', 'IQR', 'YeoJohnson', 'Robust', False),
        ('Simple_None_None_MinMax_OneHot', 'Simple', 'None', 'None', 'MinMax', False),
        ('KNN_Log_None_Standard_OneHot', 'KNN', 'None', 'Log1p', 'Standard', False),
        ('Iterative_YeoJohnson_IQR_MinMax_OneHot', 'Iterative', 'IQR', 'YeoJohnson', 'MinMax', False),
        ('Simple_YeoJohnson_Winsor_Robust_OneHot', 'Simple', 'Winsor', 'YeoJohnson', 'Robust', False),
        ('KNN_None_IQR_MinMax_OneHot', 'KNN', 'IQR', 'None', 'MinMax', False),
        ('Simple_Log_IQR_Standard_Ordinal', 'Simple', 'IQR', 'Log1p', 'Standard', True),
        ('Iterative_None_IQR_Robust_OneHot', 'Iterative', 'IQR', 'None', 'Robust', False),
        ('KNN_YeoJohnson_Winsor_MinMax_Ordinal', 'KNN', 'Winsor', 'YeoJohnson', 'MinMax', True),
        ('Simple_Log_None_Robust_OneHot', 'Simple', 'None', 'Log1p', 'Robust', False),
    ]


def create_pipeline_from_config_tuple(
    config_tuple: tuple,
    num_cols: List[str],
    cat_cols: List[str]
) -> ColumnTransformer:

    _, imp, out, skw, scl, ord_enc = config_tuple
    
    # Build numerical pipeline
    num_steps = [
        ('impute', IMPUTERS[imp.lower()]),
        ('outlier', OUTLIER_HANDLERS[out.lower()]),
        ('skew', SKEW_TRANSFORMERS[skw.lower()]),
        ('scale', SCALERS[scl.lower()])
    ]
    num_pipeline = Pipeline([s for s in num_steps if s[1] != 'passthrough'])
    
    # Build categorical transformers
    ordinal_cols = [c for c in ORDINAL_COLS if c in cat_cols]
    nominal_cols = [c for c in cat_cols if c not in ordinal_cols]
    
    if ord_enc and len(ordinal_cols) > 0:
        cat_transformers = [
            ('ordinal', OrdinalEncoder(
                categories=ORDINAL_CATEGORIES[:len(ordinal_cols)],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ), ordinal_cols),
            ('onehot', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ), nominal_cols)
        ]
    else:
        cat_transformers = [
            ('onehot', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ), cat_cols)
        ]
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols)
    ] + cat_transformers)
    
    return preprocessor