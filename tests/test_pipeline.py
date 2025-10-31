import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_loader import prepare_data
from src.feature_engineering import add_features, winsorize_func, iqr_clip_func
from src.models import calculate_rae, evaluate_pipeline, get_models_to_test
from src.preprocessing import PipelineBuilder


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    
    data = {
        'Id': range(1, n+1),
        'MSZoning': np.random.choice(['RL', 'RM', 'FV'], n),
        'LotArea': np.random.randint(5000, 20000, n),
        'OverallQual': np.random.randint(1, 11, n),
        'YearBuilt': np.random.randint(1950, 2010, n),
        'YrSold': np.random.randint(2006, 2011, n),
        'GrLivArea': np.random.randint(800, 3000, n),
        'TotalBsmtSF': np.random.randint(0, 1500, n),
        'FullBath': np.random.randint(1, 4, n),
        'HalfBath': np.random.randint(0, 2, n),
        'BsmtFullBath': np.random.randint(0, 2, n),
        'BsmtHalfBath': np.random.randint(0, 2, n),
        'BedroomAbvGr': np.random.randint(1, 6, n),
        'ExterQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa'], n),
        'KitchenQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa'], n),
        'SalePrice': np.random.randint(100000, 400000, n)
    }
    
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        'data': {
            'test_size': 0.25,
            'random_state': 42
        },
        'preprocessing': {
            'imputer': 'simple',
            'outlier': 'winsor',
            'skew': 'log1p',
            'scaler': 'standard',
            'use_ordinal': True,
            'use_fe': True
        },
        'model': {
            'name': 'ridge',
            'params': {}
        },
        'tuning': {
            'enabled': False,
            'n_trials': 5
        }
    }


class TestDataPreparation:    
    def test_prepare_data(self, sample_data):
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data(
            sample_data, test_size=0.25, random_state=42
        )
        
        # Check shapes
        assert len(X_train) + len(X_test) == len(sample_data)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Check column types
        assert len(num_cols) > 0
        assert len(cat_cols) > 0
        assert 'SalePrice' not in X_train.columns
        
        # Check Id is dropped
        assert 'Id' not in X_train.columns


class TestFeatureEngineering:    
    def test_add_features(self, sample_data):
        X_train, X_test, _, _, num_cols, _ = prepare_data(sample_data)
        X_train_fe, X_test_fe, num_cols_fe = add_features(X_train, X_test, num_cols)
        
        # Check new features exist
        assert 'Age' in X_train_fe.columns
        assert 'TotalSF' in X_train_fe.columns
        assert 'TotalBath' in X_train_fe.columns
        
        # Check Age calculation
        assert (X_train_fe['Age'] == X_train_fe['YrSold'] - X_train_fe['YearBuilt']).all()
        
        # Check updated num_cols
        assert len(num_cols_fe) > len(num_cols)
    
    def test_winsorize_func(self):
        X = np.array([[1, 2, 3, 4, 100]]).T  # Outlier at 100
        X_winsorized = winsorize_func(X)
        
        # Check outlier is clipped
        assert X_winsorized.max() < 100
        assert X_winsorized.min() >= X.min()
    
    def test_iqr_clip_func(self):
        X = np.array([[1, 2, 3, 4, 5, 100]]).T  # Outlier at 100
        X_clipped = iqr_clip_func(X)
        
        # Check outlier is clipped
        assert X_clipped.max() < 100


class TestModels:    
    def test_calculate_rae(self):
        y_true = np.array([100, 200, 300, 400])
        y_pred = np.array([110, 190, 310, 390])
        
        rae = calculate_rae(y_true, y_pred)
        
        # RAE should be positive
        assert rae > 0
        # Perfect prediction would have RAE = 0
        assert calculate_rae(y_true, y_true) == 0
    
    def test_get_models_to_test(self):
        models = get_models_to_test()
        
        assert len(models) == 7
        assert 'ridge' in models
        assert 'lasso' in models
        assert 'elasticnet' in models
    
    def test_evaluate_pipeline(self, sample_data, sample_config):
        # Prepare data
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data(sample_data)
        
        # Create pipeline
        builder = PipelineBuilder(sample_config)
        preprocessor = builder.create_pipeline(num_cols, cat_cols)
        
        # Get model
        models = get_models_to_test()
        model = models['ridge']
        
        # Evaluate
        metrics, X_train_proc, X_test_proc = evaluate_pipeline(
            'test_pipeline',
            preprocessor,
            model,
            X_train, X_test,
            np.log1p(y_train), np.log1p(y_test),
            inverse_transform=True
        )
        
        # Check metrics exist
        assert 'Train_RMSE' in metrics
        assert 'Test_RMSE' in metrics
        assert 'Train_R2' in metrics
        assert 'Test_R2' in metrics
        
        # Check metrics are reasonable
        assert 0 <= metrics['Test_R2'] <= 1
        assert metrics['Test_RMSE'] > 0


class TestPreprocessing:    
    def test_pipeline_builder(self, sample_data, sample_config):
        X_train, X_test, _, _, num_cols, cat_cols = prepare_data(sample_data)
        
        builder = PipelineBuilder(sample_config)
        preprocessor = builder.create_pipeline(num_cols, cat_cols)
        
        # Fit and transform
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        
        # Check shapes
        assert X_train_proc.shape[0] == len(X_train)
        assert X_test_proc.shape[0] == len(X_test)
        assert X_train_proc.shape[1] == X_test_proc.shape[1]
        
        # Check no NaN values after preprocessing
        assert not np.isnan(X_train_proc).any()
        assert not np.isnan(X_test_proc).any()


class TestEndToEnd:    
    def test_full_pipeline(self, sample_data, sample_config):
        # Prepare data
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data(sample_data)
        
        # Add features
        X_train_fe, X_test_fe, num_cols_fe = add_features(X_train, X_test, num_cols)
        
        # Create pipeline
        builder = PipelineBuilder(sample_config)
        preprocessor = builder.create_pipeline(num_cols_fe, cat_cols)
        
        # Train model
        models = get_models_to_test()
        model = models['ridge']
        
        # Evaluate
        metrics, _, _ = evaluate_pipeline(
            'full_test',
            preprocessor,
            model,
            X_train_fe, X_test_fe,
            np.log1p(y_train), np.log1p(y_test),
            inverse_transform=True
        )
        
        # Check reasonable performance
        assert metrics['Test_R2'] > 0.3  # At least some predictive power
        assert metrics['Test_RMSE'] < 200000  # Reasonable error for house prices


if __name__ == '__main__':
    pytest.main([__file__, '-v'])