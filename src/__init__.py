__version__ = '1.0.0'
__author__ = 'CONQ025 Team'

# Import key classes and functions for convenience
from .data_loader import download_data_if_needed, load_csv, quick_eda, prepare_data
from .feature_engineering import add_features, winsorize_func, iqr_clip_func
from .preprocessing import PipelineBuilder, get_benchmark_configs
from .models import (
    get_models_to_test,
    calculate_rae,
    evaluate_pipeline,
    tune_model_optuna,
    create_model_from_params
)
from .trainer import Trainer
from .utils import setup_logging, plot_results, save_model, load_model

__all__ = [
    # Data
    'download_data_if_needed',
    'load_csv',
    'quick_eda',
    'prepare_data',
    
    # Feature Engineering
    'add_features',
    'winsorize_func',
    'iqr_clip_func',
    
    # Preprocessing
    'PipelineBuilder',
    'get_benchmark_configs',
    
    # Models
    'get_models_to_test',
    'calculate_rae',
    'evaluate_pipeline',
    'tune_model_optuna',
    'create_model_from_params',
    
    # Training
    'Trainer',
    
    # Utils
    'setup_logging',
    'plot_results',
    'print_summary',
    'save_model',
    'load_model'
]