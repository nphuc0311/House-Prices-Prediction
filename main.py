import argparse
import os
import sys
import warnings
import yaml

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import download_data_if_needed, load_csv, quick_eda, prepare_data
from src.trainer import Trainer
from src.utils import setup_logging, plot_results, print_summary, save_model

import logging
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def mode_eda(args):
    logger.info("Running EDA mode...")
    
    # Download data if needed
    download_data_if_needed(args.data_path)
    
    # Load and analyze
    df = load_csv(args.data_path)
    print("=" * 60)
    quick_eda(df)
    print("=" * 60)


def mode_train(args):
    logger.info("Running TRAIN mode...")
    
    # Load config
    config = load_config(args.config)
    
    # Download data if needed
    data_path = args.data_path or 'data/raw/train-house-prices-advanced-regression-techniques.csv'
    download_data_if_needed(data_path)
    
    # Load and prepare data
    df = load_csv(data_path)
    data_cfg = config.get('data', {})
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data(
        df,
        test_size=data_cfg.get('test_size', 0.25),
        random_state=data_cfg.get('random_state', 42),
        drop_cols=data_cfg.get('drop_columns')
    )
    
    # Train
    trainer = Trainer(X_train, X_test, y_train, y_test, num_cols, cat_cols, config)
    preprocessor, model, metrics = trainer.train_single_pipeline(
        tune=config.get('tuning', {}).get('enabled', True)
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Save model
    if args.save_model or config.get('output', {}).get('save_model', True):
        output_path = args.output or 'models'
        save_model(preprocessor, model, config, output_path)
        logger.info(f"Model saved to {output_path}/")


def mode_benchmark(args):
    logger.info("Running BENCHMARK mode...")
    
    # Load config (or use defaults)
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = {
            'data': {'test_size': 0.25, 'random_state': 42},
            'tuning': {'enabled': True, 'n_trials': 20}
        }
    
    # Download data if needed
    data_path = args.data_path or 'data/raw/train-house-prices-advanced-regression-techniques.csv'
    download_data_if_needed(data_path)
    
    # Load and prepare data
    df = load_csv(data_path)
    data_cfg = config.get('data', {})
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data(
        df,
        test_size=data_cfg.get('test_size', 0.25),
        random_state=data_cfg.get('random_state', 42)
    )
    
    # Run benchmark
    trainer = Trainer(X_train, X_test, y_train, y_test, num_cols, cat_cols, config)
    results_df = trainer.run_benchmark()
    
    # Tune top pipelines
    results_df = trainer.tune_top_pipelines(results_df, top_n=10)
    
    # Print top 10
    print("\n" + "=" * 80)
    print("TOP 10 PIPELINE PERFORMANCES (with tuning):")
    print("=" * 80)
    display_cols = ['Pipeline', 'With_FE', 'Test_R2', 'Tuned_Test_R2', 
                   'Test_RMSE', 'Tuned_Test_RMSE']
    print(results_df.head(10)[display_cols].to_string(index=False))
    
    # Create visualizations
    output_path = args.output or 'results'
    os.makedirs(output_path, exist_ok=True)
    plot_results(results_df, output_path=output_path)
    
    # Save results
    results_path = f"{output_path}/pipeline_benchmark_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description='House Prices Prediction - ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['eda', 'train', 'benchmark'],
        help='Execution mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_pipeline.yaml',
        help='Path to config YAML file (default: config/default.yaml)'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to data CSV file (auto-downloads if not present)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results/models'
    )
    
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='Save trained model (train mode)'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(getattr(logging, args.log_level))
    
    # Route to appropriate mode
    if args.mode == 'eda':
        mode_eda(args)
    elif args.mode == 'train':
        mode_train(args)
    elif args.mode == 'benchmark':
        mode_benchmark(args)


if __name__ == '__main__':
    main()