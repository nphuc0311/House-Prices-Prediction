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
from src.utils import setup_logging, plot_results, save_model
from src.xai import explain_model_pipeline, ExplainabilityAnalyzer

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

    print("=" * 80)
    
    # Run explainability if requested
    if args.explain:
        logger.info("\nRunning explainability analysis...")
        
        # Save test data for explanation
        import numpy as np
        temp_dir = args.output or 'models'
        X_test_path = f"{temp_dir}/X_test_explain.csv"
        y_test_path = f"{temp_dir}/y_test_explain.npy"

        # Apply feature engineering if enabled
        use_fe = config.get('preprocessing', {}).get('use_fe', False)
        if use_fe:
            from src.feature_engineering import add_features
            _, X_test, _ = add_features(X_train, X_test, num_cols)
        
        # Save the test data which already has engineered features if they were enabled
        X_test.to_csv(X_test_path, index=False)
        np.save(y_test_path, y_test)
        
        # Run explanation
        explain_output = args.explain_output or f"{temp_dir}/explain"
        
        analyzer = ExplainabilityAnalyzer(
            preprocessor=preprocessor,
            model=model,
            feature_names_in=X_test.columns.tolist(),
            target_log_transformed=trainer.y_log_transform
        )
        
        shap_values, _ = analyzer.compute_shap(
            X_test, explain_output, 
            sample_size=config.get('explainability', {}).get('sample_size', 1000)
        )
        
        perm_importance = analyzer.compute_permutation_importance(
            X_test, y_test, explain_output,
            n_repeats=config.get('explainability', {}).get('n_repeats', 30)
        )
        
        analyzer.generate_report(explain_output, shap_values, perm_importance)
        
        logger.info(f"✓ Explainability results saved to {explain_output}/")


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


def mode_explain(args):
    logger.info("Running EXPLAIN mode...")
    
    if not args.model_path:
        logger.error("--model_path is required for explain mode")
        print("Error: --model_path must be specified for explain mode")
        print("Example: python main.py --mode explain --model_path models/best_pipeline.joblib --X_path data/X_test.csv --y_path data/y_test.npy")
        sys.exit(1)
    
    if not args.X_path:
        logger.error("--X_path is required for explain mode")
        print("Error: --X_path must be specified for explain mode")
        sys.exit(1)
    
    if not args.y_path:
        logger.error("--y_path is required for explain mode")
        print("Error: --y_path must be specified for explain mode")
        sys.exit(1)
    
    # Load config for explainability settings
    config = {}
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    
    xai_config = config.get('explainability', {})
    sample_size = xai_config.get('sample_size', 1000)
    n_repeats = xai_config.get('n_repeats', 30)
    
    # Set output directory
    output_dir = args.output or 'results/explain'
    
    # Run explainability pipeline
    results = explain_model_pipeline(
        model_path=args.model_path,
        X_path=args.X_path,
        y_path=args.y_path,
        output_dir=output_dir,
        sample_size=sample_size,
        n_repeats=n_repeats
    )
    
    print("\n" + "=" * 80)
    print("EXPLAINABILITY ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Number of features: {len(results['feature_names'])}")
    print(f"Samples explained: {results['shap_values'].shape[0]}")
    print("\nGenerated files:")
    print("  - shap_values.csv")
    print("  - shap_summary_bar.png")
    print("  - shap_summary_beeswarm.png")
    print("  - shap_dependence_*.png")
    print("  - permutation_importance.csv")
    print("  - permutation_importance.png")
    print("  - explainability_report.txt")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='House Prices Prediction - ML Pipeline with Explainability',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['eda', 'train', 'benchmark', 'explain'],
        help='Execution mode (explain mode requires --model_path, --X_path, --y_path)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
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
        '--explain',
        action='store_true',
        help='Run explainability analysis after training (train mode)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to saved model joblib file (explain mode)'
    )
    
    parser.add_argument(
        '--X_path',
        type=str,
        default=None,
        help='Path to features CSV (explain mode)'
    )
    
    parser.add_argument(
        '--y_path',
        type=str,
        default=None,
        help='Path to target CSV or NPY file (explain mode)'
    )
    
    parser.add_argument(
        '--explain_output',
        type=str,
        default=None,
        help='Output directory for explainability results (train mode with --explain)'
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
    elif args.mode == 'explain':
        mode_explain(args)


if __name__ == '__main__':
    main()