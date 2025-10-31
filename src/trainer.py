import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats

from sklearn.linear_model import Ridge

from .preprocessing import (
    get_benchmark_configs,
    create_pipeline_from_config_tuple,
    PipelineBuilder
)
from .models import (
    get_models_to_test,
    evaluate_pipeline,
    tune_model_optuna,
    create_model_from_params
)
from .feature_engineering import add_features
from .utils import plot_results

logger = logging.getLogger(__name__)


class Trainer:
    
    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        num_cols: list,
        cat_cols: list,
        config: Dict[str, Any]
    ):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.config = config
        
        # Check if target needs log transform
        self.y_skew = stats.skew(y_train)
        self.y_log_transform = abs(self.y_skew) > 1
        
        if self.y_log_transform:
            self.y_train_transformed = np.log1p(y_train)
            self.y_test_transformed = np.log1p(y_test)
            logger.info(f"Target skew: {self.y_skew:.2f} - Applied log1p transform")
        else:
            self.y_train_transformed = y_train
            self.y_test_transformed = y_test
            logger.info(f"Target skew: {self.y_skew:.2f} - No transform needed")
    
    def run_benchmark(self) -> pd.DataFrame:
        logger.info("Starting benchmark...")
        
        pipeline_configs = get_benchmark_configs()
        models_to_test = get_models_to_test()
        results = []
        
        # Phase 1: Evaluate all configs without FE
        logger.info("Phase 1: Evaluating pipelines without feature engineering...")
        for i, config_tuple in enumerate(pipeline_configs, 1):
            name = config_tuple[0]
            preprocessor = create_pipeline_from_config_tuple(
                config_tuple, self.num_cols, self.cat_cols
            )
            
            for model_name, model in models_to_test.items():
                full_name = f"{name}_{model_name}"
                try:
                    metrics, _, _ = evaluate_pipeline(
                        full_name, preprocessor, model,
                        self.X_train, self.X_test,
                        self.y_train_transformed, self.y_test_transformed,
                        inverse_transform=self.y_log_transform
                    )
                    metrics['With_FE'] = False
                    results.append(metrics)
                except Exception as e:
                    logger.error(f"Failed to evaluate {full_name}: {e}")
            
            if i % 5 == 0:
                logger.info(f"Completed {i}/{len(pipeline_configs)} configurations...")
        
        results_df = pd.DataFrame(results).sort_values('Test_R2', ascending=False)
        
        # Phase 2: Apply FE to top 10 configs
        logger.info("\nPhase 2: Applying feature engineering to top 10 configs...")
        X_train_fe, X_test_fe, num_cols_fe = add_features(
            self.X_train, self.X_test, self.num_cols
        )
        
        top_10_configs = results_df['Pipeline'].str.rsplit('_', n=1).str[0].unique()[:10]
        fe_results = []
        
        for config_name in top_10_configs:
            matching_configs = [c for c in pipeline_configs if c[0] == config_name]
            if not matching_configs:
                continue
            
            config_tuple = matching_configs[0]
            preprocessor = create_pipeline_from_config_tuple(
                config_tuple, num_cols_fe, self.cat_cols
            )
            
            for model_name, model in models_to_test.items():
                full_name = f"{config_name}_{model_name}"
                try:
                    metrics, _, _ = evaluate_pipeline(
                        full_name, preprocessor, model,
                        X_train_fe, X_test_fe,
                        self.y_train_transformed, self.y_test_transformed,
                        inverse_transform=self.y_log_transform
                    )
                    metrics['With_FE'] = True
                    fe_results.append(metrics)
                except Exception as e:
                    logger.error(f"Failed to evaluate {full_name} with FE: {e}")
        
        results_df = pd.concat([results_df, pd.DataFrame(fe_results)], ignore_index=True)
        results_df = results_df.sort_values('Test_R2', ascending=False)
        
        logger.info(f"\nBenchmark complete! Total pipelines evaluated: {len(results_df)}")
        return results_df
    
    def tune_top_pipelines(
        self,
        results_df: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        logger.info(f"\nPhase 3: Tuning top {top_n} pipelines...")
        
        tuning_cfg = self.config.get('tuning', {})
        n_trials = tuning_cfg.get('n_trials', 20)
        
        pipeline_configs = get_benchmark_configs()
        tuning_results = []
        top_rows = results_df.head(top_n)
        
        # Prepare FE data if needed
        X_train_fe, X_test_fe, num_cols_fe = add_features(
            self.X_train, self.X_test, self.num_cols
        )
        
        for idx, row in top_rows.iterrows():
            pipeline_name = row['Pipeline']
            with_fe = row['With_FE']
            
            # Extract model name and config
            model_name = pipeline_name.split('_')[-1]
            config_name = '_'.join(pipeline_name.split('_')[:-1])
            
            # Find matching config
            matching_configs = [c for c in pipeline_configs if c[0] == config_name]
            if not matching_configs:
                logger.warning(f"Config not found for {pipeline_name}")
                continue
            
            config_tuple = matching_configs[0]
            
            # Select appropriate data
            X_tr = X_train_fe if with_fe else self.X_train
            X_te = X_test_fe if with_fe else self.X_test
            num_cols_list = num_cols_fe if with_fe else self.num_cols
            
            # Create preprocessor
            preprocessor = create_pipeline_from_config_tuple(
                config_tuple, num_cols_list, self.cat_cols
            )
            
            # Tune
            try:
                best_params, best_cv_rmse = tune_model_optuna(
                    model_name, preprocessor, X_tr,
                    self.y_train_transformed, n_trials=n_trials
                )
                
                # Evaluate with best params
                X_tr_processed = preprocessor.fit_transform(X_tr)
                X_te_processed = preprocessor.transform(X_te)
                
                tuned_model = create_model_from_params(model_name, best_params)
                tuned_model.fit(X_tr_processed, self.y_train_transformed)
                y_te_pred = tuned_model.predict(X_te_processed)
                
                if self.y_log_transform:
                    y_te_pred = np.expm1(y_te_pred)
                    y_te_actual = np.expm1(self.y_test_transformed)
                else:
                    y_te_actual = self.y_test_transformed
                
                from sklearn.metrics import r2_score, mean_squared_error
                tuned_r2 = r2_score(y_te_actual, y_te_pred)
                tuned_rmse = np.sqrt(mean_squared_error(y_te_actual, y_te_pred))
                
                tuning_results.append({
                    'Pipeline': pipeline_name,
                    'Tuned_Params': str(best_params),
                    'Tuned_Test_R2': tuned_r2,
                    'Tuned_Test_RMSE': tuned_rmse
                })
                
                logger.info(f"✓ {pipeline_name}: R²={tuned_r2:.4f}, RMSE={tuned_rmse:.2f}")
                
            except Exception as e:
                logger.error(f"✗ {pipeline_name}: Tuning failed - {str(e)}")
                tuning_results.append({
                    'Pipeline': pipeline_name,
                    'Tuned_Params': 'Failed',
                    'Tuned_Test_R2': row['Test_R2'],
                    'Tuned_Test_RMSE': row['Test_RMSE']
                })
        
        # Merge tuning results
        tuning_df = pd.DataFrame(tuning_results)
        results_df = results_df.merge(tuning_df, on='Pipeline', how='left')
        results_df['Tuned_Test_R2'] = results_df['Tuned_Test_R2'].fillna(results_df['Test_R2'])
        results_df['Tuned_Test_RMSE'] = results_df['Tuned_Test_RMSE'].fillna(results_df['Test_RMSE'])
        results_df = results_df.sort_values('Tuned_Test_R2', ascending=False)
        
        return results_df
    
    def train_single_pipeline(
        self,
        tune: bool = True
    ) -> Tuple[Any, Any, Dict[str, Any]]:

        logger.info("Training single pipeline from config...")
        
        preproc_cfg = self.config.get('preprocessing', {})
        model_cfg = self.config.get('model', {})
        
        # Apply FE if configured
        if preproc_cfg.get('use_fe', True):
            X_tr, X_te, num_cols_list = add_features(
                self.X_train, self.X_test, self.num_cols
            )
        else:
            X_tr, X_te, num_cols_list = self.X_train, self.X_test, self.num_cols
        
        # Create pipeline
        builder = PipelineBuilder(self.config)
        preprocessor = builder.create_pipeline(num_cols_list, self.cat_cols)
        
        # Get model
        model_name = model_cfg.get('name', 'ridge').lower()
        models_dict = get_models_to_test()
        model = models_dict.get(model_name, Ridge())
        
        # Tune if requested
        if tune and self.config.get('tuning', {}).get('enabled', True):
            n_trials = self.config.get('tuning', {}).get('n_trials', 20)
            best_params, _ = tune_model_optuna(
                model_name, preprocessor, X_tr,
                self.y_train_transformed, n_trials=n_trials
            )
            model = create_model_from_params(model_name, best_params)
            logger.info(f"Model tuned with params: {best_params}")
        
        # Train and evaluate
        metrics, _, _ = evaluate_pipeline(
            f"SinglePipeline_{model_name}",
            preprocessor, model,
            X_tr, X_te,
            self.y_train_transformed, self.y_test_transformed,
            inverse_transform=self.y_log_transform
        )
        
        logger.info(f"Training complete! Test R²: {metrics['Test_R2']:.4f}, "
                   f"Test RMSE: {metrics['Test_RMSE']:.2f}")
        
        return preprocessor, model, metrics