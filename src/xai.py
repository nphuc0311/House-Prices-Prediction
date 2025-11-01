import os
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Suppress verbose SHAP logging
logging.getLogger('shap').setLevel(logging.WARNING)
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
import joblib

logger = logging.getLogger(__name__)


class ExplainabilityAnalyzer:    
    def __init__(
        self,
        preprocessor: ColumnTransformer,
        model: BaseEstimator,
        feature_names_in: List[str],
        target_log_transformed: bool = False
    ):
        self.preprocessor = preprocessor
        self.model = model
        self.feature_names_in = feature_names_in
        self.target_log_transformed = target_log_transformed
        self.transformed_feature_names = self._extract_feature_names()
        
        logger.info(f"Initialized ExplainabilityAnalyzer with {len(self.transformed_feature_names)} transformed features")
    
    def _extract_feature_names(self) -> List[str]:
        feature_names = []
        
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'remainder' or transformer == 'drop':
                continue
            
            if hasattr(transformer, 'get_feature_names_out'):
                # OneHotEncoder, OrdinalEncoder with get_feature_names_out
                try:
                    names = transformer.get_feature_names_out(columns)
                    feature_names.extend(names)
                except Exception:
                    # Fallback for pipelines
                    feature_names.extend(columns)
            elif hasattr(transformer, 'named_steps'):
                # Pipeline - use original column names
                feature_names.extend(columns)
            else:
                # Simple transformer
                feature_names.extend(columns)
        
        logger.info(f"Extracted {len(feature_names)} feature names from preprocessor")
        return feature_names
    
    def compute_shap(
        self,
        X: pd.DataFrame,
        output_dir: str,
        sample_size: int = 1000,
        background_size: int = 100
    ) -> Tuple[np.ndarray, shap.Explanation]:
        
        logger.info("Computing SHAP values...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample data if needed
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        # Transform data
        X_transformed = self.preprocessor.transform(X_sample)
        
        # Detect model type and choose explainer
        model_name = self.model.__class__.__name__.lower()
        
        if any(tree_model in model_name for tree_model in ['forest', 'tree', 'xgb', 'lgbm', 'catboost', 'gbm']):
            logger.info("Using TreeExplainer (tree-based model detected)")
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_transformed)
        elif any(linear_model in model_name for linear_model in ['linear', 'ridge', 'lasso', 'elasticnet']):
            logger.info("Using LinearExplainer (linear model detected)")
            explainer = shap.LinearExplainer(self.model, X_transformed)
            shap_values = explainer.shap_values(X_transformed)
        else:
            logger.info("Using KernelExplainer (model-agnostic approach)")
            # Use smaller background for KernelExplainer (computationally expensive)
            if len(X_transformed) > background_size:
                background = shap.sample(X_transformed, background_size, random_state=42)
            else:
                background = X_transformed
            
            explainer = shap.KernelExplainer(self.model.predict, background)
            shap_values = explainer.shap_values(X_transformed, nsamples=100)
        
        # Create explanation object with proper base values
        if hasattr(explainer, 'expected_value'):
            base_values = explainer.expected_value
            # Handle case where expected_value is a single float
            if isinstance(base_values, (float, int)):
                base_values = np.array([base_values] * len(X_transformed))
            # Handle case where expected_value is a numpy array
            elif isinstance(base_values, np.ndarray) and base_values.ndim == 0:
                base_values = np.array([float(base_values)] * len(X_transformed))
        else:
            # Use mean prediction as base value for all samples
            base_values = np.array([self.model.predict(X_transformed).mean()] * len(X_transformed))

        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_values,
            data=X_transformed,
            feature_names=self.transformed_feature_names
        )
        
        # Save SHAP values to CSV
        shap_df = pd.DataFrame(
            shap_values,
            columns=self.transformed_feature_names
        )
        shap_df.to_csv(f"{output_dir}/shap_values.csv", index=False)
        logger.info(f"Saved SHAP values to {output_dir}/shap_values.csv")
        
        # Generate visualizations
        self._plot_shap_summary(explanation, output_dir)
        self._plot_shap_dependence(explanation, output_dir, top_n=3)
        
        return shap_values, explanation
    
    def _plot_shap_summary(self, explanation: shap.Explanation, output_dir: str):
        """Generate SHAP summary plots (bar and beeswarm)."""
        # Bar plot (mean absolute SHAP)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(explanation, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_summary_bar.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP bar plot to {output_dir}/shap_summary_bar.png")
        
        # Beeswarm plot (detailed distribution)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(explanation, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_summary_beeswarm.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP beeswarm plot to {output_dir}/shap_summary_beeswarm.png")
    
    def _plot_shap_dependence(self, explanation: shap.Explanation, output_dir: str, top_n: int = 3):
        """Generate SHAP dependence plots for top N features."""
        # Get top features by mean absolute SHAP
        mean_abs_shap = np.abs(explanation.values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        
        for idx in top_indices:
            feature_name = self.transformed_feature_names[idx]
            safe_name = feature_name.replace('/', '_').replace(' ', '_')
            
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                idx,
                explanation.values,
                explanation.data,
                feature_names=self.transformed_feature_names,
                show=False
            )
            plt.tight_layout()
            plt.savefig(f"{output_dir}/shap_dependence_{safe_name}.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved SHAP dependence plot for {feature_name}")
    
    def compute_permutation_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        output_dir: str,
        n_repeats: int = 30,
        random_state: int = 42
    ) -> pd.DataFrame:
        
        logger.info(f"Computing permutation importance ({n_repeats} repeats)...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Transform data
        X_transformed = self.preprocessor.transform(X)
        
        # Inverse transform target if needed
        if self.target_log_transformed:
            y_eval = y  # Use transformed target for scoring
        else:
            y_eval = y
        
        # Compute permutation importance
        result = permutation_importance(
            self.model,
            X_transformed,
            y_eval,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.transformed_feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        })
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        
        # Save to CSV
        importance_df.to_csv(f"{output_dir}/permutation_importance.csv", index=False)
        logger.info(f"Saved permutation importance to {output_dir}/permutation_importance.csv")
        
        # Plot
        self._plot_permutation_importance(importance_df, output_dir)
        
        return importance_df
    
    def _plot_permutation_importance(self, importance_df: pd.DataFrame, output_dir: str, top_n: int = 20):
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance_mean'], 
                xerr=top_features['importance_std'], color='steelblue', alpha=0.8)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Permutation Importance (RMSE decrease)', fontsize=12)
        plt.title(f'Top {top_n} Features by Permutation Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/permutation_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved permutation importance plot to {output_dir}/permutation_importance.png")
    
    def generate_report(self, output_dir: str, shap_values: np.ndarray, perm_importance_df: pd.DataFrame):
        """Generate summary report."""
        report_path = f"{output_dir}/explainability_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL EXPLAINABILITY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Model: {self.model.__class__.__name__}\n")
            f.write(f"Target Log-Transformed: {self.target_log_transformed}\n")
            f.write(f"Number of Features: {len(self.transformed_feature_names)}\n")
            f.write(f"Number of Samples Explained: {shap_values.shape[0]}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("TOP 10 FEATURES BY SHAP (Mean Absolute Impact)\n")
            f.write("-" * 80 + "\n")
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_shap_indices = np.argsort(mean_abs_shap)[-10:][::-1]
            for rank, idx in enumerate(top_shap_indices, 1):
                f.write(f"{rank:2d}. {self.transformed_feature_names[idx]:40s} {mean_abs_shap[idx]:.6f}\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("TOP 10 FEATURES BY PERMUTATION IMPORTANCE\n")
            f.write("-" * 80 + "\n")
            for rank, row in enumerate(perm_importance_df.head(10).itertuples(), 1):
                f.write(f"{rank:2d}. {row.feature:40s} {row.importance_mean:.6f} ± {row.importance_std:.6f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("FILES GENERATED:\n")
            f.write("=" * 80 + "\n")
            f.write("- shap_values.csv: SHAP values for all samples\n")
            f.write("- shap_summary_bar.png: Mean absolute SHAP values\n")
            f.write("- shap_summary_beeswarm.png: Detailed SHAP distribution\n")
            f.write("- shap_dependence_*.png: Dependence plots for top features\n")
            f.write("- permutation_importance.csv: Feature importance scores\n")
            f.write("- permutation_importance.png: Visual importance ranking\n")
            f.write("- explainability_report.txt: This report\n")
        
        logger.info(f"Saved explainability report to {report_path}")


def explain_model_pipeline(
    model_path: str,
    X_path: str,
    y_path: str,
    output_dir: str,
    sample_size: int = 1000,
    n_repeats: int = 30
) -> Dict[str, Any]:

    logger.info("Starting explainability pipeline...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and preprocessor
    logger.info(f"Loading model from {model_path}")
    pipeline = joblib.load(model_path)
    preprocessor = pipeline['preprocessor']
    model = pipeline['model']
    
    # Load data
    logger.info(f"Loading data from {X_path} and {y_path}")
    X = pd.read_csv(X_path)
    
    if y_path.endswith('.npy'):
        y = np.load(y_path)
    elif y_path.endswith('.csv'):
        y_df = pd.read_csv(y_path)
        y = y_df.values.ravel() if len(y_df.columns) == 1 else y_df.iloc[:, 0].values
    else:
        raise ValueError(f"Unsupported target file format: {y_path}")
    
    # Detect if target was log-transformed (heuristic: check if all positive and skewed)
    from scipy import stats
    y_skew = stats.skew(y)
    target_log_transformed = abs(y_skew) > 1
    
    # Initialize analyzer
    analyzer = ExplainabilityAnalyzer(
        preprocessor=preprocessor,
        model=model,
        feature_names_in=X.columns.tolist(),
        target_log_transformed=target_log_transformed
    )
    
    # Compute SHAP
    shap_values, explanation = analyzer.compute_shap(X, output_dir, sample_size=sample_size)
    
    # Compute Permutation Importance
    perm_importance_df = analyzer.compute_permutation_importance(
        X, y, output_dir, n_repeats=n_repeats
    )
    
    # Generate report
    analyzer.generate_report(output_dir, shap_values, perm_importance_df)
    
    logger.info(f"✓ Explainability analysis complete! Results saved to {output_dir}/")
    
    return {
        'shap_values': shap_values,
        'permutation_importance': perm_importance_df,
        'output_dir': output_dir,
        'feature_names': analyzer.transformed_feature_names
    }


# Convenience functions for backward compatibility
def compute_shap(
    model: BaseEstimator,
    X: pd.DataFrame,
    feature_names: List[str],
    output_dir: str,
    sample_size: int = 1000
) -> np.ndarray:

    logger.warning("Using simplified compute_shap. Consider using ExplainabilityAnalyzer for full pipeline support.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42) if isinstance(X, pd.DataFrame) else X[:sample_size]
    else:
        X_sample = X
    
    model_name = model.__class__.__name__.lower()
    
    if any(linear in model_name for linear in ['linear', 'ridge', 'lasso', 'elasticnet']):
        explainer = shap.LinearExplainer(model, X_sample)
    else:
        explainer = shap.KernelExplainer(model.predict, X_sample[:100])
    
    shap_values = explainer.shap_values(X_sample)
    
    # Save and plot
    pd.DataFrame(shap_values, columns=feature_names).to_csv(f"{output_dir}/shap_values.csv", index=False)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return shap_values


def compute_permutation_importance(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    n_repeats: int = 30
) -> pd.DataFrame:

    logger.warning("Using simplified compute_permutation_importance. Consider using ExplainabilityAnalyzer for full pipeline support.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    importance_df.to_csv(f"{output_dir}/permutation_importance.csv", index=False)
    
    plt.figure(figsize=(10, 8))
    top20 = importance_df.head(20)
    plt.barh(range(len(top20)), top20['importance_mean'], xerr=top20['importance_std'])
    plt.yticks(range(len(top20)), top20['feature'])
    plt.xlabel('Permutation Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/permutation_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return importance_df