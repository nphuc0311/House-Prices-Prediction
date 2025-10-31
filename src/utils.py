import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def plot_results(results_df: pd.DataFrame, output_path: str = None) -> None:
    logger.info("Creating visualization plots...")
    
    top10 = results_df.head(10)
    
    # Figure 1: Top 10 tuned metrics (horizontal bars)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # RMSE plot
    axes[0].barh(range(len(top10)), top10['Tuned_Test_RMSE'], color='coral')
    axes[0].set_yticks(range(len(top10)))
    labels = [f"{p[:30]}... ({'FE' if w else 'NoFE'})" if len(p) > 30 
              else f"{p} ({'FE' if w else 'NoFE'})"
              for p, w in zip(top10['Pipeline'], top10['With_FE'])]
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel('Tuned Test RMSE')
    axes[0].set_title('Tuned Test RMSE (Lower is Better)')
    axes[0].invert_yaxis()
    
    # Add text labels
    for i, val in enumerate(top10['Tuned_Test_RMSE']):
        axes[0].text(val + 0.01 * max(top10['Tuned_Test_RMSE']), i, 
                    f'{val:.0f}', va='center', ha='left', fontsize=8)
    axes[0].set_xlim(0, max(top10['Tuned_Test_RMSE']) * 1.1)
    
    # R² plot
    axes[1].barh(range(len(top10)), top10['Tuned_Test_R2'], color='lightgreen')
    axes[1].set_yticks(range(len(top10)))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel('Tuned Test R²')
    axes[1].set_title('Tuned Test R² (Higher is Better)')
    axes[1].invert_yaxis()
    
    # Add text labels
    for i, val in enumerate(top10['Tuned_Test_R2']):
        axes[1].text(val + 0.01, i, f'{val*100:.2f}%',
                    va='center', ha='left', fontsize=8)
    axes[1].set_xlim(0, 1.1)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(f"{output_path}/top10_metrics.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Test vs Tuned comparison for top 5
    top5 = results_df.head(5)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    x = np.arange(len(top5))
    width = 0.35
    
    # RMSE comparison
    axes[0].bar(x - width/2, top5['Test_RMSE'], width, 
               label='Test', alpha=0.8, color='coral')
    axes[0].bar(x + width/2, top5['Tuned_Test_RMSE'], width,
               label='Tuned', alpha=0.8, color='darkred')
    axes[0].set_xlabel('Pipeline Rank')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE: Test vs Tuned (Lower is Better)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(range(1, 6))
    axes[0].legend()
    
    # Add labels
    max_rmse = max(top5['Test_RMSE'].max(), top5['Tuned_Test_RMSE'].max())
    for i, (test_val, tuned_val) in enumerate(zip(top5['Test_RMSE'], top5['Tuned_Test_RMSE'])):
        axes[0].text(i - width/2, test_val + 0.02 * max_rmse, f'{test_val:.0f}',
                    ha='center', va='bottom', fontsize=8)
        axes[0].text(i + width/2, tuned_val + 0.02 * max_rmse, f'{tuned_val:.0f}',
                    ha='center', va='bottom', fontsize=8)
    axes[0].set_ylim(0, max_rmse * 1.1)
    
    # R² comparison
    axes[1].bar(x - width/2, top5['Test_R2'], width,
               label='Test', alpha=0.8, color='lightgreen')
    axes[1].bar(x + width/2, top5['Tuned_Test_R2'], width,
               label='Tuned', alpha=0.8, color='darkgreen')
    axes[1].set_xlabel('Pipeline Rank')
    axes[1].set_ylabel('R²')
    axes[1].set_title('R²: Test vs Tuned (Higher is Better)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(range(1, 6))
    axes[1].legend()
    
    # Add labels
    for i, (test_val, tuned_val) in enumerate(zip(top5['Test_R2'], top5['Tuned_Test_R2'])):
        axes[1].text(i - width/2, test_val + 0.02, f'{test_val*100:.2f}%',
                    ha='center', va='bottom', fontsize=8)
        axes[1].text(i + width/2, tuned_val + 0.02, f'{tuned_val*100:.2f}%',
                    ha='center', va='bottom', fontsize=8)
    axes[1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(f"{output_path}/test_vs_tuned.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Figure 3: Line plot for RMSE improvement
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top10))
    
    ax.plot(x, top10['Test_RMSE'], 'o-', label='Before Tuning', 
           linewidth=2, markersize=8)
    ax.plot(x, top10['Tuned_Test_RMSE'], 's-', label='After Tuning',
           linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i+1}" for i in range(len(top10))], fontsize=10)
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.set_xlabel('Pipeline Rank', fontsize=12)
    ax.set_title('RMSE Improvement: Before vs After Hyperparameter Tuning',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(f"{output_path}/rmse_improvement.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    logger.info("Plots created successfully!")
    

def save_model(preprocessor, model, config: dict, output_path: str) -> None:
    import joblib
    import yaml
    import os
    
    os.makedirs(output_path, exist_ok=True)
    
    # Save model pipeline
    model_path = f"{output_path}/best_pipeline.joblib"
    joblib.dump({'preprocessor': preprocessor, 'model': model}, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save config
    config_path = f"{output_path}/best_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Config saved to {config_path}")


def load_model(model_path: str):
    import joblib
    
    pipeline = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return pipeline['preprocessor'], pipeline['model']