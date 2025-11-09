# House Prices Prediction - Production ML Pipeline with Explainability

A modular, CLI-driven, and extensible Python project for House Prices prediction using scikit-learn pipelines. This project refactors a Jupyter Notebook into production-ready code with comprehensive benchmarking, hyperparameter tuning, **model explainability (XAI)**, and interactive demo capabilities.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns across data loading, preprocessing, modeling, and training
- **CLI Interface**: Command-line driven for easy automation and integration
- **YAML Configuration**: Flexible pipeline configuration without code changes
- **Comprehensive Benchmarking**: Tests 20 preprocessing configurations × 7 models
- **Feature Engineering**: Automated creation of Age, TotalSF, and TotalBath features
- **Hyperparameter Tuning**: Bayesian optimization with Optuna (20 trials, 3-fold CV)
- **Model Persistence**: Save/load trained pipelines with joblib
- **Visualization**: Automated plotting of benchmark results
- **🆕 Model Explainability (XAI)**: SHAP values, permutation importance, feature analysis

## 🔍 New: Explainability Module

The project now includes comprehensive model explainability using:
- **SHAP (SHapley Additive exPlanations)**: Feature importance and interaction analysis
- **Permutation Importance**: Model-agnostic feature ranking
- **Rich Visualizations**: Summary plots, dependence plots, importance rankings

**Quick start:**
```bash
# Train model with explainability
python main.py --mode train --config config/default.yaml --save_model --explain

# Explain existing model
python main.py --mode explain \
  --model_path models/best_pipeline.joblib \
  --X_path data/X_test.csv \
  --y_path data/y_test.npy
```

📖 **Full documentation**: [`docs/XAI_MODULE.md`](docs/XAI_MODULE.md) | [`docs/XAI_QUICKSTART.md`](docs/XAI_QUICKSTART.md)

## 📁 Project Structure

```
house_prices_project/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml       # Pipeline configuration (includes XAI settings)
├── data/
│   └── raw/                         # Raw CSV data (auto-downloaded)
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data loading & EDA
│   ├── feature_engineering.py      # FE & outlier handling
│   ├── preprocessing.py            # Pipeline construction
│   ├── models.py                   # Model training & tuning
│   ├── trainer.py                  # Orchestration
│   ├── utils.py                    # Visualization & helpers
│   └── xai.py                      # 🆕 Explainability module
├── models/                          # Saved models
├── results/                         # Benchmark results & plots
├── notebooks/                       # Original notebook
├── tests/
│   ├── test_pipeline.py            # Unit tests
├── main.py                         # CLI entry point (updated)
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone repository
git clone <repository-url>
cd house_prices_project

# Install dependencies (includes SHAP)
pip install -r requirements.txt
```

## 📊 Usage

### 1. Exploratory Data Analysis (EDA)

```bash
python main.py --mode eda --data_path data/raw/train-house-prices-advanced-regression-techniques.csv
```

Features:
- Dataset shape and target statistics
- Distribution plots
- Missing value analysis (top 15)
- Skewness analysis (|skew| > 1)
- Correlation with target (top 8)

### 2. Train Single Pipeline

```bash
python main.py --mode train --config config/default.yaml --save_model
```

This will:
- Load data (auto-download if needed)
- Apply preprocessing based on YAML config
- Apply feature engineering
- Train model with hyperparameter tuning (Optuna, 20 trials)
- Save model to `models/best_pipeline.joblib`
- Save config to `models/best_config.yaml`

**Example Output:**
```
TRAINING RESULTS
================================================================================
Pipeline: SinglePipeline_ridge
Train_RMSE: 18234.56
Test_RMSE: 24567.89
Train_R2: 0.9234
Test_R2: 0.8912
```

### 3. Train with Explainability 🆕

```bash
python main.py --mode train \
  --config config/default.yaml \
  --save_model \
  --explain
```

This adds:
- SHAP value computation
- Permutation importance analysis
- Visualization generation
- Summary report

**Output:** `models/explain/` directory with plots and CSVs

### 4. Explain Existing Model 🆕

```bash
python main.py --mode explain \
  --model_path models/best_pipeline.joblib \
  --X_path data/X_test.csv \
  --y_path data/y_test.npy \
  --output results/explain
```

**Requirements:**
- Saved model (joblib format)
- Features CSV (original, unprocessed)
- Target CSV or NPY file

### 5. Full Benchmark

```bash
python main.py --mode benchmark --data_path data/raw/train-house-prices-advanced-regression-techniques.csv
```

This executes:
1. **Phase 1**: Evaluate all pipeline configs without FE (~260 runs)
2. **Phase 2**: Apply FE to top 10 configs and re-evaluate
3. **Phase 3**: Tune hyperparameters for top 10 pipelines (Optuna)
4. **Output**: 
   - `results/pipeline_benchmark_results.csv`
   - Visualization plots (top 10 metrics, test vs tuned, RMSE improvement)
   - Comprehensive summary report

## ⚙️ Configuration

Edit `config/default.yaml`:

```yaml
data:
  test_size: 0.25
  random_state: 42
  drop_columns: [Id, Alley, PoolQC, Fence, MiscFeature]

preprocessing:
  imputer: iterative     # simple, iterative, knn
  outlier: winsor        # none, winsor, iqr
  skew: log1p            # none, log1p, yeojohnson
  scaler: standard       # minmax, standard, robust
  use_ordinal: true      # Use ordinal encoding for quality features
  use_fe: true           # Apply feature engineering

model:
  name: ridge            # linearreg, ridge, lasso, elasticnet, huber, ransac, quantile
  params: {}

tuning:
  enabled: true
  n_trials: 20
  cv_folds: 3

# 🆕 Explainability settings
explainability:
  sample_size: 1000      # Number of samples for SHAP (↓ for speed)
  n_repeats: 30          # Permutation importance repeats
  background_size: 100   # KernelExplainer background samples
  top_features: 20       # Features to display in plots
```

## 🔬 Pipeline Components

### Data Preparation
- **Drop columns**: Id, high-missing columns (>50% null)
- **Train/test split**: 75/25
- **Target transform**: log1p if |skew| > 1

### Feature Engineering
- **Age**: YrSold - YearBuilt
- **TotalSF**: GrLivArea + TotalBsmtSF
- **TotalBath**: FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath

### Preprocessing Pipeline
1. **Imputation**: SimpleImputer (mean), IterativeImputer (10 iter), KNNImputer (k=5)
2. **Outlier Handling**: Winsorize (2%/5% tails), IQR clipping (1.5×IQR)
3. **Skew Transform**: log1p, Yeo-Johnson
4. **Scaling**: MinMax, Standard, Robust
5. **Encoding**: 
   - Ordinal: ExterQual, BsmtQual, KitchenQual, etc. (Po→Fa→TA→Gd→Ex)
   - OneHot: Remaining categorical features

### Models
- LinearRegression
- Ridge (alpha: 0.01-10.0)
- Lasso (alpha: 0.0001-0.1)
- ElasticNet (alpha: 0.0001-0.1, l1_ratio: 0.1-0.9)
- HuberRegressor (epsilon: 1.1-2.0)
- RANSACRegressor (min_samples: 0.5-0.9)
- QuantileRegressor (alpha: 0.0001-0.1)

### Evaluation Metrics
- **RMSE**: Root Mean Squared Error (lower is better)
- **RAE**: Relative Absolute Error (ratio-based metric)
- **R²**: Coefficient of determination (higher is better)

### 🆕 Explainability Metrics
- **SHAP Values**: Feature contribution to predictions
- **Permutation Importance**: Model reliance on features
- **Dependence Plots**: Feature interaction analysis

## 📈 Expected Performance

Based on Kaggle kernels and typical results:

| Metric | Without FE | With FE | After Tuning |
|--------|-----------|---------|--------------|
| Test R² | 0.86-0.88 | 0.88-0.90 | 0.89-0.91 |
| Test RMSE | $26k-$28k | $24k-$26k | $22k-$25k |

**Best Configuration** (typical):
- Imputer: Iterative
- Outlier: Winsor
- Skew: Log1p
- Scaler: Robust
- Model: Ridge (alpha ~1.0-3.0)
- With FE: Yes

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_pipeline.py -v
pytest tests/test_xai_smoke.py -v  # 🆕 XAI tests

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🆕 Explainability Output Examples

### SHAP Summary (Beeswarm)
Shows how each feature impacts predictions:
- Red dots: High feature values
- Blue dots: Low feature values
- X-axis: SHAP value (impact on prediction)

### Permutation Importance
Ranks features by model reliance:
```
TOP 10 FEATURES BY PERMUTATION IMPORTANCE
1. num_GrLivArea           0.054321 ± 0.002341
2. num_OverallQual         0.041234 ± 0.001890
3. num_TotalBsmtSF         0.032145 ± 0.001456
...
```

### Explainability Report
Text summary with:
- Model metadata
- Top 10 features (SHAP + Permutation)
- Files generated
- Interpretation guidance

## 📄 Extending the Project

### Add New Model
Edit `src/models.py`:

```python
def get_models_to_test() -> Dict[str, Any]:
    return {
        # ... existing models ...
        'xgboost': XGBRegressor(n_estimators=100, random_state=42)
    }
```

### Add New Preprocessing Step
Edit `src/preprocessing.py`:

```python
NEW_COMPONENT = {
    'method1': Transformer1(),
    'method2': Transformer2()
}
```

### Customize Feature Engineering
Edit `src/feature_engineering.py` → `add_features()` function

### 🆕 Add Custom Explainability
Edit `src/xai.py`:

```python
class ExplainabilityAnalyzer:
    def compute_custom_metric(self, X, y, output_dir):
        # Your custom explainability logic
        pass
```

## 📦 Data

**Source**: Kaggle House Prices - Advanced Regression Techniques

**Auto-download**: The project automatically downloads data using gdown if not present:
```python
# Google Drive ID: 1Dh_y7gFDUa2sD72_cKIa209dhbMVoGEd
```

**Manual download**:
```bash
gdown 1Dh_y7gFDUa2sD72_cKIa209dhbMVoGEd -O data/raw/train-house-prices-advanced-regression-techniques.csv
```

## 🛠 Troubleshooting

### XAI-Specific Issues

**Issue**: `ModuleNotFoundError: No module named 'shap'`
```bash
pip install shap
```

**Issue**: `SHAP computation is too slow`
```yaml
# Edit config/default.yaml
explainability:
  sample_size: 200  # Reduce from 1000
```

**Issue**: `Feature names don't match`
- Use **original** features (not preprocessed) for `X_path`
- Let `ExplainabilityAnalyzer` handle preprocessing

### General Issues

**Issue**: `Model file not found`
```bash
# Train a model first
python main.py --mode train --config config/default.yaml --save_model
```

**Issue**: `Data file not found`
```bash
# The project auto-downloads, but if it fails:
mkdir -p data/raw
gdown 1Dh_y7gFDUa2sD72_cKIa209dhbMVoGEd -O data/raw/train-house-prices-advanced-regression-techniques.csv
```

## 📁 Advanced Usage

### Batch Explainability
```python
from src.xai import explain_model_pipeline

models = ['ridge', 'lasso', 'elasticnet']
for model_name in models:
    explain_model_pipeline(
        model_path=f'models/{model_name}.joblib',
        X_path='data/X_test.csv',
        y_path='data/y_test.npy',
        output_dir=f'results/explain_{model_name}'
    )
```

### Custom SHAP Analysis
```python
from src.xai import ExplainabilityAnalyzer
import joblib

pipeline = joblib.load('models/best_pipeline.joblib')
analyzer = ExplainabilityAnalyzer(
    preprocessor=pipeline['preprocessor'],
    model=pipeline['model'],
    feature_names_in=X_test.columns.tolist(),
    target_log_transformed=True
)

# Explain specific samples
shap_values, explanation = analyzer.compute_shap(
    X_test.iloc[:100],  # First 100 samples
    output_dir='results/sample_explain',
    sample_size=100
)
```

## 🎯 Key Design Decisions

1. **Log Transform Target**: Applied when |skew| > 1 to normalize distribution
2. **Categorical Handling**: Separate ordinal (ordered quality features) from nominal
3. **Outlier Strategy**: Winsorize preferred over removal to preserve sample size
4. **CV Strategy**: 3-fold CV during tuning balances accuracy and speed
5. **Metric Focus**: RMSE for interpretability (same units as $), R² for variance explained
6. **🆕 XAI Integration**: Pipeline-aware explainability with automatic feature name extraction

## 📚 References

- Original Kaggle Competition: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- scikit-learn Documentation: [sklearn.pipeline](https://scikit-learn.org/stable/modules/compose.html)
- Optuna Documentation: [optuna.org](https://optuna.org/)
- **🆕 SHAP Documentation**: [shap.readthedocs.io](https://shap.readthedocs.io/)
- **🆕 Lundberg & Lee (2017)**: [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## ✨ Acknowledgments

- Original Jupyter Notebook: Module 5 House Prices prediction project
- Kaggle community for dataset and insights
- scikit-learn, Optuna, and **SHAP** teams for excellent tools
- Lundberg & Lee for SHAP methodology

---

## 🆕 What's New in v2.0

- ✅ **Explainability Module**: SHAP + Permutation Importance
- ✅ **New CLI Mode**: `python main.py --mode explain`
- ✅ **XAI Tests**: Comprehensive test coverage for explainability
- ✅ **Documentation**: Full XAI module documentation + quick start guide
- ✅ **Config Updates**: Explainability settings in YAML
- ✅ **Backward Compatible**: All existing features still work

**Upgrade from v1.x:**
```bash
pip install -r requirements.txt  # Installs SHAP
# No code changes needed - existing scripts work as before
```

---

**Contact**: For questions or issues, please open a GitHub issue or contact the maintainers.

📖 **Documentation**: [`docs/XAI_MODULE.md`](docs/XAI_MODULE.md) | [`docs/XAI_QUICKSTART.md`](docs/XAI_QUICKSTART.md)
