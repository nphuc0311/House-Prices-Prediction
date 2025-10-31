# House Prices Prediction - Production ML Pipeline

A modular, CLI-driven, and extensible Python project for House Prices prediction using scikit-learn pipelines. This project refactors a Jupyter Notebook into production-ready code with comprehensive benchmarking, hyperparameter tuning, and interactive demo capabilities.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns across data loading, preprocessing, modeling, and training
- **CLI Interface**: Command-line driven for easy automation and integration
- **YAML Configuration**: Flexible pipeline configuration without code changes
- **Comprehensive Benchmarking**: Tests 20 preprocessing configurations × 7 models
- **Feature Engineering**: Automated creation of Age, TotalSF, and TotalBath features
- **Hyperparameter Tuning**: Bayesian optimization with Optuna (20 trials, 3-fold CV)
- **Model Persistence**: Save/load trained pipelines with joblib
- **Visualization**: Automated plotting of benchmark results

## 📁 Project Structure

```
house_prices_project/
├── README.md
├── requirements.txt
├── config/
│   └── default_pipeline.yaml       # Pipeline configuration
├── data/
│   └── raw/                         # Raw CSV data (auto-downloaded)
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data loading & EDA
│   ├── feature_engineering.py      # FE & outlier handling
│   ├── preprocessing.py            # Pipeline construction
│   ├── models.py                   # Model training & tuning
│   ├── trainer.py                  # Orchestration
│   └── utils.py                    # Visualization & helpers
├── models/                          # Saved models
├── results/                         # Benchmark results & plots
├── notebooks/                       # Original notebook
├── tests/
│   └── test_pipeline.py            # Unit tests
├── main.py                         # CLI entry point
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

# Install dependencies
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

### 3. Full Benchmark

```bash
python main.py --mode benchmark --data_path data/raw/train-house-prices-advanced-regression-techniques.csv
```

This executes:
1. **Phase 1**: Evaluate 20 pipeline configs × 7 models without FE (~140 runs)
2. **Phase 2**: Apply FE to top 10 configs and re-evaluate
3. **Phase 3**: Tune hyperparameters for top 10 pipelines (Optuna)
4. **Output**: 
   - `results/pipeline_benchmark_results.csv`
   - Visualization plots (top 10 metrics, test vs tuned, RMSE improvement)
   - Comprehensive summary report

## ⚙️ Configuration

Edit `config/default_pipeline.yaml`:

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
# Run unit tests
pytest tests/test_pipeline.py -v

# Test specific function
pytest tests/test_pipeline.py::test_evaluate_pipeline -v
```

## 🔄 Extending the Project

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

## 🐛 Troubleshooting

**Issue**: `Model file not found`
```bash
# Solution: Train a model first
python main.py --mode train --config config/default_pipeline.yaml --save_model
```

**Issue**: `Data file not found`
```bash
# Solution: The project auto-downloads, but if it fails:
mkdir -p data/raw
gdown 1Dh_y7gFDUa2sD72_cKIa209dhbMVoGEd -O data/raw/train-house-prices-advanced-regression-techniques.csv
```

**Issue**: Gradio demo errors
```bash
# Solution: Ensure model exists and all dependencies are installed
pip install --upgrade gradio
```

## 📝 Advanced Usage

### Save specific benchmark config as model
1. Run benchmark to find best config
2. Update `config/default_pipeline.yaml` with best parameters
3. Run `python main.py --mode train --save_model`

### Batch predictions
```python
from src.utils import load_model
import pandas as pd

preprocessor, model = load_model('models/best_pipeline.joblib')
X_new = pd.read_csv('new_houses.csv')
X_processed = preprocessor.transform(X_new)
predictions = model.predict(X_processed)
predictions = np.expm1(predictions)  # Inverse log transform
```

### Custom benchmarking
Edit `src/preprocessing.py` → `get_benchmark_configs()` to add/remove configurations

## 🎯 Key Design Decisions

1. **Log Transform Target**: Applied when |skew| > 1 to normalize distribution
2. **Categorical Handling**: Separate ordinal (ordered quality features) from nominal
3. **Outlier Strategy**: Winsorize preferred over removal to preserve sample size
4. **CV Strategy**: 3-fold CV during tuning balances accuracy and speed
5. **Metric Focus**: RMSE for interpretability (same units as $), R² for variance explained

## 📚 References

- Original Kaggle Competition: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- scikit-learn Documentation: [sklearn.pipeline](https://scikit-learn.org/stable/modules/compose.html)
- Optuna Documentation: [optuna.org](https://optuna.org/)

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
- scikit-learn, Optuna, and Gradio teams

---

**Contact**: For questions or issues, please open a GitHub issue or contact the maintainers.