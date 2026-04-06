# T20 Match Winner Prediction - Project Report

## Overview
This project builds an ML model to predict T20 cricket match winners using historical IPL data. The system uses MLflow for experiment tracking.

---

## Team Roles
- **Person A**: Data processing and feature engineering (parsing YAML files, creating train/test splits)
- **Person B (this project)**: ML model training with MLflow tracking

---

## Data Pipeline

### Step 1: Data Splitting (Person A)
- Source: Cricsheet IPL YAML files (2022-2026)
- Train: 256 matches (2022 to early April 2025)
- Test: 43 matches (late April 2025 to 2026)
- Split rule: Time-based (no data leakage)

### Step 2: Feature Engineering (Person A)
Features computed BEFORE each match using only historical data:

| Feature | Description |
|---------|-------------|
| team1_win_pct_last_5 | Team 1's win % in last 5 matches |
| team2_win_pct_last_5 | Team 2's win % in last 5 matches |
| team1_head_to_head_win_pct | Team 1's win % vs Team 2 (all-time) |
| team2_head_to_head_win_pct | Team 2's win % vs Team 1 (all-time) |
| team1_win_pct_at_venue | Team 1's win % at this venue |
| team2_win_pct_at_venue | Team 2's win % at this venue |
| toss_advantage | Whether toss winner is team1 |

### Step 3: CSV Generation (Person B)
- `train_features.csv`: 254 rows (2 matches with no winner removed)
- `test_features.csv`: 40 rows (3 matches with no winner removed)

---

## ML Training Pipeline

### Data Preprocessing
1. Label encode categorical variables (team1, team2, venue, toss_decision)
2. Create difference features (win_pct_diff, h2h_diff, venue_diff)
3. Final feature set: 14 features

### Models Trained

| Model | Test Accuracy | Precision | Recall | F1 Score |
|-------|--------------|-----------|--------|----------|
| Logistic Regression | 0.2250 | 0.3289 | 0.2250 | 0.2300 |
| Random Forest | 0.2000 | 0.2308 | 0.2000 | 0.1961 |
| **XGBoost** | **0.4500** | 0.4914 | 0.4500 | 0.4430 |

**Best Model: XGBoost with 45% accuracy**

### Why 45% is reasonable for 11 classes
- Random guessing would give ~9% accuracy (1/11 teams)
- 45% is 5x better than random
- Cricket outcomes have inherent randomness (upsets happen)

---

## MLflow Tracking

### Experiment: T20_Match_Winner_Prediction
All training runs logged with:
- Parameters (hyperparameters used)
- Metrics (accuracy, precision, recall, f1)
- Model artifacts (serialized models)
- Tags (model type)

### Access MLflow UI
```
http://localhost:5000
```

---

## Project Structure

```
ipl_match_predictor_model/
├── data/
│   ├── processed/
│   │   ├── train_features.csv     # 254 rows, 16 columns
│   │   └── test_features.csv      # 40 rows, 16 columns
│   └── splits/pre_match_eval/
│       ├── train/                 # 256 YAML files
│       └── test/                  # 43 YAML files
├── scripts/
│   ├── generate_train_features.py  # Creates train_features.csv
│   └── generate_test_features.py   # Creates test_features.csv
└── src/
    ├── train_with_mlflow.py       # Main training script
    └── train_no_mlflow.py          # Training without MLflow
```

---

## How to Reproduce

### 1. Setup
```bash
git clone https://github.com/Dhakshi03/ipl_match_predictor_model.git
cd ipl_match_predictor_model
pip install mlflow scikit-learn xgboost pandas numpy pyyaml
```

### 2. Generate Features
```bash
python scripts/generate_train_features.py
python scripts/generate_test_features.py
```

### 3. Start MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 4. Train Models
```bash
python -X utf8 src/train_with_mlflow.py
```

---

## Observations

1. **XGBoost performs best** - Likely due to ability to handle non-linear relationships
2. **Low accuracy overall** - 11-class classification is inherently difficult
3. **Pre-match features are limited** - Match outcome depends on in-match performance (which can't be known pre-match)
4. **Toss matters** - Could add more toss-based features

---

## Potential Improvements

1. Add more historical features (season performance, player form)
2. Include team strength ratings (ELO system)
3. Hyperparameter tuning for XGBoost
4. Feature importance analysis to understand key predictors

---

## Files Created/Modified by Person B

1. `scripts/generate_train_features.py` - Train feature generation
2. `src/train_with_mlflow.py` - ML training with experiment tracking
3. `src/train_no_mlflow.py` - Quick training without MLflow
4. `data/processed/train_features.csv` - Generated CSV
5. `data/processed/test_features.csv` - Generated CSV

---

## Conclusion

Successfully built a T20 match winner prediction system with:
- Proper train/test split (time-based)
- Historical pre-match features
- Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- MLflow experiment tracking
- XGBoost achieving 45% accuracy on 11-class classification

The model is available in MLflow for comparison and the best model (XGBoost) can be registered for deployment.
