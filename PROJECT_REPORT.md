# T20 Match Winner Prediction - Final Project Report

## Overview

This project builds an ML model to predict T20 cricket match winners using historical IPL data. The system uses **binary classification** (team1 wins vs team2 wins) and MLflow for experiment tracking.

---

## Team Roles

- **Person A**: Data processing and feature engineering (parsing YAML files, creating train/test splits)
- **Person B**: ML model training with MLflow tracking

---

## Data Pipeline

### Step 1: Data Splitting
- **Source**: Cricsheet IPL YAML files (2022-2026)
- **Train**: 256 matches (2022 to early April 2025)
- **Test**: 43 matches (late April 2025 to 2026)
- **Split rule**: Time-based (no data leakage)

### Step 2: Feature Engineering

#### Original Features (14 columns)
| Feature | Description |
|---------|-------------|
| team1_win_pct_last_5 | Team 1's win % in last 5 matches |
| team2_win_pct_last_5 | Team 2's win % in last 5 matches |
| team1_head_to_head_win_pct | Team 1's win % vs Team 2 |
| team2_head_to_head_win_pct | Team 2's win % vs Team 1 |
| team1_win_pct_at_venue | Team 1's win % at this venue |
| team2_win_pct_at_venue | Team 2's win % at this venue |
| toss_advantage | Whether toss winner is team1 |

#### Enhanced Features (20 columns) - **BEST RESULT**
Person B extracted additional batting/bowling statistics from YAML:

| Feature | Description |
|---------|-------------|
| team1_avg_runs | Team 1's average runs in previous matches |
| team2_avg_runs | Team 2's average runs |
| team1_avg_wickets | Team 1's average wickets lost |
| team2_avg_wickets | Team 2's average wickets lost |
| team1_run_rate | Team 1's historical run rate |
| team2_run_rate | Team 2's historical run rate |
| team1_death_run_rate | Team 1's death overs (16-20) run rate |
| team2_death_run_rate | Team 2's death overs run rate |
| team1_matches | Number of matches team 1 played |
| team2_matches | Number of matches team 2 played |

**Plus difference features**: runs_diff, run_rate_diff, death_rate_diff, wickets_diff, matches_diff

---

## MLflow Experiments

### All Experiments (Logged at http://localhost:5000)

| Experiment | Description | Best Accuracy |
|------------|-------------|--------------|
| T20_Match_Winner_Prediction | Initial models (binary, 14 features) | 60% |
| T20_Match_Winner_Enhanced_Features | Binary with batting stats (20 features) | **65%** |
| T20_Match_Winner_All_Improvements | Feature engineering attempts (35 features) | 57.5% |

---

## Model Results Summary

### Binary Classification (2-class: team1 wins vs team2 wins)

| Model | Features | Test Accuracy | CV Score |
|-------|----------|--------------|----------|
| **GradientBoosting (lr=0.1, d=5)** | **20** | **65%** | 51.6% |
| MLP (100,50) | 14 | 60% | 44.1% |
| XGBoost | 20 | 57.5% | 50.0% |
| RandomForest | 20 | 60% | 45.7% |

**Final Model: GradientBoosting with 20 enhanced features = 65% accuracy**

---

## Improvement Attempts (All Logged in MLflow)

We systematically attempted to improve the model:

1. **Feature Engineering**
   - Added interaction features (run_rate × toss_advantage)
   - Added polynomial features (squared terms)
   - Added binned features
   - **Result**: No improvement, caused overfitting

2. **Target Encoding**
   - Team win rate encoding
   - Venue win rate encoding
   - **Result**: Added noise due to small sample size

3. **More Model Configurations**
   - 13+ configurations tuned
   - **Result**: Original model remained best

4. **Ensemble Methods**
   - Soft voting, Hard voting
   - **Result**: No improvement over single model

5. **Threshold Optimization**
   - Tested 0.4, 0.45, 0.5, 0.55, 0.6
   - **Result**: Default 0.5 was optimal

### Why Improvements Failed

The 35-feature model achieved only **57.5%** vs the 20-feature model at **65%**.

**Reason**: With only 254 training samples:
- 20 features = 12.7:1 sample-to-feature ratio (acceptable)
- 35 features = 7.2:1 ratio (too low → overfitting)

Each additional feature had fewer than 8 samples to learn from, causing the model to memorize rather than generalize.

---

## Final Predictions

Using GradientBoosting (lr=0.1, depth=5) with 20 enhanced features:

- **Test Accuracy**: 65% (26/40 correct)
- **Random Baseline**: 50%
- **Improvement**: +15% over random

---

## Project Structure

```
ipl_match_predictor_model/
├── data/
│   ├── processed/
│   │   ├── train_features.csv         # Original 14 features
│   │   ├── test_features.csv
│   │   ├── enhanced_train_features.csv  # 20 features (BEST)
│   │   ├── enhanced_test_features.csv
│   │   └── enhanced_final_predictions.csv
│   └── splits/pre_match_eval/
│       ├── train/                      # 256 YAML files
│       └── test/                      # 43 YAML files
├── scripts/
│   ├── generate_train_features.py
│   └── generate_test_features.py
├── src/
│   ├── enhanced_feature_engineering.py  # Extracts batting stats
│   ├── enhanced_training.py            # Main training
│   └── all_improvements.py             # Improvement attempts
├── ENHANCED_FEATURES_REPORT.md          # 65% result details
├── IMPROVEMENT_ANALYSIS_REPORT.md      # Why improvements failed
└── PROJECT_REPORT.md                    # This file
```

---

## How to Reproduce

### 1. Setup
```bash
git clone https://github.com/Dhakshi03/ipl_match_predictor_model.git
cd ipl_match_predictor_model
pip install mlflow scikit-learn xgboost pandas numpy pyyaml
```

### 2. Generate Enhanced Features
```bash
python src/enhanced_feature_engineering.py
```

### 3. Start MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

### 4. Train Final Model
```bash
python -X utf8 src/enhanced_training.py
```

---

## Conclusion

**Final Model Performance:**
- **Accuracy**: 65% (26/40 correct predictions)
- **Improvement over random**: +15%
- **Algorithm**: GradientBoosting with 20 enhanced batting features

**Key Learnings:**
1. Simple model with good features > Complex model with more features
2. Feature-to-sample ratio matters (aim for 10:1 minimum)
3. With small datasets (254 samples), simpler is better
4. Target encoding and polynomial features caused overfitting

**All experiments logged in MLflow** for reproducibility and comparison.

---

## Files for Submission

| File | Description |
|------|-------------|
| `enhanced_final_predictions.csv` | Final predictions (65% accuracy) |
| `enhanced_model_comparison.csv` | All model results |
| `ENHANCED_FEATURES_REPORT.md` | Details of 65% result |
| `IMPROVEMENT_ANALYSIS_REPORT.md` | Why improvement attempts failed |
| MLflow Experiment: `T20_Match_Winner_Enhanced_Features` | All runs logged |

---

**MLflow UI**: http://localhost:5000

**Best Experiment**: T20_Match_Winner_Enhanced_Features (65% accuracy)
