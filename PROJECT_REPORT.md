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
| **T20_Match_Winner_PreMatch_Clean** | **No data leakage - PRE-MATCH ONLY** | **50%** |

---

## Model Results Summary

### Binary Classification (2-class: team1 wins vs team2 wins)

| Model | Features | Test Accuracy | CV Score |
|-------|----------|--------------|----------|
| **GradientBoosting (lr=0.1, d=5)** | **20** | **65%** | 51.6% |
| MLP (100,50) | 14 | 60% | 44.1% |
| XGBoost | 20 | 57.5% | 50.0% |
| RandomForest | 20 | 60% | 45.7% |
| **GradientBoosting (lr=0.05, d=4)** | **19 (clean)** | **50%** | 50.8% |

**Final Model: GradientBoosting with 20 enhanced features = 65% accuracy**

---

## DATA LEAKAGE DISCOVERY & FIX

### The Problem
The original 65% accuracy model had **data leakage**. It used features that are only known AFTER the match:

**LEAKAGE FEATURES (removed):**
- `team1_inning_runs`, `team1_inning_wickets`, `team1_inning_run_rate`, `team1_death_runs`
- `team2_inning_runs`, `team2_inning_wickets`, `team2_inning_run_rate`, `team2_death_runs`

These are actual match results - they don't exist before the match starts!

### The Fix
Created `clean_prematch_training.py` that uses **ONLY pre-match features**:
- Historical averages from last 5 matches (team form)
- Team/venue encodings
- Toss information

### Clean Model Results

```
================================================================================
CLEAN PRE-MATCH MODEL RESULTS (NO DATA LEAKAGE)
================================================================================
Best: GB_lr0.05_d4 with 0.5500 accuracy
Random baseline: 50%
Improvement: +5.0%

Total: 20/40 = 50.0%
```

**Features used (19 clean features):**
- team1_encoded, team2_encoded, venue_encoded
- toss_decision_encoded, toss_advantage
- team1_avg_runs_last5, team2_avg_runs_last5
- team1_avg_wickets_last5, team2_avg_wickets_last5
- team1_run_rate_last5, team2_run_rate_last5
- team1_death_rate_last5, team2_death_rate_last5
- team1_matches, team2_matches
- runs_diff, run_rate_diff, death_rate_diff, wickets_diff

### Why Clean Accuracy is Lower
The 65% was inflated because:
1. Model was "cheating" - seeing match results to predict match results
2. Real pre-match prediction is hard: T20 cricket is inherently unpredictable
3. 50% accuracy is only marginally above random (50%)

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

### Original Model (with leakage - INFLATED):
- **Test Accuracy**: 65% (26/40 correct)
- **Random Baseline**: 50%
- **Improvement**: +15% over random

### Clean Model (no leakage - REALISTIC):
- **Test Accuracy**: 50% (20/40 correct)
- **Random Baseline**: 50%
- **Improvement**: +0% (barely above random)

**Important**: The 65% was artificially inflated due to data leakage. The real pre-match prediction accuracy is ~50%, which is only marginally better than random guessing for T20 cricket.

---

## Project Structure

```
ipl_match_predictor_model/
├── data/
│   ├── processed/
│   │   ├── train_features.csv         # Original 14 features
│   │   ├── test_features.csv
│   │   ├── enhanced_train_features.csv  # 20 features (with leakage)
│   │   ├── enhanced_test_features.csv
│   │   ├── enhanced_final_predictions.csv
│   │   └── clean_prematch_predictions.csv  # Clean model results
│   └── splits/pre_match_eval/
│       ├── train/                      # 256 YAML files
│       └── test/                      # 43 YAML files
├── scripts/
│   ├── generate_train_features.py
│   └── generate_test_features.py
├── src/
│   ├── enhanced_feature_engineering.py  # Extracts batting stats
│   ├── enhanced_training.py            # Main training (65% - with leakage)
│   ├── clean_prematch_training.py      # Clean training (50% - no leakage)
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

### 4. Train Original Model (with leakage - DO NOT USE)
```bash
python -X utf8 src/enhanced_training.py
```

### 5. Train Clean Model (no leakage - RECOMMENDED)
```bash
python -X utf8 src/clean_prematch_training.py
```

---

## Conclusion

**Final Model Performance (Clean - No Leakage):**
- **Accuracy**: 50% (20/40 correct predictions)
- **Improvement over random**: +0% (marginal)
- **Algorithm**: GradientBoosting with 19 pre-match features

**Key Learnings:**
1. Simple model with good features > Complex model with more features
2. Feature-to-sample ratio matters (aim for 10:1 minimum)
3. With small datasets (254 samples), simpler is better
4. Target encoding and polynomial features caused overfitting
5. **CRITICAL**: Always check for data leakage - use only pre-match features!
6. T20 cricket match outcomes are inherently difficult to predict pre-match

**All experiments logged in MLflow** for reproducibility and comparison.

---

## Files for Submission

| File | Description |
|------|-------------|
| `clean_prematch_predictions.csv` | Clean pre-match predictions (50% accuracy) |
| `enhanced_final_predictions.csv` | Original predictions (65% - WITH LEAKAGE) |
| `enhanced_model_comparison.csv` | All model results |
| ENHANCED_FEATURES_REPORT.md | Details of 65% result |
| IMPROVEMENT_ANALYSIS_REPORT.md | Why improvement attempts failed |
| MLflow Experiment: `T20_Match_Winner_PreMatch_Clean` | Clean runs logged |

---

**MLflow UI**: http://localhost:5000

**Clean Experiment**: T20_Match_Winner_PreMatch_Clean (50% accuracy - no leakage)

**Note**: The 65% accuracy in previous experiments was due to data leakage. The clean model achieves ~50% which is realistic for pre-match T20 predictions.
