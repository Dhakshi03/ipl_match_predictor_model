from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://localhost:5000")
EXPERIMENT_NAME = "T20_Match_Winner_Prediction"
mlflow.set_experiment(EXPERIMENT_NAME)


def load_and_preprocess(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = train_df.dropna(subset=['winner'])
    test_df = test_df.dropna(subset=['winner'])

    le_team = LabelEncoder()
    le_venue = LabelEncoder()
    le_toss = LabelEncoder()
    le_target = LabelEncoder()

    all_teams = pd.concat([train_df['team1'], train_df['team2'], train_df['toss_winner']]).unique()
    all_venues = pd.concat([train_df['venue'], test_df['venue']]).unique()
    all_toss = train_df['toss_decision'].dropna().unique()

    le_team.fit(all_teams)
    le_venue.fit(all_venues)
    le_toss.fit(list(all_toss) + ['unknown'])
    le_target.fit(pd.concat([train_df['winner'], test_df['winner']]).unique())

    for df in [train_df, test_df]:
        df['team1_encoded'] = le_team.transform(df['team1'])
        df['team2_encoded'] = le_team.transform(df['team2'])
        df['venue_encoded'] = le_venue.transform(df['venue'])
        df['toss_decision_encoded'] = df['toss_decision'].apply(
            lambda x: le_toss.transform([x])[0] if x in le_toss.classes_ else le_toss.transform(['unknown'])[0]
        )
        df['toss_advantage'] = (df['toss_winner'] == df['team1']).astype(int)
        df['win_pct_diff'] = df['team1_win_pct_last_5'] - df['team2_win_pct_last_5']
        df['h2h_diff'] = df['team1_head_to_head_win_pct'] - df['team2_head_to_head_win_pct']
        df['venue_diff'] = df['team1_win_pct_at_venue'] - df['team2_win_pct_at_venue']

    feature_cols = [
        'team1_encoded', 'team2_encoded', 'venue_encoded',
        'toss_decision_encoded', 'toss_advantage',
        'team1_win_pct_last_5', 'team2_win_pct_last_5',
        'team1_head_to_head_win_pct', 'team2_head_to_head_win_pct',
        'team1_win_pct_at_venue', 'team2_win_pct_at_venue',
        'win_pct_diff', 'h2h_diff', 'venue_diff'
    ]

    X_train = train_df[feature_cols]
    y_train = le_target.transform(train_df['winner'])
    X_test = test_df[feature_cols]
    y_test = le_target.transform(test_df['winner'])

    return X_train, X_test, y_train, y_test, le_target


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }

    print(f"\n{model_name} Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return metrics


def log_run(run_name, model, X_train, X_test, y_train, y_test, params, model_type):
    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, run_name)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        mlflow.set_tag("model_type", model_type)

        print(f"  Run ID: {run.info.run_id}")

    return model, metrics


def main():
    print("="*60)
    print("T20 Match Winner Prediction - ML Training with MLflow")
    print("="*60)

    print("\n[Step 1] Loading pre-processed data...")
    train_path = Path("data/processed/train_features.csv")
    test_path = Path("data/processed/test_features.csv")

    X_train, X_test, y_train, y_test, le_target = load_and_preprocess(train_path, test_path)
    print(f"  Train samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Number of features: {X_train.shape[1]}")
    print(f"  Number of classes (teams): {len(le_target.classes_)}")

    print("\n[Step 2] Starting MLflow experiment logging...")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  Tracking URI: http://localhost:5000")

    results = {}

    print("\n[Step 3] Training Model 1: Logistic Regression...")
    lr_params = {'max_iter': 1000, 'random_state': 42}
    lr_model = LogisticRegression(**lr_params)
    model, metrics = log_run(
        "logistic_regression_v1",
        lr_model, X_train, X_test, y_train, y_test,
        lr_params, "logistic_regression"
    )
    results['logistic_regression'] = {'model': model, 'metrics': metrics}

    print("\n[Step 4] Training Model 2: Random Forest...")
    rf_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
    rf_model = RandomForestClassifier(**rf_params)
    model, metrics = log_run(
        "random_forest_v1",
        rf_model, X_train, X_test, y_train, y_test,
        rf_params, "random_forest"
    )
    results['random_forest'] = {'model': model, 'metrics': metrics}

    print("\n[Step 5] Training Model 3: XGBoost...")
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    xgb_model = XGBClassifier(**xgb_params)
    model, metrics = log_run(
        "xgboost_v1",
        xgb_model, X_train, X_test, y_train, y_test,
        xgb_params, "xgboost"
    )
    results['xgboost'] = {'model': xgb_model, 'metrics': metrics}

    best_model_name = max(results, key=lambda k: results[k]['metrics']['accuracy'])
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['metrics']['accuracy']:.4f}")
    print(f"{'='*60}")

    print("\n[Step 6] Summary of all models:")
    for name, data in results.items():
        print(f"  {name}: accuracy={data['metrics']['accuracy']:.4f}")

    print(f"\nView full experiment details at: http://localhost:5000")


if __name__ == "__main__":
    main()