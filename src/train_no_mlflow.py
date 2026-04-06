from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


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


def main():
    print("Loading data...")
    train_path = Path("data/processed/train_features.csv")
    test_path = Path("data/processed/test_features.csv")

    X_train, X_test, y_train, y_test, le_target = load_and_preprocess(train_path, test_path)
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(le_target.classes_)}")

    results = {}

    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results['logistic_regression'] = {'model': lr_model, 'metrics': lr_metrics}

    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results['random_forest'] = {'model': rf_model, 'metrics': rf_metrics}

    print("\nTraining XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss', verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    results['xgboost'] = {'model': xgb_model, 'metrics': xgb_metrics}

    best_model_name = max(results, key=lambda k: results[k]['metrics']['accuracy'])
    print(f"\n{'='*50}")
    print(f"Best model: {best_model_name} with accuracy: {results[best_model_name]['metrics']['accuracy']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()