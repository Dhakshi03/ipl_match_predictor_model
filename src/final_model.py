from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://localhost:5000")
EXPERIMENT_NAME = "T20_Match_Winner_Final"
mlflow.set_experiment(EXPERIMENT_NAME)


def load_basic_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df = train_df.dropna(subset=['winner'])
    test_df = test_df.dropna(subset=['winner'])
    
    le_team = LabelEncoder()
    le_venue = LabelEncoder()
    le_toss = LabelEncoder()
    
    all_teams = pd.concat([train_df['team1'], train_df['team2'], train_df['toss_winner']]).unique()
    all_venues = pd.concat([train_df['venue'], test_df['venue']]).unique()
    all_toss = train_df['toss_decision'].dropna().unique()
    
    le_team.fit(all_teams)
    le_venue.fit(all_venues)
    le_toss.fit(list(all_toss) + ['unknown'])
    
    for df in [train_df, test_df]:
        df['team1_encoded'] = le_team.transform(df['team1'])
        df['team2_encoded'] = le_team.transform(df['team2'])
        df['venue_encoded'] = le_venue.transform(df['venue'])
        df['toss_decision_encoded'] = df['toss_decision'].apply(
            lambda x: le_toss.transform([x])[0] if x in le_toss.classes_ else le_toss.transform(['unknown'])[0]
        )
        df['toss_advantage'] = (df['toss_winner'] == df['team1']).astype(int)
        df['team1_win'] = (df['winner'] == df['team1']).astype(int)
        
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
    
    X_train = train_df[feature_cols].fillna(0.5).values
    y_train = train_df['team1_win'].values
    X_test = test_df[feature_cols].fillna(0.5).values
    y_test = test_df['team1_win'].values
    
    return X_train, X_test, y_train, y_test, feature_cols, test_df


def main():
    print("="*80)
    print("T20 MATCH WINNER - FINAL MODEL (14 Basic Features, No Leakage)")
    print("="*80)
    print("\nConfiguration: MLP (100, 50) - Same as original 60% model")
    
    print("\n[Step 1] Loading basic data...")
    train_path = Path("data/processed/train_features.csv")
    test_path = Path("data/processed/test_features.csv")
    
    X_train, X_test, y_train, y_test, feature_cols, test_df = load_basic_data(train_path, test_path)
    
    print(f"    Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"    Features: {len(feature_cols)}")
    print(f"    Feature columns: {feature_cols}")
    print(f"    Class distribution - Train: {np.sum(y_train==1)} team1 wins, {np.sum(y_train==0)} team2 wins")
    print(f"                    Test: {np.sum(y_test==1)} team1 wins, {np.sum(y_test==0)} team2 wins")
    
    print("\n[Step 2] Training MLP (100, 50) with 5-Fold CV...")
    print("="*80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )
    
    print(f"\nModel: MLPClassifier")
    print(f"hidden_layer_sizes: (100, 50)")
    print(f"max_iter: 1000")
    print(f"early_stopping: True")
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"\n5-Fold CV Scores: {cv_scores}")
    print(f"CV Mean: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("\n" + "="*80)
    print("FINAL MODEL RESULTS")
    print("="*80)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"\nCV Accuracy Mean: {np.mean(cv_scores):.4f}")
    print(f"Random Baseline: 50%")
    print(f"Improvement: +{(test_acc-0.5)*100:.1f}%")
    
    with mlflow.start_run(run_name="MLP_100_50_Final"):
        mlflow.log_params({
            'hidden_layer_sizes': '(100, 50)',
            'max_iter': 1000,
            'random_state': 42,
            'early_stopping': True,
            'num_features': len(feature_cols)
        })
        mlflow.log_metrics({
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1': test_f1,
            'test_roc_auc': roc_auc,
            'cv_accuracy_mean': np.mean(cv_scores),
            'cv_accuracy_std': np.std(cv_scores),
        })
        signature = infer_signature(X_test_scaled, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        mlflow.set_tag("model_type", "MLPClassifier")
        mlflow.set_tag("config", "(100, 50)")
        mlflow.set_tag("data_leakage", "none")
        mlflow.set_tag("features", "14_basic")
    
    print("\n[Step 3] Generating predictions...")
    test_df = test_df.reset_index(drop=True)
    predicted_winners = []
    for i in range(len(test_df)):
        if y_pred[i] == 1:
            predicted_winners.append(test_df.iloc[i]['team1'])
        else:
            predicted_winners.append(test_df.iloc[i]['team2'])
    
    results_df = test_df[['match_id', 'match_date', 'team1', 'team2', 'venue', 'winner']].copy()
    results_df.columns = ['match_id', 'match_date', 'team1', 'team2', 'venue', 'actual_winner']
    results_df['predicted_winner'] = predicted_winners
    results_df['correct'] = results_df['actual_winner'] == results_df['predicted_winner']
    
    print("\nPredictions:")
    correct_count = 0
    for i, row in results_df.iterrows():
        status = "[OK]" if row['correct'] else "[WRONG]"
        if row['correct']:
            correct_count += 1
        print(f"  {status} {row['match_date'][:10]} | {row['team1']} vs {row['team2']} | Pred: {row['predicted_winner']} | Actual: {row['actual_winner']}")
    
    total = len(results_df)
    print(f"\nTotal: {correct_count}/{total} = {correct_count/total*100:.1f}%")
    
    results_df.to_csv('data/processed/final_predictions.csv', index=False)
    print(f"\nSaved to: data/processed/final_predictions.csv")
    
    print("\n" + "="*80)
    print("MLflow logged to experiment: T20_Match_Winner_Final")
    print("="*80)


if __name__ == "__main__":
    main()
