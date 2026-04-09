from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://localhost:5000")
EXPERIMENT_NAME = "T20_Match_Winner_Ensemble"
mlflow.set_experiment(EXPERIMENT_NAME)

LEAKAGE_COLS = [
    'team1_inning_runs', 'team1_inning_wickets', 'team1_inning_run_rate', 'team1_death_runs',
    'team2_inning_runs', 'team2_inning_wickets', 'team2_inning_run_rate', 'team2_death_runs'
]


def load_enhanced_clean_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    for col in LEAKAGE_COLS:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])
    
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
        
        df['runs_diff'] = df['team1_avg_runs'] - df['team2_avg_runs']
        df['run_rate_diff'] = df['team1_run_rate'] - df['team2_run_rate']
        df['death_rate_diff'] = df['team1_death_run_rate'] - df['team2_death_run_rate']
        df['wickets_diff'] = df['team1_avg_wickets'] - df['team2_avg_wickets']
        df['matches_diff'] = df['team1_matches'] - df['team2_matches']
    
    feature_cols = [
        'team1_encoded', 'team2_encoded', 'venue_encoded',
        'toss_decision_encoded', 'toss_advantage',
        'team1_avg_runs', 'team2_avg_runs',
        'team1_avg_wickets', 'team2_avg_wickets',
        'team1_run_rate', 'team2_run_rate',
        'team1_death_run_rate', 'team2_death_run_rate',
        'team1_matches', 'team2_matches',
        'runs_diff', 'run_rate_diff', 'death_rate_diff', 'wickets_diff', 'matches_diff'
    ]
    
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['team1_win'].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['team1_win'].values
    
    return X_train, X_test, y_train, y_test, feature_cols, test_df


def train_and_evaluate(model, X_train, y_train, X_test, y_test, run_name, model_name=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        y_proba = None
        roc_auc = 0.0
    
    test_acc = accuracy_score(y_test, y_pred)
    
    return {
        'test_accuracy': test_acc,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'roc_auc': roc_auc,
        'y_pred': y_pred
    }


def main():
    print("="*80)
    print("T20 MATCH WINNER - ENSEMBLE METHODS")
    print("="*80)
    print("\n1. Voting Ensemble (Hard + Soft)")
    print("2. Stacking Ensemble")
    print("3. Compare Results")
    
    print("\n[Step 1] Loading data...")
    X_train, X_test, y_train, y_test, feature_cols, test_df = load_enhanced_clean_data(
        Path("data/processed/enhanced_train_features.csv"),
        Path("data/processed/enhanced_test_features.csv")
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {len(feature_cols)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    print("\n[Step 2] Voting Ensemble (Hard Voting - Majority Vote)...")
    voting_hard = VotingClassifier(
        estimators=[
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss', verbosity=0)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42, early_stopping=True)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=42))
        ],
        voting='hard'
    )
    
    metrics = train_and_evaluate(voting_hard, X_train_scaled, y_train, X_test_scaled, y_test, "Voting_Hard")
    results['Voting_Hard'] = metrics
    
    with mlflow.start_run(run_name="Voting_Hard"):
        mlflow.log_params({'voting': 'hard', 'n_estimators': 5, 'base_models': 'GB,RF,XGB,MLP,Ada'})
        mlflow.log_metrics({
            'test_accuracy': metrics['test_accuracy'],
            'cv_accuracy_mean': metrics['cv_mean'],
            'cv_accuracy_std': metrics['cv_std'],
            'roc_auc': metrics['roc_auc']
        })
        mlflow.set_tag("ensemble_type", "Voting")
        mlflow.set_tag("voting_type", "hard")
    
    above = "+" if metrics['test_accuracy'] > 0.5 else ""
    print(f"    Test Accuracy: {above}{metrics['test_accuracy']:.4f} | CV: {metrics['cv_mean']:.4f}")
    
    print("\n[Step 3] Voting Ensemble (Soft Voting - Average Probabilities)...")
    voting_soft = VotingClassifier(
        estimators=[
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss', verbosity=0)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42, early_stopping=True)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=42))
        ],
        voting='soft'
    )
    
    metrics = train_and_evaluate(voting_soft, X_train_scaled, y_train, X_test_scaled, y_test, "Voting_Soft")
    results['Voting_Soft'] = metrics
    
    with mlflow.start_run(run_name="Voting_Soft"):
        mlflow.log_params({'voting': 'soft', 'n_estimators': 5, 'base_models': 'GB,RF,XGB,MLP,Ada'})
        mlflow.log_metrics({
            'test_accuracy': metrics['test_accuracy'],
            'cv_accuracy_mean': metrics['cv_mean'],
            'cv_accuracy_std': metrics['cv_std'],
            'roc_auc': metrics['roc_auc']
        })
        mlflow.set_tag("ensemble_type", "Voting")
        mlflow.set_tag("voting_type", "soft")
    
    above = "+" if metrics['test_accuracy'] > 0.5 else ""
    print(f"    Test Accuracy: {above}{metrics['test_accuracy']:.4f} | CV: {metrics['cv_mean']:.4f}")
    
    print("\n[Step 4] Stacking Ensemble...")
    stacking = StackingClassifier(
        estimators=[
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss', verbosity=0))
        ],
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5
    )
    
    metrics = train_and_evaluate(stacking, X_train_scaled, y_train, X_test_scaled, y_test, "Stacking")
    results['Stacking'] = metrics
    
    with mlflow.start_run(run_name="Stacking"):
        mlflow.log_params({
            'base_estimators': 3,
            'base_models': 'GB,RF,XGB',
            'meta_learner': 'LogisticRegression'
        })
        mlflow.log_metrics({
            'test_accuracy': metrics['test_accuracy'],
            'cv_accuracy_mean': metrics['cv_mean'],
            'cv_accuracy_std': metrics['cv_std'],
            'roc_auc': metrics['roc_auc']
        })
        mlflow.set_tag("ensemble_type", "Stacking")
    
    above = "+" if metrics['test_accuracy'] > 0.5 else ""
    print(f"    Test Accuracy: {above}{metrics['test_accuracy']:.4f} | CV: {metrics['cv_mean']:.4f}")
    
    print("\n" + "="*80)
    print("ENSEMBLE RESULTS COMPARISON")
    print("="*80)
    
    baseline_acc = 0.65
    print(f"\n{'Method':<20} {'Accuracy':<12} {'CV Mean':<12} {'ROC-AUC':<10} {'vs Baseline':<12}")
    print("-"*80)
    
    for method, metrics in results.items():
        diff = metrics['test_accuracy'] - baseline_acc
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
        above = "+" if metrics['test_accuracy'] > 0.5 else ""
        print(f"{method:<20} {above}{metrics['test_accuracy']:.4f}     {metrics['cv_mean']:.4f}     {metrics['roc_auc']:.4f}     {diff_str}")
    
    best_method = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    
    print("\n" + "="*80)
    print("BEST ENSEMBLE METHOD")
    print("="*80)
    print(f"\nBest: {best_method[0]} with {best_method[1]['test_accuracy']:.4f} accuracy")
    print(f"Previous Best Single Model (GB): {baseline_acc:.4f}")
    print(f"Improvement: +{(best_method[1]['test_accuracy'] - baseline_acc)*100:.1f}%")
    
    print("\n[Step 5] Generating predictions with best method...")
    best_model_name = best_method[0]
    
    if best_model_name == 'Voting_Hard':
        best_model = voting_hard
    elif best_model_name == 'Voting_Soft':
        best_model = voting_soft
    else:
        best_model = stacking
    
    y_pred = best_method[1]['y_pred']
    
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
    
    results_df.to_csv('data/processed/ensemble_predictions.csv', index=False)
    print(f"\nSaved to: data/processed/ensemble_predictions.csv")
    
    print("\n" + "="*80)
    print("MLflow logged to experiment: T20_Match_Winner_Ensemble")
    print("="*80)


if __name__ == "__main__":
    main()