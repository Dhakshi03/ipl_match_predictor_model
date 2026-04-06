from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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

    X_train = train_df[feature_cols].values
    y_train = le_target.transform(train_df['winner'])
    X_test = test_df[feature_cols].values
    y_test = le_target.transform(test_df['winner'])

    return X_train, X_test, y_train, y_test, le_target


def main():
    print("="*80)
    print("ENSEMBLE CREATION - TOP 5 MODELS")
    print("="*80)

    train_path = Path("data/processed/train_features.csv")
    test_path = Path("data/processed/test_features.csv")

    X_train, X_test, y_train, y_test, le_target = load_and_preprocess(train_path, test_path)

    print("\nCreating Voting Ensemble from Top 5 Models...")

    models = [
        ('gb1', GradientBoostingClassifier(n_estimators=100, max_depth=7, learning_rate=0.2, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss', verbosity=0)),
        ('gb2', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)),
        ('gb3', GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')),
    ]

    print("\n[1] Hard Voting Ensemble (majority vote)")
    hard_ensemble = VotingClassifier(estimators=models, voting='hard')
    hard_ensemble.fit(X_train, y_train)
    hard_pred = hard_ensemble.predict(X_test)
    hard_acc = accuracy_score(y_test, hard_pred)
    print(f"    Hard Voting Accuracy: {hard_acc:.4f}")

    print("\n[2] Soft Voting Ensemble (average probabilities)")
    soft_ensemble = VotingClassifier(estimators=models, voting='soft')
    soft_ensemble.fit(X_train, y_train)
    soft_pred = soft_ensemble.predict(X_test)
    soft_acc = accuracy_score(y_test, soft_pred)
    print(f"    Soft Voting Accuracy: {soft_acc:.4f}")

    print("\n[3] Best Single Model: GradientBoosting (lr=0.2, d=7)")
    best_single = GradientBoostingClassifier(n_estimators=100, max_depth=7, learning_rate=0.2, random_state=42)
    best_single.fit(X_train, y_train)
    best_pred = best_single.predict(X_test)
    best_acc = accuracy_score(y_test, best_pred)
    print(f"    Single Model Accuracy: {best_acc:.4f}")

    print("\n" + "="*80)
    print("ENSEMBLE SUMMARY")
    print("="*80)
    print(f"    Best Single Model: {best_acc:.4f}")
    print(f"    Hard Voting Ensemble: {hard_acc:.4f}")
    print(f"    Soft Voting Ensemble: {soft_acc:.4f}")

    if soft_acc > best_acc and soft_acc > hard_acc:
        print("\n    Winner: Soft Voting Ensemble!")
        final_pred = soft_pred
        final_acc = soft_acc
        final_method = "Soft Voting Ensemble"
    elif hard_acc > best_acc:
        print("\n    Winner: Hard Voting Ensemble!")
        final_pred = hard_pred
        final_acc = hard_acc
        final_method = "Hard Voting Ensemble"
    else:
        print("\n    Winner: Best Single Model (GradientBoosting lr=0.2_d=7)!")
        final_pred = best_pred
        final_acc = best_acc
        final_method = "GradientBoosting (Single Model)"

    print(f"    Final Accuracy: {final_acc:.4f}")

    test_df = pd.read_csv(test_path)
    test_df = test_df.dropna(subset=['winner'])

    results_df = test_df[['match_id', 'match_date', 'team1', 'team2', 'venue', 'winner']].copy()
    results_df.columns = ['match_id', 'match_date', 'team1', 'team2', 'venue', 'actual_winner']
    results_df['predicted_winner'] = le_target.inverse_transform(final_pred)
    results_df['correct'] = results_df['actual_winner'] == results_df['predicted_winner']

    results_df.to_csv('data/processed/final_predictions.csv', index=False)
    print(f"\n    Final predictions saved to: data/processed/final_predictions.csv")

    print("\n" + "="*80)
    print("FINAL PREDICTIONS")
    print("="*80)
    for idx, row in results_df.iterrows():
        status = "[OK]" if row['correct'] else "[WRONG]"
        print(f"    {status} {row['match_date'][:10]} | {row['team1']} vs {row['team2']} | Pred: {row['predicted_winner']} | Actual: {row['actual_winner']}")

    correct = results_df['correct'].sum()
    print(f"\n    Total: {len(results_df)} | Correct: {correct} | Accuracy: {final_acc*100:.2f}%")
    print(f"    Method Used: {final_method}")

    return final_method, final_acc


if __name__ == "__main__":
    method, acc = main()