from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def load_enhanced_data(train_path, test_path):
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

    X_train = train_df[feature_cols].values
    y_train = train_df['team1_win'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['team1_win'].values

    return X_train, X_test, y_train, y_test, test_df


def main():
    print("="*80)
    print("FINAL PREDICTIONS - ENHANCED FEATURES (BEST MODEL)")
    print("="*80)

    train_path = Path("data/processed/enhanced_train_features.csv")
    test_path = Path("data/processed/enhanced_test_features.csv")

    X_train, X_test, y_train, y_test, test_df = load_enhanced_data(train_path, test_path)

    print("\nTraining best model: GradientBoosting (lr=0.1, depth=5)...")
    model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"Random baseline: 50%")
    print(f"Improvement: +{(acc-0.5)*100:.1f}%")

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

    print("\n" + "="*80)
    print("PREDICTIONS")
    print("="*80)

    for i, row in results_df.iterrows():
        status = "[OK]" if row['correct'] else "[WRONG]"
        print(f"    {status} {row['match_date'][:10]} | {row['team1']} vs {row['team2']}")
        print(f"        Predicted: {row['predicted_winner']} | Actual: {row['actual_winner']}")

    correct = results_df['correct'].sum()
    total = len(results_df)

    print(f"\n{'='*80}")
    print(f"SUMMARY: {correct}/{total} correct | Accuracy: {acc*100:.1f}%")
    print(f"Improvement over random: +{(acc-0.5)*100:.1f}%")
    print(f"{'='*80}")

    results_df.to_csv('data/processed/enhanced_final_predictions.csv', index=False)
    print(f"\nPredictions saved to: data/processed/enhanced_final_predictions.csv")


if __name__ == "__main__":
    main()