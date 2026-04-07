from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_binary(train_path, test_path):
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
        df['win_pct_diff'] = df['team1_win_pct_last_5'] - df['team2_win_pct_last_5']
        df['h2h_diff'] = df['team1_head_to_head_win_pct'] - df['team2_head_to_head_win_pct']
        df['venue_diff'] = df['team1_win_pct_at_venue'] - df['team2_win_pct_at_venue']
        df['team1_win'] = (df['winner'] == df['team1']).astype(int)

    feature_cols = [
        'team1_encoded', 'team2_encoded', 'venue_encoded',
        'toss_decision_encoded', 'toss_advantage',
        'team1_win_pct_last_5', 'team2_win_pct_last_5',
        'team1_head_to_head_win_pct', 'team2_head_to_head_win_pct',
        'team1_win_pct_at_venue', 'team2_win_pct_at_venue',
        'win_pct_diff', 'h2h_diff', 'venue_diff'
    ]

    X_train = train_df[feature_cols].values
    y_train = train_df['team1_win'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['team1_win'].values

    return X_train, X_test, y_train, y_test, le_team, test_df, feature_cols


def main():
    print("="*80)
    print("FINAL PREDICTIONS - BEST MODEL (MLP 100,50)")
    print("="*80)

    train_path = Path("data/processed/train_features.csv")
    test_path = Path("data/processed/test_features.csv")

    X_train, X_test, y_train, y_test, le_team, test_df, feature_cols = load_and_preprocess_binary(train_path, test_path)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining best model: MLPClassifier (hidden_layer_sizes=(100, 50))...")
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, early_stopping=True)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nMetrics:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:   {rec:.4f}")
    print(f"  F1:       {f1:.4f}")
    print(f"  Baseline: 50.0% (random coin flip)")

    test_df = test_df.reset_index(drop=True)
    y_pred_series = pd.Series(y_pred, name='prediction')

    results_df = test_df[['match_id', 'match_date', 'team1', 'team2', 'venue', 'winner']].copy()
    results_df.columns = ['match_id', 'match_date', 'team1', 'team2', 'venue', 'actual_winner']

    predicted_winners = []
    for i, row in results_df.iterrows():
        if y_pred[i] == 1:
            predicted_winners.append(row['team1'])
        else:
            predicted_winners.append(row['team2'])

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

    print("\n" + "="*80)
    print(f"SUMMARY: {correct}/{total} correct | Accuracy: {acc*100:.1f}%")
    print(f"Improvement over random: +{(acc-0.5)*100:.1f}%")
    print("="*80)

    results_df.to_csv('data/processed/final_predictions_binary.csv', index=False)
    print(f"\nPredictions saved to: data/processed/final_predictions_binary.csv")


if __name__ == "__main__":
    main()