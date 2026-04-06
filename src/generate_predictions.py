from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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

    return X_train, X_test, y_train, y_test, le_target, test_df


def main():
    print("="*80)
    print("T20 MATCH WINNER PREDICTIONS - XGBOOST (BEST MODEL)")
    print("="*80)

    train_path = Path("data/processed/train_features.csv")
    test_path = Path("data/processed/test_features.csv")

    X_train, X_test, y_train, y_test, le_target, test_df = load_and_preprocess(train_path, test_path)

    print(f"\nTotal test matches: {len(test_df)}")
    print(f"Unique teams: {len(le_target.classes_)}")

    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss', verbosity=0)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    y_pred_labels = le_target.inverse_transform(y_pred)
    y_actual_labels = le_target.inverse_transform(y_test)

    results_df = test_df[['match_id', 'match_date', 'team1', 'team2', 'venue', 'winner']].copy()
    results_df.columns = ['match_id', 'match_date', 'team1', 'team2', 'venue', 'actual_winner']
    results_df['predicted_winner'] = y_pred_labels
    results_df['correct'] = results_df['actual_winner'] == results_df['predicted_winner']

    print("\n" + "="*80)
    print("PREDICTIONS vs ACTUAL RESULTS")
    print("="*80)

    for idx, row in results_df.iterrows():
        status = "[OK]" if row['correct'] else "[WRONG]"
        print(f"\n{status} Match {row['match_id']} | {row['match_date'][:10]}")
        print(f"   {row['team1']} vs {row['team2']} at {row['venue']}")
        print(f"   Predicted: {row['predicted_winner']}")
        print(f"   Actual:    {row['actual_winner']}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    correct = results_df['correct'].sum()
    total = len(results_df)
    accuracy = correct / total * 100
    print(f"Total matches: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    output_path = Path("data/processed/predictions.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nFull predictions saved to: {output_path}")

    print("\n" + "="*80)
    print("INCORRECT PREDICTIONS")
    print("="*80)
    incorrect = results_df[~results_df['correct']]
    for idx, row in incorrect.iterrows():
        print(f"\n{row['team1']} vs {row['team2']}")
        print(f"   Predicted: {row['predicted_winner']} | Actual: {row['actual_winner']}")


if __name__ == "__main__":
    main()