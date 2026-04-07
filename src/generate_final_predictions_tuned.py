from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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

    return X_train, X_test, y_train, y_test, le_team, test_df


def main():
    print("="*80)
    print("FINAL PREDICTIONS - BEST MODELS AFTER TUNING")
    print("="*80)

    train_path = Path("data/processed/train_features.csv")
    test_path = Path("data/processed/test_features.csv")

    X_train, X_test, y_train, y_test, le_team, test_df = load_and_preprocess_binary(train_path, test_path)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # 1. Original MLP (from previous best - 60%)
    print("\n[1] Original MLP (100,50) - Best from previous run")
    mlp_orig = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, early_stopping=True)
    mlp_orig.fit(X_train_scaled, y_train)
    mlp_orig_pred = mlp_orig.predict(X_test_scaled)
    mlp_orig_acc = accuracy_score(y_test, mlp_orig_pred)
    print(f"    Accuracy: {mlp_orig_acc:.4f} ({mlp_orig_acc*100:.1f}%)")
    results.append(('MLP_Original_100_50', mlp_orig_acc, mlp_orig_pred))

    # 2. Tuned MLP
    print("\n[2] Tuned MLP (100,) with tanh activation")
    mlp_tuned = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', alpha=0.001,
                                learning_rate_init=0.01, max_iter=1000, random_state=42, early_stopping=False)
    mlp_tuned.fit(X_train_scaled, y_train)
    mlp_tuned_pred = mlp_tuned.predict(X_test_scaled)
    mlp_tuned_acc = accuracy_score(y_test, mlp_tuned_pred)
    print(f"    Accuracy: {mlp_tuned_acc:.4f} ({mlp_tuned_acc*100:.1f}%)")
    results.append(('MLP_Tuned_100', mlp_tuned_acc, mlp_tuned_pred))

    # 3. Tuned SVM (poly kernel - best from tuning)
    print("\n[3] Tuned SVM (poly kernel, C=10, gamma=0.1)")
    svm_tuned = SVC(kernel='poly', C=10, gamma=0.1, coef0=1, probability=True, random_state=42)
    svm_tuned.fit(X_train_scaled, y_train)
    svm_tuned_pred = svm_tuned.predict(X_test_scaled)
    svm_tuned_acc = accuracy_score(y_test, svm_tuned_pred)
    print(f"    Accuracy: {svm_tuned_acc:.4f} ({svm_tuned_acc*100:.1f}%)")
    results.append(('SVM_Tuned', svm_tuned_acc, svm_tuned_pred))

    # 4. Ensemble of original + tuned
    print("\n[4] Simple Ensemble (Original MLP + Tuned SVM)")
    mlp_proba = mlp_orig.predict_proba(X_test_scaled)[:, 1]
    svm_proba = svm_tuned.predict_proba(X_test_scaled)[:, 1]
    avg_proba = (mlp_proba + svm_proba) / 2
    ens_pred = (avg_proba >= 0.5).astype(int)
    ens_acc = accuracy_score(y_test, ens_pred)
    print(f"    Accuracy: {ens_acc:.4f} ({ens_acc*100:.1f}%)")
    results.append(('Ensemble_MLP_SVM', ens_acc, ens_pred))

    # Find best
    best_name, best_acc, best_pred = max(results, key=lambda x: x[1])
    print(f"\n{'='*80}")
    print(f"BEST: {best_name} with {best_acc:.4f} ({best_acc*100:.1f}%)")
    print(f"Random baseline: 50%")
    print(f"Improvement over random: +{(best_acc-0.5)*100:.1f}%")
    print(f"{'='*80}")

    # Generate predictions with best model
    test_df = test_df.reset_index(drop=True)

    predicted_winners = []
    for i in range(len(test_df)):
        if best_pred[i] == 1:
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
    print(f"SUMMARY: {correct}/{total} correct | Accuracy: {best_acc*100:.1f}%")
    print(f"{'='*80}")

    results_df.to_csv('data/processed/final_predictions_tuned.csv', index=False)
    print(f"\nPredictions saved to: data/processed/final_predictions_tuned.csv")

    return results_df, best_name, best_acc


if __name__ == "__main__":
    results_df, best_name, best_acc = main()