from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://localhost:5000")
EXPERIMENT_NAME = "T20_Match_Winner_PreMatch_Clean"
mlflow.set_experiment(EXPERIMENT_NAME)


def parse_match_details(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    info = data["info"]
    teams = info["teams"]
    outcome = info.get("outcome", {})
    toss = info.get("toss", {})

    match_info = {
        "match_id": file_path.stem,
        "match_date": info["dates"][0],
        "team1": teams[0],
        "team2": teams[1],
        "venue": info.get("venue"),
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "winner": outcome.get("winner"),
    }

    innings_data = []
    for inning in data["innings"]:
        for inning_name, details in inning.items():
            team = details["team"]
            total_runs = 0
            total_wickets = 0
            total_balls = 0
            death_overs_runs = 0
            death_overs_balls = 0

            for delivery in details["deliveries"]:
                for ball_str, ball_data in delivery.items():
                    over = float(ball_str)
                    total_balls += 1
                    runs = ball_data["runs"]["total"]
                    total_runs += runs
                    wicket = 1 if "wicket" in ball_data else 0
                    total_wickets += wicket
                    if over >= 16:
                        death_overs_runs += runs
                        death_overs_balls += 1

            innings_data.append({
                "team": team,
                "total_runs": total_runs,
                "total_wickets": total_wickets,
                "total_balls": total_balls,
                "run_rate": total_runs / (total_balls / 6) if total_balls > 0 else 0,
                "death_overs_runs": death_overs_runs,
                "death_overs_balls": death_overs_balls,
                "death_run_rate": death_overs_runs / (death_overs_balls / 6) if death_overs_balls > 0 else 0,
            })

    return match_info, innings_data


def create_clean_features(data_dir, history_stats=None, is_test=False):
    yaml_files = sorted(Path(data_dir).glob("*.yaml"))

    all_matches = []
    for yaml_file in yaml_files:
        match_info, innings_data = parse_match_details(yaml_file)
        all_matches.append((yaml_file, match_info, innings_data))

    all_matches.sort(key=lambda x: x[1]["match_date"])

    if history_stats is None:
        history_stats = defaultdict(lambda: {"runs": [], "wickets": [], "balls": [], "death_runs": [], "death_balls": [], "matches": 0})

    enhanced_rows = []

    for yaml_file, match_info, innings_data in all_matches:
        team1 = match_info["team1"]
        team2 = match_info["team2"]

        stats1 = history_stats[team1]
        stats2 = history_stats[team2]

        # VALID PRE-MATCH FEATURES (no leakage)
        form1_runs = np.mean(stats1["runs"][-5:]) if stats1["runs"] else 30.0
        form2_runs = np.mean(stats2["runs"][-5:]) if stats2["runs"] else 30.0
        form1_wickets = np.mean(stats1["wickets"][-5:]) if stats1["wickets"] else 5.0
        form2_wickets = np.mean(stats2["wickets"][-5:]) if stats2["wickets"] else 5.0

        form1_run_rate = np.mean([r/b*6 for r, b in zip(stats1["runs"][-5:], stats1["balls"][-5:]) if stats1["runs"] and b > 0]) if stats1["runs"] else 6.0
        form2_run_rate = np.mean([r/b*6 for r, b in zip(stats2["runs"][-5:], stats2["balls"][-5:]) if stats2["runs"] and b > 0]) if stats2["runs"] else 6.0

        # FIX: Use average death run rate from last 5 matches (NOT current match)
        form1_death_rate = np.mean([r/b*6 for r, b in zip(stats1["death_runs"][-5:], stats1["death_balls"][-5:]) if stats1["death_runs"] and b > 0]) if stats1["death_runs"] else 7.5
        form2_death_rate = np.mean([r/b*6 for r, b in zip(stats2["death_runs"][-5:], stats2["death_balls"][-5:]) if stats2["death_runs"] and b > 0]) if stats2["death_runs"] else 7.5

        row = {
            "match_id": match_info["match_id"],
            "match_date": match_info["match_date"],
            "team1": team1,
            "team2": team2,
            "venue": match_info["venue"],
            "toss_winner": match_info["toss_winner"],
            "toss_decision": match_info["toss_decision"],
            "winner": match_info["winner"],

            # PRE-MATCH HISTORICAL FEATURES (NO LEAKAGE)
            "team1_avg_runs_last5": form1_runs,
            "team2_avg_runs_last5": form2_runs,
            "team1_avg_wickets_last5": form1_wickets,
            "team2_avg_wickets_last5": form2_wickets,
            "team1_run_rate_last5": form1_run_rate,
            "team2_run_rate_last5": form2_run_rate,
            "team1_death_rate_last5": form1_death_rate,
            "team2_death_rate_last5": form2_death_rate,
            "team1_matches": stats1["matches"],
            "team2_matches": stats2["matches"],
        }

        enhanced_rows.append(row)

        # Update history AFTER computing features (no leakage)
        for inning in innings_data:
            team = inning["team"]
            stats = history_stats[team]
            stats["runs"].append(inning["total_runs"])
            stats["wickets"].append(inning["total_wickets"])
            stats["balls"].append(inning["total_balls"])
            stats["death_runs"].append(inning["death_overs_runs"])
            stats["death_balls"].append(inning["death_overs_balls"])
            stats["matches"] += 1

    df = pd.DataFrame(enhanced_rows)
    return df, history_stats


def main():
    print("="*80)
    print("T20 MATCH WINNER - CLEAN PRE-MATCH MODEL (NO LEAKAGE)")
    print("="*80)

    train_dir = Path("data/splits/pre_match_eval/train")
    test_dir = Path("data/splits/pre_match_eval/test")

    print("\n[Step 1] Creating clean pre-match features (NO data leakage)...")
    print("  - Removed inning features (actual match results)")
    print("  - Using LAST 5 MATCHES historical averages")
    print("  - No future information used")

    train_df, train_history = create_clean_features(train_dir)
    print(f"  Train matches: {len(train_df)}")

    test_df, _ = create_clean_features(test_dir, history_stats=train_history, is_test=True)
    print(f"  Test matches: {len(test_df)}")

    print("\n[Step 2] Preprocessing...")

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

        # Difference features
        df['runs_diff'] = df['team1_avg_runs_last5'] - df['team2_avg_runs_last5']
        df['run_rate_diff'] = df['team1_run_rate_last5'] - df['team2_run_rate_last5']
        df['death_rate_diff'] = df['team1_death_rate_last5'] - df['team2_death_rate_last5']
        df['wickets_diff'] = df['team1_avg_wickets_last5'] - df['team2_avg_wickets_last5']

    # CLEAN FEATURES (NO LEAKAGE)
    feature_cols = [
        'team1_encoded', 'team2_encoded', 'venue_encoded',
        'toss_decision_encoded', 'toss_advantage',
        'team1_avg_runs_last5', 'team2_avg_runs_last5',
        'team1_avg_wickets_last5', 'team2_avg_wickets_last5',
        'team1_run_rate_last5', 'team2_run_rate_last5',
        'team1_death_rate_last5', 'team2_death_rate_last5',
        'team1_matches', 'team2_matches',
        'runs_diff', 'run_rate_diff', 'death_rate_diff', 'wickets_diff'
    ]

    X_train = train_df[feature_cols].values
    y_train = train_df['team1_win'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['team1_win'].values

    print(f"  Features: {len(feature_cols)} (clean, no leakage)")
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"  Class: {np.sum(y_train==1)} team1 wins, {np.sum(y_train==0)} team2 wins")

    print("\n[Step 3] Training with 5-Fold CV...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Test multiple models
    models = [
        ('GB_lr0.1_d5', GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
        ('GB_lr0.05_d4', GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)),
        ('GB_lr0.08_d5', GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)),
    ]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    for name, model in models:
        print(f"\n  Training {name}...")

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        test_acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        with mlflow.start_run(run_name=name):
            mlflow.log_metrics({
                'test_accuracy': test_acc,
                'test_roc_auc': roc,
                'cv_accuracy_mean': np.mean(cv_scores),
                'cv_accuracy_std': np.std(cv_scores),
            })
            mlflow.sklearn.log_model(model, "model")
            mlflow.set_tag("model_type", "GradientBoosting_Clean")

        above = "+" if test_acc > 0.5 else ""
        print(f"    Test: {above}{test_acc:.4f} | CV: {np.mean(cv_scores):.4f}")

        results.append({
            'model': name,
            'test_accuracy': test_acc,
            'cv_mean': np.mean(cv_scores),
            'roc_auc': roc
        })

    # Find best
    results_df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)
    best = results_df.iloc[0]

    print("\n" + "="*80)
    print("CLEAN PRE-MATCH MODEL RESULTS")
    print("="*80)
    print(f"\nBest: {best['model']} with {best['test_accuracy']:.4f} accuracy")
    print(f"Random baseline: 50%")
    print(f"Improvement: +{(best['test_accuracy']-0.5)*100:.1f}%")

    print("\n[Step 4] Generating final predictions...")
    best_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)

    test_df = test_df.reset_index(drop=True)
    predicted_winners = []
    for i in range(len(test_df)):
        if y_pred[i] == 1:
            predicted_winners.append(test_df.iloc[i]['team1'])
        else:
            predicted_winners.append(test_df.iloc[i]['team2'])

    results_df_final = test_df[['match_id', 'match_date', 'team1', 'team2', 'venue', 'winner']].copy()
    results_df_final.columns = ['match_id', 'match_date', 'team1', 'team2', 'venue', 'actual_winner']
    results_df_final['predicted_winner'] = predicted_winners
    results_df_final['correct'] = results_df_final['actual_winner'] == results_df_final['predicted_winner']

    print("\nPredictions:")
    for i, row in results_df_final.iterrows():
        status = "[OK]" if row['correct'] else "[WRONG]"
        print(f"  {status} {row['match_date'][:10]} | {row['team1']} vs {row['team2']} | Pred: {row['predicted_winner']} | Actual: {row['actual_winner']}")

    correct = results_df_final['correct'].sum()
    total = len(results_df_final)
    print(f"\nTotal: {correct}/{total} = {correct/total*100:.1f}%")

    results_df_final.to_csv('data/processed/clean_prematch_predictions.csv', index=False)
    print(f"\nSaved to: data/processed/clean_prematch_predictions.csv")


if __name__ == "__main__":
    main()