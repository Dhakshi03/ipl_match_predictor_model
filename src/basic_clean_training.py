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
EXPERIMENT_NAME = "T20_Match_Winner_Basic_Clean"
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

    return match_info


def create_basic_features(data_dir, team_history=None, h2h_history=None, venue_history=None, is_test=False):
    yaml_files = sorted(Path(data_dir).glob("*.yaml"))
    
    all_matches = []
    for yaml_file in yaml_files:
        match_info = parse_match_details(yaml_file)
        all_matches.append((yaml_file, match_info))
    
    all_matches.sort(key=lambda x: x[1]["match_date"])
    
    if team_history is None:
        team_history = defaultdict(lambda: {"wins": 0, "matches": 0, "recent_results": []})
    if h2h_history is None:
        h2h_history = defaultdict(lambda: {"wins": 0, "matches": 0})
    if venue_history is None:
        venue_history = defaultdict(lambda: {"wins": 0, "matches": 0})
    
    enhanced_rows = []
    
    for yaml_file, match_info in all_matches:
        team1 = match_info["team1"]
        team2 = match_info["team2"]
        venue = match_info["venue"]
        
        hist1 = team_history[team1]
        hist2 = team_history[team2]
        
        team1_recent = hist1["recent_results"]
        team2_recent = hist2["recent_results"]
        
        team1_win_pct_last5 = sum(team1_recent[-5:]) / len(team1_recent[-5:]) if len(team1_recent) >= 5 else (hist1["wins"] / hist1["matches"] if hist1["matches"] > 0 else 0.5)
        team2_win_pct_last5 = sum(team2_recent[-5:]) / len(team2_recent[-5:]) if len(team2_recent) >= 5 else (hist2["wins"] / hist2["matches"] if hist2["matches"] > 0 else 0.5)
        
        pair = tuple(sorted([team1, team2]))
        h2h = h2h_history[pair]
        team1_h2h = h2h["wins"] / h2h["matches"] if h2h["matches"] > 0 else 0.5
        team2_h2h = 1 - team1_h2h
        
        venue_hist = venue_history[venue]
        team1_venue = venue_hist.get(team1, {"wins": 0, "matches": 0})
        team2_venue = venue_hist.get(team2, {"wins": 0, "matches": 0})
        
        team1_venue_pct = team1_venue["wins"] / team1_venue["matches"] if team1_venue["matches"] > 0 else 0.5
        team2_venue_pct = team2_venue["wins"] / team2_venue["matches"] if team2_venue["matches"] > 0 else 0.5
        
        row = {
            "match_id": match_info["match_id"],
            "match_date": match_info["match_date"],
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "toss_winner": match_info["toss_winner"],
            "toss_decision": match_info["toss_decision"],
            "winner": match_info["winner"],
            
            "team1_win_pct_last_5": team1_win_pct_last5,
            "team2_win_pct_last_5": team2_win_pct_last5,
            "team1_head_to_head_win_pct": team1_h2h,
            "team2_head_to_head_win_pct": team2_h2h,
            "team1_win_pct_at_venue": team1_venue_pct,
            "team2_win_pct_at_venue": team2_venue_pct,
        }
        
        enhanced_rows.append(row)
        
        winner = match_info["winner"]
        if winner:
            for team, hist in [(team1, hist1), (team2, hist2)]:
                hist["matches"] += 1
                if winner == team:
                    hist["wins"] += 1
                    hist["recent_results"].append(1)
                else:
                    hist["recent_results"].append(0)
                if len(hist["recent_results"]) > 20:
                    hist["recent_results"] = hist["recent_results"][-20:]
            
            h2h["matches"] += 1
            h2h["wins"] += 1 if winner == team1 else 0
            
            for team, vhist in [(team1, team1_venue), (team2, team2_venue)]:
                if venue not in venue_history:
                    venue_history[venue] = {}
                if team not in venue_history[venue]:
                    venue_history[venue][team] = {"wins": 0, "matches": 0}
                vh = venue_history[venue][team]
                vh["matches"] += 1
                if winner == team:
                    vh["wins"] += 1
    
    df = pd.DataFrame(enhanced_rows)
    return df, team_history, h2h_history, venue_history


def main():
    print("="*80)
    print("T20 MATCH WINNER - BASIC PRE-MATCH MODEL (NO LEAKAGE)")
    print("="*80)
    
    train_dir = Path("data/splits/pre_match_eval/train")
    test_dir = Path("data/splits/pre_match_eval/test")
    
    print("\n[Step 1] Creating basic pre-match features...")
    train_df, team_hist, h2h_hist, venue_hist = create_basic_features(train_dir)
    print(f"  Train matches: {len(train_df)}")
    
    test_df, _, _, _ = create_basic_features(test_dir, team_hist, h2h_hist, venue_hist, is_test=True)
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
    
    print(f"  Features: {len(feature_cols)} (basic pre-match, no leakage)")
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"  Class: {np.sum(y_train==1)} team1 wins, {np.sum(y_train==0)} team2 wins")
    
    print("\n[Step 3] Training with 5-Fold CV...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = [
        ('GB_lr0.1_d5', GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
        ('GB_lr0.05_d4', GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)),
        ('GB_lr0.1_d3', GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42)),
        ('GB_lr0.02_d3', GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.02, random_state=42)),
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
            mlflow.set_tag("model_type", "GradientBoosting_Basic")
        
        above = "+" if test_acc > 0.5 else ""
        print(f"    Test: {above}{test_acc:.4f} | CV: {np.mean(cv_scores):.4f}")
        
        results.append({
            'model': name,
            'test_accuracy': test_acc,
            'cv_mean': np.mean(cv_scores),
            'roc_auc': roc
        })
    
    results_df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)
    best = results_df.iloc[0]
    
    print("\n" + "="*80)
    print("BASIC PRE-MATCH MODEL RESULTS (NO LEAKAGE)")
    print("="*80)
    print(f"\nBest: {best['model']} with {best['test_accuracy']:.4f} accuracy")
    print(f"Random baseline: 50%")
    print(f"Improvement: +{(best['test_accuracy']-0.5)*100:.1f}%")


if __name__ == "__main__":
    main()
