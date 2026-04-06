from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def parse_match_metadata(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    info = data["info"]
    teams = info["teams"]
    outcome = info.get("outcome", {})
    toss = info.get("toss", {})

    winner = outcome.get("winner")
    result = outcome.get("result")
    
    return {
        "match_id": file_path.stem,
        "match_date": info["dates"][0],
        "season": int(str(info["dates"][0])[:4]),
        "team1": teams[0],
        "team2": teams[1],
        "venue": info.get("venue"),
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "winner": winner,
        "result": result
    }


def win_pct_last_n(history, team, n=5):
    team_history = history[(history["team1"] == team) | (history["team2"] == team)].tail(n)
    if team_history.empty:
        return 0.5
    return (team_history["winner"] == team).mean()


def head_to_head_win_pct(history, team, opponent):
    h2h = history[
        ((history["team1"] == team) & (history["team2"] == opponent)) |
        ((history["team1"] == opponent) & (history["team2"] == team))
    ]
    if h2h.empty:
        return 0.5
    return (h2h["winner"] == team).mean()


def venue_win_pct(history, team, venue):
    venue_history = history[
        (((history["team1"] == team) | (history["team2"] == team)) & (history["venue"] == venue))
    ]
    if venue_history.empty:
        return 0.5
    return (venue_history["winner"] == team).mean()


def create_features(yaml_dir, history_df=None, is_test=False):
    yaml_files = sorted(Path(yaml_dir).glob("*.yaml"))
    
    matches = [parse_match_metadata(path) for path in yaml_files]
    matches_df = pd.DataFrame(matches)
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])
    matches_df = matches_df.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    
    if history_df is not None:
        matches_df = pd.concat([history_df, matches_df], ignore_index=True)
        matches_df = matches_df.drop_duplicates(subset=['match_id'], keep='last')
        matches_df = matches_df.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    
    feature_rows = []
    
    for idx, row in matches_df.iterrows():
        if is_test and history_df is not None:
            check_idx = history_df.shape[0] + idx
        else:
            check_idx = idx
        
        history = matches_df.iloc[:check_idx]
        
        team1 = row["team1"]
        team2 = row["team2"]
        venue = row["venue"]

        feature_rows.append({
            "match_id": row["match_id"],
            "match_date": row["match_date"],
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "season": row["season"],
            "toss_winner": row["toss_winner"],
            "toss_decision": row["toss_decision"],
            "team1_win_pct_last_5": win_pct_last_n(history, team1),
            "team2_win_pct_last_5": win_pct_last_n(history, team2),
            "team1_head_to_head_win_pct": head_to_head_win_pct(history, team1, team2),
            "team2_head_to_head_win_pct": head_to_head_win_pct(history, team2, team1),
            "team1_win_pct_at_venue": venue_win_pct(history, team1, venue),
            "team2_win_pct_at_venue": venue_win_pct(history, team2, venue),
            "winner": row["winner"],
            "result": row.get("result")
        })

    feature_df = pd.DataFrame(feature_rows)
    
    if is_test and history_df is not None:
        feature_df = feature_df.iloc[history_df.shape[0]:].reset_index(drop=True)
    
    return feature_df


def encode_features(df, le_team1=None, le_team2=None, le_venue=None, le_toss=None, fit=True):
    df = df.copy()
    
    if fit:
        le_team1 = LabelEncoder()
        le_team2 = LabelEncoder()
        le_venue = LabelEncoder()
        le_toss = LabelEncoder()
        
        all_teams = pd.concat([df['team1'], df['team2'], df['toss_winner']]).unique()
        all_venues = df['venue'].unique()
        all_toss = df['toss_decision'].dropna().unique()
        
        le_team1.fit(all_teams)
        le_team2.fit(all_teams)
        le_venue.fit(all_venues)
        le_toss.fit(all_toss)
    
    df['team1_encoded'] = le_team1.transform(df['team1'])
    df['team2_encoded'] = le_team2.transform(df['team2'])
    df['venue_encoded'] = le_venue.transform(df['venue'])
    df['toss_decision_encoded'] = le_toss.transform(df['toss_decision'].fillna('unknown'))
    
    df['toss_advantage'] = (df['toss_winner'] == df['team1']).astype(int)
    
    df['win_pct_diff'] = df['team1_win_pct_last_5'] - df['team2_win_pct_last_5']
    df['h2h_diff'] = df['team1_head_to_head_win_pct'] - df['team2_head_to_head_win_pct']
    df['venue_diff'] = df['team1_win_pct_at_venue'] - df['team2_win_pct_at_venue']
    
    return df, le_team1, le_team2, le_venue, le_toss


def main():
    base_dir = Path(__file__).parent.parent / "data" / "splits" / "pre_match_eval"
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating train features...")
    train_df = create_features(base_dir / "train")
    print(f"Train matches: {len(train_df)}")
    print(f"Train columns: {train_df.columns.tolist()}")
    print(f"Sample winner values: {train_df['winner'].head()}")
    
    print("\nCreating test features...")
    test_df = create_features(base_dir / "test", history_df=train_df, is_test=True)
    print(f"Test matches: {len(test_df)}")
    
    train_df.to_csv(output_dir / "train_features.csv", index=False)
    test_df.to_csv(output_dir / "test_features.csv", index=False)
    
    print(f"\nSaved to {output_dir}")
    print(f"Train: {output_dir / 'train_features.csv'}")
    print(f"Test: {output_dir / 'test_features.csv'}")


if __name__ == "__main__":
    main()