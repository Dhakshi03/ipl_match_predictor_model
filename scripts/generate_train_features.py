from pathlib import Path
import yaml
import pandas as pd


def parse_match_metadata(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    info = data["info"]
    teams = info["teams"]
    outcome = info.get("outcome", {})
    toss = info.get("toss", {})

    return {
        "match_id": file_path.stem,
        "match_date": info["dates"][0],
        "season": int(str(info["dates"][0])[:4]),
        "team1": teams[0],
        "team2": teams[1],
        "venue": info.get("venue"),
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "winner": outcome.get("winner"),
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


def generate_train_features():
    train_dir = Path("data/splits/pre_match_eval/train")
    yaml_files = sorted(train_dir.glob("*.yaml"))

    matches = [parse_match_metadata(path) for path in yaml_files]
    matches_df = pd.DataFrame(matches)
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])
    matches_df = matches_df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    feature_rows = []

    for idx, row in matches_df.iterrows():
        history = matches_df.iloc[:idx]
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
        })

    feature_df = pd.DataFrame(feature_rows)
    output_path = Path("data/processed/train_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    print(f"Generated {len(feature_df)} train feature rows at {output_path}")
    return feature_df


if __name__ == "__main__":
    generate_train_features()