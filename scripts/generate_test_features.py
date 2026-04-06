from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_TRAIN_DIR = Path("data/splits/pre_match_eval/train")
DEFAULT_TEST_DIR = Path("data/splits/pre_match_eval/test")
DEFAULT_OUTPUT = Path("data/splits/pre_match_eval/test_features.csv")


def parse_match_metadata(file_path: Path) -> dict[str, object]:
    with file_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

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


def load_matches_from_dir(directory: Path) -> pd.DataFrame:
    yaml_files = sorted(directory.glob("*.yaml"))
    matches = [parse_match_metadata(path) for path in yaml_files]
    matches_df = pd.DataFrame(matches)
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])
    return matches_df.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def win_pct_last_n(history: pd.DataFrame, team: str, n: int = 5) -> float:
    team_history = history[(history["team1"] == team) | (history["team2"] == team)].tail(n)
    if team_history.empty:
        return 0.5
    return float((team_history["winner"] == team).mean())


def head_to_head_win_pct(history: pd.DataFrame, team: str, opponent: str) -> float:
    h2h = history[
        ((history["team1"] == team) & (history["team2"] == opponent))
        | ((history["team1"] == opponent) & (history["team2"] == team))
    ]
    if h2h.empty:
        return 0.5
    return float((h2h["winner"] == team).mean())


def venue_win_pct(history: pd.DataFrame, team: str, venue: str | None) -> float:
    venue_history = history[
        ((history["team1"] == team) | (history["team2"] == team)) & (history["venue"] == venue)
    ]
    if venue_history.empty:
        return 0.5
    return float((venue_history["winner"] == team).mean())


def build_test_feature_df(train_matches_df: pd.DataFrame, test_matches_df: pd.DataFrame) -> pd.DataFrame:
    feature_rows: list[dict[str, object]] = []
    history = train_matches_df.copy()

    for _, row in test_matches_df.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]
        venue = row["venue"]

        feature_rows.append(
            {
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
            }
        )

        history = pd.concat([history, row.to_frame().T], ignore_index=True)

    return pd.DataFrame(feature_rows)


def generate_test_features(
    train_dir: Path = DEFAULT_TRAIN_DIR,
    test_dir: Path = DEFAULT_TEST_DIR,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    train_matches_df = load_matches_from_dir(train_dir)
    test_matches_df = load_matches_from_dir(test_dir)
    feature_df = build_test_feature_df(train_matches_df, test_matches_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    return feature_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pre-match features for the test split using full train history."
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_df = generate_test_features(args.train_dir, args.test_dir, args.output)
    print(f"Generated {len(feature_df)} test feature rows at {args.output}")


if __name__ == "__main__":
    main()
