from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict


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
        "season": int(str(info["dates"][0])[:4]),
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
                    ball_num = int((over - int(over)) * 10)
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
                "overs": total_balls / 6,
                "run_rate": total_runs / (total_balls / 6) if total_balls > 0 else 0,
                "death_overs_runs": death_overs_runs,
                "death_overs_balls": death_overs_balls,
                "death_overs_run_rate": death_overs_runs / (death_overs_balls / 6) if death_overs_balls > 0 else 0,
            })

    return match_info, innings_data


def create_enhanced_features(data_dir, is_train=True, history_stats=None):
    yaml_files = sorted(Path(data_dir).glob("*.yaml"))

    all_matches = []
    for yaml_file in yaml_files:
        match_info, innings_data = parse_match_details(yaml_file)
        all_matches.append((yaml_file, match_info, innings_data))

    all_matches.sort(key=lambda x: x[1]["match_date"])

    enhanced_rows = []

    if history_stats is None:
        history_stats = defaultdict(lambda: {
            "runs": [], "wickets": [], "balls": [], "death_runs": [], "death_balls": [], "matches": 0, "wins": 0
        })

    for yaml_file, match_info, innings_data in all_matches:
        team1 = match_info["team1"]
        team2 = match_info["team2"]
        match_date = match_info["match_date"]

        stats1 = history_stats[team1]
        stats2 = history_stats[team2]

        form1_runs = np.mean(stats1["runs"]) if stats1["runs"] else 30.0
        form2_runs = np.mean(stats2["runs"]) if stats2["runs"] else 30.0
        form1_wickets = np.mean(stats1["wickets"]) if stats1["wickets"] else 5.0
        form2_wickets = np.mean(stats2["wickets"]) if stats2["wickets"] else 5.0

        form1_run_rate = np.mean([r/b*6 for r, b in zip(stats1["runs"], stats1["balls"])]) if stats1["runs"] else 6.0
        form2_run_rate = np.mean([r/b*6 for r, b in zip(stats2["runs"], stats2["balls"])]) if stats2["runs"] else 6.0

        form1_death_rate = np.mean([r/b*6 for r, b in zip(stats1["death_runs"], stats1["death_balls"]) if b > 0]) if stats1["death_runs"] and any(b > 0 for b in stats1["death_balls"]) else 7.5
        form2_death_rate = np.mean([r/b*6 for r, b in zip(stats2["death_runs"], stats2["death_balls"]) if b > 0]) if stats2["death_runs"] and any(b > 0 for b in stats2["death_balls"]) else 7.5

        inning1 = next((i for i in innings_data if i["team"] == team1), None)
        inning2 = next((i for i in innings_data if i["team"] == team2), None)

        row = {
            "match_id": match_info["match_id"],
            "match_date": match_date,
            "team1": team1,
            "team2": team2,
            "venue": match_info["venue"],
            "toss_winner": match_info["toss_winner"],
            "toss_decision": match_info["toss_decision"],
            "winner": match_info["winner"],

            "team1_avg_runs": form1_runs,
            "team2_avg_runs": form2_runs,
            "team1_avg_wickets": form1_wickets,
            "team2_avg_wickets": form2_wickets,
            "team1_run_rate": form1_run_rate,
            "team2_run_rate": form2_run_rate,
            "team1_death_run_rate": form1_death_rate,
            "team2_death_run_rate": form2_death_rate,
            "team1_matches": stats1["matches"],
            "team2_matches": stats2["matches"],
        }

        if inning1:
            row["team1_inning_runs"] = inning1["total_runs"]
            row["team1_inning_wickets"] = inning1["total_wickets"]
            row["team1_inning_run_rate"] = inning1["run_rate"]
            row["team1_death_runs"] = inning1["death_overs_runs"]
        else:
            row["team1_inning_runs"] = 0
            row["team1_inning_wickets"] = 0
            row["team1_inning_run_rate"] = 0
            row["team1_death_runs"] = 0

        if inning2:
            row["team2_inning_runs"] = inning2["total_runs"]
            row["team2_inning_wickets"] = inning2["total_wickets"]
            row["team2_inning_run_rate"] = inning2["run_rate"]
            row["team2_death_runs"] = inning2["death_overs_runs"]
        else:
            row["team2_inning_runs"] = 0
            row["team2_inning_wickets"] = 0
            row["team2_inning_run_rate"] = 0
            row["team2_death_runs"] = 0

        enhanced_rows.append(row)

        for inning in innings_data:
            team = inning["team"]
            stats = history_stats[team]
            stats["runs"].append(inning["total_runs"])
            stats["wickets"].append(inning["total_wickets"])
            stats["balls"].append(inning["total_balls"])
            stats["death_runs"].append(inning["death_overs_runs"])
            stats["death_balls"].append(inning["death_overs_balls"])
            stats["matches"] += 1
            if match_info["winner"] == team:
                stats["wins"] += 1

    df = pd.DataFrame(enhanced_rows)
    return df, history_stats


def main():
    print("="*80)
    print("ENHANCED FEATURE ENGINEERING")
    print("="*80)

    train_dir = Path("data/splits/pre_match_eval/train")
    test_dir = Path("data/splits/pre_match_eval/test")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Step 1] Processing TRAIN data...")
    train_df, history_stats = create_enhanced_features(train_dir, is_train=True)
    print(f"  Train matches: {len(train_df)}")

    print("\n[Step 2] Processing TEST data (using train as history)...")
    test_df, _ = create_enhanced_features(test_dir, is_train=False, history_stats=history_stats)
    print(f"  Test matches: {len(test_df)}")

    train_df.to_csv(output_dir / "enhanced_train_features.csv", index=False)
    test_df.to_csv(output_dir / "enhanced_test_features.csv", index=False)

    print(f"\n[Step 3] Saving...")
    print(f"  Enhanced train: {output_dir / 'enhanced_train_features.csv'}")
    print(f"  Enhanced test: {output_dir / 'enhanced_test_features.csv'}")

    print("\n" + "="*80)
    print("NEW FEATURES SUMMARY")
    print("="*80)

    new_cols = ['team1_avg_runs', 'team2_avg_runs', 'team1_avg_wickets', 'team2_avg_wickets',
                'team1_run_rate', 'team2_run_rate', 'team1_death_run_rate', 'team2_death_run_rate',
                'team1_matches', 'team2_matches', 'team1_inning_runs', 'team1_inning_wickets',
                'team1_inning_run_rate', 'team1_death_runs', 'team2_inning_runs', 'team2_inning_wickets',
                'team2_inning_run_rate', 'team2_death_runs']

    print(f"\nTotal columns: {len(train_df.columns)}")
    print(f"Total rows: Train={len(train_df)}, Test={len(test_df)}")

    print("\n[Step 4] Feature statistics (sample):")
    for col in new_cols:
        if col in train_df.columns:
            print(f"  {col}: mean={train_df[col].mean():.2f}, min={train_df[col].min():.2f}, max={train_df[col].max():.2f}")

    print("\n" + "="*80)
    print("SAMPLE DATA (first 5 rows, key columns)")
    print("="*80)

    sample_cols = ['match_id', 'team1', 'team2', 'team1_avg_runs', 'team2_avg_runs',
                  'team1_run_rate', 'team2_run_rate', 'team1_death_run_rate', 'team2_death_run_rate']

    print(train_df[sample_cols].head().to_string())

    print("\n" + "="*80)
    print("INNING FEATURES (actual match performance)")
    print("="*80)

    inning_cols = ['match_id', 'team1', 'team2', 'team1_inning_runs', 'team1_inning_wickets',
                   'team1_inning_run_rate', 'team1_death_runs', 'team2_inning_runs', 'team2_inning_wickets',
                   'team2_inning_run_rate', 'team2_death_runs']

    print(train_df[inning_cols].head().to_string())

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = main()