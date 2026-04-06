from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "generate_test_features.py"
)


def load_script_module():
    spec = importlib.util.spec_from_file_location("generate_test_features", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GenerateTestFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script_module()

    def test_generates_one_row_per_test_match_and_writes_csv(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        train_dir = project_root / "data" / "splits" / "pre_match_eval" / "train"
        test_dir = project_root / "data" / "splits" / "pre_match_eval" / "test"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_features.csv"
            feature_df = self.module.generate_test_features(train_dir, test_dir, output_path)

            expected_count = len(list(test_dir.glob("*.yaml")))

            self.assertEqual(len(feature_df), expected_count)
            self.assertTrue(output_path.exists())

            written_df = pd.read_csv(output_path)
            self.assertEqual(len(written_df), expected_count)
            self.assertEqual(feature_df.iloc[0]["match_id"], written_df.iloc[0]["match_id"])

    def test_first_test_row_uses_train_history(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        train_dir = project_root / "data" / "splits" / "pre_match_eval" / "train"
        test_dir = project_root / "data" / "splits" / "pre_match_eval" / "test"

        train_matches_df = self.module.load_matches_from_dir(train_dir)
        test_matches_df = self.module.load_matches_from_dir(test_dir)
        feature_df = self.module.build_test_feature_df(train_matches_df, test_matches_df)

        first_test_match = test_matches_df.iloc[0]
        expected_team1_last_5 = self.module.win_pct_last_n(train_matches_df, first_test_match["team1"])
        expected_team2_last_5 = self.module.win_pct_last_n(train_matches_df, first_test_match["team2"])

        self.assertEqual(feature_df.iloc[0]["match_id"], first_test_match["match_id"])
        self.assertEqual(feature_df.iloc[0]["team1_win_pct_last_5"], expected_team1_last_5)
        self.assertEqual(feature_df.iloc[0]["team2_win_pct_last_5"], expected_team2_last_5)


if __name__ == "__main__":
    unittest.main()
