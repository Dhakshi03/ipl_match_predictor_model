from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://localhost:5000")
EXPERIMENT_NAME = "T20_Match_Winner_Hyperparameter_Tuning"
mlflow.set_experiment(EXPERIMENT_NAME)


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

    return X_train, X_test, y_train, y_test, le_team, feature_cols


def main():
    print("="*80)
    print("T20 MATCH WINNER - HYPERPARAMETER TUNING")
    print("="*80)

    print("\n[Step 1] Loading data...")
    train_path = Path("data/processed/train_features.csv")
    test_path = Path("data/processed/test_features.csv")

    X_train, X_test, y_train, y_test, le_team, feature_cols = load_and_preprocess_binary(train_path, test_path)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"    Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"    Features: {X_train.shape[1]}")
    print(f"    Baseline (random): 50%")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    # 1. MLP TUNING
    print("\n" + "="*80)
    print("[Step 2] Tuning MLP Neural Network")
    print("="*80)

    mlp_param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100), (150, 100), (100, 50, 25), (200, 100), (150, 75, 30)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01, 0.00001],
        'learning_rate_init': [0.001, 0.01, 0.0001],
        'early_stopping': [True, False]
    }

    mlp_base = MLPClassifier(max_iter=1000, random_state=42, learning_rate='adaptive')

    print("    Running RandomizedSearchCV for MLP...")
    mlp_search = RandomizedSearchCV(
        mlp_base, mlp_param_grid, n_iter=40, cv=cv, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=0
    )
    mlp_search.fit(X_train_scaled, y_train)

    print(f"    Best MLP params: {mlp_search.best_params_}")
    print(f"    Best MLP CV score: {mlp_search.best_score_:.4f}")

    mlp_best = mlp_search.best_estimator_
    mlp_pred = mlp_best.predict(X_test_scaled)
    mlp_acc = accuracy_score(y_test, mlp_pred)

    with mlflow.start_run(run_name="MLP_Best"):
        mlflow.log_params(mlp_search.best_params_)
        mlflow.log_metrics({
            'test_accuracy': mlp_acc,
            'cv_accuracy_mean': mlp_search.best_score_,
            'cv_accuracy_std': np.std(mlp_search.cv_results_['std_test_score']),
        })
        mlflow.sklearn.log_model(mlp_best, "model")
        mlflow.set_tag("model_type", "MLP")

    print(f"    MLP Test Accuracy: {mlp_acc:.4f}")
    results.append(('MLP', mlp_acc, mlp_search.best_params_, mlp_search.best_score_))

    # 2. GRADIENT BOOSTING TUNING
    print("\n" + "="*80)
    print("[Step 3] Tuning Gradient Boosting")
    print("="*80)

    gb_param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }

    gb_base = GradientBoostingClassifier(random_state=42)

    print("    Running RandomizedSearchCV for GradientBoosting...")
    gb_search = RandomizedSearchCV(
        gb_base, gb_param_grid, n_iter=50, cv=cv, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=0
    )
    gb_search.fit(X_train_scaled, y_train)

    print(f"    Best GB params: {gb_search.best_params_}")
    print(f"    Best GB CV score: {gb_search.best_score_:.4f}")

    gb_best = gb_search.best_estimator_
    gb_pred = gb_best.predict(X_test_scaled)
    gb_acc = accuracy_score(y_test, gb_pred)

    with mlflow.start_run(run_name="GradientBoosting_Best"):
        mlflow.log_params({k: str(v) for k, v in gb_search.best_params_.items()})
        mlflow.log_metrics({
            'test_accuracy': gb_acc,
            'cv_accuracy_mean': gb_search.best_score_,
            'cv_accuracy_std': np.std(gb_search.cv_results_['std_test_score']),
        })
        mlflow.sklearn.log_model(gb_best, "model")
        mlflow.set_tag("model_type", "GradientBoosting")

    print(f"    GradientBoosting Test Accuracy: {gb_acc:.4f}")
    results.append(('GradientBoosting', gb_acc, gb_search.best_params_, gb_search.best_score_))

    # 3. SVM TUNING
    print("\n" + "="*80)
    print("[Step 4] Tuning SVM")
    print("="*80)

    svm_param_grid = {
        'C': [0.01, 0.1, 1, 10, 50, 100],
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'coef0': [0, 1, 2]
    }

    svm_base = SVC(probability=True, random_state=42)

    print("    Running RandomizedSearchCV for SVM...")
    svm_search = RandomizedSearchCV(
        svm_base, svm_param_grid, n_iter=40, cv=cv, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=0
    )
    svm_search.fit(X_train_scaled, y_train)

    print(f"    Best SVM params: {svm_search.best_params_}")
    print(f"    Best SVM CV score: {svm_search.best_score_:.4f}")

    svm_best = svm_search.best_estimator_
    svm_pred = svm_best.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)

    with mlflow.start_run(run_name="SVM_Best"):
        mlflow.log_params(svm_search.best_params_)
        mlflow.log_metrics({
            'test_accuracy': svm_acc,
            'cv_accuracy_mean': svm_search.best_score_,
            'cv_accuracy_std': np.std(svm_search.cv_results_['std_test_score']),
        })
        mlflow.sklearn.log_model(svm_best, "model")
        mlflow.set_tag("model_type", "SVM")

    print(f"    SVM Test Accuracy: {svm_acc:.4f}")
    results.append(('SVM', svm_acc, svm_search.best_params_, svm_search.best_score_))

    # 4. RANDOM FOREST TUNING
    print("\n" + "="*80)
    print("[Step 5] Tuning Random Forest")
    print("="*80)

    rf_param_grid = {
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    }

    rf_base = RandomForestClassifier(random_state=42)

    print("    Running RandomizedSearchCV for RandomForest...")
    rf_search = RandomizedSearchCV(
        rf_base, rf_param_grid, n_iter=40, cv=cv, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train_scaled, y_train)

    print(f"    Best RF params: {rf_search.best_params_}")
    print(f"    Best RF CV score: {rf_search.best_score_:.4f}")

    rf_best = rf_search.best_estimator_
    rf_pred = rf_best.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)

    with mlflow.start_run(run_name="RandomForest_Best"):
        mlflow.log_params({k: str(v) for k, v in rf_search.best_params_.items()})
        mlflow.log_metrics({
            'test_accuracy': rf_acc,
            'cv_accuracy_mean': rf_search.best_score_,
            'cv_accuracy_std': np.std(rf_search.cv_results_['std_test_score']),
        })
        mlflow.sklearn.log_model(rf_best, "model")
        mlflow.set_tag("model_type", "RandomForest")

    print(f"    RandomForest Test Accuracy: {rf_acc:.4f}")
    results.append(('RandomForest', rf_acc, rf_search.best_params_, rf_search.best_score_))

    # 5. VOTING ENSEMBLE OF BEST MODELS
    print("\n" + "="*80)
    print("[Step 6] Creating Voting Ensemble of Best Models")
    print("="*80)

    ensemble = VotingClassifier(
        estimators=[
            ('mlp', mlp_best),
            ('gb', gb_best),
            ('svm', svm_best),
            ('rf', rf_best)
        ],
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_train)
    ens_pred = ensemble.predict(X_test_scaled)
    ens_acc = accuracy_score(y_test, ens_pred)

    with mlflow.start_run(run_name="VotingEnsemble_Best"):
        mlflow.log_metrics({'test_accuracy': ens_acc})
        mlflow.set_tag("model_type", "VotingEnsemble")
        mlflow.set_tag("components", "MLP_GB_SVM_RF")

    print(f"    Voting Ensemble Test Accuracy: {ens_acc:.4f}")
    results.append(('VotingEnsemble', ens_acc, 'soft_voting_4_models', 0))

    # SUMMARY
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*80)

    results_df = pd.DataFrame(results, columns=['Model', 'Test_Accuracy', 'Best_Params', 'CV_Score'])
    results_df = results_df.sort_values('Test_Accuracy', ascending=False)

    print("\nFinal Rankings:")
    print("-"*80)
    for idx, row in results_df.iterrows():
        print(f"    {row['Model']:20} | Test: {row['Test_Accuracy']:.4f} | CV: {row['CV_Score']:.4f}")

    best = results_df.iloc[0]
    print(f"\n    BEST OVERALL: {best['Model']} with {best['Test_Accuracy']:.4f} accuracy")

    print(f"\nMLflow: http://localhost:5000")
    print(f"Experiment: {EXPERIMENT_NAME}")

    results_df.to_csv('data/processed/hyperparameter_tuning_results.csv', index=False)

    return results_df, mlp_best, gb_best, svm_best, rf_best, ensemble


if __name__ == "__main__":
    results_df, mlp_best, gb_best, svm_best, rf_best, ensemble = main()