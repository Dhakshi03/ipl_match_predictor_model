from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://localhost:5000")
EXPERIMENT_NAME = "T20_Match_Winner_Prediction_Phase2"
mlflow.set_experiment(EXPERIMENT_NAME)

MODELS_REQUIRE_SCALING = ['LogisticRegression', 'RidgeClassifier', 'SVC', 'MLPClassifier']


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

    X_train = train_df[feature_cols].values
    y_train = le_target.transform(train_df['winner'])
    X_test = test_df[feature_cols].values
    y_test = le_target.transform(test_df['winner'])

    return X_train, X_test, y_train, y_test, le_target, feature_cols


def get_model_configs():
    configs = []

    configs.append(('LogisticRegression', 'C=0.1', LogisticRegression, {'C': 0.1, 'max_iter': 2000, 'random_state': 42, 'class_weight': 'balanced'}))
    configs.append(('LogisticRegression', 'C=1.0', LogisticRegression, {'C': 1.0, 'max_iter': 2000, 'random_state': 42, 'class_weight': 'balanced'}))
    configs.append(('LogisticRegression', 'C=10.0', LogisticRegression, {'C': 10.0, 'max_iter': 2000, 'random_state': 42, 'class_weight': 'balanced'}))

    configs.append(('RandomForest', 'depth=5', RandomForestClassifier, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42, 'class_weight': 'balanced'}))
    configs.append(('RandomForest', 'depth=10', RandomForestClassifier, {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'class_weight': 'balanced'}))
    configs.append(('RandomForest', 'depth=15', RandomForestClassifier, {'n_estimators': 100, 'max_depth': 15, 'random_state': 42, 'class_weight': 'balanced'}))

    configs.append(('XGBoost', 'lr=0.01_d=4', None, {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.01, 'random_state': 42, 'eval_metric': 'mlogloss', 'verbosity': 0}))
    configs.append(('XGBoost', 'lr=0.1_d=6', None, {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'eval_metric': 'mlogloss', 'verbosity': 0}))
    configs.append(('XGBoost', 'lr=0.3_d=8', None, {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.3, 'random_state': 42, 'eval_metric': 'mlogloss', 'verbosity': 0}))

    configs.append(('AdaBoost', 'n_est=50', AdaBoostClassifier, {'n_estimators': 50, 'random_state': 42}))
    configs.append(('AdaBoost', 'n_est=100', AdaBoostClassifier, {'n_estimators': 100, 'random_state': 42}))
    configs.append(('AdaBoost', 'n_est=200', AdaBoostClassifier, {'n_estimators': 200, 'random_state': 42}))

    configs.append(('SVM', 'C=0.1', SVC, {'C': 0.1, 'kernel': 'rbf', 'random_state': 42, 'class_weight': 'balanced'}))
    configs.append(('SVM', 'C=1.0', SVC, {'C': 1.0, 'kernel': 'rbf', 'random_state': 42, 'class_weight': 'balanced'}))
    configs.append(('SVM', 'C=10.0', SVC, {'C': 10.0, 'kernel': 'rbf', 'random_state': 42, 'class_weight': 'balanced'}))

    configs.append(('GradientBoosting', 'lr=0.05_d=3', GradientBoostingClassifier, {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'random_state': 42}))
    configs.append(('GradientBoosting', 'lr=0.1_d=5', GradientBoostingClassifier, {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42}))
    configs.append(('GradientBoosting', 'lr=0.2_d=7', GradientBoostingClassifier, {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.2, 'random_state': 42}))

    configs.append(('RidgeClassifier', 'alpha=0.1', RidgeClassifier, {'alpha': 0.1, 'random_state': 42, 'class_weight': 'balanced'}))
    configs.append(('RidgeClassifier', 'alpha=1.0', RidgeClassifier, {'alpha': 1.0, 'random_state': 42, 'class_weight': 'balanced'}))
    configs.append(('RidgeClassifier', 'alpha=10.0', RidgeClassifier, {'alpha': 10.0, 'random_state': 42, 'class_weight': 'balanced'}))

    configs.append(('NaiveBayes', 'Gaussian', GaussianNB, {}))

    configs.append(('MLPClassifier', 'hidden=(50,)', MLPClassifier, {'hidden_layer_sizes': (50,), 'max_iter': 1000, 'random_state': 42, 'early_stopping': True}))
    configs.append(('MLPClassifier', 'hidden=(100,50)', MLPClassifier, {'hidden_layer_sizes': (100, 50), 'max_iter': 1000, 'random_state': 42, 'early_stopping': True}))
    configs.append(('MLPClassifier', 'hidden=(100,100,50)', MLPClassifier, {'hidden_layer_sizes': (100, 100, 50), 'max_iter': 1000, 'random_state': 42, 'early_stopping': True}))

    try:
        from catboost import CatBoostClassifier
        configs.append(('CatBoost', 'lr=0.03_d=4', CatBoostClassifier, {'iterations': 100, 'depth': 4, 'learning_rate': 0.03, 'random_state': 42, 'verbose': 0, 'auto_class_weights': 'Balanced'}))
        configs.append(('CatBoost', 'lr=0.1_d=6', CatBoostClassifier, {'iterations': 100, 'depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'verbose': 0, 'auto_class_weights': 'Balanced'}))
        configs.append(('CatBoost', 'lr=0.3_d=8', CatBoostClassifier, {'iterations': 100, 'depth': 8, 'learning_rate': 0.3, 'random_state': 42, 'verbose': 0, 'auto_class_weights': 'Balanced'}))
        HAS_CATBOOST = True
    except ImportError:
        print("CatBoost not available, skipping...")
        HAS_CATBOOST = False

    return configs, HAS_CATBOOST


def main():
    print("="*80)
    print("T20 MATCH WINNER PREDICTION - PHASE 2 (CONTINUED)")
    print("="*80)

    print("\n[Step 1] Loading data...")
    train_path = Path("data/processed/train_features.csv")
    test_path = Path("data/processed/test_features.csv")

    X_train, X_test, y_train, y_test, le_target, feature_cols = load_and_preprocess(train_path, test_path)
    print(f"    Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    print("\n[Step 2] Getting model configurations...")
    configs, has_catboost = get_model_configs()
    print(f"    Total configurations: {len(configs)}")

    print("\n[Step 3] Training with 5-Fold CV...")
    print("="*80)

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i, (model_type, config_name, model_class, params) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {model_type} ({config_name})...")

        needs_scaling = model_class.__name__ in MODELS_REQUIRE_SCALING if model_class else False

        if needs_scaling:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        try:
            if model_class is None:
                if model_type == 'XGBoost':
                    from xgboost import XGBClassifier
                    model = XGBClassifier(**params)
                elif model_type == 'CatBoost':
                    from catboost import CatBoostClassifier
                    model = CatBoostClassifier(**params)
            else:
                model = model_class(**params)

            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            }

            with mlflow.start_run(run_name=f"{model_type}_{config_name}") as run:
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metrics({
                    'test_accuracy': test_metrics['accuracy'],
                    'test_precision': test_metrics['precision'],
                    'test_recall': test_metrics['recall'],
                    'test_f1': test_metrics['f1'],
                    'cv_accuracy_mean': np.mean(cv_scores),
                    'cv_accuracy_std': np.std(cv_scores),
                })
                signature = infer_signature(X_test_scaled, y_pred)
                mlflow.sklearn.log_model(model, "model", signature=signature)
                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("config_name", config_name)

            results.append({
                'model_type': model_type,
                'config_name': config_name,
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1': test_metrics['f1'],
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'needs_scaling': needs_scaling,
                'status': 'success'
            })

            print(f"    Test: {test_metrics['accuracy']:.4f} | CV: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

        except Exception as e:
            print(f"    FAILED: {str(e)[:50]}")
            results.append({
                'model_type': model_type,
                'config_name': config_name,
                'test_accuracy': 0,
                'test_precision': 0,
                'test_recall': 0,
                'test_f1': 0,
                'cv_mean': 0,
                'cv_std': 0,
                'needs_scaling': needs_scaling,
                'status': 'failed',
                'error': str(e)[:100]
            })

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_accuracy', ascending=False)

    print("\n[Step 4] Top 10 Models:")
    print("-"*80)
    for idx, row in results_df.head(10).iterrows():
        status = "[OK]" if row['status'] == 'success' else "[FAIL]"
        print(f"    {status} {row['model_type']:20} | {row['config_name']:20} | Test: {row['test_accuracy']:.4f} | CV: {row['cv_mean']:.4f}")

    results_df.to_csv('data/processed/model_comparison_phase2.csv', index=False)
    print(f"\n    Results saved to: data/processed/model_comparison_phase2.csv")
    print(f"    View MLflow at: http://localhost:5000")

    return results_df, le_target, feature_cols, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    results_df, le_target, feature_cols, X_train, X_test, y_train, y_test = main()