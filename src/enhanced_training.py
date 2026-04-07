from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://localhost:5000")
EXPERIMENT_NAME = "T20_Match_Winner_Enhanced_Features"
mlflow.set_experiment(EXPERIMENT_NAME)

MODELS_REQUIRE_SCALING = ['LogisticRegression', 'RidgeClassifier', 'SVC', 'MLPClassifier']


def load_enhanced_data(train_path, test_path):
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
        df['team1_win'] = (df['winner'] == df['team1']).astype(int)
        
        df['runs_diff'] = df['team1_avg_runs'] - df['team2_avg_runs']
        df['run_rate_diff'] = df['team1_run_rate'] - df['team2_run_rate']
        df['death_rate_diff'] = df['team1_death_run_rate'] - df['team2_death_run_rate']
        df['wickets_diff'] = df['team1_avg_wickets'] - df['team2_avg_wickets']
        df['matches_diff'] = df['team1_matches'] - df['team2_matches']

    feature_cols = [
        'team1_encoded', 'team2_encoded', 'venue_encoded',
        'toss_decision_encoded', 'toss_advantage',
        'team1_avg_runs', 'team2_avg_runs',
        'team1_avg_wickets', 'team2_avg_wickets',
        'team1_run_rate', 'team2_run_rate',
        'team1_death_run_rate', 'team2_death_run_rate',
        'team1_matches', 'team2_matches',
        'runs_diff', 'run_rate_diff', 'death_rate_diff', 'wickets_diff', 'matches_diff'
    ]

    X_train = train_df[feature_cols].values
    y_train = train_df['team1_win'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['team1_win'].values

    return X_train, X_test, y_train, y_test, le_team, feature_cols, test_df


def get_model_configs():
    configs = []

    configs.append(('LogisticRegression', 'C=1.0', LogisticRegression, {'C': 1.0, 'max_iter': 2000, 'random_state': 42}))
    configs.append(('RandomForest', 'depth=5', RandomForestClassifier, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}))
    configs.append(('RandomForest', 'depth=10', RandomForestClassifier, {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}))
    
    configs.append(('XGBoost', 'lr=0.1_d=6', None, {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0}))
    configs.append(('XGBoost', 'lr=0.05_d=4', None, {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0}))
    
    configs.append(('AdaBoost', 'n_est=100', AdaBoostClassifier, {'n_estimators': 100, 'random_state': 42}))
    configs.append(('AdaBoost', 'n_est=200', AdaBoostClassifier, {'n_estimators': 200, 'random_state': 42}))
    
    configs.append(('SVM', 'C=1.0', SVC, {'C': 1.0, 'kernel': 'rbf', 'random_state': 42, 'probability': True}))
    configs.append(('SVM', 'C=10.0', SVC, {'C': 10.0, 'kernel': 'rbf', 'random_state': 42, 'probability': True}))
    
    configs.append(('GradientBoosting', 'lr=0.1_d=5', GradientBoostingClassifier, {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42}))
    configs.append(('GradientBoosting', 'lr=0.05_d=3', GradientBoostingClassifier, {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'random_state': 42}))
    
    configs.append(('MLPClassifier', 'hidden=(50,)', MLPClassifier, {'hidden_layer_sizes': (50,), 'max_iter': 1000, 'random_state': 42, 'early_stopping': True}))
    configs.append(('MLPClassifier', 'hidden=(100,50)', MLPClassifier, {'hidden_layer_sizes': (100, 50), 'max_iter': 1000, 'random_state': 42, 'early_stopping': True}))
    configs.append(('MLPClassifier', 'hidden=(100,)', MLPClassifier, {'hidden_layer_sizes': (100,), 'max_iter': 1000, 'random_state': 42, 'early_stopping': True}))

    try:
        from catboost import CatBoostClassifier
        configs.append(('CatBoost', 'lr=0.1_d=6', CatBoostClassifier, {'iterations': 100, 'depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'verbose': 0}))
        configs.append(('CatBoost', 'lr=0.05_d=4', CatBoostClassifier, {'iterations': 100, 'depth': 4, 'learning_rate': 0.05, 'random_state': 42, 'verbose': 0}))
    except:
        pass

    return configs


def main():
    print("="*80)
    print("T20 MATCH WINNER - ENHANCED FEATURES (WITH NEW BATTING STATS)")
    print("="*80)

    print("\n[Step 1] Loading enhanced data...")
    train_path = Path("data/processed/enhanced_train_features.csv")
    test_path = Path("data/processed/enhanced_test_features.csv")

    X_train, X_test, y_train, y_test, le_team, feature_cols, test_df = load_enhanced_data(train_path, test_path)

    print(f"    Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"    Features: {X_train.shape[1]}")
    print(f"    Feature columns: {feature_cols}")
    print(f"    Class distribution - Train: {np.sum(y_train==1)} team1 wins, {np.sum(y_train==0)} team2 wins")
    print(f"                    Test: {np.sum(y_test==1)} team1 wins, {np.sum(y_test==0)} team2 wins")

    print("\n[Step 2] Training with 5-Fold CV...")
    configs = get_model_configs()
    print(f"    Total configurations: {len(configs)}")
    print(f"    Random baseline: 50%")
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
            else:
                model = model_class(**params)

            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            try:
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            except:
                roc_auc = 0.0

            test_acc = accuracy_score(y_test, y_pred)
            test_prec = precision_score(y_test, y_pred, zero_division=0)
            test_rec = recall_score(y_test, y_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)

            with mlflow.start_run(run_name=f"{model_type}_{config_name}") as run:
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metrics({
                    'test_accuracy': test_acc,
                    'test_precision': test_prec,
                    'test_recall': test_rec,
                    'test_f1': test_f1,
                    'test_roc_auc': roc_auc,
                    'cv_accuracy_mean': np.mean(cv_scores),
                    'cv_accuracy_std': np.std(cv_scores),
                })
                signature = infer_signature(X_test_scaled, y_pred)
                mlflow.sklearn.log_model(model, "model", signature=signature)
                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("config_name", config_name)

            above = "+" if test_acc > 0.5 else ""
            print(f"    Test: {above}{test_acc:.4f} | CV: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f} | ROC: {roc_auc:.4f}")

            results.append({
                'model_type': model_type,
                'config_name': config_name,
                'test_accuracy': test_acc,
                'test_precision': test_prec,
                'test_recall': test_rec,
                'test_f1': test_f1,
                'test_roc_auc': roc_auc,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
            })

        except Exception as e:
            print(f"    FAILED: {str(e)[:60]}")
            results.append({
                'model_type': model_type,
                'config_name': config_name,
                'test_accuracy': 0,
                'test_precision': 0,
                'test_recall': 0,
                'test_f1': 0,
                'test_roc_auc': 0,
                'cv_mean': 0,
                'cv_std': 0,
            })

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_accuracy', ascending=False)

    print("\n[Step 3] Top Models (sorted by Test Accuracy):")
    print("-"*80)
    for idx, row in results_df.head(10).iterrows():
        above = "+" if row['test_accuracy'] > 0.5 else ""
        print(f"    {row['model_type']:20} | {row['config_name']:20} | Test: {above}{row['test_accuracy']:.4f} | CV: {row['cv_mean']:.4f}")

    best = results_df.iloc[0]
    print(f"\n    BEST: {best['model_type']} ({best['config_name']}) - Accuracy: {best['test_accuracy']:.4f}")

    results_df.to_csv('data/processed/enhanced_model_comparison.csv', index=False)
    print(f"\n    Results saved to: data/processed/enhanced_model_comparison.csv")
    print(f"    MLflow: http://localhost:5000")

    return results_df, le_team, X_train, X_test, y_train, y_test, feature_cols, test_df


if __name__ == "__main__":
    results_df, le_team, X_train, X_test, y_train, y_test, feature_cols, test_df = main()