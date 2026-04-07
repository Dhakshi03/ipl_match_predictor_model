from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://localhost:5000")
EXPERIMENT_NAME = "T20_Match_Winner_Comparison"
mlflow.set_experiment(EXPERIMENT_NAME)

LEAKAGE_COLS = [
    'team1_inning_runs', 'team1_inning_wickets', 'team1_inning_run_rate', 'team1_death_runs',
    'team2_inning_runs', 'team2_inning_wickets', 'team2_inning_run_rate', 'team2_death_runs'
]

MODELS_REQUIRING_SCALING = ['MLPClassifier', 'LogisticRegression', 'SVC']


def load_basic_data(train_path, test_path):
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
    
    return X_train, X_test, y_train, y_test, feature_cols


def load_enhanced_clean_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    for col in LEAKAGE_COLS:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])
    
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
    
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['team1_win'].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['team1_win'].values
    
    return X_train, X_test, y_train, y_test, feature_cols


def get_model_configs():
    configs = []
    
    configs.append(('MLP', '100_50', MLPClassifier, {'hidden_layer_sizes': (100, 50), 'max_iter': 1000, 'random_state': 42, 'early_stopping': True}))
    configs.append(('MLP', '100', MLPClassifier, {'hidden_layer_sizes': (100,), 'max_iter': 1000, 'random_state': 42, 'early_stopping': True}))
    configs.append(('MLP', '50', MLPClassifier, {'hidden_layer_sizes': (50,), 'max_iter': 1000, 'random_state': 42, 'early_stopping': True}))
    
    configs.append(('GradientBoosting', 'lr0.1_d5', GradientBoostingClassifier, {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42}))
    configs.append(('GradientBoosting', 'lr0.05_d4', GradientBoostingClassifier, {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'random_state': 42}))
    configs.append(('GradientBoosting', 'lr0.1_d3', GradientBoostingClassifier, {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'random_state': 42}))
    configs.append(('GradientBoosting', 'lr0.05_d3', GradientBoostingClassifier, {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'random_state': 42}))
    
    configs.append(('RandomForest', '100_d5', RandomForestClassifier, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}))
    configs.append(('RandomForest', '100_d10', RandomForestClassifier, {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}))
    configs.append(('RandomForest', '200_d5', RandomForestClassifier, {'n_estimators': 200, 'max_depth': 5, 'random_state': 42}))
    
    configs.append(('XGBoost', 'lr0.1_d6', XGBClassifier, {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0}))
    configs.append(('XGBoost', 'lr0.05_d4', XGBClassifier, {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0}))
    
    configs.append(('LogisticRegression', 'C1', LogisticRegression, {'C': 1.0, 'max_iter': 2000, 'random_state': 42}))
    
    configs.append(('SVM', 'C1_rbf', SVC, {'C': 1.0, 'kernel': 'rbf', 'random_state': 42, 'probability': True}))
    configs.append(('SVM', 'C10_rbf', SVC, {'C': 10.0, 'kernel': 'rbf', 'random_state': 42, 'probability': True}))
    
    configs.append(('AdaBoost', '100', AdaBoostClassifier, {'n_estimators': 100, 'random_state': 42}))
    configs.append(('AdaBoost', '200', AdaBoostClassifier, {'n_estimators': 200, 'random_state': 42}))
    
    configs.append(('GaussianNB', 'default', GaussianNB, {}))
    
    return configs


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, dataset_name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = 0.0
    
    test_acc = accuracy_score(y_test, y_pred)
    
    return {
        'test_accuracy': test_acc,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'roc_auc': roc_auc
    }


def main():
    print("="*80)
    print("T20 MATCH WINNER - COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    print("\nTesting 18 configurations on 2 datasets (Basic + Enhanced Clean)")
    print("Leakage features REMOVED from Enhanced dataset")
    
    configs = get_model_configs()
    print(f"\nTotal configurations: {len(configs)}")
    
    print("\n[Step 1] Loading datasets...")
    
    X_train_basic, X_test_basic, y_train_basic, y_test_basic, features_basic = load_basic_data(
        Path("data/processed/train_features.csv"),
        Path("data/processed/test_features.csv")
    )
    print(f"  Basic: {len(features_basic)} features, Train: {len(X_train_basic)}, Test: {len(X_test_basic)}")
    
    X_train_enh, X_test_enh, y_train_enh, y_test_enh, features_enh = load_enhanced_clean_data(
        Path("data/processed/enhanced_train_features.csv"),
        Path("data/processed/enhanced_test_features.csv")
    )
    print(f"  Enhanced Clean: {len(features_enh)} features, Train: {len(X_train_enh)}, Test: {len(X_test_enh)}")
    
    print("\n[Step 2] Scaling datasets...")
    scaler_basic = StandardScaler()
    X_train_basic_scaled = scaler_basic.fit_transform(X_train_basic)
    X_test_basic_scaled = scaler_basic.transform(X_test_basic)
    
    scaler_enh = StandardScaler()
    X_train_enh_scaled = scaler_enh.fit_transform(X_train_enh)
    X_test_enh_scaled = scaler_enh.transform(X_test_enh)
    
    results = []
    
    print("\n[Step 3] Training all models...")
    print("="*80)
    
    for i, (model_type, config_name, model_class, params) in enumerate(configs):
        model_key = f"{model_type}_{config_name}"
        
        print(f"\n[{i+1}/{len(configs)}] {model_type} ({config_name})")
        
        needs_scaling = model_type in MODELS_REQUIRING_SCALING
        
        if needs_scaling:
            X_tr_b, X_te_b = X_train_basic_scaled, X_test_basic_scaled
            X_tr_e, X_te_e = X_train_enh_scaled, X_test_enh_scaled
        else:
            X_tr_b, X_te_b = X_train_basic, X_test_basic
            X_tr_e, X_te_e = X_train_enh, X_test_enh
        
        model_basic = model_class(**params) if model_class != XGBClassifier else XGBClassifier(**params)
        model_enh = model_class(**params) if model_class != XGBClassifier else XGBClassifier(**params)
        
        print(f"  Training on Basic dataset...")
        metrics_basic = train_and_evaluate(model_basic, X_tr_b, y_train_basic, X_te_b, y_test_basic, model_key, "Basic")
        
        print(f"  Training on Enhanced Clean dataset...")
        metrics_enh = train_and_evaluate(model_enh, X_tr_e, y_train_enh, X_te_e, y_test_enh, model_key, "Enhanced_Clean")
        
        with mlflow.start_run(run_name=f"{model_key}_Basic"):
            mlflow.log_params({**params, 'dataset': 'Basic', 'num_features': len(features_basic)})
            mlflow.log_metrics({
                'test_accuracy': metrics_basic['test_accuracy'],
                'cv_accuracy_mean': metrics_basic['cv_mean'],
                'cv_accuracy_std': metrics_basic['cv_std'],
                'roc_auc': metrics_basic['roc_auc']
            })
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("config", config_name)
            mlflow.set_tag("dataset", "Basic")
        
        with mlflow.start_run(run_name=f"{model_key}_Enhanced_Clean"):
            mlflow.log_params({**params, 'dataset': 'Enhanced_Clean', 'num_features': len(features_enh)})
            mlflow.log_metrics({
                'test_accuracy': metrics_enh['test_accuracy'],
                'cv_accuracy_mean': metrics_enh['cv_mean'],
                'cv_accuracy_std': metrics_enh['cv_std'],
                'roc_auc': metrics_enh['roc_auc']
            })
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("config", config_name)
            mlflow.set_tag("dataset", "Enhanced_Clean")
        
        above_b = "+" if metrics_basic['test_accuracy'] > 0.5 else ""
        above_e = "+" if metrics_enh['test_accuracy'] > 0.5 else ""
        
        print(f"    Basic: {above_b}{metrics_basic['test_accuracy']:.4f} | Enhanced Clean: {above_e}{metrics_enh['test_accuracy']:.4f}")
        
        results.append({
            'model': model_type,
            'config': config_name,
            'basic_accuracy': metrics_basic['test_accuracy'],
            'basic_cv': metrics_basic['cv_mean'],
            'enh_accuracy': metrics_enh['test_accuracy'],
            'enh_cv': metrics_enh['cv_mean']
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('basic_accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS TABLE")
    print("="*80)
    
    print(f"\n{'Model':<25} {'Config':<15} {'Basic':<12} {'Enhanced':<12} {'Diff':<8}")
    print("-"*80)
    
    for _, row in results_df.iterrows():
        diff = row['enh_accuracy'] - row['basic_accuracy']
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
        print(f"{row['model']:<25} {row['config']:<15} {row['basic_accuracy']:.4f}     {row['enh_accuracy']:.4f}     {diff_str}")
    
    best_basic = results_df.iloc[0]
    best_enh = results_df.sort_values('enh_accuracy', ascending=False).iloc[0]
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nBest Basic: {best_basic['model']} ({best_basic['config']}) = {best_basic['basic_accuracy']:.4f}")
    print(f"Best Enhanced Clean: {best_enh['model']} ({best_enh['config']}) = {best_enh['enh_accuracy']:.4f}")
    print(f"\nRandom baseline: 50%")
    
    results_df.to_csv('data/processed/comparison_results.csv', index=False)
    print(f"\nResults saved to: data/processed/comparison_results.csv")
    print("="*80)


if __name__ == "__main__":
    main()
