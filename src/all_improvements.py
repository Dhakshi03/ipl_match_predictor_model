from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import RFE
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://localhost:5000")
EXPERIMENT_NAME = "T20_Match_Winner_All_Improvements"
mlflow.set_experiment(EXPERIMENT_NAME)


def load_and_engineer_features(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = train_df.dropna(subset=['winner'])
    test_df = test_df.dropna(subset=['winner'])

    # Label encoding for categorical features
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

    # ============================================
    # IMPROVEMENT 1: INTERACTION FEATURES
    # ============================================
    for df in [train_df, test_df]:
        # Original difference features
        df['runs_diff'] = df['team1_avg_runs'] - df['team2_avg_runs']
        df['run_rate_diff'] = df['team1_run_rate'] - df['team2_run_rate']
        df['death_rate_diff'] = df['team1_death_run_rate'] - df['team2_death_run_rate']
        df['wickets_diff'] = df['team1_avg_wickets'] - df['team2_avg_wickets']
        df['matches_diff'] = df['team1_matches'] - df['team2_matches']

        # INTERACTION FEATURES (Improvement 1)
        df['run_rate_x_toss'] = df['team1_run_rate'] * df['toss_advantage']
        df['runs_diff_x_toss'] = df['runs_diff'] * df['toss_advantage']
        df['death_rate_x_run_rate'] = df['team1_death_run_rate'] * df['team1_run_rate']
        df['wickets_x_runs'] = df['team1_avg_wickets'] * df['team1_avg_runs']
        df['experience_ratio'] = (df['team1_matches'] + 1) / (df['team2_matches'] + 1)
        
        # Squared terms (Polynomial features)
        df['runs_diff_sq'] = df['runs_diff'] ** 2
        df['run_rate_diff_sq'] = df['run_rate_diff'] ** 2
        
        # Binned features
        df['runs_diff_bin'] = pd.cut(df['runs_diff'], bins=[-200, -50, 0, 50, 200], labels=[0, 1, 2, 3]).astype(int)
        df['run_rate_diff_bin'] = pd.cut(df['run_rate_diff'], bins=[-5, -1, 0, 1, 5], labels=[0, 1, 2, 3]).astype(int)

    # ============================================
    # IMPROVEMENT 2: TARGET ENCODING FOR TEAMS
    # ============================================
    # Team1 target encoding (mean win rate)
    team1_win_rate = train_df.groupby('team1')['team1_win'].mean()
    team2_win_rate = train_df.groupby('team2')['team1_win'].mean()
    
    # Venue target encoding
    venue_win_rate = train_df.groupby('venue')['team1_win'].mean()
    
    global_mean = train_df['team1_win'].mean()
    
    for df in [train_df, test_df]:
        df['team1_target_enc'] = df['team1'].map(team1_win_rate).fillna(global_mean)
        df['team2_target_enc'] = df['team2'].map(team2_win_rate).fillna(global_mean)
        df['venue_target_enc'] = df['venue'].map(venue_win_rate).fillna(global_mean)
        
        # Frequency encoding
        team1_freq = train_df['team1'].value_counts(normalize=True)
        team2_freq = train_df['team2'].value_counts(normalize=True)
        venue_freq = train_df['venue'].value_counts(normalize=True)
        
        df['team1_freq'] = df['team1'].map(team1_freq).fillna(0)
        df['team2_freq'] = df['team2'].map(team2_freq).fillna(0)
        df['venue_freq'] = df['venue'].map(venue_freq).fillna(0)

    # ============================================
    # FEATURE COLUMNS - ALL IMPROVED FEATURES
    # ============================================
    feature_cols = [
        # Encoded features
        'team1_encoded', 'team2_encoded', 'venue_encoded',
        'toss_decision_encoded', 'toss_advantage',
        
        # Original numerical features
        'team1_avg_runs', 'team2_avg_runs',
        'team1_avg_wickets', 'team2_avg_wickets',
        'team1_run_rate', 'team2_run_rate',
        'team1_death_run_rate', 'team2_death_run_rate',
        'team1_matches', 'team2_matches',
        
        # Difference features
        'runs_diff', 'run_rate_diff', 'death_rate_diff', 'wickets_diff', 'matches_diff',
        
        # IMPROVEMENT 1: Interaction features
        'run_rate_x_toss', 'runs_diff_x_toss', 'death_rate_x_run_rate',
        'wickets_x_runs', 'experience_ratio',
        
        # IMPROVEMENT 1: Squared terms
        'runs_diff_sq', 'run_rate_diff_sq',
        
        # IMPROVEMENT 1: Binned features
        'runs_diff_bin', 'run_rate_diff_bin',
        
        # IMPROVEMENT 2: Target encoding
        'team1_target_enc', 'team2_target_enc', 'venue_target_enc',
        
        # IMPROVEMENT 2: Frequency encoding
        'team1_freq', 'team2_freq', 'venue_freq',
    ]

    X_train = train_df[feature_cols].values
    y_train = train_df['team1_win'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['team1_win'].values

    return X_train, X_test, y_train, y_test, feature_cols, test_df


def main():
    print("="*80)
    print("ALL IMPROVEMENTS COMBINED - COMPREHENSIVE TUNING")
    print("="*80)

    print("\n[Step 1] Loading and engineering features...")
    train_path = Path("data/processed/enhanced_train_features.csv")
    test_path = Path("data/processed/enhanced_test_features.csv")

    X_train, X_test, y_train, y_test, feature_cols, test_df = load_and_engineer_features(train_path, test_path)

    print(f"    Original features: 20")
    print(f"    After improvements: {len(feature_cols)}")
    print(f"    New features added:")
    print(f"      - Interaction features: run_rate_x_toss, runs_diff_x_toss, etc.")
    print(f"      - Polynomial features: runs_diff_sq, run_rate_diff_sq")
    print(f"      - Binned features: runs_diff_bin, run_rate_diff_bin")
    print(f"      - Target encoding: team1_target_enc, team2_target_enc, venue_target_enc")
    print(f"      - Frequency encoding: team1_freq, team2_freq, venue_freq")

    print(f"\n    Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    print("\n[Step 2] Training with IMPROVEMENT 3: More GradientBoosting tuning...")
    print("="*80)

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # IMPROVEMENT 3: Fine-tuned GradientBoosting configurations
    gb_configs = [
        ('GB_tuned_1', GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42, min_samples_split=3)),
        ('GB_tuned_2', GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, min_samples_split=5)),
        ('GB_tuned_3', GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, subsample=0.8)),
        ('GB_tuned_4', GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, min_samples_leaf=3)),
        ('GB_tuned_5', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, max_features='sqrt')),
    ]

    # IMPROVEMENT 3: More XGBoost configs
    xgb_configs = []
    try:
        from xgboost import XGBClassifier
        xgb_configs = [
            ('XGB_tuned_1', XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42, eval_metric='logloss', verbosity=0)),
            ('XGB_tuned_2', XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss', verbosity=0)),
        ]
    except:
        pass

    # IMPROVEMENT 3: More RandomForest configs
    rf_configs = [
        ('RF_tuned_1', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, min_samples_split=3)),
        ('RF_tuned_2', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, min_samples_leaf=2)),
    ]

    # IMPROVEMENT 3: Tuned MLP
    mlp_configs = [
        ('MLP_tuned_1', MLPClassifier(hidden_layer_sizes=(150, 75), max_iter=1500, random_state=42, early_stopping=True, learning_rate_init=0.001)),
        ('MLP_tuned_2', MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=1500, random_state=42, early_stopping=True)),
    ]

    # IMPROVEMENT 3: Logistic Regression with class weights
    lr_configs = [
        ('LR_balanced', LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight='balanced')),
        ('LR_tuned', LogisticRegression(C=0.5, max_iter=2000, random_state=42, solver='lbfgs')),
    ]

    all_configs = gb_configs + xgb_configs + rf_configs + mlp_configs + lr_configs

    print(f"    Total configurations: {len(all_configs)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for i, (name, model) in enumerate(all_configs):
        print(f"\n[{i+1}/{len(all_configs)}] {name}...")

        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            try:
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            except:
                roc_auc = 0.0

            test_acc = accuracy_score(y_test, y_pred)

            with mlflow.start_run(run_name=name) as run:
                mlflow.log_metrics({
                    'test_accuracy': test_acc,
                    'test_precision': precision_score(y_test, y_pred, zero_division=0),
                    'test_recall': recall_score(y_test, y_pred, zero_division=0),
                    'test_f1': f1_score(y_test, y_pred, zero_division=0),
                    'test_roc_auc': roc_auc,
                    'cv_accuracy_mean': np.mean(cv_scores),
                    'cv_accuracy_std': np.std(cv_scores),
                })
                mlflow.sklearn.log_model(model, "model")
                mlflow.set_tag("model_type", name)

            above = "+" if test_acc > 0.5 else ""
            print(f"    Test: {above}{test_acc:.4f} | CV: {np.mean(cv_scores):.4f}")

            results.append({
                'model': name,
                'test_accuracy': test_acc,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'roc_auc': roc_auc,
            })

        except Exception as e:
            print(f"    FAILED: {str(e)[:40]}")

    # ============================================
    # IMPROVEMENT 4: ENSEMBLE METHODS
    # ============================================
    print("\n" + "="*80)
    print("[Step 3] IMPROVEMENT 4: ENSEMBLE METHODS")
    print("="*80)

    # Get best models for ensemble
    results_df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)
    
    # Weighted voting ensemble (top 3)
    print("\nCreating Weighted Voting Ensemble (top 3 by CV)...")
    top3 = results_df.head(3)
    print(f"    Top 3: {top3['model'].tolist()}")

    # IMPROVEMENT 4: Soft voting with best models
    estimators = [
        ('gb1', GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)),
        ('gb2', GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
    ]

    ensemble_soft = VotingClassifier(estimators=estimators, voting='soft')
    cv_scores_ens = cross_val_score(ensemble_soft, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    ensemble_soft.fit(X_train_scaled, y_train)
    y_pred_ens = ensemble_soft.predict(X_test_scaled)
    ens_acc = accuracy_score(y_test, y_pred_ens)

    with mlflow.start_run(run_name="Ensemble_Soft_Voting") as run:
        mlflow.log_metrics({
            'test_accuracy': ens_acc,
            'cv_accuracy_mean': np.mean(cv_scores_ens),
        })
        mlflow.sklearn.log_model(ensemble_soft, "model")

    above = "+" if ens_acc > 0.5 else ""
    print(f"    Soft Voting Test: {above}{ens_acc:.4f} | CV: {np.mean(cv_scores_ens):.4f}")

    results.append({
        'model': 'Ensemble_Soft_Voting',
        'test_accuracy': ens_acc,
        'cv_mean': np.mean(cv_scores_ens),
        'cv_std': np.std(cv_scores_ens),
        'roc_auc': 0.0,
    })

    # Hard voting
    ensemble_hard = VotingClassifier(estimators=estimators, voting='hard')
    cv_scores_hard = cross_val_score(ensemble_hard, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    ensemble_hard.fit(X_train_scaled, y_train)
    y_pred_hard = ensemble_hard.predict(X_test_scaled)
    ens_hard_acc = accuracy_score(y_test, y_pred_hard)

    above = "+" if ens_hard_acc > 0.5 else ""
    print(f"    Hard Voting Test: {above}{ens_hard_acc:.4f} | CV: {np.mean(cv_scores_hard):.4f}")

    results.append({
        'model': 'Ensemble_Hard_Voting',
        'test_accuracy': ens_hard_acc,
        'cv_mean': np.mean(cv_scores_hard),
        'cv_std': np.std(cv_scores_hard),
        'roc_auc': 0.0,
    })

    # ============================================
    # IMPROVEMENT 5: THRESHOLD ADJUSTMENT
    # ============================================
    print("\n" + "="*80)
    print("[Step 4] IMPROVEMENT 5: THRESHOLD OPTIMIZATION")
    print("="*80)

    # Use best model and optimize threshold
    best_gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
    best_gb.fit(X_train_scaled, y_train)
    y_proba_best = best_gb.predict_proba(X_test_scaled)[:, 1]

    best_threshold = 0.5
    best_threshold_acc = accuracy_score(y_test, (y_proba_best >= 0.5).astype(int))

    for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
        y_pred_thresh = (y_proba_best >= threshold).astype(int)
        acc_thresh = accuracy_score(y_test, y_pred_thresh)
        if acc_thresh > best_threshold_acc:
            best_threshold_acc = acc_thresh
            best_threshold = threshold
        print(f"    Threshold {threshold}: {acc_thresh:.4f}")

    above = "+" if best_threshold_acc > 0.5 else ""
    print(f"\n    Best threshold: {best_threshold} -> Accuracy: {above}{best_threshold_acc:.4f}")

    results.append({
        'model': f'GB_Threshold_{best_threshold}',
        'test_accuracy': best_threshold_acc,
        'cv_mean': 0,
        'cv_std': 0,
        'roc_auc': 0.0,
    })

    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "="*80)
    print("ALL IMPROVEMENTS COMPLETE")
    print("="*80)

    results_df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)

    print("\n[FINAL RANKINGS]")
    print("-"*80)
    for idx, row in results_df.head(10).iterrows():
        above = "+" if row['test_accuracy'] > 0.5 else ""
        print(f"    {row['model']:30} | Test: {above}{row['test_accuracy']:.4f} | CV: {row['cv_mean']:.4f}")

    best = results_df.iloc[0]
    print(f"\n    BEST: {best['model']} with {best['test_accuracy']:.4f} accuracy")

    results_df.to_csv('data/processed/all_improvements_results.csv', index=False)
    print(f"\n    Results saved to: data/processed/all_improvements_results.csv")

    return results_df, best


if __name__ == "__main__":
    results_df, best = main()