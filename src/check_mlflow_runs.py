import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

exp = mlflow.get_experiment_by_name("T20_Match_Winner_Prediction_Phase2")
if exp:
    client = mlflow.MlflowClient()
    runs = client.search_runs(exp.experiment_id, max_results=100)
    print(f"Phase 2 Experiment runs: {len(runs)}")
    for r in runs:
        name = r.data.tags.get("mlflow.runName", "unnamed")
        acc = r.data.metrics.get("test_accuracy", 0)
        print(f"  - {name} | Accuracy: {acc:.4f}")
else:
    print("Phase 2 experiment not found")

print()

exp1 = mlflow.get_experiment_by_name("T20_Match_Winner_Prediction")
if exp1:
    client = mlflow.MlflowClient()
    runs = client.search_runs(exp1.experiment_id, max_results=100)
    print(f"Phase 1 Experiment runs: {len(runs)}")
    for r in runs:
        name = r.data.tags.get("mlflow.runName", "unnamed")
        acc = r.data.metrics.get("test_accuracy", 0)
        print(f"  - {name} | Accuracy: {acc:.4f}")
else:
    print("Phase 1 experiment not found")