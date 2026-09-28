import os
import pandas as pd

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "safety_evaluation_results.csv"
)

df = pd.read_csv(RESULTS_PATH)

df["correct"] = df["actual"] == df["predicted"]

summary = df.groupby("benchmark_label").agg(
    total=("benchmark_label", "size"),
    predicted_suicide=("predicted", "sum"),
    correct=("correct", "sum")
)

summary["accuracy"] = summary["correct"] / summary["total"]

summary["false_positive_rate"] = (
    summary["predicted_suicide"] / summary["total"]
)

print("\n========== CATEGORY ANALYSIS ==========\n")

print(summary.to_string())

print("\n\nPredicted suicide rate by category:")

print(
    summary["false_positive_rate"]
    .sort_values(ascending=False)
    .to_string()
)