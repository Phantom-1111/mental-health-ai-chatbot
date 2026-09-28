import os
import sys
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
ML_PATH = os.path.join(PROJECT_ROOT, "ml")

sys.path.append(ML_PATH)

from safety.predict_function import analyze_safety


TEST_PATH = os.path.join(
    os.path.dirname(__file__),
    "test.parquet"
)

ANNOTATION_PATH = os.path.join(
    os.path.dirname(__file__),
    "gpt-4o-mini.parquet"
)


test_df = pd.read_parquet(TEST_PATH)
annotation_df = pd.read_parquet(ANNOTATION_PATH)


df = test_df.merge(
    annotation_df[["example_id", "final_label"]],
    on="example_id",
    how="inner"
)


print("Total evaluation examples:", len(df))


results = []

for _, row in df.iterrows():

    text = row["inputs_joined"]

    prediction = analyze_safety(text)

    benchmark_label = row["final_label"]

    actual = 1 if benchmark_label == "suicidal_ideation" else 0
    predicted = 1 if prediction["label"] == "suicide" else 0

    results.append({
        "example_id": row["example_id"],
        "benchmark_label": benchmark_label,
        "actual": actual,
        "predicted": predicted,
        "suicide_probability": prediction["suicide_probability"],
        "non_suicide_probability": prediction["non_suicide_probability"]
    })


results_df = pd.DataFrame(results)


y_true = results_df["actual"]
y_pred = results_df["predicted"]
y_score = results_df["suicide_probability"]


accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)

false_negative_rate = fn / (fn + tp)

roc_auc = roc_auc_score(
    y_true,
    y_score
)


print("\n========== SAFETY EVALUATION ==========")

print(f"Accuracy:           {accuracy:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"Recall:             {recall:.4f}")
print(f"F1 Score:           {f1:.4f}")
print(f"Specificity:        {specificity:.4f}")
print(f"False Negative Rate:{false_negative_rate:.4f}")
print(f"ROC-AUC:            {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["non-suicidal", "suicidal"],
        zero_division=0
    )
)


output_path = os.path.join(
    os.path.dirname(__file__),
    "safety_evaluation_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nDetailed results saved to:")
print(output_path)