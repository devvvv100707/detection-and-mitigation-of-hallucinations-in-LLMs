"""Score manual labels when every label cell is filled.

If any label is empty, write outputs/label_metrics.json with status
labels_missing and exit 0. This script does not invent labels.
"""

from pathlib import Path
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "annotations/manual_labels.csv"
KEY = ROOT / "annotations/answer_key.csv"
OUTPUT = ROOT / "outputs/label_metrics.json"
FIGURES = ROOT / "outputs/figures"

POSITIVE = "Hallucinated or contradicted"
UNCERTAIN = "Uncertain/insufficient evidence"
ALLOWED = {POSITIVE, "Supported/correct", UNCERTAIN}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_missing():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"status": "labels_missing"}, indent=2), encoding="utf-8")
    print("labels missing, metrics cannot be calculated")


def confusion(rows):
    """Detector flag vs label. Positive class is Hallucinated or contradicted."""
    labels = ["not_hallucinated", "hallucinated"]
    matrix = {"not_hallucinated": {"not_hallucinated": 0, "hallucinated": 0},
              "hallucinated": {"not_hallucinated": 0, "hallucinated": 0}}
    for row in rows:
        actual = "hallucinated" if row["label"] == POSITIVE else "not_hallucinated"
        predicted = "hallucinated" if row["detector_flag"] == "true" else "not_hallucinated"
        matrix[actual][predicted] += 1
    tp = matrix["hallucinated"]["hallucinated"]
    fp = matrix["not_hallucinated"]["hallucinated"]
    fn = matrix["hallucinated"]["not_hallucinated"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall):
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "labels": labels,
        "matrix_actual_by_predicted": matrix,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": matrix["not_hallucinated"]["not_hallucinated"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_plots(scores):
    FIGURES.mkdir(parents=True, exist_ok=True)
    values = [scores["precision"], scores["recall"], scores["f1"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Precision", "Recall", "F1"], [0 if value is None else value for value in values], color="#3498db")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Detector vs manual labels (n shown in label_metrics.json)\n20-question pilot")
    fig.tight_layout()
    fig.savefig(FIGURES / "label_prf.png", dpi=150)
    plt.close(fig)

    matrix = scores["matrix_actual_by_predicted"]
    grid = np.array([
        [matrix["not_hallucinated"]["not_hallucinated"], matrix["not_hallucinated"]["hallucinated"]],
        [matrix["hallucinated"]["not_hallucinated"], matrix["hallucinated"]["hallucinated"]],
    ])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(grid, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted not hallucinated", "Predicted hallucinated"])
    ax.set_yticks([0, 1], ["Labeled not hallucinated", "Labeled hallucinated"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(grid[i, j]), ha="center", va="center")
    ax.set_title("Confusion matrix, Uncertain labels excluded\n20-question pilot")
    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(FIGURES / "label_confusion.png", dpi=150)
    plt.close(fig)


def main():
    if not LABELS.exists() or not KEY.exists():
        write_missing()
        return 0
    labels = read_csv(LABELS)
    key = {row["item_id"]: row for row in read_csv(KEY)}
    if not labels or any(row.get("label", "").strip() not in ALLOWED for row in labels):
        write_missing()
        return 0
    joined = []
    for row in labels:
        info = key[row["item_id"]]
        joined.append({**row, **info})
    student_rows = [row for row in joined if row["answer_role"] == "student_with_retrieval"]
    uncertain_count = sum(row["label"] == UNCERTAIN for row in student_rows)
    scored = [row for row in student_rows if row["label"] != UNCERTAIN]
    detector = confusion(scored)

    def hallucinated_share(role):
        chosen = [row for row in joined if row["answer_role"] == role]
        count = sum(row["label"] == POSITIVE for row in chosen)
        by_dataset = {}
        for dataset in ("pubmed", "healthcaremagic"):
            subset = [row for row in chosen if row["dataset"] == dataset]
            hits = sum(row["label"] == POSITIVE for row in subset)
            by_dataset[dataset] = {"hallucinated": hits, "n": len(subset)}
        return {"hallucinated": count, "n": len(chosen), "by_dataset": by_dataset}

    e_rows = [row for row in joined if row["answer_role"] == "e_final"]
    uncertainty_text = "The available evidence is insufficient to answer this reliably."
    e_abstained = [row for row in e_rows if row["answer_to_label"] == uncertainty_text]
    labeled_uncertain = sum(row["label"] == UNCERTAIN for row in e_abstained)
    payload = {
        "status": "calculated",
        "pilot_note": "20 test questions is a small pilot. These scores are not a significance test.",
        "detector_vs_student_retrieval_labels": {
            **detector,
            "uncertain_labels_excluded": uncertain_count,
            "n_student_answers": len(student_rows),
            "n_in_matrix": len(scored),
        },
        "hallucinated_final_answers": {
            "D": hallucinated_share("d_final"),
            "E": hallucinated_share("e_final"),
        },
        "e_uncertainty_statements_labeled_uncertain": {
            "uncertainty_statements": len(e_abstained),
            "labeled_uncertain_insufficient": labeled_uncertain,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_plots(detector)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
