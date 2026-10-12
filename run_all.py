"""Rebuild tables and figures from saved experiment files.

Does not load the student or teacher models and does not change thresholds.
Commands that would rerun the models are documented in the README only.
"""

from pathlib import Path
import csv
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

CONFIGS = [
    ("A", "A_student_only"),
    ("B", "B_teacher_only"),
    ("C", "C_student_retrieval"),
    ("D", "D_baseline_mitigation"),
    ("E-nov", "E-nov"),
    ("E", "E"),
]


def wilson(successes, n, z=1.96):
    """95% Wilson interval for a proportion. Returns (low, high)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n * n)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def read_csv(name):
    with (OUTPUTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ad_block(metrics, config_key, split):
    block = metrics["results"][config_key][split]["all"]
    pubmed = metrics["results"][config_key][split]["pubmed"]
    hcm = metrics["results"][config_key][split]["healthcaremagic"]
    wins = block["similarity_pass_count"]
    n = block["n"]
    low, high = wilson(wins, n)
    return {
        "wins": wins,
        "n": n,
        "similarity_pass_rate": block["similarity_pass_rate"],
        "wilson_low": low,
        "wilson_high": high,
        "mean_similarity": block["mean_similarity_to_reference"],
        "mean_latency_seconds": block["mean_latency_seconds"],
        "detector_flags": block["detector_flags"],
        "pubmed_wins": pubmed["similarity_pass_count"],
        "pubmed_n": pubmed["n"],
        "pubmed_rate": pubmed["similarity_pass_rate"],
        "pubmed_wilson": wilson(pubmed["similarity_pass_count"], pubmed["n"]),
        "hcm_wins": hcm["similarity_pass_count"],
        "hcm_n": hcm["n"],
        "hcm_rate": hcm["similarity_pass_rate"],
        "hcm_wilson": wilson(hcm["similarity_pass_count"], hcm["n"]),
    }


def e_block(metrics, key):
    block = metrics["results"][key]["all"]
    pubmed = metrics["results"][key]["pubmed"]
    hcm = metrics["results"][key]["healthcaremagic"]
    wins = block["similarity_pass_count_uncertainty_not_a_win"]
    n = block["n"]
    low, high = wilson(wins, n)
    return {
        "wins": wins,
        "n": n,
        "similarity_pass_rate": block["similarity_pass_rate_including_uncertainty_in_denominator"],
        "wilson_low": low,
        "wilson_high": high,
        "mean_similarity": block["mean_similarity_to_reference_all"],
        "mean_latency_seconds": block["mean_latency_seconds"],
        "detector_flags": None,
        "uncertain_count": block["uncertain_count"],
        "tier_counts": block["tier_counts"],
        "rate_excluding_uncertainty": block["similarity_pass_rate_excluding_uncertainty"],
        "n_excluding_uncertainty": block["n_excluding_uncertainty"],
        "pubmed_wins": pubmed["similarity_pass_count_uncertainty_not_a_win"],
        "pubmed_n": pubmed["n"],
        "pubmed_rate": pubmed["similarity_pass_rate_including_uncertainty_in_denominator"],
        "pubmed_wilson": wilson(pubmed["similarity_pass_count_uncertainty_not_a_win"], pubmed["n"]),
        "hcm_wins": hcm["similarity_pass_count_uncertainty_not_a_win"],
        "hcm_n": hcm["n"],
        "hcm_rate": hcm["similarity_pass_rate_including_uncertainty_in_denominator"],
        "hcm_wilson": wilson(hcm["similarity_pass_count_uncertainty_not_a_win"], hcm["n"]),
    }


def collect():
    ad = json.loads((OUTPUTS / "ad_val20_test20_metrics.json").read_text())
    e = json.loads((OUTPUTS / "e_val20_test20_metrics.json").read_text())
    reproduced = json.loads((OUTPUTS / "baseline_metrics.json").read_text())
    table = {"validation": {}, "test": {}}
    for split, e_key_nov, e_key in (
        ("validation", "E-nov_validation", "E_validation_chosen_cutoff"),
        ("test", "E-nov_test", "E_test"),
    ):
        for label, key in CONFIGS[:4]:
            table[split][label] = ad_block(ad, key, split)
        table[split]["E-nov"] = e_block(e, e_key_nov)
        table[split]["E"] = e_block(e, e_key)
    return table, reproduced, e


def plot_comparison(table):
    for split in ("validation", "test"):
        labels = [name for name, _ in CONFIGS]
        rates = [table[split][name]["similarity_pass_rate"] for name in labels]
        lows = [table[split][name]["similarity_pass_rate"] - table[split][name]["wilson_low"] for name in labels]
        highs = [table[split][name]["wilson_high"] - table[split][name]["similarity_pass_rate"] for name in labels]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, rates, color="#3498db", yerr=[lows, highs], capsize=4)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Similarity pass rate")
        ax.set_xlabel("Configuration")
        ax.set_title(f"Similarity pass rate by configuration ({split})\n95% Wilson intervals; not factual accuracy")
        fig.tight_layout()
        fig.savefig(FIGURES / f"comparison_{split}.png", dpi=150)
        plt.close(fig)


def plot_dataset(table):
    for split in ("validation", "test"):
        labels = [name for name, _ in CONFIGS]
        x = np.arange(len(labels))
        width = 0.36
        pubmed = [table[split][name]["pubmed_rate"] for name in labels]
        hcm = [table[split][name]["hcm_rate"] for name in labels]
        pubmed_err = [
            [table[split][name]["pubmed_rate"] - table[split][name]["pubmed_wilson"][0] for name in labels],
            [table[split][name]["pubmed_wilson"][1] - table[split][name]["pubmed_rate"] for name in labels],
        ]
        hcm_err = [
            [table[split][name]["hcm_rate"] - table[split][name]["hcm_wilson"][0] for name in labels],
            [table[split][name]["hcm_wilson"][1] - table[split][name]["hcm_rate"] for name in labels],
        ]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(x - width / 2, pubmed, width, yerr=pubmed_err, capsize=3, label="PubMedQA", color="#3498db")
        ax.bar(x + width / 2, hcm, width, yerr=hcm_err, capsize=3, label="HealthCareMagic", color="#1abc9c")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Similarity pass rate")
        ax.set_xlabel("Configuration")
        ax.set_title(f"Dataset-wise similarity pass rate ({split})\n95% Wilson intervals; not factual accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / f"dataset_{split}.png", dpi=150)
        plt.close(fig)


def plot_before_after(table):
    derived_path = OUTPUTS / "derived_e_student_similarity.json"
    derived = json.loads(derived_path.read_text()) if derived_path.exists() else {}
    ad_rows = read_csv("ad_val20_test20_predictions.csv")
    for split in ("validation", "test"):
        c_vals = [
            float(row["similarity_to_reference"])
            for row in ad_rows
            if row["split"] == split and row["configuration"] == "C_student_retrieval"
        ]
        d_vals = [
            float(row["similarity_to_reference"])
            for row in ad_rows
            if row["split"] == split and row["configuration"] == "D_baseline_mitigation"
        ]
        e_final = table[split]["E"]["mean_similarity"]
        e_student = derived.get(split, {}).get("mean_student_similarity_to_reference")
        labels = ["D student\n(with retrieval)", "D final", "E student", "E final"]
        values = [float(np.mean(c_vals)), float(np.mean(d_vals)), e_student, e_final]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        shown = [0 if value is None else value for value in values]
        ax.bar(labels, shown, color=["#3498db", "#2ecc71", "#9b59b6", "#16a085"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean similarity to reference text")
        ax.set_title(
            f"Similarity before and after mitigation ({split})\n"
            "Mean cosine similarity, not a similarity pass rate and not factual accuracy"
        )
        if e_student is None:
            ax.text(0.5, 0.9, "E student similarity file missing", transform=ax.transAxes, ha="center")
        fig.tight_layout()
        fig.savefig(FIGURES / f"similarity_before_after_{split}.png", dpi=150)
        plt.close(fig)


def plot_tiers(table):
    counts = table["test"]["E"]["tier_counts"]
    # Saved tiers use accept / corrected / uncertain.
    order = [("accept", "accept"), ("corrected", "corrected"), ("uncertain", "uncertain")]
    labels = [label for label, _ in order]
    values = [counts.get(key, 0) for _, key in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#2ecc71", "#3498db", "#e67e22"])
    ax.set_ylabel("Test questions")
    ax.set_xlabel("E tier")
    ax.set_title("E tier counts on the test split (n=20)\nNot a similarity pass rate")
    fig.tight_layout()
    fig.savefig(FIGURES / "tiers_test.png", dpi=150)
    plt.close(fig)


def plot_latency(table):
    for split in ("validation", "test"):
        labels = [name for name, _ in CONFIGS]
        values = [table[split][name]["mean_latency_seconds"] for name in labels]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values, color="#8e44ad")
        ax.set_ylabel("Mean latency (seconds)")
        ax.set_xlabel("Configuration")
        ax.set_title(f"Mean latency per configuration ({split})\nNot a similarity pass rate")
        fig.tight_layout()
        fig.savefig(FIGURES / f"latency_{split}.png", dpi=150)
        plt.close(fig)


def write_settings(e_metrics):
    settings = {
        "seed": 42,
        "validation_questions": 20,
        "test_questions": 20,
        "per_dataset_each_split": {"pubmed": 10, "healthcaremagic": 10},
        "models": {
            "student": "google/flan-t5-small",
            "teacher": "google/flan-t5-base",
            "embedder": "all-MiniLM-L6-v2",
        },
        "thresholds": {
            "retrieval_min_score": 0.25,
            "detection": 0.38,
            "blend": 0.55,
            "prefer_teacher": 0.70,
            "similarity_pass": 0.52,
        },
        "verify_cutoff_grid": [0.30, 0.40, 0.50, 0.60, 0.70],
        "chosen_verify_cutoff": e_metrics["chosen_verify_threshold"],
        "library_versions_from_reproduction": {
            "torch": "2.14.1+cpu",
            "transformers": "4.40.1",
            "sentence_transformers": "3.1.1",
            "datasets": "4.8.2",
        },
        "note": "Figures are rebuilt from saved outputs. This file does not record a new model run.",
    }
    (OUTPUTS / "run_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")


def write_summary(table, reproduced):
    payload = {
        "similarity_pass_rate_definition": "Share of answers with cosine similarity to the reference text strictly above 0.52. Not factual accuracy.",
        "reproduced_baseline": {
            "n": reproduced["n_questions"],
            "detector_flags": reproduced["detector_flags"],
            "mean_student_similarity": reproduced["mean_student_similarity"],
            "mean_teacher_similarity": reproduced["mean_teacher_similarity"],
            "mean_final_similarity": reproduced["mean_final_similarity"],
            "similarity_pass_rate": reproduced["similarity_pass_rate"],
            "pubmed_pass_rate": reproduced["pubmed_pass_rate"],
            "hcm_pass_rate": reproduced["hcm_pass_rate"],
            "threshold_failure_rate": reproduced["threshold_failure_rate"],
        },
        "comparison": table,
    }
    (OUTPUTS / "comparison_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    table, reproduced, e_metrics = collect()
    plot_comparison(table)
    plot_dataset(table)
    plot_before_after(table)
    plot_tiers(table)
    plot_latency(table)
    write_settings(e_metrics)
    write_summary(table, reproduced)
    from label_metrics import main as label_main
    label_main()
    print("Rebuilt figures and summary from saved outputs.")


if __name__ == "__main__":
    main()
