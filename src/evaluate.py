"""Evaluate baseline mode and write new files under outputs/.

Does not write results.csv, hallucination_results.png, or eval_results.png.
"""

from pathlib import Path
import csv
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (
    BaselinePipeline,
    PipelineConfig,
    build_experiment_splits,
    load_sources,
    select_device,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

PRESERVED = {
    "n_questions": 20,
    "detector_flags": 5,
    "mean_student_similarity": 0.462,
    "mean_teacher_similarity": 0.569,
    "mean_final_similarity": 0.556,
    "similarity_pass_rate": 0.65,
    "pubmed_pass_rate": 0.70,
    "hcm_pass_rate": 0.60,
    "threshold_failure_rate": 0.35,
}


def plot_results(rows, metrics, path: Path):
    questions = [row["question_id"] for row in rows]
    s_sims = [row["student_similarity"] for row in rows]
    t_sims = [row["teacher_similarity"] for row in rows]
    f_sims = [row["final_similarity"] for row in rows]
    sources = [row["source"] for row in rows]

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(
        "Baseline-mode rerun (not the preserved figure)\n"
        "Similarity-threshold pass rate is not factual accuracy",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    labels = [
        "Similarity\npass rate",
        "Threshold\nfailure rate",
        "PubMedQA\npass rate",
        "HealthCareMagic\npass rate",
    ]
    values = [
        metrics["similarity_pass_rate"],
        metrics["threshold_failure_rate"],
        metrics["pubmed_pass_rate"],
        metrics["hcm_pass_rate"],
    ]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]
    bars = ax1.bar(labels, values, color=colors, width=0.45, edgecolor="white")
    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{val:.0%}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    ax1.set_ylim(0, 1.25)
    ax1.set_ylabel("Rate")
    ax1.set_title("Similarity-threshold rates (cutoff 0.52)")
    ax1.spines[["top", "right"]].set_visible(False)

    model_labels = ["Student", "Teacher", "Final"]
    means = [
        metrics["mean_student_similarity"],
        metrics["mean_teacher_similarity"],
        metrics["mean_final_similarity"],
    ]
    mbars = ax2.bar(model_labels, means, color=["#3498db", "#9b59b6", "#2ecc71"], width=0.45)
    for bar, val in zip(mbars, means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Cosine similarity to reference text")
    ax2.set_title("Mean similarity to reference text")
    ax2.text(
        0.97, 0.97,
        f"Detector flags:\n{metrics['detector_flags']}/{metrics['n_questions']}",
        transform=ax2.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeeba"),
    )
    ax2.spines[["top", "right"]].set_visible(False)

    x = np.arange(len(questions))
    width = 0.25
    ax3.bar(x - width, s_sims, width, label="Student", color="#3498db", alpha=0.85)
    ax3.bar(x, t_sims, width, label="Teacher", color="#9b59b6", alpha=0.85)
    ax3.bar(x + width, f_sims, width, label="Final", color="#2ecc71", alpha=0.85)
    ax3.axhline(0.52, color="#e67e22", linestyle="--", linewidth=1.2, label="Pass cutoff 0.52")
    ax3.set_xticks(x)
    ax3.set_xticklabels(questions, fontsize=8, rotation=45)
    ax3.set_ylim(0, 1.15)
    ax3.set_ylabel("Similarity to reference text")
    ax3.set_xlabel("Question")
    ax3.legend(fontsize=8)
    ax3.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate(config: PipelineConfig | None = None):
    config = config or PipelineConfig(mode="baseline")
    device, _ = select_device()
    knowledge_data, test_data = load_sources(config)
    pipe = BaselinePipeline(config, knowledge_data)

    rows = []
    for i, item in enumerate(test_data):
        started = time.perf_counter()
        result = pipe.full_pipeline(item["question"], source_hint=item["source"])
        elapsed = time.perf_counter() - started
        s_sim = pipe.similarity_score(result["student_ans"], item["true_answer"])
        t_sim = pipe.similarity_score(result["teacher_ans"], item["true_answer"])
        f_sim = pipe.similarity_score(result["final_ans"], item["true_answer"])
        cutoff = (
            config.pubmed_pass_threshold if item["source"] == "pubmed"
            else config.hcm_pass_threshold
        )
        rows.append({
            "question_id": f"Q{i + 1}",
            "source": item["source"],
            "question": item["question"],
            "true_answer": item["true_answer"],
            "student_answer": result["student_ans"],
            "teacher_answer": result["teacher_ans"],
            "final_answer": result["final_ans"],
            "mitigation_msg": result["mitigation_msg"],
            "detector_flag": bool(result["hallucination"]["is_hallucinating"]),
            "student_teacher_sim": result["hallucination"]["student_teacher_sim"],
            "student_context_sim": result["hallucination"]["student_context_sim"],
            "too_short": bool(result["hallucination"]["too_short"]),
            "student_similarity": s_sim,
            "teacher_similarity": t_sim,
            "final_similarity": f_sim,
            "passes_similarity_threshold": f_sim > cutoff,
            "latency_seconds": elapsed,
        })
        print(
            f"{rows[-1]['question_id']} {item['source']} "
            f"flag={rows[-1]['detector_flag']} "
            f"student={s_sim:.3f} teacher={t_sim:.3f} final={f_sim:.3f} "
            f"{result['mitigation_msg']}"
        )

    pubmed = [row for row in rows if row["source"] == "pubmed"]
    hcm = [row for row in rows if row["source"] == "healthcaremagic"]
    n = len(rows)
    pubmed_pass = sum(row["passes_similarity_threshold"] for row in pubmed) / len(pubmed)
    hcm_pass = sum(row["passes_similarity_threshold"] for row in hcm) / len(hcm)
    pass_rate = sum(row["passes_similarity_threshold"] for row in rows) / n
    metrics = {
        "mode": config.mode,
        "note": (
            "Fresh baseline-mode run. Not claimed to match hallucination_results.png. "
            "Similarity-threshold pass rate is not factual accuracy."
        ),
        "device": str(device),
        "n_questions": n,
        "n_knowledge": len(knowledge_data),
        "seed_recorded": config.seed,
        "seed_used_for_sampling": False,
        "sample_size_config": config.sample_size,
        "models": {
            "student": config.student_model_name,
            "teacher": config.teacher_model_name,
            "embedder": config.embedder_name,
        },
        "thresholds": {
            "retrieval_min_score": config.retrieval_min_score,
            "detection": config.detection_threshold,
            "blend": config.blend_threshold,
            "prefer_teacher": config.prefer_teacher_threshold,
            "pubmed_pass": config.pubmed_pass_threshold,
            "hcm_pass": config.hcm_pass_threshold,
        },
        "detector_flags": int(sum(row["detector_flag"] for row in rows)),
        "mean_student_similarity": float(np.mean([row["student_similarity"] for row in rows])),
        "mean_teacher_similarity": float(np.mean([row["teacher_similarity"] for row in rows])),
        "mean_final_similarity": float(np.mean([row["final_similarity"] for row in rows])),
        "similarity_pass_rate": float(pass_rate),
        "pubmed_pass_rate": float(pubmed_pass),
        "hcm_pass_rate": float(hcm_pass),
        "threshold_failure_rate": float(1 - pass_rate),
        "mean_latency_seconds": float(np.mean([row["latency_seconds"] for row in rows])),
    }
    metrics["difference_from_preserved_figure"] = {
        key: metrics[key] - PRESERVED[key] for key in PRESERVED
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT_DIR / "baseline_predictions.csv"
    metrics_path = OUTPUT_DIR / "baseline_metrics.json"
    plot_path = OUTPUT_DIR / "baseline_similarity.png"
    with pred_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_results(rows, metrics, plot_path)
    print(json.dumps(metrics, indent=2))
    return metrics


def _rates(rows, pubmed_cutoff, hcm_cutoff):
    def pack(subset):
        if not subset:
            return {"n": 0, "detector_flags": None, "similarity_pass_rate": None}
        flagged = [row["detector_flag"] for row in subset]
        if any(flag == "not_applicable" for flag in flagged):
            detector_flags = None
        else:
            detector_flags = int(sum(flag == "true" for flag in flagged))
        passed = 0
        for row in subset:
            cutoff = pubmed_cutoff if row["dataset"] == "pubmed" else hcm_cutoff
            if row["similarity_to_reference"] > cutoff:
                passed += 1
        without_fallback = [row for row in subset if row["fallback_used"] == "false"]
        passed_without_fallback = 0
        for row in without_fallback:
            cutoff = pubmed_cutoff if row["dataset"] == "pubmed" else hcm_cutoff
            if row["similarity_to_reference"] > cutoff:
                passed_without_fallback += 1
        return {
            "n": len(subset),
            "similarity_pass_count": passed,
            "detector_flags": detector_flags,
            "similarity_pass_rate": passed / len(subset),
            "mean_similarity_to_reference": float(np.mean([row["similarity_to_reference"] for row in subset])),
            "fallback_used_count": int(sum(row["fallback_used"] == "true" for row in subset)),
            "n_excluding_fallback": len(without_fallback),
            "similarity_pass_count_excluding_fallback": passed_without_fallback,
            "similarity_pass_rate_excluding_fallback": (
                passed_without_fallback / len(without_fallback) if without_fallback else None
            ),
            "prompt_truncation_count": int(sum(int(row["truncated_generations"]) for row in subset)),
            "mean_latency_seconds": float(np.mean([float(row["latency_seconds"]) for row in subset])),
        }

    return {
        "all": pack(rows),
        "pubmed": pack([row for row in rows if row["dataset"] == "pubmed"]),
        "healthcaremagic": pack([row for row in rows if row["dataset"] == "healthcaremagic"]),
    }


def run_configurations_ad(val_size=20, test_size=20, seed=42):
    """Run configs A-D on one seeded validation/test split. Config E is not run."""
    config = PipelineConfig(mode="baseline", val_size=val_size, test_size=test_size, seed=seed)
    knowledge_data, questions, split_meta = build_experiment_splits(config)
    pipe = BaselinePipeline(config, knowledge_data)
    rows = []
    for item in questions:
        query = item["question"]
        source = item["source"]
        truth = item["true_answer"]
        context = pipe.retrieve_context(query)

        started = time.perf_counter()
        answer_a, fallback_a, trunc_a, tokens_a = pipe.generate_answer(pipe.student_model, query, "", source)
        latency_a = time.perf_counter() - started

        started = time.perf_counter()
        answer_b, fallback_b, trunc_b, tokens_b = pipe.generate_answer(pipe.teacher_model, query, "", source)
        latency_b = time.perf_counter() - started

        started = time.perf_counter()
        answer_c, fallback_c, trunc_c, tokens_c = pipe.generate_answer(pipe.student_model, query, context, source)
        latency_c = time.perf_counter() - started

        started = time.perf_counter()
        result_d = pipe.full_pipeline(query, source_hint=source)
        latency_d = time.perf_counter() - started

        d_truncated = result_d["student_prompt_truncated"] or result_d["teacher_prompt_truncated"]
        d_tokens = max(result_d["student_prompt_tokens"], result_d["teacher_prompt_tokens"])
        d_truncations = int(result_d["student_prompt_truncated"]) + int(result_d["teacher_prompt_truncated"])
        outputs = [
            ("A_student_only", answer_a, fallback_a, latency_a, "not_applicable", trunc_a, tokens_a, int(trunc_a)),
            ("B_teacher_only", answer_b, fallback_b, latency_b, "not_applicable", trunc_b, tokens_b, int(trunc_b)),
            ("C_student_retrieval", answer_c, fallback_c, latency_c, "not_applicable", trunc_c, tokens_c, int(trunc_c)),
            (
                "D_baseline_mitigation",
                result_d["final_ans"],
                result_d["student_fallback_used"] or result_d["teacher_fallback_used"],
                latency_d,
                "true" if result_d["hallucination"]["is_hallucinating"] else "false",
                d_truncated,
                d_tokens,
                d_truncations,
            ),
        ]
        for name, answer, fallback, latency, flag, truncated, prompt_tokens, truncations in outputs:
            rows.append({
                "question_id": item["question_id"],
                "split": item["split"],
                "dataset": source,
                "row_index": item["row_index"],
                "configuration": name,
                "answer": answer,
                "similarity_to_reference": pipe.similarity_score(answer, truth),
                "detector_flag": flag,
                "fallback_used": "true" if fallback else "false",
                "prompt_truncated": "true" if truncated else "false",
                "prompt_tokens": prompt_tokens,
                "truncated_generations": truncations,
                "latency_seconds": latency,
            })
            print(
                f"{item['split']} {name} {item['question_id']} "
                f"sim={rows[-1]['similarity_to_reference']:.3f} flag={flag}"
            )

    by_config = {}
    for name in ("A_student_only", "B_teacher_only", "C_student_retrieval", "D_baseline_mitigation"):
        selected = [row for row in rows if row["configuration"] == name]
        by_config[name] = {
            "validation": _rates(
                [row for row in selected if row["split"] == "validation"],
                config.pubmed_pass_threshold,
                config.hcm_pass_threshold,
            ),
            "test": _rates(
                [row for row in selected if row["split"] == "test"],
                config.pubmed_pass_threshold,
                config.hcm_pass_threshold,
            ),
        }

    metrics = {
        "configurations_run": ["A_student_only", "B_teacher_only", "C_student_retrieval", "D_baseline_mitigation"],
        "configurations_not_run": ["E_improved"],
        "note": (
            "similarity_pass_rate is the share of answers with cosine similarity "
            "to the reference text strictly above 0.52. It is not factual accuracy. "
            "detector_flags is null when that configuration does not run the student-teacher detector. "
            "Thresholds were not tuned on these questions."
        ),
        "models": {
            "student": config.student_model_name,
            "teacher": config.teacher_model_name,
            "embedder": config.embedder_name,
        },
        "thresholds": {
            "retrieval_min_score": config.retrieval_min_score,
            "detection": config.detection_threshold,
            "blend": config.blend_threshold,
            "prefer_teacher": config.prefer_teacher_threshold,
            "similarity_pass": config.pubmed_pass_threshold,
        },
        "split": split_meta,
        "results": by_config,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT_DIR / f"ad_val{val_size}_test{test_size}_predictions.csv"
    metrics_path = OUTPUT_DIR / f"ad_val{val_size}_test{test_size}_metrics.json"
    with pred_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


UNCERTAIN_TEXT = "The available evidence is insufficient to answer this reliably."


def _e_row(item, result, similarity, latency, threshold):
    uncertain = result["tier"] == "uncertain" or result["final_answer"] == UNCERTAIN_TEXT
    cutoff = 0.52
    similarity_win = (not uncertain) and similarity > cutoff
    return {
        "question_id": item["question_id"],
        "split": item["split"],
        "dataset": item["source"],
        "row_index": item["row_index"],
        "configuration": result["configuration"],
        "verify_threshold": threshold if result["configuration"] == "E" else "",
        "student_answer": result["student_answer"],
        "teacher_answer": result["teacher_answer"],
        "student_verification": result["student_verification"],
        "teacher_verification": result["teacher_verification"],
        "student_support_similarity": result["student_support_similarity"],
        "teacher_support_similarity": result["teacher_support_similarity"],
        "tier": result["tier"],
        "answer": result["final_answer"],
        "similarity_to_reference": similarity,
        "similarity_win_excluding_uncertainty": similarity_win,
        "uncertain": uncertain,
        "fallback_used": result["fallback_used"],
        "fallback_suppressed": result["fallback_suppressed"],
        "prompt_truncated": result["prompt_truncated"],
        "truncated_generations": result["truncated_generations"],
        "prompt_tokens": result["prompt_tokens"],
        "latency_seconds": latency,
    }


def _summarize_e(rows):
    def pack(subset):
        if not subset:
            return {"n": 0}
        wins = [row for row in subset if row["similarity_win_excluding_uncertainty"]]
        uncertain = [row for row in subset if row["uncertain"]]
        answered = [row for row in subset if not row["uncertain"]]
        tiers = {}
        for row in subset:
            tiers[row["tier"]] = tiers.get(row["tier"], 0) + 1
        return {
            "n": len(subset),
            "tier_counts": tiers,
            "uncertain_count": len(uncertain),
            "similarity_pass_count_uncertainty_not_a_win": len(wins),
            "similarity_pass_rate_including_uncertainty_in_denominator": len(wins) / len(subset),
            "n_excluding_uncertainty": len(answered),
            "similarity_pass_count_excluding_uncertainty": len(wins),
            "similarity_pass_rate_excluding_uncertainty": (len(wins) / len(answered) if answered else None),
            "mean_similarity_to_reference_all": float(np.mean([row["similarity_to_reference"] for row in subset])),
            "mean_similarity_to_reference_excluding_uncertainty": (
                float(np.mean([row["similarity_to_reference"] for row in answered])) if answered else None
            ),
            "mean_latency_seconds": float(np.mean([row["latency_seconds"] for row in subset])),
            "truncated_generations": int(sum(int(row["truncated_generations"]) for row in subset)),
            "fallback_used_count": int(sum(bool(row["fallback_used"]) for row in subset)),
            "fallback_suppressed_count": int(sum(bool(row["fallback_suppressed"]) for row in subset)),
        }

    summary = {}
    for dataset in ("all", "pubmed", "healthcaremagic"):
        if dataset == "all":
            chosen = rows
        else:
            chosen = [row for row in rows if row["dataset"] == dataset]
        summary[dataset] = pack(chosen)
    return summary


def run_configuration_e(val_size=20, test_size=20, seed=42):
    """Choose the verify cutoff on validation, then run E and E-nov once on test.

    Selection rule, fixed before any test score is computed: on the validation
    split, pick the grid value with the most similarity wins. An uncertainty
    statement is never a win. The denominator is every validation question.
    If two cutoffs tie, keep the larger cutoff.
    """
    config = PipelineConfig(mode="baseline", val_size=val_size, test_size=test_size, seed=seed)
    grid = list(config.verify_grid)
    if len(grid) > 5:
        raise ValueError("verify grid must contain at most 5 values")
    knowledge_data, questions, split_meta = build_experiment_splits(config)
    validation = [item for item in questions if item["split"] == "validation"]
    test = [item for item in questions if item["split"] == "test"]
    pipe = BaselinePipeline(config, knowledge_data)
    rows = []

    print("Validation grid. Test questions are not scored in this loop.")
    print("Grid:", grid)
    grid_summary = []
    for threshold in grid:
        threshold_rows = []
        for item in validation:
            started = time.perf_counter()
            result = pipe.run_e(item["question"], item["source"], threshold=threshold)
            latency = time.perf_counter() - started
            similarity = pipe.similarity_score(result["final_answer"], item["true_answer"])
            row = _e_row(item, result, similarity, latency, threshold)
            threshold_rows.append(row)
            rows.append(row)
            print(
                f"val E thr={threshold} {item['question_id']} tier={result['tier']} "
                f"sim={similarity:.3f} trunc={result['truncated_generations']}"
            )
        summary = _summarize_e(threshold_rows)["all"]
        grid_summary.append({
            "threshold": threshold,
            "validation": summary,
        })
        print(
            "validation-only",
            threshold,
            "wins",
            summary["similarity_pass_count_uncertainty_not_a_win"],
            "uncertain",
            summary["uncertain_count"],
        )

    def sort_key(entry):
        stats = entry["validation"]
        return (stats["similarity_pass_count_uncertainty_not_a_win"], entry["threshold"])

    chosen = max(grid_summary, key=sort_key)
    chosen_threshold = chosen["threshold"]
    print(f"Chosen verify cutoff from validation only: {chosen_threshold}")

    for item in validation:
        started = time.perf_counter()
        result = pipe.run_enov(item["question"], item["source"])
        latency = time.perf_counter() - started
        similarity = pipe.similarity_score(result["final_answer"], item["true_answer"])
        rows.append(_e_row(item, result, similarity, latency, ""))
        print(f"val E-nov {item['question_id']} sim={similarity:.3f}")

    print("Test split starts after the cutoff was chosen.")
    for item in test:
        started = time.perf_counter()
        result = pipe.run_e(item["question"], item["source"], threshold=chosen_threshold)
        latency = time.perf_counter() - started
        similarity = pipe.similarity_score(result["final_answer"], item["true_answer"])
        rows.append(_e_row(item, result, similarity, latency, chosen_threshold))
        print(f"test E {item['question_id']} tier={result['tier']} sim={similarity:.3f}")

        started = time.perf_counter()
        result = pipe.run_enov(item["question"], item["source"])
        latency = time.perf_counter() - started
        similarity = pipe.similarity_score(result["final_answer"], item["true_answer"])
        rows.append(_e_row(item, result, similarity, latency, ""))
        print(f"test E-nov {item['question_id']} sim={similarity:.3f}")

    def subset(configuration, split_name, threshold=None):
        selected = [
            row for row in rows
            if row["configuration"] == configuration and row["split"] == split_name
        ]
        if threshold is not None:
            selected = [row for row in selected if float(row["verify_threshold"]) == float(threshold)]
        return selected

    metrics = {
        "configurations_run": ["E", "E-nov"],
        "context_cap": (
            "E and E-nov shorten retrieved context until the full prompt is within "
            "512 tokens. A-D still use the 2000-character context and can truncate."
        ),
        "selection_rule": (
            "On validation only, choose the grid value with the most similarity wins. "
            "Uncertainty statements are never wins. Denominator is all validation questions. "
            "Ties keep the larger cutoff. Test scores are not used."
        ),
        "verify_grid": grid,
        "validation_grid_results": grid_summary,
        "chosen_verify_threshold": chosen_threshold,
        "models": {
            "student": config.student_model_name,
            "teacher": config.teacher_model_name,
            "embedder": config.embedder_name,
        },
        "similarity_pass_cutoff": 0.52,
        "note": (
            "similarity pass rate is cosine similarity to the reference text above 0.52. "
            "It is not factual accuracy. Uncertainty statements are never counted as wins."
        ),
        "split": split_meta,
        "results": {
            "E_validation_chosen_cutoff": _summarize_e(subset("E", "validation", chosen_threshold)),
            "E_test": _summarize_e(subset("E", "test", chosen_threshold)),
            "E-nov_validation": _summarize_e(subset("E-nov", "validation")),
            "E-nov_test": _summarize_e(subset("E-nov", "test")),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT_DIR / f"e_val{val_size}_test{test_size}_predictions.csv"
    metrics_path = OUTPUT_DIR / f"e_val{val_size}_test{test_size}_metrics.json"
    with pred_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-ad", action="store_true")
    parser.add_argument("--config-e", action="store_true")
    parser.add_argument("--val-size", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=20)
    args = parser.parse_args()
    if args.config_e:
        run_configuration_e(val_size=args.val_size, test_size=args.test_size)
    elif args.configs_ad:
        run_configurations_ad(val_size=args.val_size, test_size=args.test_size)
    else:
        evaluate()
