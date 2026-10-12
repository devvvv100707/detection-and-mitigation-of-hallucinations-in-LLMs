"""Score saved E student answers against reference text.

This is not a new experiment. It only embeds answers that are already in
outputs/e_val20_test20_predictions.csv. The result is stored so run_all.py
can draw the before/after figure without loading a model again.
"""

from pathlib import Path
import csv
import json
import sys

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_annotations import hcm_lookup, pubmed_lookup


def main():
    rows = list(csv.DictReader((ROOT / "outputs/e_val20_test20_predictions.csv").open(encoding="utf-8")))
    chosen = [
        row for row in rows
        if row["configuration"] == "E" and row["verify_threshold"] == "0.3"
        and row["split"] in ("validation", "test")
    ]
    lookups = {"pubmed": pubmed_lookup(), "healthcaremagic": hcm_lookup()}
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    grouped = {"validation": [], "test": []}
    per_question = []
    for row in chosen:
        reference = lookups[row["dataset"]][int(row["row_index"])]["reference"]
        e1 = embedder.encode(row["student_answer"], convert_to_tensor=True)
        e2 = embedder.encode(reference, convert_to_tensor=True)
        score = max(0.0, util.cos_sim(e1, e2).item())
        grouped[row["split"]].append(score)
        per_question.append({
            "question_id": row["question_id"],
            "split": row["split"],
            "student_similarity_to_reference": score,
            "final_similarity_to_reference": float(row["similarity_to_reference"]),
        })
    payload = {
        "note": "MiniLM cosine similarity of the saved E student answer to the dataset reference. Not a new generation run.",
        "validation": {"mean_student_similarity_to_reference": sum(grouped["validation"]) / len(grouped["validation"])},
        "test": {"mean_student_similarity_to_reference": sum(grouped["test"]) / len(grouped["test"])},
        "per_question": per_question,
    }
    path = ROOT / "outputs/derived_e_student_similarity.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(payload["validation"], payload["test"])


if __name__ == "__main__":
    main()
