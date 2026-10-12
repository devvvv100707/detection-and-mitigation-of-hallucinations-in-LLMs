"""Build the blinded test-question label sheet from saved predictions.

Loads dataset rows only to recover question text for saved row indexes.
Does not run the student or teacher models and does not change thresholds.
"""

from pathlib import Path
import csv
import random
import sys

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEED = 42
ROLES = (
    ("student_with_retrieval", "C_student_retrieval"),
    ("d_final", "D_baseline_mitigation"),
    ("e_final", "E"),
)


def pubmed_lookup():
    rows = {}
    dataset = load_dataset("pubmed_qa", "pqa_labeled")["train"]
    for index, item in enumerate(dataset):
        rows[index] = {
            "question": item["question"],
            "reference": item["long_answer"],
            "final_decision": item.get("final_decision") or "",
        }
    return rows


def hcm_lookup():
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    dataset = dataset.filter(
        lambda x: x["input"] and x["output"]
        and len(x["input"].split()) > 5
        and len(x["output"].split()) > 20
    )
    rows = {}
    for index, item in enumerate(dataset):
        if len(item["output"].split()) < 20:
            continue
        rows[index] = {
            "question": item["input"].strip(),
            "reference": item["output"].strip(),
            "final_decision": "",
        }
    return rows


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    ad_rows = [
        row for row in read_csv(ROOT / "outputs/ad_val20_test20_predictions.csv")
        if row["split"] == "test"
    ]
    e_rows = [
        row for row in read_csv(ROOT / "outputs/e_val20_test20_predictions.csv")
        if row["split"] == "test" and row["configuration"] == "E" and row["verify_threshold"] == "0.3"
    ]
    by_question = {}
    for row in ad_rows:
        by_question.setdefault(row["question_id"], {})[row["configuration"]] = row
    for row in e_rows:
        by_question.setdefault(row["question_id"], {})["E"] = row

    lookups = {"pubmed": pubmed_lookup(), "healthcaremagic": hcm_lookup()}
    sheet = []
    for question_id, configs in sorted(by_question.items()):
        sample = configs["D_baseline_mitigation"]
        source = sample["dataset"]
        looked = lookups[source][int(sample["row_index"])]
        for role, config_name in ROLES:
            saved = configs[config_name]
            sheet.append({
                "question_id": question_id,
                "dataset": source,
                "question": looked["question"],
                "reference_answer": looked["reference"],
                "pubmedqa_final_decision": looked["final_decision"] if source == "pubmed" else "",
                "answer_to_label": saved["answer"],
                "answer_role": role,
                "detector_flag": sample["detector_flag"] if role == "student_with_retrieval" else "",
            })

    random.Random(SEED).shuffle(sheet)
    label_path = ROOT / "annotations/manual_labels.csv"
    key_path = ROOT / "annotations/answer_key.csv"
    label_fields = [
        "item_id", "dataset", "question", "reference_answer",
        "pubmedqa_final_decision", "answer_to_label", "label",
    ]
    key_fields = ["item_id", "question_id", "dataset", "answer_role", "detector_flag"]
    with label_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=label_fields)
        writer.writeheader()
        for number, row in enumerate(sheet, start=1):
            writer.writerow({
                "item_id": f"item_{number:03d}",
                "dataset": row["dataset"],
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "pubmedqa_final_decision": row["pubmedqa_final_decision"],
                "answer_to_label": row["answer_to_label"],
                "label": "",
            })
    with key_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=key_fields)
        writer.writeheader()
        for number, row in enumerate(sheet, start=1):
            writer.writerow({
                "item_id": f"item_{number:03d}",
                "question_id": row["question_id"],
                "dataset": row["dataset"],
                "answer_role": row["answer_role"],
                "detector_flag": row["detector_flag"],
            })
    print(f"Wrote {len(sheet)} blinded rows")


if __name__ == "__main__":
    main()
