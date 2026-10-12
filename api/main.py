"""HTTP API for the saved experiments and configuration E.

Models load once in a background thread. A failed load leaves the server up
and /api/health reports the error.
"""

from pathlib import Path
import json
import sys
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import BaselinePipeline, PipelineConfig, build_experiment_splits

FIGURES = ROOT / "outputs" / "figures"
FIGURE_NAMES = {
    "comparison_validation.png",
    "comparison_test.png",
    "dataset_validation.png",
    "dataset_test.png",
    "similarity_before_after_validation.png",
    "similarity_before_after_test.png",
    "tiers_test.png",
    "latency_validation.png",
    "latency_test.png",
    "label_prf.png",
    "label_confusion.png",
}
TIER_NAMES = {"accept": "accepted", "corrected": "revised", "uncertain": "uncertain"}
VERIFY_CUTOFF = 0.30

app = FastAPI(title="HallucinationGuard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

STATE = {"ready": False, "error": None, "pipe": None, "example_question": None}


class QuestionIn(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def check_question(cls, value):
        text = value.strip()
        if not text:
            raise ValueError("Question is empty.")
        if len(text) > 500:
            raise ValueError("Question must be at most 500 characters.")
        return text


def _load_models():
    try:
        config = PipelineConfig(mode="baseline", val_size=20, test_size=20, seed=42)
        knowledge, questions, _meta = build_experiment_splits(config)
        STATE["pipe"] = BaselinePipeline(config, knowledge)
        test_questions = [item["question"] for item in questions if item["split"] == "test"]
        STATE["example_question"] = test_questions[0] if test_questions else None
        STATE["ready"] = True
        STATE["error"] = None
    except Exception as exc:
        STATE["ready"] = False
        STATE["error"] = f"Model loading failed: {exc}"


@app.on_event("startup")
def startup():
    threading.Thread(target=_load_models, daemon=True).start()


@app.get("/api/health")
def health():
    return {"ready": STATE["ready"], "error": STATE["error"]}


def _reasons(result, evidence, cutoff):
    student_score = result["student_support_similarity"]
    reasons = []
    if evidence is None:
        reasons.append("No knowledge-base row was available.")
    else:
        reasons.append(
            f"Top retrieved evidence came from {evidence['source']} "
            f"with similarity {evidence['similarity']:.3f}. "
            f"The retrieval cutoff is {STATE['pipe'].config.retrieval_min_score}."
        )
        if not evidence["included"]:
            reasons.append("That row was below the retrieval cutoff, so the prompt had no evidence.")
    if student_score is None:
        reasons.append("The student answer had no evidence text to score.")
    else:
        reasons.append(
            f"Student evidence similarity was {float(student_score):.3f} "
            f"against the verify cutoff {cutoff:.2f} "
            f"({result['student_verification']})."
        )
    if result["tier"] == "accept":
        reasons.append("The student answer was Supported, so the accept tier kept it.")
    elif result["tier"] == "corrected":
        teacher_score = result["teacher_support_similarity"]
        reasons.append(
            f"The student answer was not Supported. Teacher evidence similarity was "
            f"{float(teacher_score):.3f} ({result['teacher_verification']})."
        )
        reasons.append("The teacher answer was Supported, so the corrected tier used it.")
    else:
        if result["teacher_verification"]:
            teacher_score = result["teacher_support_similarity"]
            shown = "none" if teacher_score is None else f"{float(teacher_score):.3f}"
            reasons.append(
                f"The teacher answer was {result['teacher_verification']} "
                f"(evidence similarity {shown})."
            )
        reasons.append("Neither answer was Supported, so the uncertain tier returned the fixed warning.")
    if result["fallback_suppressed"]:
        reasons.append("A short answer would have copied retrieved text. That copy was not used.")
    return reasons


@app.post("/api/analyze")
def analyze(body: QuestionIn):
    if not STATE["ready"] or STATE["pipe"] is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "Models are still loading.")
    pipe = STATE["pipe"]
    evidence = pipe.top_evidence(body.question)
    result = pipe.run_e(body.question, source_hint="general", threshold=VERIFY_CUTOFF)
    teacher_answer = result["teacher_answer"] or None
    student_teacher = None
    if teacher_answer:
        student_teacher = pipe.similarity_score(result["student_answer"], teacher_answer)
    tier = TIER_NAMES[result["tier"]]
    return {
        "student_answer": result["student_answer"],
        "teacher_answer": teacher_answer,
        "final_answer": result["final_answer"],
        "tier": tier,
        "student_verification": result["student_verification"],
        "teacher_verification": result["teacher_verification"] or None,
        "similarities": {
            "student_teacher": student_teacher,
            "student_evidence": result["student_support_similarity"],
            "teacher_evidence": result["teacher_support_similarity"] if result["teacher_support_similarity"] != "" else None,
        },
        "evidence": None if evidence is None else {
            "text": evidence["text"],
            "source": evidence["source"],
            "similarity": evidence["similarity"],
        },
        "reasons": _reasons(result, evidence, VERIFY_CUTOFF),
        "latency": result["step_latency"],
        "settings": {"verify_cutoff": VERIFY_CUTOFF, "top_k": pipe.config.top_k},
    }


@app.get("/api/results")
def results():
    summary_path = ROOT / "outputs" / "comparison_summary.json"
    labels_path = ROOT / "outputs" / "label_metrics.json"
    original_path = ROOT / "baselines" / "baseline_original.json"
    if not summary_path.exists() or not original_path.exists():
        raise HTTPException(status_code=503, detail="Saved result files are missing. Run run_all.py first.")
    summary = json.loads(summary_path.read_text())
    original = json.loads(original_path.read_text())
    label_metrics = {"status": "labels_missing"}
    if labels_path.exists():
        label_metrics = json.loads(labels_path.read_text())
    return {
        "example_question": STATE["example_question"],
        "original_experiment": original,
        "reproduced_baseline": summary["reproduced_baseline"],
        "comparison": summary["comparison"],
        "similarity_pass_rate_definition": summary["similarity_pass_rate_definition"],
        "label_metrics": label_metrics,
        "figures": sorted(name for name in FIGURE_NAMES if (FIGURES / name).exists()),
    }


@app.get("/api/figures/{name}")
def figure(name: str):
    if name not in FIGURE_NAMES or "/" in name or ".." in name:
        raise HTTPException(status_code=404, detail="Figure not found.")
    path = FIGURES / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Figure not found.")
    return FileResponse(path, media_type="image/png")
