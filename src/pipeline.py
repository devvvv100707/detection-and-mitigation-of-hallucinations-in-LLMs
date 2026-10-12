"""Student–teacher retrieval pipeline.

Baseline mode follows main.py: same prompts, thresholds, and mitigation
branches. Improved mitigation is not implemented here.
"""

from dataclasses import dataclass
import random
import re

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch


@dataclass
class PipelineConfig:
    """Runtime settings.

    In mode "baseline", data selection ignores sample_size and seed and uses
    the same index slices as main.py. Threshold defaults match main.py.
    """

    mode: str = "baseline"
    sample_size: int = 20
    seed: int = 42
    retrieval_min_score: float = 0.25
    top_k: int = 5
    detection_threshold: float = 0.38
    blend_threshold: float = 0.55
    prefer_teacher_threshold: float = 0.70
    pubmed_pass_threshold: float = 0.52
    hcm_pass_threshold: float = 0.52
    student_model_name: str = "google/flan-t5-small"
    teacher_model_name: str = "google/flan-t5-base"
    embedder_name: str = "all-MiniLM-L6-v2"
    student_max_new_tokens: int = 180
    teacher_max_new_tokens: int = 200
    kb_size_per_source: int = 150
    test_size_per_source: int = 10
    val_size: int = 20
    test_size: int = 20


def select_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device_id = 0 if device.type in ("mps", "cuda") else -1
    return device, device_id


def load_sources(config: PipelineConfig):
    """Load the same PubMedQA and HealthCareMagic slices as main.py.

    Baseline mode always returns 10 + 10 test questions. sample_size is stored
    for later experiments and is not applied in this mode.
    """
    if config.mode != "baseline":
        raise ValueError(
            "Only mode='baseline' is implemented. Improved mode is not part of this step."
        )

    pubmed_dataset = load_dataset("pubmed_qa", "pqa_labeled")
    pubmed_train = pubmed_dataset["train"].select(range(config.kb_size_per_source))
    pubmed_test = pubmed_dataset["train"].select(
        range(config.kb_size_per_source, config.kb_size_per_source + config.test_size_per_source)
    )

    hcm_dataset = load_dataset(
        "lavita/ChatDoctor-HealthCareMagic-100k",
        split="train",
    )
    hcm_dataset = hcm_dataset.filter(
        lambda x: x["input"] and x["output"]
        and len(x["input"].split()) > 5
        and len(x["output"].split()) > 20
    )
    hcm_train = hcm_dataset.select(range(config.kb_size_per_source))
    hcm_test = hcm_dataset.select(
        range(config.kb_size_per_source, config.kb_size_per_source + config.test_size_per_source)
    )

    knowledge_data = []
    for item in pubmed_train:
        knowledge_data.append({
            "question": item["question"],
            "context": " ".join(item["context"]["contexts"])[:800],
            "answer": item["long_answer"],
            "source": "pubmed",
        })
    for item in hcm_train:
        if len(item["output"].split()) >= 20:
            knowledge_data.append({
                "question": item["input"].strip(),
                "context": item["input"].strip()[:800],
                "answer": item["output"].strip()[:800],
                "source": "healthcaremagic",
            })

    test_data = []
    for item in pubmed_test:
        test_data.append({
            "question": item["question"],
            "true_answer": item["long_answer"],
            "source": "pubmed",
        })
    for item in hcm_test:
        test_data.append({
            "question": item["input"].strip(),
            "true_answer": item["output"].strip(),
            "source": "healthcaremagic",
        })
    return knowledge_data, test_data


class BaselinePipeline:
    """Encapsulates main.py retrieval, generation, detection, and mitigation."""

    def __init__(self, config: PipelineConfig, knowledge_data):
        self.config = config
        self.knowledge_data = knowledge_data
        _, device_id = select_device()
        self.student_model = pipeline(
            "text2text-generation",
            model=config.student_model_name,
            device=device_id,
            max_new_tokens=config.student_max_new_tokens,
        )
        self.teacher_model = pipeline(
            "text2text-generation",
            model=config.teacher_model_name,
            device=device_id,
            max_new_tokens=config.teacher_max_new_tokens,
        )
        self.embedder = SentenceTransformer(config.embedder_name)
        kb_questions = [d["question"] for d in knowledge_data]
        self.kb_embeddings = self.embedder.encode(
            kb_questions,
            convert_to_tensor=True,
            batch_size=64,
            show_progress_bar=False,
        )

    def retrieve_context(self, query, top_k=None):
        top_k = self.config.top_k if top_k is None else top_k
        q_emb = self.embedder.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self.kb_embeddings)[0]
        top_scores, top_idx = torch.topk(scores, k=min(top_k, len(self.knowledge_data)))

        parts = []
        for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
            if score > self.config.retrieval_min_score:
                entry = self.knowledge_data[idx]
                parts.append(
                    f"[{entry['source'].upper()}] "
                    f"Q: {entry['question']}\n"
                    f"Context: {entry['context'][:300]}\n"
                    f"A: {entry['answer']}"
                )
        return "\n---\n".join(parts)[:2000] if parts else ""

    def generate_answer(self, model, query, context, source_hint="general"):
        """Generate an answer with source-aware prompting."""
        if source_hint == "pubmed":
            domain_note = (
                "You are a medical researcher. "
                "Base your answer on clinical evidence and research findings."
            )
        else:
            domain_note = (
                "You are a clinical doctor answering a patient's question. "
                "Give a practical, clear explanation."
            )

        if context:
            prompt = (
                f"{domain_note}\n\n"
                f"Reference cases from the knowledge base:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Think step by step. Provide a detailed, accurate medical answer "
                f"using the reference cases above:"
            )
        else:
            prompt = (
                f"{domain_note}\n\n"
                f"Question: {query}\n\n"
                f"Think step by step. Provide a detailed, accurate medical answer:"
            )

        prompt_tokens = len(model.tokenizer(prompt, add_special_tokens=True)["input_ids"])
        truncated = prompt_tokens > model.tokenizer.model_max_length
        result = model(prompt, do_sample=False)
        ans = result[0]["generated_text"].strip()
        ans = re.sub(r'\b(\w+)( \1\b)+', r'\1', ans)
        fallback_used = False
        if len(ans.split()) < 8 and context:
            lines = [line for line in context.split("\n") if line.startswith("A:")]
            if lines:
                ans = lines[0].replace("A:", "").strip()
                fallback_used = True
        return ans, fallback_used, truncated, prompt_tokens

    def similarity_score(self, a, b):
        e1 = self.embedder.encode(a, convert_to_tensor=True)
        e2 = self.embedder.encode(b, convert_to_tensor=True)
        return max(0.0, util.cos_sim(e1, e2).item())

    def detect_hallucination(self, student_ans, teacher_ans, context, thr=None):
        thr = self.config.detection_threshold if thr is None else thr
        st_sim = self.similarity_score(student_ans, teacher_ans)
        ctx_sim = self.similarity_score(student_ans, context) if context else 0.0
        too_short = len(student_ans.split()) < 10
        is_hallucinating = (st_sim < thr and ctx_sim < thr) or too_short
        return {
            "is_hallucinating": is_hallucinating,
            "student_teacher_sim": st_sim,
            "student_context_sim": ctx_sim,
            "too_short": too_short,
        }

    def mitigate(self, student_ans, teacher_ans, info):
        st_sim = info["student_teacher_sim"]
        if info["is_hallucinating"]:
            return teacher_ans, "Hallucination detected → Replaced with Teacher"
        elif st_sim < self.config.blend_threshold:
            teacher_words = teacher_ans.split()[:20]
            student_words = student_ans.split()
            blended = " ".join(teacher_words) + ". " + " ".join(student_words[:60])
            return blended, "Low confidence → Blended Student + Teacher"
        elif st_sim < self.config.prefer_teacher_threshold:
            return teacher_ans, "Moderate confidence → Preferred Teacher"
        else:
            return student_ans, "High confidence → Student answer accepted"

    def full_pipeline(self, query, source_hint="general"):
        context = self.retrieve_context(query)
        student_ans, student_fallback, student_truncated, student_tokens = self.generate_answer(
            self.student_model, query, context, source_hint
        )
        teacher_ans, teacher_fallback, teacher_truncated, teacher_tokens = self.generate_answer(
            self.teacher_model, query, context, source_hint
        )
        info = self.detect_hallucination(student_ans, teacher_ans, context)
        final, msg = self.mitigate(student_ans, teacher_ans, info)
        return {
            "context": context,
            "student_ans": student_ans,
            "teacher_ans": teacher_ans,
            "final_ans": final,
            "mitigation_msg": msg,
            "hallucination": info,
            "student_fallback_used": student_fallback,
            "teacher_fallback_used": teacher_fallback,
            "student_prompt_truncated": student_truncated,
            "teacher_prompt_truncated": teacher_truncated,
            "student_prompt_tokens": student_tokens,
            "teacher_prompt_tokens": teacher_tokens,
        }


def normalize_question(text):
    return " ".join(str(text).lower().split())


def _dedupe(rows):
    seen = set()
    kept = []
    dropped = 0
    for row in rows:
        key = normalize_question(row["question"])
        if not key or key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, dropped


def _load_pool():
    """Load both datasets and drop duplicate question text."""
    pubmed = load_dataset("pubmed_qa", "pqa_labeled")["train"]
    pubmed_rows = []
    for index, item in enumerate(pubmed):
        pubmed_rows.append({
            "question": item["question"],
            "true_answer": item["long_answer"],
            "kb_context": " ".join(item["context"]["contexts"])[:800],
            "kb_answer": item["long_answer"],
            "source": "pubmed",
            "row_index": index,
        })
    hcm = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    hcm = hcm.filter(
        lambda x: x["input"] and x["output"]
        and len(x["input"].split()) > 5
        and len(x["output"].split()) > 20
    )
    hcm_rows = []
    for index, item in enumerate(hcm):
        if len(item["output"].split()) < 20:
            continue
        hcm_rows.append({
            "question": item["input"].strip(),
            "true_answer": item["output"].strip(),
            "kb_context": item["input"].strip()[:800],
            "kb_answer": item["output"].strip()[:800],
            "source": "healthcaremagic",
            "row_index": index,
        })
    pubmed_rows, pubmed_dropped = _dedupe(pubmed_rows)
    hcm_rows, hcm_dropped = _dedupe(hcm_rows)
    return pubmed_rows, hcm_rows, pubmed_dropped, hcm_dropped


def build_experiment_splits(config: PipelineConfig):
    """Seeded validation/test split whose questions are absent from the KB.

    This is not the original adjacent-slice split. main.py uses PubMedQA
    train[0:150] as the knowledge base and train[150:160] as the only test
    slice, with no shuffle and no validation split. The same pattern is used
    after filtering HealthCareMagic. This function shuffles with config.seed,
    holds out val_size validation questions and test_size test questions.
    Each of those sizes is split evenly across the two sources. The knowledge
    base is built only from other rows.
    """
    if config.val_size < 2 or config.val_size % 2 != 0 or config.test_size < 2 or config.test_size % 2 != 0:
        raise ValueError("val_size and test_size must be even so each source contributes the same count")
    pubmed_rows, hcm_rows, pubmed_dropped, hcm_dropped = _load_pool()
    rng = random.Random(config.seed)
    val_per_source = config.val_size // 2
    test_per_source = config.test_size // 2
    per_source = val_per_source + test_per_source
    heldout = []
    knowledge = []
    for source_rows in (pubmed_rows, hcm_rows):
        order = list(range(len(source_rows)))
        rng.shuffle(order)
        chosen = order[:per_source]
        rest = order[per_source:]
        if len(chosen) < per_source:
            raise ValueError("Not enough unique questions for the requested validation and test sizes")
        for position, index in enumerate(chosen):
            row = dict(source_rows[index])
            row["split"] = "validation" if position < val_per_source else "test"
            heldout.append(row)
        kb_added = 0
        for index in rest:
            if kb_added >= config.kb_size_per_source:
                break
            knowledge.append(source_rows[index])
            kb_added += 1

    held_keys = {normalize_question(row["question"]) for row in heldout}
    knowledge = [row for row in knowledge if normalize_question(row["question"]) not in held_keys]
    kb_keys = {normalize_question(row["question"]) for row in knowledge}
    leaked = held_keys & kb_keys
    print(f"Knowledge-base leakage overlaps: {len(leaked)}")
    if leaked:
        raise RuntimeError(f"Knowledge base contains {len(leaked)} validation or test questions")

    knowledge_data = [{
        "question": row["question"],
        "context": row["kb_context"],
        "answer": row["kb_answer"],
        "source": row["source"],
    } for row in knowledge]
    questions = []
    for number, row in enumerate(heldout, start=1):
        questions.append({
            "question_id": f"{row['source']}_{row['split']}_{row['row_index']}",
            "split": row["split"],
            "source": row["source"],
            "row_index": row["row_index"],
            "question": row["question"],
            "true_answer": row["true_answer"],
        })
    meta = {
        "seed": config.seed,
        "val_size": config.val_size,
        "test_size": config.test_size,
        "duplicates_removed": {"pubmed": pubmed_dropped, "healthcaremagic": hcm_dropped},
        "n_validation": sum(row["split"] == "validation" for row in questions),
        "n_test": sum(row["split"] == "test" for row in questions),
        "n_knowledge": len(knowledge_data),
        "leakage_count": 0,
        "split_difference": (
            "Original main.py split is unshuffled and has no validation set: "
            "knowledge base indices 0-149 and test indices 150-159 within each source. "
            "This split shuffles with the configured seed, holds out val_size validation questions "
            "and test_size test questions (each split evenly across the two sources), "
            "and excludes those questions from the knowledge base."
        ),
    }
    return knowledge_data, questions, meta
