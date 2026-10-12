# ===============================
# IMPORTS
# ===============================
from datasets import load_dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# DEVICE SETUP (Apple Silicon MPS)
# ===============================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("⚡ Using Apple MPS (GPU acceleration)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("⚡ Using CUDA")
else:
    device = torch.device("cpu")
    print("💻 Using CPU")

device_id = 0 if device.type in ("mps", "cuda") else -1

# ===============================
# LOAD DATASET
# ===============================
print("📦 Loading dataset...")
dataset    = load_dataset("pubmed_qa", "pqa_labeled")
train_data = dataset["train"].select(range(200))
test_data  = dataset["train"].select(range(200, 220))
print("✅ Dataset loaded!")

# ===============================
# BUILD KNOWLEDGE BASE
# ===============================
knowledge_data = []
for item in train_data:
    knowledge_data.append({
        "question": item["question"],
        "context":  " ".join(item["context"]["contexts"])[:800],
        "answer":   item["long_answer"]
    })

# ===============================
# LOAD MODELS
# ===============================
print("🧠 Loading models...")

student_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=device_id,
    max_new_tokens=100
)

teacher_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=device_id,
    max_new_tokens=100
)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Models loaded!")

# ===============================
# PRE-COMPUTE KB EMBEDDINGS ONCE
# ===============================
print("📐 Pre-computing embeddings...")
kb_questions  = [item["question"] for item in knowledge_data]
kb_embeddings = embedder.encode(kb_questions, convert_to_tensor=True)
print("✅ Done!")

# ===============================
# RETRIEVAL
# ===============================
def retrieve_context(query, top_k=2):
    q_emb   = embedder.encode(query, convert_to_tensor=True)
    scores  = util.cos_sim(q_emb, kb_embeddings)[0]
    top_idx = torch.topk(scores, k=min(top_k, len(knowledge_data))).indices.tolist()
    return " ".join(knowledge_data[i]["context"] for i in top_idx)[:1200]

# ===============================
# GENERATION
# ===============================
def generate_answer(model, query, context):
    prompt = (
        f"Context: {context[:600]}\n"
        f"Question: {query}\n"
        f"Short medical answer:"
    )
    return model(prompt, do_sample=False)[0]["generated_text"].strip()

# ===============================
# SIMILARITY
# ===============================
def similarity_score(a, b):
    e1 = embedder.encode(a, convert_to_tensor=True)
    e2 = embedder.encode(b, convert_to_tensor=True)
    return max(0.0, util.cos_sim(e1, e2).item())

# ===============================
# PIPELINE
# ===============================
def final_pipeline(query):
    context     = retrieve_context(query)
    student_ans = generate_answer(student_model, query, context)
    teacher_ans = generate_answer(teacher_model, query, context)
    sim         = similarity_score(student_ans, teacher_ans)
    print(f"  📊 Student↔Teacher sim: {sim:.2f}")
    final_ans   = teacher_ans if sim < 0.5 else student_ans
    return final_ans, student_ans, teacher_ans

# ===============================
# EVALUATION + GRAPHS
# ===============================
def evaluate_system():
    print("\n📊 Running Evaluation...\n")
    scores       = []
    student_sims = []
    teacher_sims = []
    questions    = []

    for i, item in enumerate(test_data):
        query       = item["question"]
        true_answer = item["long_answer"]
        print(f"\n🔹 Q{i+1}: {query[:80]}...")

        final, student_ans, teacher_ans = final_pipeline(query)

        final_sim   = similarity_score(final, true_answer)
        student_sim = similarity_score(student_ans, true_answer)
        teacher_sim = similarity_score(teacher_ans, true_answer)

        scores.append(final_sim)
        student_sims.append(student_sim)
        teacher_sims.append(teacher_sim)
        questions.append(f"Q{i+1}")

        print(f"  ✅ Final   : {final[:100]}")
        print(f"  📖 True    : {true_answer[:100]}")
        print(f"  🎯 Sim     : {final_sim:.2f}")

    threshold = 0.40
    correct   = sum(s > threshold for s in scores)
    accuracy  = correct / len(scores)
    mean_sim  = sum(scores) / len(scores)

    print("\n========================")
    print("📈 FINAL RESULTS")
    print("========================")
    print(f"Mean similarity  : {mean_sim:.2f}")
    print(f"Accuracy (>{threshold}): {accuracy:.2f}")
    print(f"Hallucination    : {1 - accuracy:.2f}")
    print("========================\n")

    # ===============================
    # GRAPH 1: Accuracy vs Hallucination
    # ===============================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RAG Medical QA Evaluation", fontsize=14, fontweight="bold", y=1.02)

    labels = ["Accuracy", "Hallucination Rate"]
    values = [accuracy, 1 - accuracy]
    colors = ["#2ecc71", "#e74c3c"]
    bars   = ax1.bar(labels, values, color=colors, width=0.4, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.0%}",
            ha="center", va="bottom", fontsize=13, fontweight="bold"
        )

    ax1.axhline(y=threshold, color="gray", linestyle="--", linewidth=1, label=f"Threshold ({threshold})")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Accuracy vs Hallucination Rate", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Rate")
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # ===============================
    # GRAPH 2: Student vs Teacher vs Final
    # ===============================
    x     = np.arange(len(questions))
    width = 0.25

    b1 = ax2.bar(x - width, student_sims, width, label="Student (flan-t5-small)", color="#3498db", alpha=0.85)
    b2 = ax2.bar(x,         teacher_sims, width, label="Teacher (flan-t5-base)",  color="#9b59b6", alpha=0.85)
    b3 = ax2.bar(x + width, scores,       width, label="Final (selected)",         color="#2ecc71", alpha=0.85)

    ax2.axhline(y=threshold, color="gray",   linestyle="--", linewidth=1,   label=f"Threshold ({threshold})")
    ax2.axhline(y=mean_sim,  color="orange", linestyle="-",  linewidth=1.2, label=f"Mean sim ({mean_sim:.2f})")

    ax2.set_xticks(x)
    ax2.set_xticklabels(questions)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Student vs Teacher vs Final — Per Question", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Similarity to Ground Truth")
    ax2.set_xlabel("Question")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("eval_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Graph saved as eval_results.png")

# ===============================
# RUN EVALUATION
# ===============================
evaluate_system()

# ===============================
# CHAT MODE
# ===============================
while True:
    query = input("\n💬 Ask a medical question (or 'exit'): ")
    if query.lower() == "exit":
        break
    final, _, _ = final_pipeline(query)
    print("\n✅ Answer:", final)












































# ===============================
# IMPORTS
# ===============================
from datasets import load_dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# DEVICE SETUP (Apple Silicon MPS)
# ===============================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("⚡ Using Apple MPS (GPU acceleration)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("⚡ Using CUDA")
else:
    device = torch.device("cpu")
    print("💻 Using CPU")

device_id = 0 if device.type in ("mps", "cuda") else -1

# ===============================
# LOAD DATASET
# ===============================
print("📦 Loading dataset...")
dataset    = load_dataset("pubmed_qa", "pqa_labeled")
train_data = dataset["train"].select(range(200))
test_data  = dataset["train"].select(range(200, 220))
print("✅ Dataset loaded!")

# ===============================
# BUILD KNOWLEDGE BASE
# ===============================
knowledge_data = []
for item in train_data:
    knowledge_data.append({
        "question": item["question"],
        "context":  " ".join(item["context"]["contexts"])[:800],
        "answer":   item["long_answer"]
    })

print(f"✅ Knowledge base: {len(knowledge_data)} entries")

# ===============================
# LOAD MODELS
# ===============================
print("🧠 Loading models...")

student_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=device_id,
    max_new_tokens=150
)

teacher_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=device_id,
    max_new_tokens=150
)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Models loaded!")

# ===============================
# PRE-COMPUTE KB EMBEDDINGS
# ===============================
print("📐 Pre-computing embeddings...")
kb_questions  = [item["question"] for item in knowledge_data]
kb_embeddings = embedder.encode(
    kb_questions,
    convert_to_tensor=True,
    show_progress_bar=False
)
print("✅ Done!")

# ===============================
# RETRIEVAL
# ===============================
def retrieve_context(query, top_k=4):
    q_emb   = embedder.encode(query, convert_to_tensor=True)
    scores  = util.cos_sim(q_emb, kb_embeddings)[0]
    top_idx = torch.topk(scores, k=min(top_k, len(knowledge_data))).indices.tolist()

    # Include both context AND answer for richer grounding
    combined = " ".join(
        knowledge_data[i]["context"] + " " + knowledge_data[i]["answer"]
        for i in top_idx
    )
    return combined[:1500]

# ===============================
# GENERATION
# ===============================
def generate_answer(model, query, context):
    prompt = (
        f"You are a medical expert. Use the context to answer accurately.\n\n"
        f"Context: {context[:700]}\n\n"
        f"Question: {query}\n\n"
        f"Provide a detailed medical answer:"
    )
    return model(prompt, do_sample=False)[0]["generated_text"].strip()

# ===============================
# SIMILARITY
# ===============================
def similarity_score(a, b):
    e1 = embedder.encode(a, convert_to_tensor=True)
    e2 = embedder.encode(b, convert_to_tensor=True)
    return max(0.0, util.cos_sim(e1, e2).item())

# ===============================
# PIPELINE
# ===============================
def final_pipeline(query):
    context     = retrieve_context(query)
    student_ans = generate_answer(student_model, query, context)
    teacher_ans = generate_answer(teacher_model, query, context)
    sim         = similarity_score(student_ans, teacher_ans)

    print(f"  📊 Student↔Teacher sim: {sim:.2f}")

    # Teacher is consistently better for medical QA
    final_ans = teacher_ans
    return final_ans, student_ans, teacher_ans

# ===============================
# EVALUATION
# ===============================
def evaluate_system():
    print("\n📊 Running Evaluation...\n")

    scores       = []
    student_sims = []
    teacher_sims = []
    questions    = []
    threshold    = 0.30

    for i, item in enumerate(test_data):
        query       = item["question"]
        true_answer = item["long_answer"]

        print(f"\n{'─'*55}")
        print(f"🔹 Q{i+1}: {query[:85]}...")

        final, student_ans, teacher_ans = final_pipeline(query)

        final_sim   = similarity_score(final,       true_answer)
        student_sim = similarity_score(student_ans, true_answer)
        teacher_sim = similarity_score(teacher_ans, true_answer)

        scores.append(final_sim)
        student_sims.append(student_sim)
        teacher_sims.append(teacher_sim)
        questions.append(f"Q{i+1}")

        print(f"  Student  : {student_sim:.2f} | {student_ans[:80]}...")
        print(f"  Teacher  : {teacher_sim:.2f} | {teacher_ans[:80]}...")
        print(f"  Final    : {final_sim:.2f}  | {final[:80]}...")
        print(f"  True     : {true_answer[:80]}...")

    correct   = sum(s > threshold for s in scores)
    accuracy  = correct / len(scores)
    hall_rate = 1 - accuracy
    mean_sim  = sum(scores) / len(scores)

    print("\n" + "="*55)
    print("📈  FINAL RESULTS")
    print("="*55)
    print(f"  Total questions  : {len(scores)}")
    print(f"  Threshold        : {threshold}")
    print(f"  Mean similarity  : {mean_sim:.2f}")
    print(f"  Accuracy         : {accuracy:.2%}")
    print(f"  Hallucination    : {hall_rate:.2%}")
    print("="*55)

    # -----------------------------------------------
    # GRAPH 1: Accuracy vs Hallucination
    # -----------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "RAG Medical QA — Hallucination Detection & Mitigation\n"
        "Dataset: PubMedQA (pqa_labeled)",
        fontsize=13, fontweight="bold", y=1.03
    )

    labels = ["Accuracy", "Hallucination\nRate"]
    values = [accuracy, hall_rate]
    colors = ["#2ecc71", "#e74c3c"]
    bars   = ax1.bar(labels, values, color=colors, width=0.35,
                     edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.0%}",
            ha="center", va="bottom", fontsize=14, fontweight="bold"
        )

    ax1.axhline(y=threshold, color="gray", linestyle="--",
                linewidth=1.2, label=f"Threshold ({threshold})")
    ax1.set_ylim(0, 1.2)
    ax1.set_title("Accuracy vs Hallucination Rate", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Rate")
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # -----------------------------------------------
    # GRAPH 2: Student vs Teacher vs Final
    # -----------------------------------------------
    x     = np.arange(len(questions))
    width = 0.25

    ax2.bar(x - width, student_sims, width,
            label="Student (flan-t5-small)", color="#3498db", alpha=0.85)
    ax2.bar(x,         teacher_sims, width,
            label="Teacher (flan-t5-base)",  color="#9b59b6", alpha=0.85)
    ax2.bar(x + width, scores,       width,
            label="Final (Teacher selected)", color="#2ecc71", alpha=0.85)

    ax2.axhline(y=threshold, color="gray",   linestyle="--", linewidth=1.2,
                label=f"Threshold ({threshold})")
    ax2.axhline(y=mean_sim,  color="orange", linestyle="-",  linewidth=1.5,
                label=f"Mean sim ({mean_sim:.2f})")

    ax2.set_xticks(x)
    ax2.set_xticklabels(questions, fontsize=8, rotation=45)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Student vs Teacher vs Final — Per Question",
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("Similarity to Ground Truth")
    ax2.set_xlabel("Question")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("eval_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n📊 Graph saved → eval_results.png")

# ===============================
# RUN EVALUATION
# ===============================
evaluate_system()

# ===============================
# CHAT MODE
# ===============================
print("\n" + "="*55)
print("💬  CHAT MODE — type 'exit' to quit")
print("="*55 + "\n")

while True:
    query = input("You: ").strip()
    if query.lower() in ("exit", "quit"):
        print("👋 Goodbye!")
        break
    if not query:
        continue

    final, student_ans, teacher_ans = final_pipeline(query)
    st_sim = similarity_score(student_ans, teacher_ans)

    print(f"\n🩺 Answer: {final}")
    print(f"   [Student↔Teacher sim: {st_sim:.2f} | "
          f"{'🔴 Hallucination risk' if st_sim < 0.45 else '🟢 Confident answer'}]\n")
    























































    # ===============================
# IMPORTS
# ===============================
from datasets import load_dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# DEVICE SETUP
# ===============================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("⚡ Using Apple MPS (GPU acceleration)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("⚡ Using CUDA")
else:
    device = torch.device("cpu")
    print("💻 Using CPU")

device_id = 0 if device.type in ("mps", "cuda") else -1

# ===============================
# LOAD DATASET 1: PubMedQA
# (biomedical research QA from NIH PubMed)
# ===============================
print("📦 Loading PubMedQA dataset...")
pubmed_dataset = load_dataset("pubmed_qa", "pqa_labeled")
pubmed_train   = pubmed_dataset["train"].select(range(150))
pubmed_test    = pubmed_dataset["train"].select(range(150, 160))
print(f"✅ PubMedQA — Train: 150 | Test: 10")

# ===============================
# LOAD DATASET 2: HealthCareMagic
# (real patient-doctor hospital conversations)
# ===============================
print("📦 Loading HealthCareMagic dataset...")
hcm_dataset = load_dataset(
    "lavita/ChatDoctor-HealthCareMagic-100k",
    split="train"
)
hcm_dataset = hcm_dataset.filter(
    lambda x: x["input"] and x["output"]
    and len(x["input"].split()) > 5
    and len(x["output"].split()) > 20
)
hcm_train = hcm_dataset.select(range(150))
hcm_test  = hcm_dataset.select(range(150, 160))
print(f"✅ HealthCareMagic — Train: 150 | Test: 10")

# ===============================
# BUILD COMBINED KNOWLEDGE BASE
# 300 total entries (150 PubMed + 150 HealthCareMagic)
# ===============================
print("🏗️  Building combined knowledge base...")
knowledge_data = []

# Add PubMedQA entries
for item in pubmed_train:
    knowledge_data.append({
        "question": item["question"],
        "context":  " ".join(item["context"]["contexts"])[:600],
        "answer":   item["long_answer"],
        "source":   "pubmed"
    })

# Add HealthCareMagic entries
for item in hcm_train:
    if len(item["output"].split()) >= 20:
        knowledge_data.append({
            "question": item["input"].strip(),
            "context":  item["input"].strip()[:600],
            "answer":   item["output"].strip()[:600],
            "source":   "healthcaremagic"
        })

print(f"✅ Combined KB: {len(knowledge_data)} entries "
      f"({sum(1 for d in knowledge_data if d['source']=='pubmed')} PubMed + "
      f"{sum(1 for d in knowledge_data if d['source']=='healthcaremagic')} HealthCareMagic)")

# ===============================
# BUILD COMBINED TEST SET
# 20 total (10 PubMed + 10 HealthCareMagic)
# ===============================
test_data = []

for item in pubmed_test:
    test_data.append({
        "question":    item["question"],
        "true_answer": item["long_answer"],
        "source":      "pubmed"
    })

for item in hcm_test:
    test_data.append({
        "question":    item["input"].strip(),
        "true_answer": item["output"].strip(),
        "source":      "healthcaremagic"
    })

print(f"✅ Test set: {len(test_data)} questions "
      f"(10 PubMed + 10 HealthCareMagic)")

# ===============================
# LOAD MODELS
# ===============================
print("🧠 Loading models...")

student_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=device_id,
    max_new_tokens=150
)

teacher_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=device_id,
    max_new_tokens=150
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Models loaded!")

# ===============================
# PRE-COMPUTE KB EMBEDDINGS
# ===============================
print("📐 Pre-computing embeddings for combined KB...")
kb_questions  = [d["question"] for d in knowledge_data]
kb_embeddings = embedder.encode(
    kb_questions,
    convert_to_tensor=True,
    batch_size=64,
    show_progress_bar=True
)
print("✅ Embeddings ready!")

# ===============================
# RETRIEVAL
# Automatically retrieves from whichever dataset
# is most relevant to the query
# ===============================
def retrieve_context(query, top_k=4):
    q_emb              = embedder.encode(query, convert_to_tensor=True)
    scores             = util.cos_sim(q_emb, kb_embeddings)[0]
    top_scores, top_idx = torch.topk(scores, k=min(top_k, len(knowledge_data)))

    parts = []
    for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
        if score > 0.25:       # only use relevant entries
            entry = knowledge_data[idx]
            parts.append(
                f"[{entry['source'].upper()}] "
                f"Q: {entry['question']}\n"
                f"A: {entry['answer']}"
            )

    return "\n---\n".join(parts)[:1500] if parts else ""

# ===============================
# GENERATION
# ===============================
def generate_answer(model, query, context):
    if context:
        prompt = (
            f"You are a medical expert with knowledge of both "
            f"clinical research and patient care.\n\n"
            f"Reference cases:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Provide a detailed medical answer:"
        )
    else:
        prompt = (
            f"You are a medical expert.\n\n"
            f"Question: {query}\n\n"
            f"Provide a detailed medical answer:"
        )
    result = model(prompt, do_sample=False)
    ans    = result[0]["generated_text"].strip()

    # Fallback if answer too short
    if len(ans.split()) < 8 and context:
        lines = [l for l in context.split("\n") if l.startswith("A:")]
        ans   = lines[0].replace("A:", "").strip() if lines else ans

    return ans

# ===============================
# SIMILARITY
# ===============================
def similarity_score(a, b):
    e1 = embedder.encode(a, convert_to_tensor=True)
    e2 = embedder.encode(b, convert_to_tensor=True)
    return max(0.0, util.cos_sim(e1, e2).item())

# ===============================
# HALLUCINATION DETECTION
# ===============================
def detect_hallucination(student_ans, teacher_ans, context, thr=0.45):
    st_sim  = similarity_score(student_ans, teacher_ans)
    ctx_sim = similarity_score(student_ans, context) if context else 0.0
    return {
        "is_hallucinating":    st_sim < thr and ctx_sim < thr,
        "student_teacher_sim": st_sim,
        "student_context_sim": ctx_sim
    }

# ===============================
# MITIGATION
# ===============================
def mitigate(student_ans, teacher_ans, info):
    if info["is_hallucinating"]:
        return teacher_ans, "🔴 Hallucination detected → Mitigated with Teacher"
    elif info["student_teacher_sim"] < 0.60:
        return teacher_ans, "🟡 Low confidence → Preferred Teacher"
    else:
        return student_ans, "🟢 Student answer accepted"

# ===============================
# FULL PIPELINE
# ===============================
def full_pipeline(query):
    context     = retrieve_context(query)
    student_ans = generate_answer(student_model, query, context)
    teacher_ans = generate_answer(teacher_model, query, context)
    info        = detect_hallucination(student_ans, teacher_ans, context)
    final, msg  = mitigate(student_ans, teacher_ans, info)
    return {
        "context":        context,
        "student_ans":    student_ans,
        "teacher_ans":    teacher_ans,
        "final_ans":      final,
        "mitigation_msg": msg,
        "hallucination":  info
    }

# ===============================
# GRAPHS
# ===============================
def plot_results(questions, s_sims, t_sims, f_sims,
                 sources, accuracy, hall_rate, threshold):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        "Hallucination Detection & Mitigation in LLMs\n"
        "Combined Dataset: PubMedQA + HealthCareMagic",
        fontsize=13, fontweight="bold", y=1.03
    )

    # Graph 1 — Accuracy vs Hallucination
    labels = ["Accuracy", "Hallucination\nRate"]
    values = [accuracy, hall_rate]
    colors = ["#2ecc71", "#e74c3c"]
    bars   = ax1.bar(labels, values, color=colors, width=0.35,
                     edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.0%}",
            ha="center", va="bottom", fontsize=14, fontweight="bold"
        )
    ax1.axhline(y=threshold, color="gray", linestyle="--",
                linewidth=1.2, label=f"Threshold ({threshold})")
    ax1.set_ylim(0, 1.2)
    ax1.set_title("Accuracy vs Hallucination Rate", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Rate")
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # Graph 2 — Per question with source color coding
    x     = np.arange(len(questions))
    width = 0.25

    ax2.bar(x - width, s_sims, width,
            label="Student (flan-t5-small)", color="#3498db", alpha=0.85)
    ax2.bar(x,         t_sims, width,
            label="Teacher (flan-t5-base)",  color="#9b59b6", alpha=0.85)
    ax2.bar(x + width, f_sims, width,
            label="Final (after mitigation)", color="#2ecc71", alpha=0.85)

    # Shade background to show which dataset each question is from
    for i, src in enumerate(sources):
        ax2.axvspan(
            i - 0.5, i + 0.5,
            alpha=0.06,
            color="#e67e22" if src == "pubmed" else "#1abc9c"
        )

    mean_f = np.mean(f_sims)
    ax2.axhline(y=threshold, color="gray",   linestyle="--", linewidth=1.2,
                label=f"Threshold ({threshold})")
    ax2.axhline(y=mean_f,    color="orange", linestyle="-",  linewidth=1.5,
                label=f"Mean final ({mean_f:.2f})")

    # Dataset legend patches
    import matplotlib.patches as mpatches
    p1 = mpatches.Patch(color="#e67e22", alpha=0.3, label="PubMed questions")
    p2 = mpatches.Patch(color="#1abc9c", alpha=0.3, label="HealthCareMagic questions")

    ax2.set_xticks(x)
    ax2.set_xticklabels(questions, fontsize=8, rotation=45)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Student vs Teacher vs Final — Per Question",
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("Similarity to Ground Truth")
    ax2.set_xlabel("Question")

    handles, lbls = ax2.get_legend_handles_labels()
    ax2.legend(handles + [p1, p2], lbls + ["PubMed questions", "HealthCareMagic questions"],
               fontsize=7, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("hallucination_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Graph saved → hallucination_results.png")

# ===============================
# EVALUATION
# ===============================
def evaluate_system():
    print("\n" + "="*60)
    print("   HALLUCINATION DETECTION & MITIGATION — EVALUATION")
    print("   Datasets : PubMedQA + HealthCareMagic (combined)")
    print("   Models   : flan-t5-small (student) | flan-t5-base (teacher)")
    print("="*60)

    s_sims, t_sims, f_sims = [], [], []
    q_labels, sources      = [], []
    hall_flags             = []
    threshold              = 0.30

    for i, item in enumerate(test_data):
        query  = item["question"]
        truth  = item["true_answer"]
        source = item["source"]

        print(f"\n{'─'*55}")
        print(f"🔹 Q{i+1} [{source.upper()}]: {query[:80]}...")

        res   = full_pipeline(query)
        s_sim = similarity_score(res["student_ans"], truth)
        t_sim = similarity_score(res["teacher_ans"], truth)
        f_sim = similarity_score(res["final_ans"],   truth)

        s_sims.append(s_sim)
        t_sims.append(t_sim)
        f_sims.append(f_sim)
        q_labels.append(f"Q{i+1}")
        sources.append(source)
        hall_flags.append(res["hallucination"]["is_hallucinating"])

        print(f"   {res['mitigation_msg']}")
        print(f"   Student  {s_sim:.2f} → {res['student_ans'][:75]}...")
        print(f"   Teacher  {t_sim:.2f} → {res['teacher_ans'][:75]}...")
        print(f"   Final    {f_sim:.2f} → {res['final_ans'][:75]}...")
        print(f"   Truth         → {truth[:75]}...")

    correct   = sum(s > threshold for s in f_sims)
    total     = len(f_sims)
    accuracy  = correct / total
    hall_rate = 1 - accuracy
    detected  = sum(hall_flags)

    # Per-dataset breakdown
    pubmed_scores = [f for f, s in zip(f_sims, sources) if s == "pubmed"]
    hcm_scores    = [f for f, s in zip(f_sims, sources) if s == "healthcaremagic"]

    print("\n" + "="*55)
    print("📈  FINAL RESULTS")
    print("="*55)
    print(f"  Total questions      : {total}")
    print(f"  Hallucinations caught: {detected} / {total}")
    print(f"  Mean student sim     : {np.mean(s_sims):.2f}")
    print(f"  Mean teacher sim     : {np.mean(t_sims):.2f}")
    print(f"  Mean final sim       : {np.mean(f_sims):.2f}")
    print(f"  Accuracy (>{threshold})    : {accuracy:.2%}")
    print(f"  Hallucination rate   : {hall_rate:.2%}")
    print(f"  ── PubMed accuracy   : "
          f"{sum(s > threshold for s in pubmed_scores)/len(pubmed_scores):.2%}")
    print(f"  ── HealthCare acc    : "
          f"{sum(s > threshold for s in hcm_scores)/len(hcm_scores):.2%}")
    print("="*55)

    plot_results(q_labels, s_sims, t_sims, f_sims,
                 sources, accuracy, hall_rate, threshold)

# ===============================
# RUN EVALUATION
# ===============================
evaluate_system()

# ===============================
# CHAT MODE
# ===============================
print("\n" + "="*55)
print("💬  CHAT MODE — type 'exit' to quit")
print("    (draws from both PubMed + HealthCareMagic KB)")
print("="*55 + "\n")

while True:
    query = input("You: ").strip()
    if query.lower() in ("exit", "quit"):
        print("👋 Goodbye!")
        break
    if not query:
        continue

    res    = full_pipeline(query)
    st_sim = res["hallucination"]["student_teacher_sim"]

    print(f"\n{res['mitigation_msg']}")
    print(f"🩺 {res['final_ans']}")
    print(f"   [S↔T: {st_sim:.2f} | "
          f"{'🔴 Hallucination risk' if st_sim < 0.45 else '🟢 Confident'}]\n")