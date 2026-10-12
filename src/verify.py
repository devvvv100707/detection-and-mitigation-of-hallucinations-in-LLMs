"""Lightweight evidence check.

Compares a generated answer with the retrieved context using MiniLM cosine
similarity. Outcomes are only "Supported" and "Insufficient evidence".
This module does not assign "Contradicted". Similarity cannot show that a
context contradicts an answer.

The default cutoff is provisional. It was not selected by looking at the
20-question reproduced baseline. Tune it later on a validation split only.
"""

SUPPORTED = "Supported"
INSUFFICIENT = "Insufficient evidence"

# Provisional. Not tuned on the reproduced 20-question baseline.
DEFAULT_SUPPORT_THRESHOLD = 0.50


def verify_support(answer, context, similarity_fn, threshold=DEFAULT_SUPPORT_THRESHOLD):
    """Return a support verdict for one answer.

    similarity_fn(a, b) must return a cosine similarity in about [-1, 1],
    such as BaselinePipeline.similarity_score. An empty answer or empty
    context is Insufficient evidence. Otherwise the answer is Supported when
    similarity >= threshold.
    """
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")
    if not answer or not str(answer).strip() or not context or not str(context).strip():
        return {
            "verdict": INSUFFICIENT,
            "similarity": None,
            "threshold": threshold,
            "reason": "empty answer or empty retrieved context",
        }
    similarity = float(similarity_fn(answer, context))
    if similarity >= threshold:
        verdict = SUPPORTED
        reason = "answer-context cosine similarity is at least the provisional cutoff"
    else:
        verdict = INSUFFICIENT
        reason = "answer-context cosine similarity is below the provisional cutoff"
    return {
        "verdict": verdict,
        "similarity": similarity,
        "threshold": threshold,
        "reason": reason,
    }


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer, util

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def similarity_fn(a, b):
        e1 = embedder.encode(a, convert_to_tensor=True)
        e2 = embedder.encode(b, convert_to_tensor=True)
        return max(0.0, util.cos_sim(e1, e2).item())

    related = verify_support(
        "Fever and cough are common symptoms.",
        "Common symptoms include fever and cough.",
        similarity_fn,
    )
    unrelated = verify_support(
        "The capital of France is Paris.",
        "Insulin resistance can cause type 2 diabetes.",
        similarity_fn,
    )
    empty = verify_support("An answer.", "", similarity_fn)
    print(related)
    print(unrelated)
    print(empty)
    assert related["verdict"] in (SUPPORTED, INSUFFICIENT)
    assert unrelated["verdict"] in (SUPPORTED, INSUFFICIENT)
    assert empty["verdict"] == INSUFFICIENT
    assert "Contradicted" not in (related["verdict"], unrelated["verdict"], empty["verdict"])
