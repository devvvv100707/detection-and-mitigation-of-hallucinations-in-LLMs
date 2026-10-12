# Preserved baseline

This note records the experiment that produced `hallucination_results.png`. That figure is the preserved measurement. An exact rerun is not claimed. `main.py` and the original PNG files were not modified or executed while this note was written. `main2.py` must not be run, because it would overwrite the original figures.

`results.csv` is empty. No script writes it. Per-question scores behind the means are not stored, so the figure cannot be reconstructed from a results table.

## Which script

`main.py` is the pipeline that matches the figure. The figure title is the string passed to `plot_results` in `main.py` (lines 291–295): “Improved Pipeline (top_k=5, adaptive prompts, 3-tier mitigation)”, with PubMed and HealthCareMagic thresholds of 0.52.

`eval_results.png` is a different, earlier PubMedQA-only run from the middle copy inside `main2.py` (threshold 0.30, final answer always the teacher, mean similarity about 0.33, pass rate 60%). It is not this baseline.

`knowledge.py` is not imported by `main.py`.

## Configuration taken from `main.py`

| Setting | Value | Where |
| --- | --- | --- |
| Student | `google/flan-t5-small`, `max_new_tokens=180` | lines 111–116 |
| Teacher | `google/flan-t5-base`, `max_new_tokens=200` | lines 118–123 |
| Embedder | `all-MiniLM-L6-v2` | line 125 |
| Decoding | `do_sample=False` | line 194 |
| Device | MPS, else CUDA, else CPU | lines 18–28 |
| PubMedQA | `pubmed_qa` / `pqa_labeled`, knowledge `train[0:150]`, test `train[150:160]` | lines 34–36, 89–94 |
| HealthCareMagic | `lavita/ChatDoctor-HealthCareMagic-100k` train, filtered, knowledge indices 0–149, test 150–159 | lines 43–53, 96–101 |
| Retrieval | question embeddings, `top_k=5`, keep score `> 0.25`, context cap 2000 characters | lines 144–160 |
| Detection | flag if student answer has fewer than 10 words, or both student–teacher and student–context cosine are `< 0.38` | lines 221–226 |
| Mitigation | flagged → teacher; student–teacher `< 0.55` → first 20 teacher words plus first 60 student words; `< 0.70` → teacher; else student | lines 239–259 |
| Similarity pass cutoff | final cosine to `true_answer` `> 0.52` for both sources | lines 420–421, 452–453 |
| Random seed | none | — |
| Validation split | none | — |

How 0.25, 0.38, 0.52, 0.55, and 0.70 were chosen cannot be determined from the code. Comments at lines 419–421 say 0.52 was relaxed or tuned “to be achievable.”

Reference text is PubMedQA `long_answer` and the HealthCareMagic doctor `output`. PubMedQA `final_decision` is unused.

## Metric definitions

All similarities are MiniLM cosine similarity, floored at 0 (`similarity_score`, lines 212–215).

- **Total questions:** 10 PubMedQA + 10 HealthCareMagic = 20 (`evaluate_system`).
- **Hallucinations detected:** count of `is_hallucinating`. The figure prints “Hallucinations caught: 5/20”. This is the detector flag count. It is not the red bar.
- **Mean student / teacher / final similarity:** mean cosine similarity of each answer to `true_answer`. The figure shows 0.462, 0.569, and 0.556.
- **Similarity-threshold pass rate:** share of final similarities strictly greater than 0.52. The figure labels this “Overall Accuracy” and shows 65%. PubMedQA 70% and HealthCareMagic 60% use the same cutoff inside each source (7/10 and 6/10).
- **Threshold-derived failure rate:** `1 - pass rate` (`main.py` line 458). The figure labels this “Hallucination Rate” and shows 35%.

These two rates answer different questions. Five of 20 answers were flagged by the detector (25%). Seven of 20 final answers fell at or below the 0.52 similarity cutoff (35%).

## Preserved numbers (from the figure, not from a new run)

- Questions: 20
- Detector flags: 5/20
- Mean student similarity: 0.462
- Mean teacher similarity: 0.569
- Mean final similarity: 0.556
- Similarity-threshold pass rate: 65% overall, 70% PubMedQA, 60% HealthCareMagic
- Threshold-derived failure rate: 35%

A later baseline-mode reimplementation should be compared with these figures and any difference reported. A match is not assumed. Dataset revisions, model weight revisions, and the lack of a seed or saved predictions can change a fresh run.
