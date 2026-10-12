# Detection and mitigation of hallucinations in healthcare QA

This project studies whether a small student model, a larger teacher model, and retrieved medical text can flag and replace unreliable answers. The web app runs the improved pipeline on a new question and displays the saved experiment. It is a research prototype, not medical advice.

## Architecture

- Student: `google/flan-t5-small`
- Teacher: `google/flan-t5-base`
- Embeddings and retrieval: `all-MiniLM-L6-v2` cosine similarity over a 300-row knowledge base
- Configuration D: the original three-way mitigation (replace, blend, or keep)
- Configuration E: token-capped context, then accept a supported student answer, else a supported teacher answer, else a fixed uncertainty statement. No blending and no copied retrieved answer.
- E-nov: D’s mitigation on the token-capped context, without the support check

`knowledge.py` is a static list and is not used. The datasets are PubMedQA `pqa_labeled` and `lavita/ChatDoctor-HealthCareMagic-100k` only.

## Setup

Python packages are listed in `requirements.txt`, including `fastapi` and `uvicorn`. The reproduction environment used torch 2.14.1+cpu, transformers 4.40.1, sentence-transformers 3.1.1, and datasets 4.8.2.

Node.js 22 is the version used for the frontend (`node -v`). In `frontend/`:

```bash
npm install
```

## Run the experiments again

These commands call the models. They are not required to rebuild the paper figures.

```bash
python3 src/evaluate.py
python3 src/evaluate.py --configs-ad --val-size 20 --test-size 20
python3 src/evaluate.py --config-e --val-size 20 --test-size 20
```

Do not run `main2.py`. It would overwrite the original figures.

Rebuild tables and figures from the saved files, without the student or teacher models:

```bash
python3 run_all.py
```

`outputs/run_settings.json` records the seed (42), split sizes (20 validation and 20 test, 10 of each dataset in each split), model names, thresholds, the verify grid 0.30–0.70, and the chosen cutoff 0.30.

## Metric definitions

A similarity pass is cosine similarity to the reference text strictly above 0.52. It is not factual accuracy. The detector flag count is separate: D flags a student answer that is shorter than 10 words, or that is below 0.38 similarity to both the teacher answer and the retrieved context. Uncertainty statements are not counted as similarity wins. Wilson intervals on the charts are 95% intervals for those pass rates. No significance test is claimed.

Precision, recall, F1, and the confusion matrix are pending manual labels. `python3 src/label_metrics.py` prints `labels missing, metrics cannot be calculated` until `annotations/manual_labels.csv` is filled. See `annotations/LABELING_GUIDE.md`.

## Preserved original baseline

The original figure `hallucination_results.png` reports 20 questions, detector flags 5/20, mean similarities 0.462 / 0.569 / 0.556, similarity pass rate 65% (13/20), PubMedQA 70% (7/10), HealthCareMagic 60% (6/10), and a threshold-derived failure rate of 35% (7/20). It is not reproducible: no seed and no per-question predictions were saved.

The reproduced baseline in `outputs/baseline_metrics.json` is the reference for later comparisons: detector flags 6/20, mean similarities 0.450 / 0.476 / 0.514, similarity pass rate 55% (11/20), PubMedQA 60% (6/10), HealthCareMagic 50% (5/10), failure rate 45% (9/20).

## Saved comparison

Seed 42. Each split has 20 questions. Similarity pass counts are wins over 20, not accuracy.

| Config | Validation pass | Validation mean similarity | Test pass | Test mean similarity | Test PubMedQA | Test HealthCareMagic |
| --- | --- | --- | --- | --- | --- | --- |
| A student only | 4/20 | 0.332 | 5/20 | 0.379 | 5/10 | 0/10 |
| B teacher only | 11/20 | 0.500 | 6/20 | 0.404 | 4/10 | 2/10 |
| C student with retrieval | 5/20 | 0.381 | 5/20 | 0.365 | 5/10 | 0/10 |
| D baseline mitigation | 9/20 (flags 6) | 0.501 | 8/20 (flags 8) | 0.481 | 5/10 | 3/10 |
| E-nov | 10/20 | 0.513 | 6/20 | 0.441 | 4/10 | 2/10 |
| E | 10/20 | 0.450 | 7/20 | 0.386 | 4/10 | 3/10 |

On the test split, E is below D (7/20 versus 8/20; mean similarity 0.386 versus 0.481). Six of E’s test answers are the uncertainty statement. Excluding them, the pass count is 7/14, which uses a smaller denominator.

## What changed from the original script

`main.py` is unchanged. New code adds a seeded validation/test split with no retrieval leakage, configurations A–D, a 512-token context cap, configuration E, saved predictions, figures, and the web app.

## Limitations

The pilot is small. Similarity is not factual correctness. Doctor replies are not verified truth. MiniLM cannot show a contradiction. The verify cutoff was chosen by validation similarity to the reference, which favors a permissive cutoff. E and D also differ in prompt length; E-nov is the ablation for that cap. E scored lower than D on similarity, and that result stands.

## Web app

Node 22. From the repository root, in one terminal:

```bash
uvicorn api.main:app --reload
```

In another:

```bash
cd frontend && npm run dev
```

Vite proxies `/api` to port 8000. Open `http://127.0.0.1:5173`.

Screenshots of the interface, at 1280px and 390px, are in `docs/screenshots/`:

- Home before a question: `home-empty-1280.png`, `home-empty-390.png`
- Home while a request is running: `home-loading-1280.png`, `home-loading-390.png`
- Accepted, revised, and uncertainty results: `home-accepted-*.png`, `home-revised-*.png`, `home-uncertain-*.png`
- Failed request: `home-error-*.png`
- Comparison modal: `modal-*.png`
- Research Results: `research-validation-*.png`, `research-test-*.png`
- About: `about-*.png`

`python3 scripts/smoke_test.py` checks the live API. The first request waits until the models finish loading.

## API

- `GET /api/health`
- `POST /api/analyze` with `{"question": "..."}` (1–500 characters)
- `GET /api/results`
- `GET /api/figures/{name}` for a whitelisted PNG
