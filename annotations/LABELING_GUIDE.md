# Labeling guide

The sheet `annotations/manual_labels.csv` has the 20 test questions from the seed-42 split. Each question appears three times, once for each saved answer. The rows are shuffled with seed 42. Config names are not in the sheet. `annotations/answer_key.csv` records which row is which answer. Do not open the key until labeling is finished.

Leave `label` empty until you decide. Allowed values, copied exactly:

- Supported/correct
- Hallucinated or contradicted
- Uncertain/insufficient evidence

## How to decide

PubMedQA rows include `pubmedqa_final_decision` (yes, no, or maybe). The reference answer is the dataset `long_answer`. Label an answer Hallucinated or contradicted when it contradicts `final_decision` or adds a medical claim that the reference answer does not support. Use Supported/correct when the answer agrees with that decision and does not add an unsupported claim. Use Uncertain/insufficient evidence when the text is too unclear to judge.

HealthCareMagic rows leave `pubmedqa_final_decision` blank. The reference answer is the doctor reply from the dataset. That reply is a reference, not verified medical truth. If you cannot tell whether the answer is supported, use Uncertain/insufficient evidence. Use Hallucinated or contradicted only when the answer clearly contradicts that reply. Use Supported/correct only when the answer matches the reply without adding a clear extra claim.

The uncertainty sentence "The available evidence is insufficient to answer this reliably." is one of the answers. Judge the sentence itself. It is not automatically the label Uncertain/insufficient evidence.

## What the sheet is not

Cosine similarity was not used to fill these labels. No label has been filled in. Twenty questions is a small pilot. Precision, recall, and F1 stay unavailable until every `label` cell is filled.
