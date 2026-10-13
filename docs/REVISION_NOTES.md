# Revision notes

Revised manuscript: `docs/Detection_and_Mitigation_of_Hallucination_IEEE.docx`

The source manuscript was already a complete paper. This revision does not replace the argument, the configurations, or the reported numbers.

## Stage 1 audit (applied only where listed)

Kept as written: title, authors, affiliation, abstract, index terms, introduction, related-work synthesis, methodology (datasets, retrieval, models, heuristic, configurations), experimental protocol, metrics, results discussion, limitations, conclusion, and the existing reference list.

Essential corrections:

- The literature-survey section was a landscape section. Both pages are now portrait, with the same five-column tables. Column widths were scaled to the portrait text width so the table stays on the page. Wording inside the table was not rewritten.
- Figure 1 labels extended outside their boxes because the figure was stored very short and wide. The same steps were redrawn with the text inside each box, including the final-output step.
- In-text citations such as [1] were not links. Each numbered citation now links to the URL or DOI already present in that reference. References [7], [30], and [31] have no URL or DOI in the manuscript, so those citations jump to the reference-list entry.
- The home-screen and results-dashboard screenshots repeated the interface and the saved charts. They were removed so the paper keeps one prototype screenshot (now Fig. 4) plus the two result charts (Figs. 2 and 3).

Not treated as errors: separate A–D and E/E-nov artifacts, the non-reproducible 65% historical figure, incomplete manual labels, and the statement that similarity is not factual accuracy.

## Change log

1. Literature-survey section set to portrait A4, matching the rest of the paper. Reason: the landscape section was the reported page-orientation problem.
2. Literature tables resized to the portrait content width (9,663 twips) without editing cell text. Reason: the old column grid was the landscape width and would overflow.
3. Figure 1 replaced with a redraw of the same pipeline so every label sits inside its box. Reason: text was leaking out of the boxes.
4. In-text bracketed citations colored blue and hyperlinked to the paper URL or DOI already cited. DOI-only entries use `https://doi.org/` plus the printed DOI. Reason: citations were not linked.
5. Removed the home-screen and dashboard screenshots and renumbered the remaining analysis screenshot to Fig. 4. Updated the two paragraphs that referred to Figs. 4–6. Reason: those screenshots duplicated the prototype and the result charts.
6. Reference-list URLs and DOIs are clickable. Reason: the bibliographic links were plain text.

No experimental number, threshold, model name, dataset name, or configuration definition was changed.

## Unresolved items

- [MISSING: URL or DOI for reference [7] (Mündler et al., ICLR 2024).] The citation links to the reference entry only.
- [MISSING: URL or DOI for reference [30] (Lewis et al., NeurIPS 2020).] Same treatment.
- [MISSING: URL or DOI for reference [31] (Chung et al., JMLR 2024).] Same treatment.
- [VERIFY: whether the held-out question identifiers for the A–D artifact and the E/E-nov artifact are the same seed-42 test items.] The manuscript already treats that comparison as descriptive. This revision did not merge the runs.
- [VERIFY: author order, corresponding-author email, and any required IEEE membership or funding footnotes.] None were added.
- [MISSING: target IEEE venue and two-column template.] Numbered citations follow the manuscript’s IEEE-style numbering. The file was not converted to the official two-column conference template, because that would reflow every table and figure.

## Verification report

Checked against `outputs/comparison_summary.json`, `outputs/ad_val20_test20_metrics.json`, `outputs/e_val20_test20_metrics.json`, `outputs/baseline_metrics.json`, `outputs/run_settings.json`, and `outputs/label_metrics.json`.

- Test pass counts match the saved comparison: A 5/20, B 6/20, C 5/20, D 8/20, E-nov 6/20, E 7/20.
- Mean similarities 0.379, 0.404, 0.365, 0.481, 0.441, and 0.386 match the saved values rounded to three decimals.
- Mean latencies 0.715, 2.084, 0.650, 3.145, 3.161, and 1.839 s match the saved values rounded to three decimals.
- Wilson intervals in Table 3 match the saved interval endpoints rounded to one decimal percent.
- E abstentions: `uncertain_count` 6 on the test split. The chosen support cutoff is 0.30.
- Reproduced baseline: similarity-pass rate 0.55 and mean final similarity about 0.514. Preserved original figure, via the recorded difference file: 65% (13/20) and mean final similarity 0.556. These stay separate from Table 3.
- `label_metrics.json` status is `labels_missing`. The paper still says manual labels are not available.
- Models and embedder match `run_settings.json`: `google/flan-t5-small`, `google/flan-t5-base`, `all-MiniLM-L6-v2`. Thresholds 0.25, 0.38, 0.55, 0.70, and pass cutoff 0.52 match that file.
- Every reference [1]–[31] is still present. No reference was added or removed.
- Figures kept: pipeline (Fig. 1), pass-rate chart (Fig. 2), mean-similarity chart (Fig. 3), and one analysis screenshot (Fig. 4).

## Optional improvements not applied

- Official IEEE two-column layout, running headers in the conference style, and a copyright footer. That is a template conversion, not a wording fix.
- New factuality metrics, significance tests, or a combined results table that treats every configuration as one experiment.
- Extra charts from `outputs/figures/` (dataset-wise bars, latency, tier counts, before/after similarity). The paper already states those results in prose or in Table 3; adding them would crowd the results section.
- Filling references [7], [30], and [31] with URLs that were not printed in the manuscript.
- Manual annotation, threshold re-tuning, or any change to the mitigation rules.
