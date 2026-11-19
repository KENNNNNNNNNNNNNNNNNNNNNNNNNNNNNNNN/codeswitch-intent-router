# Taglish Router

An AI-powered routing system that investigates how
code-switching between Tagalog and English affects
language model performance.

## Research Objectives

- Detect intra-sentence language switching.
- Calculate code-switch density.
- Select matching few-shot examples.
- Evaluate LLM intent-classification accuracy.
- Implement a human handoff mechanism.

The longer-term research questions are whether model confidence tracks
actual accuracy as code-switching density increases, whether
density-matched Taglish few-shot examples improve classification, and
where a sensible human-handoff threshold lies.

## Initial Research Scope

- Language pair: Tagalog-English
- Domain: Campus Registrar Inquiries
- Primary task: Intent Classification
- Programming language: Python

## Intent Categories

- enrollment_status
- document_request
- class_schedule
- tuition_payment
- student_records
- clearance

## Technology Stack

- Python
- Pandas
- Scikit-learn
- Groq API (`openai/gpt-oss-120b`, free tier)
- Matplotlib
- Streamlit
- Pytest

## Project Status

This is a compressed 7-day MVP. It does not claim to test the full research hypothesis.

- Day 1: Project initialization. Done.
- Day 2: Dataset creation. Done. 120 synthetic utterances.
- Day 3: Token-level language annotation, density formula, grouped split and
  scikit-learn baseline. Done.
- Day 4: Zero-shot LLM classification. Done.
- Day 5: Few-shot ablation (random, English-only, density-matched, max-diversity). Done.
- Day 6: Confidence vs accuracy, and the human-handoff threshold. Done.
- Day 7: Streamlit demo and this write-up. Done.

## How to Run

Requires Python 3.10+.

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q                  # tests; the LLM is mocked, no API key needed
```

The LLM steps need a free Groq API key from console.groq.com, stored in the
`GROQ_API_KEY` environment variable. Never put the key in a file in this repo.
On Windows, store it without it appearing in your command history:

```
[Environment]::SetEnvironmentVariable("GROQ_API_KEY", (Read-Host "Paste your Groq key"), "User")
```

Then open a new terminal.

| Step | Command | Output in `results/` |
|---|---|---|
| Baseline | `python experiments/baseline_sklearn.py` | `baseline_predictions.csv` |
| Zero-shot (Day 4) | `python experiments/llm_zero_shot.py --split test` (also `--split train`, `--split validation`) | `llm_zero_shot_<split>_<model>.csv` |
| Few-shot (Day 5) | `python experiments/llm_few_shot.py` | `llm_fewshot_<strategy>_test_<model>.csv`, `ablation_test_<model>.csv` |
| Handoff (Day 6) | `python experiments/confidence_handoff.py` | `confidence_by_bucket.csv`, `handoff_coverage.csv`, `handoff_threshold.csv`, `calibration_plot.png` |
| Demo (Day 7) | `streamlit run app.py` | |

LLM scripts save each answer as it arrives and skip sentences already in the
results file, so an interrupted run can be restarted without repeating calls.

## Code Layout

- `src/tagger.py`, `src/density.py`: rule-based token language tagger and code-switch density.
- `src/llm_client.py`: prompt, Groq call with retries, reply parsing (intent + confidence).
- `src/fewshot.py`: few-shot example selectors (train split only, one example per intent).
- `src/evaluate.py`: accuracy, macro F1 and 95% Wilson intervals, overall and per density bucket.
- `src/calibration.py`: reliability table, AUROC, coverage vs accuracy, threshold selection.
- `src/router.py`: one message in, intent + confidence + density + handoff decision out.
- `app.py`: Streamlit demo.

## Dataset

File: `data/taglish_dataset.csv`

Columns: `id, text, intent, language_pattern, source, split`

- 120 synthetic utterances, 20 per intent.
- `language_pattern` counts: Taglish 60, English 31, Tagalog 29.
- `source` is `synthetic` for all rows.
- `split` is a placeholder (`unassigned`) in this file; see `data/splits.csv`.

### Language pattern labels

`language_pattern` is a sentence-level label:

- **English**: no Tagalog tokens.
- **Tagalog**: no English content words.
- **Taglish**: both present.

These labels are rough categories. They are separate from token-level
language annotation, which will be used to compute code-switching density.

## Results

All numbers below come from the files in `results/`. Model `openai/gpt-oss-120b`,
temperature 0. Density buckets: single-language (density 0), low-mixed
(0, 0.3], high-mixed (> 0.3).

**The test split has only 16 sentences** (8 single-language, 3 low-mixed,
5 high-mixed). One sentence moves a bucket's accuracy by 12 to 33 points.
Differences between methods below are not statistically meaningful.

### Day 4 and 5: test split (n=16)

| Method | Correct | Accuracy | Macro F1 | Single | Low | High |
|---|---|---|---|---|---|---|
| Baseline (TF-IDF char n-grams + logistic regression) | 15/16 | 0.938 | 0.933 | 1.00 | 1.00 | 0.80 |
| LLM zero-shot | 15/16 | 0.938 | 0.933 | 0.875 | 1.00 | 1.00 |
| Few-shot, random | 14/16 | 0.875 | 0.878 | 0.75 | 1.00 | 1.00 |
| Few-shot, English-only | 15/16 | 0.938 | 0.933 | 0.875 | 1.00 | 1.00 |
| Few-shot, density-matched | 15/16 | 0.938 | 0.933 | 0.875 | 1.00 | 1.00 |
| Few-shot, max-diversity | 15/16 | 0.938 | 0.933 | 0.875 | 1.00 | 1.00 |

95% interval for 15/16: [0.72, 0.99]. Few-shot prompts hold 6 examples, one per
intent, from the train split only; strategies differ only in the language mix
of the examples. No strategy produced an invalid reply.

- Every LLM method misses the same sentence ("Can I get a certified true copy
  of my records?", labelled `student_records`, predicted `document_request`).
- Zero-shot is already at 15/16, so few-shot could gain at most one sentence.
  **This test set cannot tell the few-shot strategies apart.** That is a
  ceiling effect, not evidence that density matching does not help.

### Day 6: confidence and handoff (zero-shot, all 120 sentences)

Zero-shot never sees the train split, so it was also run on train and
validation ("dev", n=104) to get more errors to study.

| Bucket | n | Accuracy | Mean confidence | Errors |
|---|---|---|---|---|
| Single-language | 60 | 0.950 | 0.982 | 3 |
| Low-mixed | 26 | 1.000 | 0.981 | 0 |
| High-mixed | 34 | 0.941 | 0.972 | 2 |

- 5 errors in 120 sentences. No density trend is visible.
- 4 of the 5 errors are `student_records` predicted as `document_request`
  (transcript, TOR, certified copy). The main error source is the boundary
  between these two labels, not code-switching.
- Stated confidence is almost flat: 116 of 120 answers are at 0.95 or higher,
  and two wrong answers were given 0.99.
- AUROC of confidence for right vs wrong: 0.81 on dev, 0.23 on test. It rests
  on 4 and 1 errors respectively, so neither value is reliable.

**Handoff threshold.** The rule was fixed before looking at train/validation
results: use the lowest confidence threshold at which the answered messages
are at least 95% correct on dev. Dev accuracy with no handoff is 0.962, so the
rule gives **threshold 0 (no handoff)**. On test this answers 15/16 correctly
(0.938). For comparison only (not the chosen rule): a threshold of 0.97 hands
off 8/104 dev messages and catches 3 of the 4 dev errors, but catches none on
test. Handing off every high-mixed message instead hands off about 30% of
messages and catches 2 of the 4 dev errors.

**Conclusion.** On this small, synthetic and fairly easy dataset, the model is
accurate at every density level and its stated confidence is high almost
everywhere. The data is not able to test whether confidence tracks accuracy
as density increases: there are too few errors, and they come from a label
boundary rather than from code-switching.

## Limitations

- The dataset is small and synthetic, written by a small number of authors.
  It does not capture real student writing, typos, slang or spelling variants.
- The test split has 16 sentences and the whole dataset has 5 LLM errors.
  No result here is statistically meaningful, and results do not generalize
  beyond this dataset.
- Many examples are near-paraphrases of each other. The train/validation/test
  split is grouped by paraphrase to avoid data leakage.
- Borrowed terms (for example "clearance", "tuition") and affixed English
  roots (for example "i-request") are ambiguous. They are handled explicitly
  in the token-level annotation guidelines. Only 10 sentences have gold token
  tags, so the tagger's accuracy is only lightly checked.
- `student_records` and `document_request` overlap (for example, requests for
  a transcript or a certified copy). Most LLM errors fall on this boundary;
  the annotation guidelines should settle it before more data is labelled.
- Confidence is the model's own stated number in its reply, not a measured
  probability. It was almost always 0.95 or higher.
- One model was tested (`openai/gpt-oss-120b` on Groq). Hosted models change
  or are retired, so exact numbers may not be reproducible later.
- The intent descriptions in the prompt were written by the project authors
  and affect the results.
- The demo is for research only and is not suitable for real student use.
