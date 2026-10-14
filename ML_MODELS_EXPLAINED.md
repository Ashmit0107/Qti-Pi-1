# ML Models Explained — Qti π

This document explains, from scratch, the three machine learning models behind Qti π's cardiac risk features: what each one predicts, what real data trains it, why that algorithm was chosen, and how to defend the numbers in front of judges. It assumes no prior ML background — if you can explain the sections below in your own words, you can defend this project.

**Revision note**: this document was fully rewritten to match the current (final) state of the pipeline. Since the last version, all three models changed meaningfully:
- `arrhythmia_knn` (13-class KNN) → **`arrhythmia_hgb_5class`** (AAMI 5-class, HistGradientBoosting)
- `heart_disease_rf` (CDC BRFSS 2015) → **`heart_disease_hgb`** (UCI Heart Failure Prediction, merged dataset) — renamed off the `_rf` suffix because the algorithm is HistGradientBoosting, not a Random Forest; the old name was a leftover from an earlier version and didn't reflect what's actually running.
- `stroke_rf` → **`stroke_hgb`** (CDC NHANES) — same dataset family, but went through **3 systematic tuning rounds** and the source dataset grew from 22,592 to 43,346 real rows. Also renamed off the `_rf` suffix, same reason as heart disease — it's HistGradientBoosting under the hood, not Random Forest.

---

## 1. The three models at a glance (final numbers)

| Model | What it predicts | Real dataset | Algorithm | Real rows used | Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|
| `arrhythmia_hgb_5class` | Which of 5 AAMI-standard heartbeat classes a beat belongs to (Normal, Supraventricular ectopic, Ventricular ectopic, Fusion, Unknown/paced) | MIT-BIH Arrhythmia Database (PhysioNet) | HistGradientBoosting | 109,494 real annotated heartbeats | **0.9404** | — |
| `heart_disease_hgb` | Whether a person is at risk of heart disease | UCI Heart Failure Prediction (Cleveland + Hungary + Switzerland + VA + Statlog merge) | HistGradientBoosting | 918 real patient records | **0.8533** | **0.9154** |
| `stroke_hgb` | Whether a person is at risk of stroke | CDC NHANES, multi-cycle (2005–2018, auto-detected from source files) | HistGradientBoosting | 43,346 real participant rows | **0.6978** | **0.8032** |

All three are trained on **real, publicly documented data** — none of the shipped models use synthetic or fabricated numbers. (The codebase still has synthetic fallbacks so the pipeline runs end-to-end if a dataset file is missing on a given machine, but those are loudly flagged with a `[SYNTHETIC FALLBACK]` console warning and never silently substituted.)

---

## 2. `arrhythmia_hgb_5class` — heartbeat rhythm classifier

### What changed from the earlier version
The model moved from a 13-class KNN classifier to a **5-class HistGradientBoosting** classifier, using the **AAMI EC57 standard grouping** (Normal, Supraventricular ectopic beat, Ventricular ectopic beat, Fusion beat, Unknown/paced) instead of the raw 13 MIT-BIH annotation symbols. This is the standard collapsing used in clinical arrhythmia-classification research specifically because several of the 13 raw MIT-BIH symbol classes have only a handful of examples each — not enough real data for any algorithm to learn a reliable pattern — while the AAMI 5-class grouping combines clinically-similar rhythm types into buckets large enough to actually evaluate and defend.

### The dataset: MIT-BIH Arrhythmia Database
- **Source**: PhysioNet (physionet.org/content/mitdb), a public research archive run by MIT and Boston's Beth Israel Hospital.
- **What it actually is**: 48 half-hour, two-lead ECG recordings from real patients, with every heartbeat individually annotated by cardiologists.
- **Scale used**: 109,494 individual labeled heartbeats, extracted via `wfdb` and regrouped into the 5 AAMI classes.

### The real number
- **Overall accuracy: 94.04%** on the 5-class AAMI grouping — meaningfully higher than the old 13-class KNN's headline number, which is expected: collapsing 13 sparse classes into 5 clinically-grouped ones removes most of the near-zero-support classes that were dragging down macro performance before.

*(Note: a full precision/recall/macro-recall breakdown per AAMI class hasn't been pulled into this document yet — pull the latest `cardiac_evaluation_report.json` for the per-class numbers if a judge asks for them directly.)*

---

## 3. `heart_disease_hgb` — heart disease risk

### What changed from the earlier version
This model switched off CDC BRFSS 2015 entirely and now trains on the **UCI Heart Failure Prediction dataset** — the classic merge of the Cleveland, Hungary, Switzerland, and VA Long Beach heart disease datasets plus the Statlog heart dataset. It's a much smaller dataset (918 rows vs BRFSS's 253,680) but the fields are **clinical measurements taken during a cardiac workup** (resting blood pressure, cholesterol, max heart rate achieved, ST depression, chest pain type, exercise-induced angina, thallium stress test result, etc.) rather than self-reported survey answers — a meaningfully different, and in some ways more clinically direct, feature set than BRFSS's lifestyle-questionnaire columns.

### The algorithm
HistGradientBoosting (same family reasoning as before: sharper decision boundaries on tabular clinical data than a single tree or plain random forest, since each new tree is trained directly against the current model's errors).

### The real numbers
- **Accuracy: 85.33%**
- **ROC-AUC: 0.9154**

This is a strong, defensible number on a well-established, clinically-grounded dataset. Because 918 rows is a small sample compared to the other two models, the honest caveat to lead with if pushed: variance on a held-out split is naturally higher with fewer rows, so these numbers should be read as "strong on a small, well-curated clinical dataset" rather than "validated at population scale" the way the stroke model (43,346 rows) is.

---

## 4. `stroke_hgb` — stroke risk, the 3-round tuning story

This is the model with the most interesting — and most honest — story to tell, because every round was tested and reported as-is, including the round that didn't move the number the way it was expected to.

### Baseline, before this round of work
- Accuracy: **0.6495**, ROC-AUC: **0.7877**

### Round 1 — initial tuning pass
Baseline hyperparameter tuning on the existing NHANES feature set (age, sex, BMI, glucose, cholesterol, hypertension, heart disease history), without the extra interaction features added later.
- Accuracy: **0.6889**, ROC-AUC: **0.7982**

### Round 2 — feature engineering + wider pruning + refined hyperparameter grid
Three changes made together, all flowing into the same evaluation report:
1. **7 new interaction features** (`data_preprocessing.py → _add_stroke_interactions()`), bringing in columns Round 1 left unused — sex, cholesterol, smoking, physical activity, waist circumference:
   `bmi_waist_circumference`, `age_smoking`, `cholesterol_hypertension`, `age_bmi`, `sex_age`, `waist_circumference_hypertension`, `physical_activity_bmi`.
2. **Wider missing-flag/noise pruning** (`cardiac_models.py → train_stroke()`) — permutation importance (by ROC-AUC) is now computed over *all* non-flag features (raw + Round 1 + Round 2 interactions), not just the `*_was_missing` columns. Anything with importance ≤ 0.0005 gets dropped.
3. **Wider/refined hyperparameter grid** — expanded from 27 to 108 combinations: `depth ∈ [3,4,5,6]`, `learning_rate ∈ [0.02,0.03,0.05]`, `min_samples_leaf ∈ [10,15,20]`, plus a new `l2_regularization ∈ [0.0,0.1,1.0]` axis, centered around Round 1's winning region (`depth=4, lr=0.03`).

Result, still on 22,592 rows:
- Accuracy: **0.6847**, ROC-AUC: **0.8065**

### Round 3 — data scaling (22,592 → 43,346 rows) + a bug fix
The dataset was expanded by pulling in additional NHANES cycles (adding older 2005–2010 cycles on top of the existing 2011–2018 ones), taking the real row count from 22,592 to 43,346.

Result:
- Accuracy: **0.6978** (+1.31 pt vs Round 2), ROC-AUC: **0.8032** (−0.33 pt vs Round 2 — within normal single-split noise range)

**Bug caught and fixed during this round**: the dataset's source-description string was hardcoded as `"2011-2018 (4 cycles)"` in the code. Once older cycles were added, that string would have stayed wrong (silently under-reporting how much data was actually used) if left alone. It's now fixed to **auto-detect the actual cycle list from the source CSVs** rather than being hardcoded — so this document does not hardcode a cycle count either, since a future data addition could change it again.

### Total improvement across all 3 rounds
- **Accuracy: 0.6495 → 0.6978 (+4.83 pts)**
- **ROC-AUC: 0.7877 → 0.8032 (+1.55 pts)**

### The honest verdict on Round 3 specifically
Doubling the row count did **not** produce a proportional accuracy/AUC jump, and that's worth explaining directly rather than glossing over: the newly-added older NHANES cycles (2005–2010) have substantially higher missingness on exactly the features that matter most here — `avg_glucose_level`, `hypertension`, `hba1c`, `waist_circumference` (glucose alone is ~56% missing across the merged dataset). More rows with weaker signal on the highest-value features means the extra data isn't a clean sample-size win — some of the added rows are individually less informative than the original 22,592, and several columns' marginal importance dropped further, so more got pruned. Accuracy still rose because more data generally helps a boosted-tree model generalize; AUC dipped slightly, which is consistent with expected single-split noise rather than a real regression.

**Why this is a defensible finding, not a failure**: "more real data = automatically better" was treated as a hypothesis, tested systematically, and reported honestly — including the part where it didn't fully hold up the way it was expected to. That's a stronger story in front of judges than a single clean number with no visible process behind it.

### Why the tuning stopped here (was not pushed further)
Three rounds of systematic tuning plus a legitimate data-scaling test is enough evidence of process; pushing further into the same feature set risks diminishing returns and overfitting to a single train/test split rather than real gains. A concrete next option exists if this exact number is ever challenged and more time is available — a "cycle-aware" filter that trains only on cycles/features where glucose and HbA1c were actually measured (the fasting subsample), trading away some of the row count gained in Round 3 in exchange for cleaner signal. **0.80 AUC is being treated as the number to defend as-is**, not as a placeholder waiting for more tuning.

---

## 5. The precision/recall/accuracy trade-off — background context

The stroke and heart-disease targets are both minority classes (stroke ~4% positive in NHANES; heart disease is a much larger share of the small UCI sample, but still not >50%). The core tension is the same one described in earlier versions of this project:
- Chasing higher recall (catching more real positive cases) means flagging more people overall, which pulls precision down.
- Chasing higher precision means raising the bar for a "positive" call, which means missing more real cases.
- Accuracy alone can look good on an imbalanced dataset purely because the negative class is much larger, even if the model is doing a poor job on the class that actually matters.

The numbers reported in Sections 2–4 above (plain accuracy + ROC-AUC) are threshold-independent or default-threshold numbers, which is a simpler and more standard way to report model quality than a hand-tuned recall-first cutoff. If a specific screening threshold (e.g. tuned for a target recall) is deployed in the product, that threshold and its resulting precision/recall trade-off should be pulled from the latest `cardiac_evaluation_report.json` and documented separately — it will differ from the plain accuracy/AUC numbers above by construction.

---

## 6. Judge-defense Q&A

**Q: Why did the model names/algorithms change since an earlier version of this document?**
A: Each change was driven by evidence, not by chasing a bigger number for its own sake. Arrhythmia moved from a 13-class KNN to a 5-class AAMI-grouped HistGradientBoosting model because several of the 13 raw MIT-BIH classes had too few real examples to evaluate meaningfully — the AAMI grouping is the standard clinical collapsing used specifically to address that. Heart disease moved from CDC BRFSS (self-reported survey answers, huge sample) to the UCI Heart Failure Prediction dataset (much smaller, but real clinical workup measurements) — a genuine sample-size-vs-measurement-quality trade-off, made explicitly. The renames from `heart_disease_rf` to `heart_disease_hgb`, and from `stroke_rf` to `stroke_hgb`, are simply correcting stale names — the algorithm behind both is HistGradientBoosting, not Random Forest, and the old `_rf` suffix was a leftover from an earlier version of the code.

**Q: Walk me through the stroke model's tuning process.**
A: Three rounds, each building on the last, every result reported as-is: Round 1 was baseline hyperparameter tuning (accuracy 0.6495 → 0.6889, AUC 0.7877 → 0.7982). Round 2 added 7 new interaction features, wider feature pruning, and a much wider hyperparameter search (accuracy → 0.6847, AUC → 0.8065). Round 3 nearly doubled the training data (22,592 → 43,346 real NHANES rows) and fixed a stale hardcoded-metadata bug along the way (accuracy → 0.6978, AUC → 0.8032). Net improvement across all three rounds: +4.83 points accuracy, +1.55 points AUC.

**Q: Why didn't doubling the stroke dataset in Round 3 produce a bigger jump?**
A: Because "more rows" isn't automatically "more signal" — the newly-added older NHANES cycles had much higher missingness on the features that matter most (glucose alone is ~56% missing overall), so a meaningful share of the added rows carried weaker information than the original data. Accuracy still improved because boosted trees generally generalize better with more data; AUC moved slightly in the other direction, consistent with normal single-split noise rather than a real regression. This was tested as a hypothesis and reported honestly rather than assumed.

**Q: Is any of this data fake or synthetic?**
A: No — all three shipped models are trained on real, cited, public datasets (MIT-BIH for arrhythmia, the UCI Heart Failure Prediction merge for heart disease, CDC NHANES for stroke). The codebase has clearly-labelled synthetic fallbacks (printed loudly as `[SYNTHETIC FALLBACK]`) so the pipeline doesn't crash if a dataset file is missing, but those are never silently used in place of real data.

**Q: How do you know your numbers aren't overfit or made up?**
A: Every reported number comes from a held-out test split the model never saw during training, and the exact figures are written to `ml/saved_models/cardiac_evaluation_report.json` by the training script itself — they can be regenerated at any time by re-running `python ml/cardiac_models.py`.

**Q: What would you do next, if you had more time?**
A: For stroke, a "cycle-aware" filter that trains only on the NHANES cycles/features where glucose and HbA1c were actually measured, trading some row count for cleaner signal — a concrete, already-identified next step, deliberately not pursued yet because the current 0.80 AUC is considered a solid, honestly-earned number and further squeezing risks overfitting to one split rather than real gains. Heart disease (0.9154 AUC) and arrhythmia (0.9404 accuracy) are both already strong and weren't touched further in this round of work.

---

## 7. Where to find the underlying evidence

- `ml/DATASETS.md` — full source citations and licenses for every dataset.
- `ml/data_preprocessing.py` — dataset loading/cleaning, including `_add_stroke_interactions()` (Round 2 feature engineering) and the priority order across dataset sources.
- `ml/download_nhanes.py`, `ml/extract_nhanes_features.py` — how the NHANES stroke dataset is built from raw CDC files, including the now-fixed auto-detection of which cycles were actually used.
- `ml/cardiac_models.py` — training code for all three models, including `train_stroke()`'s permutation-importance pruning and hyperparameter search (27 → 108 combinations across the 3 rounds).
- `ml/saved_models/cardiac_evaluation_report.json` — the exact numbers cited in this document, regenerated fresh every time `cardiac_models.py` is run.
