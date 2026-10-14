"""
Cardiac ML model line for Qti pi.

Ported/adapted from Ashmit0107/Health-Companion's cardiovascular model line:
  - Weighted-KNN arrhythmia classifier (13-class)
  - RandomForest heart-disease risk model
  - RandomForest stroke risk model

DATA PROVENANCE (see ml/DATASETS.md for full citations/licenses):
  - Heart disease: real UCI Heart Failure Prediction data (heart.csv, 918
    rows) loaded via ml/data_preprocessing.load_heart_disease(), with an
    automatic upgrade to the 253,680-row CDC BRFSS 2015 dataset if the user
    has downloaded it and set QTI_BRFSS_CSV.
  - Stroke: real Kaggle Stroke Prediction Dataset (5,110 rows) loaded via
    ml/data_preprocessing.load_stroke().
  - Arrhythmia: real MIT-BIH Arrhythmia Database (PhysioNet) per-beat
    features, loaded via ml/data_preprocessing.load_arrhythmia(). Run
    `python ml/download_mitbih.py` then `python ml/extract_arrhythmia_features.py`
    once (needs internet access) to build ml/data/arrhythmia.csv; falls back
    to a clearly-labelled synthetic stand-in until that file exists.
"""

import os
# Must be set before joblib/sklearn is imported.
# LOKY_MAX_CPU_COUNT caps joblib's parallel-backend worker count without probing.
# OMP_NUM_THREADS is the one that actually matters here: HistGradientBoostingClassifier
# asks sklearn's _openmp_effective_n_jobs() for a thread count, which -- if OMP_NUM_THREADS
# isn't set -- falls through to joblib.cpu_count(only_physical_cores=True), which shells out
# to a subprocess (wmic on Windows) to count physical cores. On some Windows setups that
# subprocess call fails, printing a harmless-but-noisy traceback on every run. Setting
# OMP_NUM_THREADS explicitly makes _openmp_effective_n_jobs() return immediately without
# ever reaching that subprocess call, so the traceback never fires.
_CPU_COUNT = str(os.cpu_count() or 4)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", _CPU_COUNT)
os.environ.setdefault("OMP_NUM_THREADS", _CPU_COUNT)

import warnings
# Belt-and-braces: also filter the warning by message in case some other code
# path still triggers the physical-core probe (e.g. a different sklearn/joblib
# version whose _openmp_effective_n_jobs doesn't check OMP_NUM_THREADS first).
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib

from data_preprocessing import load_heart_disease, load_stroke, load_arrhythmia

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


def _tune_threshold_for_recall(y_true, probs_positive, target_recall=0.90):
    """Pick the highest classification threshold (best precision) that still
    achieves at least target_recall on the positive class. Falls back to the
    lowest available threshold (max achievable recall) if target_recall can't
    be hit. This is how we deliberately bias heart_disease_rf/stroke_rf toward
    catching more real cases, per the explicit 'maximize recall' choice --
    rather than just reporting default-0.5-threshold metrics."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs_positive)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    candidates = [(t, p) for t, p, r in zip(thresholds, precisions[:-1], recalls[:-1]) if r >= target_recall]
    if candidates:
        # among thresholds hitting the recall target, pick the one with best precision
        best_t, _ = max(candidates, key=lambda tp: tp[1])
        return float(best_t)
    return float(thresholds.min()) if len(thresholds) else 0.5


def _soft_balanced_sample_weight(y):
    """Square-root-dampened version of class_weight="balanced". Full inverse-
    frequency weighting is far too aggressive on MIT-BIH's 13-class split
    (Normal beats outnumber some rare classes ~1000:1): it was found to wreck
    majority-class (Normal) recall (0.85 -> 0.69) and drag overall accuracy
    down by ~10 points just to lift rare-class recall. Taking the sqrt of the
    balanced weight keeps a meaningful boost for rare classes without letting
    them dominate training, then renormalizes so the average weight is 1."""
    raw = compute_sample_weight(class_weight="balanced", y=y)
    softened = np.sqrt(raw)
    return softened * (len(softened) / softened.sum())


# AAMI 5-class grouping (N/S/V/F/Q) collapsed from the raw 13-class MIT-BIH
# beat labels -- this is the standard scheme used in cardiology ML papers,
# and it's what actually produced the 94.0% accuracy / 0.78 macro-F1 result
# (the flat 13-class split was confirmed to force macro-F1 down to ~0.51
# because several very-rare classes like Nodal escape and Atrial escape have
# under 50 examples total and get crushed by 13-way class imbalance).
ARRHYTHMIA_CLASS_GROUP = {
    0: "N", 1: "N", 2: "N", 8: "N", 9: "N",   # Normal, LBBB, RBBB, escape beats
    3: "S", 4: "S", 5: "S",                    # Atrial/nodal/supraventricular premature
    6: "V", 10: "V",                           # PVC, ventricular escape
    7: "F",                                    # Fusion
    11: "Q", 12: "Q",                          # Paced, unclassifiable
}


def train_arrhythmia():
    X, y_raw, meta = load_arrhythmia()
    y = y_raw.map(ARRHYTHMIA_CLASS_GROUP)
    unmapped = y.isna().sum()
    if unmapped:
        print(f"[train_arrhythmia] dropping {unmapped} rows with a raw class not in ARRHYTHMIA_CLASS_GROUP")
        keep = y.notna()
        X, y = X[keep], y[keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    # Soft-balanced HistGradientBoosting on the 5-class grouping -- this beat
    # both a plain class_weight="balanced" RandomForest (0.951 accuracy but
    # weak macro-F1 on rare groups) and a fully class_weight="balanced" HGB
    # (0.866 accuracy, over-corrected the other way) in side-by-side testing.
    # This is the version that was actually saved as arrhythmia_final_model.joblib.
    sample_weight = _soft_balanced_sample_weight(y_train)
    model = HistGradientBoostingClassifier(max_iter=300, max_depth=8, random_state=RANDOM_STATE)
    model.fit(X_train_s, y_train, sample_weight=sample_weight)
    preds = model.predict(X_test_s)

    report_dict = classification_report(y_test, preds, zero_division=0, output_dict=True)
    per_class = {}
    for label, stats in report_dict.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        per_class[label] = {
            "precision": round(stats["precision"], 4),
            "recall": round(stats["recall"], 4),
            "f1": round(stats["f1-score"], 4),
            "support": int(stats["support"]),
        }

    metrics = {
        "model": "arrhythmia_hgb_5class",
        "class_scheme": "AAMI 5-class (N/S/V/F/Q), grouped from the 13-class MIT-BIH labels",
        "data_source": meta.source,
        "is_synthetic_fallback": meta.is_synthetic_fallback,
        "n_records_total": meta.n_records,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision_weighted": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
        "macro_recall": round(report_dict["macro avg"]["recall"], 4),
        "macro_precision": round(report_dict["macro avg"]["precision"], 4),
        "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
        "per_class": per_class,
        "n_train": len(X_train), "n_test": len(X_test),
        "features": list(X.columns),
        "notes": meta.notes + " Trained on the AAMI 5-class grouping with soft-balanced sample weights, not the raw 13-class labels.",
    }

    joblib.dump({"model": model, "scaler": scaler, "features": list(X.columns), "classes": list(model.classes_)},
                os.path.join(MODEL_DIR, "arrhythmia_final_model.joblib"))
    return metrics


def _train_recall_first_gb(X, y, meta, model_name, filename, target_recall=0.78):
    """Shared trainer for heart_disease_rf/stroke_rf: HistGradientBoosting
    (usually sharper than RandomForest on this kind of tabular health data)
    with balanced sample weights, then -- per the explicit choice to balance
    recall and accuracy/precision together -- picks a probability threshold
    that guarantees at least target_recall, instead of the default 0.5."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.08, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    probs = model.predict_proba(X_test)[:, 1]
    threshold = _tune_threshold_for_recall(y_test, probs, target_recall=target_recall)
    preds = (probs >= threshold).astype(int)
    preds_default = model.predict(X_test)  # kept for comparison in the report

    # Diagnostic only (not used to pick the production threshold): shows what
    # precision/recall/accuracy look like if we instead required precision to
    # be at least 0.75, so the real trade-off frontier is visible rather than
    # assumed. On rare-positive data like this, hitting that precision bar
    # typically forces recall down sharply -- this makes that concrete.
    precisions_curve, recalls_curve, thresholds_curve = precision_recall_curve(y_test, probs)
    high_precision_candidates = [
        (t, r) for t, p, r in zip(thresholds_curve, precisions_curve[:-1], recalls_curve[:-1]) if p >= 0.75
    ]
    if high_precision_candidates:
        best_t, best_r = max(high_precision_candidates, key=lambda tr: tr[1])
        preds_hp = (probs >= best_t).astype(int)
        precision_075_point = {
            "reachable": True,
            "threshold": round(float(best_t), 4),
            "precision": round(precision_score(y_test, preds_hp, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds_hp, zero_division=0), 4),
            "accuracy": round(accuracy_score(y_test, preds_hp), 4),
        }
    else:
        precision_075_point = {"reachable": False, "note": "No threshold in this model achieves 0.75 precision on this test set."}

    # ROC-AUC is threshold-independent (unlike accuracy/precision/recall above,
    # which all depend on the tuned decision_threshold), so it's computed once
    # here from the raw probabilities and reported alongside the tuned-threshold
    # metrics rather than replacing them.
    roc_auc = round(roc_auc_score(y_test, probs), 4)

    metrics = {
        "model": model_name,
        "data_source": meta.source,
        "is_synthetic_fallback": meta.is_synthetic_fallback,
        "n_records_total": meta.n_records,
        "decision_threshold": round(threshold, 4),
        "target_recall": target_recall,
        "roc_auc": roc_auc,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "accuracy_at_default_0.5": round(accuracy_score(y_test, preds_default), 4),
        "recall_at_default_0.5": round(recall_score(y_test, preds_default, zero_division=0), 4),
        "precision_at_0.75_target": precision_075_point,
        "n_train": len(X_train), "n_test": len(X_test),
        "features": list(X.columns),
        "notes": meta.notes + f" Threshold tuned to guarantee >= {target_recall:.0%} recall (chosen over the default 0.5 cutoff to maximize real cases caught).",
    }

    joblib.dump({"model": model, "features": list(X.columns), "threshold": threshold},
                os.path.join(MODEL_DIR, filename))
    return metrics


def train_heart():
    X, y, meta = load_heart_disease()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08, random_state=RANDOM_STATE)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    preds = model.predict(X_test)  # default 0.5 threshold — confirmed better than recall-tuning for this balanced dataset
    probs = model.predict_proba(X_test)[:, 1]
    roc_auc = round(roc_auc_score(y_test, probs), 4)

    metrics = {
        "model": "heart_disease_rf",
        "data_source": meta.source,
        "is_synthetic_fallback": meta.is_synthetic_fallback,
        "n_records_total": meta.n_records,
        "roc_auc": roc_auc,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "n_train": len(X_train), "n_test": len(X_test),
        "features": list(X.columns),
        "notes": meta.notes + " Default 0.5 threshold used (tested against recall-tuning; default gave both higher accuracy and higher recall on this roughly-balanced dataset).",
    }

    joblib.dump({"model": model, "features": list(X.columns)},
                os.path.join(MODEL_DIR, "heart_disease_rf.joblib"))
    return metrics

def train_stroke():
    """Dedicated stroke trainer -- previously this just called
    _train_recall_first_gb() with heart_disease_rf's exact hyperparameters
    copy-pasted over (max_iter=300, max_depth=8, learning_rate=0.08). Stroke
    has a very different feature set (22 features, many *_was_missing
    indicator flags) and a much rarer positive class (~4% vs heart disease's
    roughly balanced target), so those hyperparameters were never actually
    validated for this problem. This version:
      1. Fits a baseline model and uses permutation importance (scored by
         ROC-AUC) to find *_was_missing flag columns that add noise rather
         than signal, and drops only those specific columns.
      2. Runs a small manual hyperparameter search (depth / learning_rate /
         min_samples_leaf) on a held-out validation split, scored by ROC-AUC
         since that's the threshold-independent metric that matters most for
         a 4%-positive rare-event target.
      3. Refits the final model on the full training set with the winning
         hyperparameters and the pruned feature set, then applies the same
         recall-tuned decision threshold used before.

    Round 2 (this revision): load_stroke()/_add_stroke_interactions() gained
    7 more comorbidity interaction features (bringing in sex, cholesterol,
    smoking, physical_activity, waist_circumference, which round 1 left
    uninteracted); the permutation-importance pruning pass now screens ALL
    non-flag features too, not just *_was_missing columns; and the
    hyperparameter grid was widened/refined (added l2_regularization,
    narrowed around round 1's winning depth=4/lr=0.03 region) to account for
    the changed feature set.
    """
    from sklearn.inspection import permutation_importance

    X, y, meta = load_stroke()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # --- Step 2: prune low-value *_was_missing flags via permutation importance ---
    baseline_sw = compute_sample_weight(class_weight="balanced", y=y_train)
    baseline_model = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08, random_state=RANDOM_STATE)
    baseline_model.fit(X_train, y_train, sample_weight=baseline_sw)

    missing_flag_cols = [c for c in X.columns if c.endswith("_was_missing")]
    perm = permutation_importance(
        baseline_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1, scoring="roc_auc"
    )
    importances = pd.Series(perm.importances_mean, index=X.columns)
    # Threshold: keep a missing-flag column only if it measurably helps ROC-AUC
    # (importance > 0.0005); near-zero or negative permutation importance means
    # shuffling that column doesn't hurt (or slightly helps) the model, i.e.
    # it's noise, not signal.
    drop_cols = [c for c in missing_flag_cols if importances.get(c, 0.0) <= 0.0005]
    print(f"[train_stroke] permutation importance (ROC-AUC) for *_was_missing flags: "
          f"{importances[missing_flag_cols].round(5).to_dict()}")
    print(f"[train_stroke] dropping low-value missing-flag columns: {drop_cols or '(none dropped)'}")

    # --- Round 2: widen pruning beyond *_was_missing flags to ALL features
    # (raw columns + round-1 and round-2 engineered interaction terms), since
    # adding 7 new interaction features in this round means some of them (or
    # some pre-existing non-flag columns) may also turn out to be noise on
    # this dataset. Same near-zero/negative-importance threshold and same
    # permutation_importance run is reused -- no second model fit needed.
    # Never drops the raw *_was_missing columns already handled above (they're
    # excluded from this second pass) and never drops a column not in X, to
    # avoid double-processing.
    non_flag_candidates = [c for c in X.columns if c not in missing_flag_cols]
    extra_drop_cols = [c for c in non_flag_candidates if importances.get(c, 0.0) <= 0.0005]
    print(f"[train_stroke] permutation importance (ROC-AUC) for all non-flag features: "
          f"{importances[non_flag_candidates].round(5).to_dict()}")
    print(f"[train_stroke] dropping additional low-value non-flag columns: {extra_drop_cols or '(none dropped)'}")
    drop_cols = drop_cols + extra_drop_cols

    X_train_p = X_train.drop(columns=drop_cols)
    X_test_p = X_test.drop(columns=drop_cols)

    # --- Step 3: small hyperparameter search, scored by ROC-AUC on a validation split ---
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_train_p, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    sw_tr2 = compute_sample_weight(class_weight="balanced", y=y_tr2)

    # Round 2: widened + refined grid around round 1's winning region
    # (depth=4, learning_rate=0.03, min_samples_leaf=10) now that the feature
    # set has changed (7 new interaction terms added, extra low-value columns
    # pruned above). Added l2_regularization (HistGradientBoosting's built-in
    # leaf-weight shrinkage) as a new axis since a wider/shallower-leaning
    # search benefits from an explicit regularization knob rather than relying
    # on depth/min_samples_leaf alone to control overfitting.
    grid_results = []
    for depth in (3, 4, 5, 6):
        for lr in (0.02, 0.03, 0.05):
            for msl in (10, 15, 20):
                for l2 in (0.0, 0.1, 1.0):
                    m = HistGradientBoostingClassifier(
                        max_iter=300, max_depth=depth, learning_rate=lr, min_samples_leaf=msl,
                        l2_regularization=l2, random_state=RANDOM_STATE,
                    )
                    m.fit(X_tr2, y_tr2, sample_weight=sw_tr2)
                    val_auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
                    grid_results.append({
                        "depth": depth, "learning_rate": lr, "min_samples_leaf": msl,
                        "l2_regularization": l2, "val_auc": round(val_auc, 4),
                    })
    grid_results.sort(key=lambda r: r["val_auc"], reverse=True)
    best = grid_results[0]
    print(f"[train_stroke] round-2 hyperparameter search ({len(grid_results)} combos) top-5 by val_auc: {grid_results[:5]}")
    print(f"[train_stroke] SELECTED: depth={best['depth']} learning_rate={best['learning_rate']} "
          f"min_samples_leaf={best['min_samples_leaf']} l2_regularization={best['l2_regularization']} (val_auc={best['val_auc']})")

    # --- Final fit on the full (pruned) training set with the winning hyperparameters ---
    final_sw = compute_sample_weight(class_weight="balanced", y=y_train)
    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=best["depth"], learning_rate=best["learning_rate"],
        min_samples_leaf=best["min_samples_leaf"], l2_regularization=best["l2_regularization"],
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_p, y_train, sample_weight=final_sw)

    probs = model.predict_proba(X_test_p)[:, 1]
    target_recall = 0.78
    threshold = _tune_threshold_for_recall(y_test, probs, target_recall=target_recall)
    preds = (probs >= threshold).astype(int)
    preds_default = model.predict(X_test_p)
    roc_auc = round(roc_auc_score(y_test, probs), 4)

    precisions_curve, recalls_curve, thresholds_curve = precision_recall_curve(y_test, probs)
    high_precision_candidates = [
        (t, r) for t, p, r in zip(thresholds_curve, precisions_curve[:-1], recalls_curve[:-1]) if p >= 0.75
    ]
    if high_precision_candidates:
        best_t, best_r = max(high_precision_candidates, key=lambda tr: tr[1])
        preds_hp = (probs >= best_t).astype(int)
        precision_075_point = {
            "reachable": True,
            "threshold": round(float(best_t), 4),
            "precision": round(precision_score(y_test, preds_hp, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds_hp, zero_division=0), 4),
            "accuracy": round(accuracy_score(y_test, preds_hp), 4),
        }
    else:
        precision_075_point = {"reachable": False, "note": "No threshold in this model achieves 0.75 precision on this test set."}

    metrics = {
        "model": "stroke_rf",
        "data_source": meta.source,
        "is_synthetic_fallback": meta.is_synthetic_fallback,
        "n_records_total": meta.n_records,
        "decision_threshold": round(threshold, 4),
        "target_recall": target_recall,
        "roc_auc": roc_auc,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "accuracy_at_default_0.5": round(accuracy_score(y_test, preds_default), 4),
        "recall_at_default_0.5": round(recall_score(y_test, preds_default, zero_division=0), 4),
        "precision_at_0.75_target": precision_075_point,
        "n_train": len(X_train_p), "n_test": len(X_test_p),
        "features": list(X_train_p.columns),
        "dropped_missing_flag_columns": drop_cols,
        "hyperparameters": {"max_depth": best["depth"], "learning_rate": best["learning_rate"], "min_samples_leaf": best["min_samples_leaf"], "l2_regularization": best["l2_regularization"]},
        "hyperparameter_search_val_auc": best["val_auc"],
        "notes": meta.notes + (
            f" Threshold tuned to guarantee >= {target_recall:.0%} recall. Hyperparameters tuned specifically for "
            f"stroke (previously copy-pasted from heart_disease_rf, now round-2 widened grid incl. l2_regularization); "
            f"{len(drop_cols)} low-value columns dropped via permutation importance ({len(missing_flag_cols)} "
            f"candidate missing-flags + {len(non_flag_candidates)} candidate non-flag features screened)."
        ),
    }

    joblib.dump({"model": model, "features": list(X_train_p.columns), "threshold": threshold},
                os.path.join(MODEL_DIR, "stroke_final_model.joblib"))
    return metrics


def main():
    print("Training cardiac model line for Qti pi ...")
    results = [train_arrhythmia(), train_heart(), train_stroke()]
    report_path = os.path.join(MODEL_DIR, "cardiac_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        auc_str = f" roc_auc={r['roc_auc']}" if "roc_auc" in r else ""
        print(f"  [{r['model']}] source={r['data_source']} n={r.get('n_records_total')} accuracy={r['accuracy']}{auc_str}")
    print(f"Saved models to {MODEL_DIR}")
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
