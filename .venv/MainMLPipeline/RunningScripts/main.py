# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
# ---

# %% [markdown]
# # Iris dataset (scikit-learn) — quick analysis
# This file is a **percent-format notebook** (`# %%` cells). You can run it in VS Code / Cursor as a notebook.
#
# The analysis is structured as:
# - Import and dependency checks
# - Loading the Iris dataset into a typed `DataFrame`
# - Quick exploratory data analysis (EDA)
# - Train/test split
# - Model comparison with cross-validation
# - Final evaluation of the best model....

# %%

# %%
import argparse
import json
import math
import os
import re
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

import warnings
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import pdb

from pathlib import Path
from typing import Dict, Any

from Utilities import (
    HEB_NIQQUD,
    SPACE_NORMALIZE,
    SENSITIVE_TERMS,
    ensure_datetime,
    load_data,
    mask_sensitive,
    normalize_hebrew,
    prepare_labels,
    preprocess_df,
    temporal_split,
)
from Classes import ColumnSelector, ClinicalFeatureExtractor
from Models import  build_vectorizer, build_model,train_with_cv,eval_probs,predict_proba_safe
from Graphics import eval_slices, drift_top_terms, top_coefficients, plot_aupr_per_class

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin, clone



# %%
# ------------------ HyperParamaters ------------------

MultyLablelMinPredictScoreForEval = 0.5
CUTOFF = pd.Timestamp("2023-10-07")


# %%

# ------------------ Utilities, data handling, split logic, and labels ------------------
# Implementations for these live in `Utilities.py` and are imported above.

# %% [markdown]
# Explain a bit what is happening here

# %%

# ------------------ Vectorizers ------------------
# ------------------ Model ------------------
# ------------------ Training with CV ------------------
# ------------------ Evaluation ------------------
# ------------------ Drift/Explainability ------------------
# ------------------ Lock feature space for drift/explainability ------------------
# ------------------ Model comparison: Detailed CV scores ------------------
# ------------------ Prepare post labels in the pre-learned class space ------------------
# ------------------ Fit a clean vectorizer on all pre text to get stable feature names for analyses ------------------
# ------------------ Detailed CV scores ------------------
# ------------------ Prepare post labels in the pre-learned class space ------------------

# %%

# %% [markdown]
# ### Feature distributions by species

# %%


# %% [markdown]
# Explain a bit what is happening here

# %% [markdown]
# Explain a bit what is happening here
# %%
# === Step 1: Parse CLI arguments ============================================================
p = argparse.ArgumentParser(description="Hebrew clinical text classification pipeline.")

p.add_argument(
    "--data",
    default=r"C:\Users\sagil\GEAH\MockDataSatatus.csv",
    help="Path to CSV or Parquet with id,text,diagnosis,date,soldier_flag",
)

p.add_argument("--multilabel", action="store_true", default=False, help="Treat diagnosis as multi-label string")
p.add_argument("--base_model", choices=["logreg", "linearsvc"], default="logreg")
p.add_argument("--use_char", action="store_true", default=False, help="Add char n-grams 3-5")
p.add_argument("--max_features_word", type=int, default=5000)
p.add_argument("--max_features_char", type=int, default=2500)
p.add_argument("--ngram_word_max", type=int, default=2)
p.add_argument("--no_calibrate", action="store_true", default=False, help="Disable probability calibration")
p.add_argument("--output_dir", default="/content/drive/MyDrive/MyPapers/PTSD-MisDiagnosis/Info/output", help="Directory to write JSON artifacts")
p.add_argument("--mask_sensitive_test", action="store_true", default=False, help="Run masking robustness test on post split")

if "ipykernel" in sys.modules:
    args = p.parse_args([])
else:
    args = p.parse_args()


# %%

# === Step 2: Ensure output directory exists ================================================
os.makedirs(args.output_dir, exist_ok=True)
graphs_dir = os.path.join(args.output_dir, "graphs")
os.makedirs(graphs_dir, exist_ok=True)

# === Step 3: Load & preprocess raw data ====================================================
# - Normalizes Hebrew text, coerces date, validates schema, boolean soldier_flag
df = load_data(args.data)
df = preprocess_df(df)
print(df.columns)


# %%
# === Step 4: Temporal split (train before 2023-10-07; evaluate on/after) ===================
pre, post = temporal_split(df)
if len(pre) == 0 or len(post) == 0:
    # We require both sides to exist to avoid leakage and to test generalization post-cutoff
    print("ERROR: Pre or Post split empty.", file=sys.stderr)
    sys.exit(1)

# %%

# === Step 5: Encode labels on pre only =====================================================
# - Prevents peeking at post distribution/classes during training
Y_pre, enc, classes = prepare_labels(pre["diagnosis"], multilabel=args.multilabel)



# %%
# === Step 6: Define vectorizer & model =====================================================
# - Word n-grams + optional char n-grams; linear model with optional calibration
vec = build_vectorizer(
    max_features_word=args.max_features_word,
    ngram_word=(1, args.ngram_word_max),
    use_char=args.use_char,
    max_features_char=args.max_features_char,
    ngram_char=(3, 5),
)
model = build_model(
    base_model=args.base_model,
    multilabel=args.multilabel,
    calibrate=not args.no_calibrate,
)



# %%
# === Step 7: Cross-validated training on pre ===============================================
# - Small grid over vectorizer and C; macro-F1 scoring; refit best pipeline


print("args.multilabel:", args.multilabel)
print("Y_pre shape:", getattr(Y_pre, "shape", None))

best, info = train_with_cv(
    pre["text"], Y_pre, vec, model, classes, args.base_model, args.multilabel
)
with open(os.path.join(args.output_dir, "cv_best.json"), "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)



# %% [markdown]
#


# %%
# === Step 8: Lock feature space for drift/explainability ===================================
# - Fit a clean vectorizer on all pre text to get stable feature names for analyses
V = build_vectorizer(
    max_features_word=args.max_features_word,
    ngram_word=(1, args.ngram_word_max),
    use_char=args.use_char,
    max_features_char=args.max_features_char,
    ngram_char=(3, 5),
)
V.fit(pre["text"])

# %% [markdown]
#     # ### Model comparison: Detailed CV scores

# %%
# === Step 9: Prepare post labels in the pre-learned class space ============================
# - Single-label: drop unseen labels; Multi-label: fix the class order using pre classes
if args.multilabel:
    y_post_list = post["diagnosis"].fillna("").map(
        lambda s: [t for t in re.split(r"[|,;/]", s) if t]
    ).tolist()
    Y_post = MultiLabelBinarizer(classes=classes).fit(classes).transform(y_post_list)
else:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_post_idx = post["diagnosis"].map(class_to_idx).fillna(-1).astype(int)
    keep = y_post_idx >= 0
    dropped = (~keep).sum()
    if dropped > 0:
        warnings.warn(f"Dropping {dropped} post rows with unseen labels.")
    post = post[keep].copy()
    Y_post = y_post_idx[keep].to_numpy()

# %% [markdown]
# Explain a bit what is happening here

# %%
# === Step 10: Evaluate best model on post split ============================================
# - Use the fitted steps from the CV best pipeline to avoid leakage
vec_step = best.named_steps["vec"]
clf_step = best.named_steps["clf"]

# Transform post text with the *trained* vectorizer
#X_post_vec = vec_step.transform(pd.DataFrame({"text": post["text"]}))
X_post_text = post["text"].astype(str).tolist()
X_post_vec = vec_step.transform(X_post_text)
#pdb.set_trace()

# Get probabilities (calibrated when enabled; logistic link fallback otherwise)
y_prob_post = (
    clf_step.predict_proba(X_post_vec)
    if hasattr(clf_step, "predict_proba")
    else predict_proba_safe(clf_step, X_post_vec)
)
print(type(Y_post), getattr(Y_post, "shape", None), len(Y_post))
print(type(y_prob_post), getattr(y_prob_post, "shape", None), len(y_prob_post))

if isinstance(y_prob_post, list):
    print("list length:", len(y_prob_post))
    print("first element type/shape:", type(y_prob_post[0]), getattr(y_prob_post[0], "shape", None))
metrics_post = eval_probs(Y_post, y_prob_post, classes, args.multilabel)
with open(os.path.join(args.output_dir, "metrics_post.json"), "w", encoding="utf-8") as f:
    json.dump(metrics_post, f, ensure_ascii=False, indent=2)

# %% [markdown]
# Explain a bit what is happening here

# %%
plot_aupr_per_class(
    Y_true=Y_post,
    Y_prob=y_prob_post,
    classes=classes,
    plots_dir=graphs_dir
)

# %%
# === Step 11: Fairness slices metrics ======================================================
# - soldier_flag / gender / age_group; includes false PTSD rate per slice
slices = eval_slices(
    post.reset_index(drop=True), Y_post, y_prob_post, classes, args.multilabel,plots_dir=graphs_dir,do_plots = False
)
with open(os.path.join(args.output_dir, "slices_post.json"), "w", encoding="utf-8") as f:
    json.dump(slices, f, ensure_ascii=False, indent=2)
# %% [markdown]
# ### Feature correlation with target (class separation)

# %%
# === Step 12: Distribution shift (drift) around top pre terms ==============================
drift = drift_top_terms(V, pre["text"], post["text"], top_k=50)
with open(os.path.join(args.output_dir, "drift.json"), "w", encoding="utf-8") as f:
    json.dump(drift, f, ensure_ascii=False, indent=2)

# %% [markdown]
# Explain a bit what is happening here


# %%
# === Step 13: Explainability via coefficients (plain model) ================================
# Refit a NON-calibrated model to expose coef_ cleanly; map to feature names

from sklearn.multiclass import OneVsRestClassifier

# 1) Start from the CV-best pipeline params
best_params = best.get_params()

# 2) Ensure the explainability vectorizer uses the SAME hyperparams as the best pipeline
vec_params = {k.replace("vec__", ""): v for k, v in best_params.items() if k.startswith("vec__")}
if vec_params:
    try:
        V.set_params(**vec_params)
    except Exception:
        # Keep V as-is if params are incompatible
        pass

# 3) Build a NON-calibrated classifier
best_clf = best.named_steps["clf"]
core_est = getattr(best_clf, "estimator", best_clf)

# Ensure the core estimator is wrapped in OneVsRest if multilabel is enabled
if args.multilabel:
    if not isinstance(core_est, OneVsRestClassifier):
        plain_clf = OneVsRestClassifier(clone(core_est))
    else:
        plain_clf = clone(core_est)
else:
    plain_clf = clone(core_est)

# Copy any clf__ params if they exist
clf_params = {k.replace("clf__", ""): v for k, v in best_params.items() if k.startswith("clf__")}
if clf_params:
    try:
        plain_clf.set_params(**clf_params)
    except Exception:
        pass

# 4) Fit explainability pipeline on all pre data
expl_pipe = Pipeline([
    ("select", ColumnSelector("text")),
    ("vec", V),
    ("clf", plain_clf),
])

# This will now accept Y_pre as a matrix when multilabel is True
expl_pipe.fit(pd.DataFrame({"text": pre["text"]}), Y_pre)

# 5) Extract top coefficients mapped to feature names
# The top_coefficients function in your code already handles OneVsRestClassifier
coef_top = top_coefficients(expl_pipe.named_steps["clf"], V, classes, k=20)

with open(os.path.join(args.output_dir, "explain_top_coeffs.json"), "w", encoding="utf-8") as f:
    json.dump(coef_top, f, ensure_ascii=False, indent=2)


# %% [markdown]
# Explain a bit what is happening here

# %%

# === Step 14: Optional masking robustness test =============================================
# - Replace context-sensitive terms (e.g., חייל/מילואים) with [MASK] and re-evaluate deltas

if args.mask_sensitive_test:
    post_masked = post.copy()
    post_masked["text"] = post_masked["text"].map(mask_sensitive)

    X_mask_text = post_masked["text"].astype(str).tolist()
    X_mask_vec = vec_step.transform(X_mask_text)

    y_prob_mask = (
        clf_step.predict_proba(X_mask_vec)
        if hasattr(clf_step, "predict_proba")
        else predict_proba_safe(clf_step, X_mask_vec)
    )

    metrics_mask = eval_probs(Y_post, y_prob_mask, classes, args.multilabel)

    delta = {}
    for k in metrics_post:
        if isinstance(metrics_post[k], dict):
            continue
        try:
            delta[k] = float(metrics_mask[k]) - float(metrics_post[k])
        except Exception:
            pass

    out = {"masked_metrics": metrics_mask, "delta_vs_unmasked": delta}
    with open(os.path.join(args.output_dir, "masking_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

