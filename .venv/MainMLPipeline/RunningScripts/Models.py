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

__all__ = ["build_vectorizer", "build_model","train_with_cv","eval_probs","predict_proba_safe"]

# %%


# ------------------ Vectorizers ------------------
def build_vectorizer(config: Dict[str, Any] = None,
                     max_features_word=None,
                     ngram_word=None,
                     use_char=None,
                     max_features_char=None,
                     ngram_char=None):
    """
    Build FeatureUnion vectorizer from config.
    Accepts either a config dict or legacy keyword arguments (for backward compatibility).
    """
    # Use config if provided, otherwise fall back to kwargs or defaults
    if config is None:
        config = {}
    
    max_features_word = max_features_word or config.get("word_max_features", 50000)
    ngram_word = ngram_word or config.get("word_ngram_range", (1, 2))
    use_char = use_char if use_char is not None else config.get("use_char", True)
    max_features_char = max_features_char or config.get("char_max_features", 30000)
    ngram_char = ngram_char or config.get("char_ngram_range", (3, 5))
    min_df = config.get("min_df", 2)

    # 1. Word TF-IDF
    word = ("word", TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_word,
        max_features=max_features_word,
        min_df=min_df
    ))

    # 2. Character TF-IDF (Optional)
    transformer_list = [word]
    if use_char:
        char = ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_char,
            max_features=max_features_char,
            min_df=min_df
        ))
        transformer_list.append(char_tfidf)

    # 3. Clinical Group Scores with Scaling
    # Using MaxAbsScaler ensures scores are scaled to [0, 1] to match TF-IDF
    clinical_groups = {
        "psychosis": ["דלוזיה", "הלוצינציה", "פרנואיד"],
        "depression": ["אנהדוניה", "ייאוש", "ירוד"],
        "anxiety": ["דרוך", "מתוח", "דופק"]
    }


    clinical_pipe = Pipeline([
        ("extractor", ClinicalFeatureExtractor(clinical_groups)),
        ("scaler", MaxAbsScaler())
    ])

    transformer_list.append(("clinical", clinical_pipe))

    return FeatureUnion(transformer_list)



    # %% [markdown]
# %%# ------------------ Model ------------------
def build_model(
    base_model: str = "logreg",
    multilabel: bool = False,
    config: Dict[str, Any] = None,
    calibrate: bool = None,
    class_weight_balanced: bool = None,
    random_state: int = None,
    calib_method: str = None,
    calib_cv: int = None,
) -> Any:
    """
    Build an estimator for single-label or multilabel.
    If multilabel=True and calibrate=True, performs per-label calibration by:
      OneVsRestClassifier(CalibratedClassifierCV(base_estimator))
    
    Accepts either a config dict or legacy keyword arguments (for backward compatibility).
    """
    # Use config if provided, otherwise fall back to kwargs or defaults
    if config is None:
        config = {}
    
    calibrate = calibrate if calibrate is not None else config.get("calibrate", True)
    class_weight_balanced = class_weight_balanced if class_weight_balanced is not None else config.get("class_weight_balanced", True)
    random_state = random_state if random_state is not None else config.get("random_state", 42)
    calib_method = calib_method or config.get("calib_method", "sigmoid")
    calib_cv = calib_cv if calib_cv is not None else config.get("calib_cv", 5)
    max_iter = config.get("max_iter", 2000)
    solver = config.get("solver", "saga")

    # 1) Choose base estimator
    if base_model == "logreg":
        base = LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            class_weight="balanced" if class_weight_balanced else None,
            random_state=random_state,
        )
    elif base_model == "linearsvc":
        base = LinearSVC()
    else:
        raise ValueError("base_model must be 'logreg' or 'linearsvc'")

    # 2) Multilabel: per-label model via OneVsRest
    if multilabel:
        if calibrate:
            # Per-label calibration: each OVR binary task gets its own calibrated classifier
            base = CalibratedClassifierCV(
                estimator=base,
                method=calib_method,
                cv=calib_cv,
            )
        clf = OneVsRestClassifier(base, n_jobs=None)
        return clf

    # 3) Single-label: optionally calibrate the multiclass classifier
    clf = base
    if calibrate:
        clf = CalibratedClassifierCV(
            estimator=clf,
            method=calib_method,
            cv=calib_cv,
        )
    return clf




# %%
# ------------------ Training with CV ------------------

def small_grid(base_model: str, config: Dict[str, Any] = None):
    """
    Input: base_model (str) specifying estimator family, optional config dict.
    Output: dict hyperparameter grid for regularization C.
    Logic: return a compact, robust grid to minimize tuning footprint.
    """
    if config is None:
        config = {}
    C_grid = config.get("C_grid", [0.003, 0.001, 0.003, 0.01, 0.03, 0.1])
    return {"C": C_grid}

# %%


# %%

def stratify_labels(y_array: np.ndarray) -> np.ndarray:
    """
    Input: y_array (ndarray) labels or multilabel indicator matrix.
    Output: 1D stratification labels for CV.
    Logic: return labels for single-label; for multilabel use row-wise label count.
    """
    if y_array.ndim == 1:
        return y_array
    else:
        return y_array.sum(axis=1)

# %%

def train_with_cv(X_text: pd.Series,
                  y_array: np.ndarray,
                  vectorizer: FeatureUnion,
                  model: Any,
                  classes: List[str],
                  base_model: str,
                  multilabel: bool,
                  config: Dict[str, Any] = None,
                  use_char: bool = None,
                  n_splits: int = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Input: texts, encoded labels, vectorizer, estimator, metadata flags, config dict.
    Output: (best_pipeline, info_dict) containing best params and macro-F1.
    Logic: build Pipeline, define grids for vectorizer and C, run StratifiedKFold GridSearchCV, refit best.
    """
    if config is None:
        config = {}
    
    grid_config = config.get("grid_search", {})
    cv_config = config.get("cv", {})
    vectorizer_config = config.get("vectorizer", {})
    
    # Get values from config with fallbacks
    use_char = use_char if use_char is not None else vectorizer_config.get("use_char", False)
    n_splits = n_splits if n_splits is not None else cv_config.get("n_splits", 5)
    shuffle = cv_config.get("shuffle", True)
    random_state = cv_config.get("random_state", 42)
    scoring = cv_config.get("scoring", "f1_macro")
    n_jobs = cv_config.get("n_jobs", -1)
    verbose = cv_config.get("verbose", 1)

    pipe = Pipeline([
        ("select", ColumnSelector("text")),
        ("vec", vectorizer),
        ("clf", model),
    ])

    # Build param_grid from config
    param_grid = {
        "vec__word__ngram_range": grid_config.get("word_ngram_range_grid", [(1, 1), (1, 2)]),
        "vec__word__max_features": grid_config.get("word_max_features_grid", [75, 100, 125, 150, 200, 225, 300, 500, 750]),
    }

    if use_char:
        param_grid.update({
            "vec__char__ngram_range": grid_config.get("char_ngram_range_grid", [(3, 5)]),
            "vec__char__max_features": grid_config.get("char_max_features_grid", [2000, 5000]),
        })

    C_vals = small_grid(base_model, grid_config)["C"]
    #for key in ["clf__base_estimator__C", "clf__estimator__C", "clf__C"]:
    #        param_grid[key] = C_vals
    #param_grid["clf__estimator__C"] = C_vals

    # Set C at the correct depth depending on wrapper(s)
    if hasattr(model, "estimator") and hasattr(getattr(model, "estimator"), "estimator"):
        # e.g. CalibratedClassifierCV(estimator=OneVsRestClassifier(estimator=LogReg))
        param_grid["clf__estimator__estimator__C"] = C_vals
    elif hasattr(model, "estimator"):
        # e.g. OneVsRestClassifier(estimator=LogReg)
        param_grid["clf__estimator__C"] = C_vals
    else:
        # e.g. plain LogisticRegression
        param_grid["clf__C"] = C_vals


    # Calculate total number of fits for progress tracking
    from itertools import product
    param_combinations = list(product(*param_grid.values()))
    total_fits = len(param_combinations) * n_splits
    
    print(f"\nStarting GridSearchCV: {len(param_combinations)} parameter combinations × {n_splits} CV folds = {total_fits} total fits")
    print(f"Progress updates every 150 fits...\n")
    
    # Create progress callback for sklearn 1.3+ (if available)
    # For older versions, we'll use verbose output
    try:
        from sklearn.model_selection import Callback
        import sklearn
        
        class ProgressCallback(Callback):
            """Callback to print progress every N fits."""
            def __init__(self, print_interval: int = 150, total_fits: int = None):
                self.print_interval = print_interval
                self.total_fits = total_fits
                self.fit_count = 0
            
            def on_fit_end(self, estimator, X, y, **kwargs):
                """Called after each fit completes."""
                self.fit_count += 1
                if self.fit_count % self.print_interval == 0:
                    total_str = f" / {self.total_fits}" if self.total_fits else ""
                    print(f"Progress: {self.fit_count}{total_str} fits completed", flush=True)
        
        # Check if sklearn version supports callbacks (1.3+)
        sklearn_version = tuple(map(int, sklearn.__version__.split(".")[:2]))
        use_callbacks = sklearn_version >= (1, 3)
    except (ImportError, AttributeError):
        use_callbacks = False
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # Build GridSearchCV with callback if available, otherwise use verbose
    gs_kwargs = {
        "param_grid": param_grid,
        "scoring": scoring,
        "cv": cv.split(X_text, stratify_labels(y_array)),
        "n_jobs": n_jobs,
        "verbose": verbose if not use_callbacks else 0,  # Reduce verbose if using callback
        "refit": True,
    }
    
    if use_callbacks:
        progress_callback = ProgressCallback(print_interval=150, total_fits=total_fits)
        gs_kwargs["callbacks"] = [progress_callback]
    
    search = GridSearchCV(pipe, **gs_kwargs)
    
    search.fit(pd.DataFrame({"text": X_text}), y_array)
    
    if use_callbacks:
        print(f"\nGridSearchCV completed: {progress_callback.fit_count} fits finished\n")
    else:
        print(f"\nGridSearchCV completed: {total_fits} fits finished\n")
    best = search.best_estimator_
    info = {
        "best_params": search.best_params_,
        "best_score_macro_f1": search.best_score_,
    }
    return best, info

# %% [markdown]
# Explain a bit what is happening here

# %%
# ------------------ Evaluation ------------------

def eval_probs(y_true, y_prob, classes, multilabel: bool) -> Dict[str, Any]:
    """
    Input: y_true (array/matrix), y_prob (array), classes list, multilabel flag.
    Output: metrics dict (macro/micro F1, per-class AUPR, PTSD metrics, Brier mean).
    Logic: derive predictions (argmax/threshold), compute aggregate F1/AUPR, PTSD slice, and Brier score.
    """
    out = {}
    if not multilabel:
        y_pred = np.argmax(y_prob, axis=1)
        out["f1_macro"] = f1_score(y_true, y_pred, average="macro")
        out["f1_micro"] = f1_score(y_true, y_pred, average="micro")
        if "PTSD" in classes:
            idx = classes.index("PTSD")
            out["f1_ptsd"] = f1_score((y_true==idx).astype(int), (y_pred==idx).astype(int))
            out["aupr_ptsd"] = average_precision_score((y_true==idx).astype(int), y_prob[:, idx])
        aupr = {}
        for i, c in enumerate(classes):
            aupr[c] = average_precision_score((y_true==i).astype(int), y_prob[:, i])
        out["aupr_per_class"] = aupr
        briers = [brier_score_loss((y_true==i).astype(int), y_prob[:, i]) for i in range(len(classes))]
        out["brier_ovr_mean"] = float(np.mean(briers))
    else:
        #y_pred_bin = (y_prob >= 0.5).astype(int)
        y_pred_bin = (y_prob >= MultyLablelMinPredictScoreForEval).astype(int)

        out["f1_macro"] = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
        out["f1_micro"] = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
        if "PTSD" in classes:
            idx = classes.index("PTSD")
            out["f1_ptsd"] = f1_score(y_true[:, idx], y_pred_bin[:, idx], zero_division=0)
            out["aupr_ptsd"] = average_precision_score(y_true[:, idx], y_prob[:, idx])
        aupr = {}
        for i, c in enumerate(classes):
            try:
                aupr[c] = average_precision_score(y_true[:, i], y_prob[:, i])
            except Exception:
                aupr[c] = float("nan")
        out["aupr_per_class"] = aupr
        bs = []
        for i in range(len(classes)):
            try:
                bs.append(brier_score_loss(y_true[:, i], y_prob[:, i]))
            except Exception:
                continue
        out["brier_ovr_mean"] = float(np.mean(bs)) if bs else float("nan")
    return out
# %%

def predict_proba_safe(model, X_df):
    """
    Input: fitted classifier (possibly calibrated/OVR) and vectorized samples.
    Output: probability array for each class/label.
    Logic: use predict_proba when available; otherwise map decision_function via logistic transform.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_df)
    if hasattr(model, "decision_function"):
        dec = model.decision_function(X_df)
        if isinstance(dec, list):
            dec = np.vstack([d for d in dec]).T
        return 1 / (1 + np.exp(-dec))
    raise ValueError("Model lacks probability outputs. Wrap with calibration.")


# %% [markdown]
# Explain a bit what is happening here


