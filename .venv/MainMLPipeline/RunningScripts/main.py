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

from pathlib import Path
from typing import Dict, Any
print("Hello World")


# %%
# ------------------ HyperParamaters ------------------

MultyLablelMinPredictScoreForEval = 0.5

# %%

# ------------------ Utilities ------------------

HEB_NIQQUD = re.compile(r'[\u0591-\u05C7]')  # Hebrew diacritics
SPACE_NORMALIZE = re.compile(r'\s+')

SENSITIVE_TERMS = [
    "חיילת","חילת","חייל","חיל", "מילואים", "מילואימני","מילואימניק", "אזאקה","אזעקה", "נפילה","מטען","פצצה","פיצוץ", "קסאם", "אירוע", "טיל",
    "מלחמה", "צבע אדום", "לחימה", "קרב","פינוי", "חטוף","חטיפה"
]

def normalize_hebrew(text: str) -> str:
    """Normalize Hebrew text: drop niqqud, unify quotes, collapse spaces."""
    if not isinstance(text, str):
        return ""
    t = text
    t = HEB_NIQQUD.sub("", t)  # remove niqqud/cantillation
    t = t.replace("”", '"').replace("“", '"').replace("׳", "'").replace("״", '"')  # unify punctuation
    t = SPACE_NORMALIZE.sub(" ", t).strip()  # de-duplicate whitespace
    return t

def mask_sensitive(text: str, mask_token: str = "[MASK]") -> str:
    """Mask context-sensitive terms with a placeholder token for robustness checks."""
    t = text
    for w in SENSITIVE_TERMS:
        # rf-string: raw regex with formatted escaped word; \b ensures full-word match
        t = re.sub(rf"\b{re.escape(w)}\b", mask_token, t)
    return t

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    """Coerce a DataFrame column to pandas datetime (invalid → NaT)."""
    return pd.to_datetime(df[col], errors="coerce", utc=False)


# %% [markdown]
# ## 

# %%

# ------------------ Data handling ------------------

def load_data(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".parquet"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return df

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:

    """Validate schema and normalize text/date; ensure soldier_flag is boolean."""
    need_cols = ["id", "text", "diagnosis", "date", "soldier_flag"]
    missing = [c for c in need_cols if c not in df.columns]
    #pdb.set_trace()

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()  # avoid mutating caller's DataFrame

    # Text normalization (PII scrubbing assumed upstream)
    df["text"] = df["text"].map(normalize_hebrew)

    # Date coercion and cleanup of invalid dates
    df["date"] = ensure_datetime(df, "date")
    if df["date"].isna().any():
        warnings.warn("Some dates could not be parsed and will be dropped.")
        df = df.dropna(subset=["date"])

    # Ensure boolean soldier flag (handles 0/1, strings → bool)
    if df["soldier_flag"].dtype != bool:
        df["soldier_flag"] = df["soldier_flag"].astype(int).astype(bool)
    return df


# %%
# ------------------ Split logic ------------------

CUTOFF = pd.Timestamp("2023-10-07")

def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pre = df[df["date"] < CUTOFF].copy()
    post = df[df["date"] >= CUTOFF].copy()
    if len(pre) == 0 or len(post) == 0:
        warnings.warn("One of the temporal splits is empty.")
    return pre, post

# %% [markdown]
# Explain a bit what is happening here


# %%
# ------------------ Labels ------------------

def prepare_labels(y: pd.Series, multilabel: bool) -> Tuple[np.ndarray, Any, List[str]]:
    """
    Input: y (Series) of labels; multilabel flag controlling encoding.
    Output: (Y, encoder_or_map, classes) where Y is indices or a multilabel binary matrix.
    Logic: use MultiLabelBinarizer for multilabel else build class->index map and transform.
    """
    if multilabel:
        y_list = y.fillna("").map(lambda s: [t for t in re.split(r"[|,;/]", s) if t]).tolist()
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(y_list)
        classes = list(mlb.classes_)
        return Y, mlb, classes
    else:
        classes = sorted(y.dropna().unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = y.map(class_to_idx).to_numpy()
        return y_idx, class_to_idx, classes
# %% [markdown]
# Explain a bit what is happening here

# %%
class ClinicalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create clinical group scores.
    Acts like 'Area' calculation by summing occurrences of related terms.
    """
    def __init__(self, feature_map: dict):
        self.feature_map = feature_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
            if not isinstance(X, pd.Series):
                if isinstance(X, pd.DataFrame):
                    X = X.iloc[:, 0]
                else:
                    X = pd.Series(np.array(X).ravel())

            features = pd.DataFrame(index=range(len(X)))

            for group_name, terms in self.feature_map.items():
                pattern = '|'.join([rf"{re.escape(t)}" for t in terms])
                # Ensure numeric count
                counts = pd.to_numeric(X.str.count(pattern), errors='coerce').fillna(0)
                features[f'{group_name}_score'] = counts

            if 'mania' in self.feature_map and 'insomnia' in self.feature_map:
                features['mania_x_insomnia'] = (
                    features['mania_score'] * features['insomnia_score']
                ).fillna(0)

            # Convert to float and replace any potential NaN/Inf with 0
            output = features.values.astype(np.float64)
            return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

    def get_feature_names_out(self, input_features=None):
        """
        Returns the names of the features produced by this transformer.
        This prevents AttributeError when called within a Pipeline or FeatureUnion.
        """
        feature_names = [f'{group_name}_score' for group_name in self.feature_map.keys()]

        if 'mania' in self.feature_map and 'insomnia' in self.feature_map:
            feature_names.append('mania_x_insomnia')

        return np.array(feature_names)

# %% [markdown]
# Explain a bit what is happening here

# %%

# ------------------ Vectorizers ------------------

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Input: initialized with 'key' (str) of column to select.
    Output: at transform, returns the selected column values.
    Logic: passthrough transformer to bridge DataFrame to vectorizers.
    """
    def __init__(self, key: str):
        """
        Input: key (str) column name to select.
        Output: ColumnSelector instance with stored key.
        Logic: save the column name for later transform.
        """
        self.key = key
    def fit(self, X, y=None):
        """
        Input: X (ignored), y (ignored).
        Output: self unchanged.
        Logic: stateless transformer; nothing to fit.
        """
        return self
    def transform(self, X):
        """
        Input: X (DataFrame-like) containing the configured column.
        Output: numpy array / Series of the selected column.
        Logic: return X[self.key].values to feed downstream steps.
        """
        return X[self.key].values

def build_vectorizer(max_features_word=50000,
                     ngram_word=(1,2),
                     use_char=True,
                     max_features_char=30000,
                     ngram_char=(3,5)):

    # 1. Word TF-IDF
    word = ("word", TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_word,
        max_features=max_features_word,
        min_df=2
    ))

    # 2. Character TF-IDF (Optional)
    transformer_list = [word]
    if use_char:
        char = ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_char,
            max_features=max_features_char,
            min_df=2
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

def build_vectorizer_old(max_features_word=50000,
                     ngram_word=(1,2),
                     use_char=True,
                     max_features_char=30000,
                     ngram_char=(3,5)):
    """
    Input: word/char TF-IDF settings: features, n-grams, toggles.
    Output: FeatureUnion combining word-level TF-IDF and optional char-level TF-IDF.
    Logic: instantiate TfidfVectorizer(s) and union them for richer feature space.
    """
    word = ("word", TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_word,
        max_features=max_features_word,
        min_df=2
    ))
    if use_char:
        char = ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_char,
            max_features=max_features_char,
            min_df=2
        ))
        vec = FeatureUnion([word, char])
    else:
        vec = FeatureUnion([word])
    return vec

# %% [markdown]
# Explain a bit what is happening here
# %%
# ------------------ Model ------------------
def build_model(
    base_model: str = "logreg",
    multilabel: bool = False,
    calibrate: bool = True,
    class_weight_balanced: bool = True,
    random_state: int = 42,
    calib_method: str = "sigmoid",
    calib_cv: int = 5,
) -> Any:
    """
    Build an estimator for single-label or multilabel.
    If multilabel=True and calibrate=True, performs per-label calibration by:
      OneVsRestClassifier(CalibratedClassifierCV(base_estimator))
    """

    # 1) Choose base estimator
    if base_model == "logreg":
        base = LogisticRegression(
            max_iter=2000,
            solver="saga",
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


# %% [markdown]
# ### Feature distributions by species

# %%
# ------------------ Training with CV ------------------

def small_grid(base_model: str):
    """
    Input: base_model (str) specifying estimator family.
    Output: dict hyperparameter grid for regularization C.
    Logic: return a compact, robust grid to minimize tuning footprint.
    """
    return {"C": [ 0.003,0.001, 0.003, 0.01, 0.03, 0.1]}

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

def train_with_cv(X_text: pd.Series,
                  y_array: np.ndarray,
                  vectorizer: FeatureUnion,
                  model: Any,
                  classes: List[str],
                  base_model: str,
                  multilabel: bool,
                  n_splits: int = 5) -> Tuple[Any, Dict[str, Any]]:
    """
    Input: texts, encoded labels, vectorizer, estimator, metadata flags, CV splits.
    Output: (best_pipeline, info_dict) containing best params and macro-F1.
    Logic: build Pipeline, define grids for vectorizer and C, run StratifiedKFold GridSearchCV, refit best.
    """


    pipe = Pipeline([
        ("select", ColumnSelector("text")),
        ("vec", vectorizer),
        ("clf", model),
    ])

    #param_grid = {
    #    "vec__transformer_list__word__1__ngram_range": [(1,1), (1,2)],
    #    "vec__transformer_list__word__1__max_features": [250,500, 1000,2500,7500],
    #}

    param_grid = {
        "vec__word__ngram_range": [(1,1), (1,2)],
        "vec__word__max_features": [75,100,125,150,200,225, 300, 500, 750],
    }

    if args.use_char:
        param_grid.update({
            "vec__char__ngram_range": [(3,5)],          # או [(3,5), (3,6)]
            "vec__char__max_features": [2000, 5000],    # טווח סביר להתחלה
        })

    C_vals = small_grid(base_model)["C"]
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


    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv.split(X_text, stratify_labels(y_array)),
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(pd.DataFrame({"text": X_text}), y_array)
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

# %%
# ------------------ Fairness slices ------------------

def eval_slices(df: pd.DataFrame, y_true, y_prob, classes, multilabel: bool, slice_cols=("soldier_flag","gender","age_group")) -> Dict[str, Any]:
    """
    Input: post DataFrame, true labels, probabilities, class list, multilabel flag, slice column names.
    Output: dict of metrics per slice value including false_ptsd_rate.
    Logic: create safe 'gender'/'age_group' columns, groupby each slice, run eval_probs, compute false PTSD rate per slice.
    """
    out = {}
    work = df.copy()
    if "gender" not in work.columns:
        work["gender"] = "NA"
    if "age" in work.columns and "age_group" not in work.columns:
        bins = [0, 25, 40, 60, 200]
        labels = ["<=25", "26-40", "41-60", "60+"]
        work["age_group"] = pd.cut(work["age"], bins=bins, labels=labels, right=True, include_lowest=True)
    elif "age_group" not in work.columns:
        work["age_group"] = "NA"
    for col in slice_cols:
        if col not in work.columns:
            continue
        for val, idx in work.groupby(col).groups.items():
            y_t = y_true[idx] if isinstance(y_true, np.ndarray) else y_true.iloc[list(idx)]
            y_p = y_prob[idx]
            try:
                metrics = eval_probs(y_t, y_p, classes, multilabel)
                if "PTSD" in classes:
                    if multilabel:
                        ptsd_idx = classes.index("PTSD")
                        #fp = np.sum((y_p[:,ptsd_idx] >= 0.5) & (y_t[:,ptsd_idx] == 0))
                        fp = np.sum((y_p[:,ptsd_idx] >= MultyLablelMinPredictScoreForEval) & (y_t[:,ptsd_idx] == 0))

                        n_neg = np.sum(y_t[:,ptsd_idx] == 0)
                        false_rate = float(fp / max(n_neg, 1))
                    else:
                        ptsd_idx = classes.index("PTSD")
                        y_pred = np.argmax(y_p, axis=1)
                        fp = np.sum((y_pred == ptsd_idx) & (y_t != ptsd_idx))
                        n_neg = np.sum(y_t != ptsd_idx)
                        false_rate = float(fp / max(n_neg, 1))
                    metrics["false_ptsd_rate"] = false_rate
                out[f"{col}={val}"] = metrics
            except Exception as e:
                out[f"{col}={val}"] = {"error": str(e)}
    return out

# %% [markdown]
# Explain a bit what is happening here

# %%
# ------------------ Fairness slices ------------------

def eval_slices(
    df: pd.DataFrame,
    y_true,
    y_prob,
    classes,
    multilabel: bool,
    slice_cols=("soldier_flag", "gender", "age_group"),
    do_plots: bool = True,
    plots_dir: str | None = None,
    plots_prefix: str = "pr",
    max_groups_per_col: int | None = None,
    figsize=(8, 6),
) -> Dict[str, Any]:
    """
    Input: DataFrame, true labels, probabilities, class list, multilabel flag.
    Output: dict of metrics per slice value including false_ptsd_rate.
    Optional: create PR plots per group via plot_aupr_per_class (show and or save).
    """
    out: Dict[str, Any] = {}
    work = df.copy()

    # Ensure slice columns exist
    if "gender" not in work.columns:
        work["gender"] = "NA"

    if "age" in work.columns and "age_group" not in work.columns:
        bins = [0, 25, 40, 60, 200]
        labels = ["<=25", "26-40", "41-60", "60+"]
        work["age_group"] = pd.cut(
            work["age"], bins=bins, labels=labels, right=True, include_lowest=True
        )
    elif "age_group" not in work.columns:
        work["age_group"] = "NA"

    # Prepare plots dir (if saving requested)
    if plots_dir is not None:
        os.makedirs(plots_dir, exist_ok=True)

    def _safe_filename(s: str) -> str:
        s = str(s)
        s = s.strip()
        s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
        s = re.sub(r"[^0-9A-Za-zא-ת_+=-]", "", s)
        return s[:120] if len(s) > 120 else s

    for col in slice_cols:
        if col not in work.columns:
            continue

        groups = list(work.groupby(col).groups.items())
        if max_groups_per_col is not None:
            groups = groups[:max_groups_per_col]

        for val, idx in groups:
            idx_list = list(idx)

            y_t = y_true[idx_list] if isinstance(y_true, np.ndarray) else y_true.iloc[idx_list]
            y_p = y_prob[idx_list]

            group_key = f"{col}={val}"

            try:
                metrics = eval_probs(y_t, y_p, classes, multilabel)

                # False PTSD rate
                if "PTSD" in classes:
                    ptsd_idx = classes.index("PTSD")
                    if multilabel:
                        fp = np.sum(
                            (y_p[:, ptsd_idx] >= MultyLablelMinPredictScoreForEval) &
                            (y_t[:, ptsd_idx] == 0)
                        )
                        n_neg = np.sum(y_t[:, ptsd_idx] == 0)
                    else:
                        y_pred = np.argmax(y_p, axis=1)
                        fp = np.sum((y_pred == ptsd_idx) & (y_t != ptsd_idx))
                        n_neg = np.sum(y_t != ptsd_idx)

                    metrics["false_ptsd_rate"] = float(fp / max(n_neg, 1))

                out[group_key] = metrics

                # Plot per group (one figure per group)
                if do_plots or plots_dir is not None:
                    safe_val = _safe_filename(val)
                    safe_col = _safe_filename(col)
                    fname = f"{plots_prefix}_{safe_col}_{safe_val}.png"

                    plot_aupr_per_class(
                        Y_true=y_t,
                        Y_prob=y_p,
                        classes=classes,
                        figsize=figsize,
                        do_plots=do_plots,
                        plots_dir=plots_dir,
                        title=f"Precision Recall | {group_key} | n={len(idx_list)}",
                        filename=fname,
                    )

            except Exception as e:
                out[group_key] = {"error": str(e)}

    return out

# %% [markdown]
# Explain a bit what is happening here

# %%
# ------------------ Drift: PSI and KL ------------------

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Input: expected and actual numeric arrays; number of bins.
    Output: float PSI value indicating distribution shift (lower ~ more stable).
    Logic: bin both on shared edges, normalize, avoid zeros, apply PSI formula sum((a-e)*ln(a/e)).
    """
    ex, bin_edges = np.histogram(expected, bins=bins)
    ac, _ = np.histogram(actual, bins=bin_edges)
    ex = ex / max(ex.sum(), 1)
    ac = ac / max(ac.sum(), 1)
    ex = np.where(ex==0, 1e-6, ex)
    ac = np.where(ac==0, 1e-6, ac)
    return float(np.sum((ac - ex) * np.log(ac / ex)))

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Input: arrays p and q representing discrete distributions over same support.
    Output: float KL divergence D(P||Q) (lower ~ more similar).
    Logic: normalize to sum 1, clip zeros to small constants, compute sum p*log(p/q).
    """
    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    p = np.where(p==0, 1e-12, p)
    q = np.where(q==0, 1e-12, q)
    return float(np.sum(p * np.log(p / q)))

def drift_top_terms(vec, X_pre_text: pd.Series, X_post_text: pd.Series, top_k: int = 50) -> Dict[str, Dict[str, float]]:
    """
    Input: vectorizer-like object, pre/post text Series, and top_k feature count.
    Output: dict feature->{pre_mean, post_mean, delta} and aggregate {_PSI, _KL}.
    Logic: clone vec and fit on pre, compute mean TF-IDF, select top_k, compare with post, compute PSI/KL over selected slice.
    """
    V = clone(vec)
    X_pre = V.fit_transform(X_pre_text)
    feature_names = []
    for name, trans in V.transformer_list:
        if hasattr(trans, "get_feature_names_out"):
            names = [f"{name}::{n}" for n in trans.get_feature_names_out()]
            feature_names.extend(names)
    pre_mean = np.asarray(X_pre.mean(axis=0)).ravel()
    top_idx = np.argsort(pre_mean)[::-1][:top_k]
    top_feats = [feature_names[i] for i in top_idx]
    X_post = V.transform(X_post_text)
    post_mean = np.asarray(X_post.mean(axis=0)).ravel()
    res = {}
    for i, feat in zip(top_idx, top_feats):
        p = pre_mean[i]
        q = post_mean[i]
        res[feat] = {"pre_mean": float(p), "post_mean": float(q), "delta": float(q - p)}
    distr_pre = pre_mean[top_idx]
    distr_post = post_mean[top_idx]
    res["_PSI"] = {"psi": psi(distr_pre, distr_post, bins=min(10, len(distr_pre)))}
    res["_KL"]  = {"kl": kl_divergence(distr_pre, distr_post)}
    return res


# %% [markdown]
# Explain a bit what is happening here

# %%
# ------------------ Explainability ------------------

def top_coefficients(model, vec, classes: List[str], k: int = 20) -> Dict[str, List[Tuple[str, float]]]:
    """
    Input: trained model (possibly wrapped), fitted vectorizer, classes list, top k.
    Output: dict mapping class->list of (feature_name, weight) or info if unavailable.
    Logic: unwrap calibrated/OVR estimators, access coef_, map to feature names, select top-k per class.
    """
    out = {}
    feature_names = []
    for name, trans in vec.transformer_list:
        if hasattr(trans, "get_feature_names_out"):
            names = [f"{name}::{n}" for n in trans.get_feature_names_out()]
            feature_names.extend(names)
    # Attempt to unwrap calibrated models
    if isinstance(model, CalibratedClassifierCV):
        base = getattr(model, "base_estimator", getattr(model, "estimator", model))
    else:
        base = model
    if isinstance(base, OneVsRestClassifier):
        for i, est in enumerate(base.estimators_):
            if hasattr(est, "coef_"):
                coefs = est.coef_.ravel()
                top_pos = np.argsort(coefs)[-k:][::-1]
                out[classes[i]] = [(feature_names[j], float(coefs[j])) for j in top_pos]
    elif hasattr(base, "coef_"):
        for i, cls in enumerate(classes):
            top_pos = np.argsort(base.coef_[i])[-k:][::-1]
            out[cls] = [(feature_names[j], float(base.coef_[i][j])) for j in top_pos]
    else:
        out["info"] = ["Coefficients unavailable for this model."]
    return out

# %% [markdown]
# Explain a bit what is happening here

# %%

# ------------------ graphics ------------------

def plot_aupr_per_class(
    Y_true,
    Y_prob,
    classes,
    figsize=(8, 6),
    do_plots: bool = True,
    plots_dir: str | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """
    Plot Precision-Recall curves and AUPR for each class.

    Parameters
    ----------
    Y_true : np.ndarray
        True labels.
        Shape (n_samples,) for single-label
        or (n_samples, n_classes) for multilabel.

    Y_prob : np.ndarray
        Predicted probabilities.
        Shape (n_samples, n_classes).

    classes : list[str]
        Class names, order must match Y_prob columns.

    do_plots : bool
        Whether to display the plot.

    plots_dir : str or None
        If provided, save plot as PNG to this directory.

    title : str or None
        Optional plot title.

    filename : str or None
        Optional filename (without path). If None, auto-generated.
    """

    if not do_plots and plots_dir is None:
        return  # nothing to do

    plt.figure(figsize=figsize)

    for i, cls in enumerate(classes):
        if Y_true.ndim == 1:
            y_true_bin = (Y_true == i).astype(int)
        else:
            y_true_bin = Y_true[:, i]

        y_prob_cls = Y_prob[:, i]

        precision, recall, _ = precision_recall_curve(
            y_true_bin,
            y_prob_cls
        )
        aupr = average_precision_score(
            y_true_bin,
            y_prob_cls
        )

        plt.plot(
            recall,
            precision,
            lw=2,
            label=f"{cls} (AUPR={aupr:.3f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title if title else "Precision–Recall Curves per Diagnosis")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()

    # ---- Save plot if requested ----
    if plots_dir is not None:
        os.makedirs(plots_dir, exist_ok=True)
        if filename is None:
            filename = "pr_curves.png"
        plt.savefig(
            os.path.join(plots_dir, filename),
            dpi=200,
            bbox_inches="tight"
        )

    # ---- Show plot if requested ----
    if do_plots:
        plt.show()

    plt.close()

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


# %% [markdown]
# === Step 4: Temporal split (train before 2023-10-07; evaluate on/after) ===================
# pre, post = temporal_split(df)
# if len(pre) == 0 or len(post) == 0:
#     # We require both sides to exist to avoid leakage and to test generalization post-cutoff
#     print("ERROR: Pre or Post split empty.", file=sys.stderr)
#     sys.exit(1)

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

