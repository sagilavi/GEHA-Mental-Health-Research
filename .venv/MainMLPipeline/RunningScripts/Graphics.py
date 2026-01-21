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
from Models import eval_probs

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

__all__ = ["eval_slices", "drift_top_terms", "top_coefficients", "plot_aupr_per_class"]



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
    MultyLablelMinPredictScoreForEval: float = 0.5,
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
