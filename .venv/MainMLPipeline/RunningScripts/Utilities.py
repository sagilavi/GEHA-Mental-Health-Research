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
# Hebrew Text Utilities

# %%

import re
import warnings
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


# ------------------ Utilities ------------------

HEB_NIQQUD = re.compile(r"[\u0591-\u05C7]")  # Hebrew diacritics
SPACE_NORMALIZE = re.compile(r"\s+")

SENSITIVE_TERMS: List[str] = [
    "חיילת",
    "חילת",
    "חייל",
    "חיל",
    "מילואים",
    "מילואימני",
    "מילואימניק",
    "אזאקה",
    "אזעקה",
    "נפילה",
    "מטען",
    "פצצה",
    "פיצוץ",
    "קסאם",
    "אירוע",
    "טיל",
    "מלחמה",
    "צבע אדום",
    "לחימה",
    "קרב",
    "פינוי",
    "חטוף",
    "חטיפה",
]


def normalize_hebrew(text: str) -> str:
    """Normalize Hebrew text: drop niqqud, unify quotes, collapse spaces."""
    if not isinstance(text, str):
        return ""
    t = text
    t = HEB_NIQQUD.sub("", t)  # remove niqqud/cantillation
    # unify punctuation
    t = (
        t.replace("”", '"')
        .replace("“", '"')
        .replace("׳", "'")
        .replace("״", '"')
    )
    # de-duplicate whitespace
    t = SPACE_NORMALIZE.sub(" ", t).strip()
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


# ------------------ Split logic ------------------

# NOTE: CUTOFF is defined in main.py; we duplicate the default here
CUTOFF = pd.Timestamp("2023-10-07")


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into pre/post by the global CUTOFF date."""
    pre = df[df["date"] < CUTOFF].copy()
    post = df[df["date"] >= CUTOFF].copy()
    if len(pre) == 0 or len(post) == 0:
        warnings.warn("One of the temporal splits is empty.")
    return pre, post


# ------------------ Labels ------------------

def prepare_labels(y: pd.Series, multilabel: bool) -> Tuple[np.ndarray, Any, List[str]]:
    """
    Input: y (Series) of labels; multilabel flag controlling encoding.
    Output: (Y, encoder_or_map, classes) where Y is indices or a multilabel binary matrix.
    Logic: use MultiLabelBinarizer for multilabel else build class->index map and transform.
    """
    if multilabel:
        y_list = (
            y.fillna("")
            .map(lambda s: [t for t in re.split(r"[|,;/]", s) if t])
            .tolist()
        )
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(y_list)
        classes = list(mlb.classes_)
        return Y, mlb, classes

    classes = sorted(y.dropna().unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = y.map(class_to_idx).to_numpy()
    return y_idx, class_to_idx, classes

