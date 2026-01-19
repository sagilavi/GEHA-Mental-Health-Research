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
from __future__ import annotations

from dataclasses import dataclass

import math
from typing import Final, Iterable, Sequence


def _require_package_imports() -> None:
    """
    Import required packages and raise a helpful error if missing.
    """
    try:
        import matplotlib.pyplot as _plt  # noqa: F401
        import numpy as _np  # noqa: F401
        import pandas as _pd  # noqa: F401
        import seaborn as _sns  # noqa: F401
        import sklearn  # noqa: F401
        import sklearn.datasets as _datasets  # noqa: F401
        import sklearn.discriminant_analysis as _discriminant_analysis  # noqa: F401
        import sklearn.ensemble as _ensemble  # noqa: F401
        import sklearn.linear_model as _linear_model  # noqa: F401
        import sklearn.metrics as _metrics  # noqa: F401
        import sklearn.model_selection as _model_selection  # noqa: F401
        import sklearn.pipeline as _pipeline  # noqa: F401
        import sklearn.preprocessing as _preprocessing  # noqa: F401
        import sklearn.svm as _svm  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing one or more required packages. Install them with:\n"
            "  pip install -U scikit-learn pandas matplotlib numpy seaborn\n"
            f"Original error: {exc}"
        ) from exc


_require_package_imports()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns

# %%
RANDOM_SEED: Final[int] = 42
TEST_SIZE: Final[float] = 0.25

# %% [markdown]
# ## Load Iris dataset into a typed DataFrame
#
# In this section we load the classic Iris dataset from `sklearn.datasets`,
# convert it into a `pandas.DataFrame`, and give the feature columns clear,
# Python-friendly names. We also keep the target labels and their human-readable
# class names together in a small immutable data container (`IrisData`).

# %%
@dataclass(frozen=True)
class IrisData:
    features: pd.DataFrame
    target: pd.Series
    target_names: tuple[str, ...]


def load_iris_dataframe() -> IrisData:
    iris = datasets.load_iris(as_frame=True)
    if iris.frame is None:
        raise ValueError("Expected Iris dataset to include a frame.")

    features_df = iris.frame.drop(columns=[iris.target.name])
    target_series = iris.frame[iris.target.name]

    features_df = features_df.rename(
        columns={
            "sepal length (cm)": "sepal_length_cm",
            "sepal width (cm)": "sepal_width_cm",
            "petal length (cm)": "petal_length_cm",
            "petal width (cm)": "petal_width_cm",
        }
    )

    if not isinstance(iris.target_names, np.ndarray):
        raise TypeError("Expected iris.target_names to be a numpy array.")

    target_names = tuple(str(x) for x in iris.target_names.tolist())

    return IrisData(
        features=features_df,
        target=target_series.astype("int64"),
        target_names=target_names,
    )


iris_data = load_iris_dataframe()
iris_data.features.head()

# %% [markdown]
# ## Quick EDA
#
# Here we create a copy of the feature data with a readable `species` column,
# compute summary statistics, and look at simple distributions and relationships
# between features. The correlation heatmap and scatter-matrix plots help you
# see how sepal and petal measurements relate to each other and to the classes.

# %%
df = iris_data.features.copy()
df["species"] = iris_data.target.map(
    {idx: name for idx, name in enumerate(iris_data.target_names)}
)

df.describe(include="all")

# %%
pd.crosstab(index=df["species"], columns="count")

# %%
corr = iris_data.features.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(6.5, 5.0))
im = ax.imshow(corr.values, vmin=-1.0, vmax=1.0, cmap="coolwarm")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
ax.set_title("Feature correlation (Pearson)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# %%
_ = pd.plotting.scatter_matrix(
    df[["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]],
    figsize=(9, 9),
    diagonal="kde",
    c=iris_data.target,
    cmap="viridis",
    alpha=0.8,
)
plt.suptitle("Scatter matrix (colored by class)", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Feature distributions by species

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
feature_cols = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
colors_map = {
    iris_data.target_names[0]: "#440154",
    iris_data.target_names[1]: "#31688e",
    iris_data.target_names[2]: "#35b779",
}

for idx, feature in enumerate(feature_cols):
    ax = axes[idx // 2, idx % 2]
    for species in iris_data.target_names:
        species_data = df[df["species"] == species][feature]
        ax.hist(
            species_data,
            alpha=0.6,
            label=species,
            bins=15,
            color=colors_map[species],
            edgecolor="black",
        )
    ax.set_xlabel(feature.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {feature.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Box plots: Feature distributions by species

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, feature in enumerate(feature_cols):
    ax = axes[idx // 2, idx % 2]
    data_by_species = [
        df[df["species"] == species][feature].values
        for species in iris_data.target_names
    ]
    bp = ax.boxplot(
        data_by_species,
        labels=iris_data.target_names,
        patch_artist=True,
        showmeans=True,
    )
    for patch, color in zip(bp["boxes"], [colors_map[s] for s in iris_data.target_names]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel(feature.replace("_", " ").title())
    ax.set_title(f"Box plot: {feature.replace('_', ' ').title()}")
    ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Violin plots: Feature distributions by species

# %%
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, feature in enumerate(feature_cols):
    ax = axes[idx // 2, idx % 2]
    sns.violinplot(
        data=df,
        x="species",
        y=feature,
        ax=ax,
        palette=[colors_map[s] for s in iris_data.target_names],
        inner="quart",
    )
    ax.set_title(f"Violin plot: {feature.replace('_', ' ').title()}")
    ax.set_xlabel("Species")
    ax.set_ylabel(feature.replace("_", " ").title())

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Pairwise feature comparisons

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
pair_combinations = [
    ("sepal_length_cm", "sepal_width_cm"),
    ("petal_length_cm", "petal_width_cm"),
    ("sepal_length_cm", "petal_length_cm"),
    ("sepal_width_cm", "petal_width_cm"),
    ("sepal_length_cm", "petal_width_cm"),
    ("sepal_width_cm", "petal_length_cm"),
]

for idx, (feat_x, feat_y) in enumerate(pair_combinations):
    ax = axes[idx // 3, idx % 3]
    for species in iris_data.target_names:
        species_df = df[df["species"] == species]
        ax.scatter(
            species_df[feat_x],
            species_df[feat_y],
            label=species,
            alpha=0.7,
            s=60,
            color=colors_map[species],
            edgecolors="black",
            linewidth=0.5,
        )
    ax.set_xlabel(feat_x.replace("_", " ").title())
    ax.set_ylabel(feat_y.replace("_", " ").title())
    ax.set_title(f"{feat_x.replace('_', ' ').title()} vs {feat_y.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### PCA 2D projection

# %%
pca = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(iris_data.features)

fig, ax = plt.subplots(figsize=(8, 6))
for idx, species in enumerate(iris_data.target_names):
    mask = iris_data.target == idx
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=species,
        alpha=0.7,
        s=80,
        color=colors_map[species],
        edgecolors="black",
        linewidth=0.5,
    )

variance_explained = pca.explained_variance_ratio_
ax.set_xlabel(f"First Principal Component ({variance_explained[0]:.1%} variance)")
ax.set_ylabel(f"Second Principal Component ({variance_explained[1]:.1%} variance)")
ax.set_title("PCA 2D Projection of Iris Features")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Train/test split
#
# We split the data into train and test sets using a **stratified** split so
# that each Iris species keeps roughly the same proportion in both sets.
# The train set is used for cross-validation and model fitting; the test set
# is held back for the final, unbiased evaluation.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    iris_data.features,
    iris_data.target,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=iris_data.target,
)

# %% [markdown]
# ## Model comparison with cross-validation
# We'll compare a few standard classifiers:
# - Logistic Regression (with scaling)
# - SVM RBF (with scaling)
# - LDA
# - Random Forest
#
# Each model is evaluated with **5-fold stratified cross-validation** on the
# training split. We compute the mean and standard deviation of the accuracy
# scores so you can see both central performance and variability across folds.

# %%
def mean_and_std(scores: Sequence[float]) -> tuple[float, float]:
    if len(scores) == 0:
        raise ValueError("scores must be non-empty.")
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return mean, std


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

models: dict[str, object] = {
    "logreg": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
        ]
    ),
    "svm_rbf": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", gamma="scale", C=1.0, random_state=RANDOM_SEED)),
        ]
    ),
    "lda": LinearDiscriminantAnalysis(),
    "rf": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
    ),
}

rows: list[dict[str, float | str]] = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    mean, std = mean_and_std(scores.tolist())
    rows.append({"model": name, "cv_mean_accuracy": mean, "cv_std": std})

results = pd.DataFrame(rows).sort_values("cv_mean_accuracy", ascending=False)
results.reset_index(drop=True, inplace=True)
results

# %%
fig, ax = plt.subplots(figsize=(7.0, 3.8))
ax.bar(results["model"], results["cv_mean_accuracy"], yerr=results["cv_std"], capsize=6)
ax.set_ylim(0.0, 1.05)
ax.set_title("Cross-validated accuracy (train split only)")
ax.set_ylabel("Accuracy")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Fit best model and evaluate on test set
#
# After comparing models on the train split, we pick the one with the highest
# cross-validated accuracy, fit it on the full training data, and then evaluate
# it once on the held-out test set. The classification report and confusion
# matrix show per-class precision, recall, F1-score, and where misclassifications
# occur.

# %%
best_model_name = str(results.iloc[0]["model"])
best_model = models[best_model_name]

best_model

# %%
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("Best model:", best_model_name)
print()
print(
    classification_report(
        y_test,
        y_pred,
        target_names=list(iris_data.target_names),
        digits=3,
    )
)

# %%
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=list(iris_data.target_names),
    cmap="Blues",
    ax=ax,
    colorbar=False,
)
ax.set_title(f"Confusion matrix — {best_model_name}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Feature importance (if Random Forest is used)

# %%
if best_model_name == "rf" or (
    hasattr(best_model, "named_steps")
    and "clf" in best_model.named_steps
    and hasattr(best_model.named_steps["clf"], "feature_importances_")
):
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model.named_steps["clf"], "feature_importances_"):
        importances = best_model.named_steps["clf"].feature_importances_
    else:
        importances = None

    if importances is not None:
        feature_names = list(iris_data.features.columns)
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(
            importance_df["feature"],
            importance_df["importance"],
            color="#31688e",
            edgecolor="black",
        )
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Feature Importance — {best_model_name}")
        ax.grid(alpha=0.3, axis="x")
        for i, (idx, row) in enumerate(importance_df.iterrows()):
            ax.text(
                row["importance"] + 0.01,
                i,
                f"{row['importance']:.3f}",
                va="center",
                fontweight="bold",
            )
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ### Model comparison: Detailed CV scores

# %%
cv_detailed_scores: dict[str, list[float]] = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_detailed_scores[name] = scores.tolist()

fig, ax = plt.subplots(figsize=(10, 6))
positions = np.arange(len(models))
width = 0.15

for fold_idx in range(5):
    fold_scores = [cv_detailed_scores[name][fold_idx] for name in models.keys()]
    offset = (fold_idx - 2) * width
    ax.bar(
        positions + offset,
        fold_scores,
        width,
        label=f"Fold {fold_idx + 1}",
        alpha=0.8,
    )

ax.set_xlabel("Model")
ax.set_ylabel("Accuracy")
ax.set_title("Cross-validation scores per fold for each model")
ax.set_xticks(positions)
ax.set_xticklabels(list(models.keys()))
ax.legend()
ax.grid(alpha=0.3, axis="y")
ax.set_ylim(0.0, 1.05)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Prediction distribution visualization

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

true_counts = pd.Series(y_test).value_counts().sort_index()
pred_counts = pd.Series(y_pred).value_counts().sort_index()

x_pos = np.arange(len(iris_data.target_names))
width = 0.35

axes[0].bar(
    x_pos - width / 2,
    [true_counts.get(i, 0) for i in range(len(iris_data.target_names))],
    width,
    label="True",
    color="#35b779",
    alpha=0.8,
    edgecolor="black",
)
axes[0].bar(
    x_pos + width / 2,
    [pred_counts.get(i, 0) for i in range(len(iris_data.target_names))],
    width,
    label="Predicted",
    color="#31688e",
    alpha=0.8,
    edgecolor="black",
)
axes[0].set_xlabel("Species")
axes[0].set_ylabel("Count")
axes[0].set_title("True vs Predicted class distribution (test set)")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(iris_data.target_names)
axes[0].legend()
axes[0].grid(alpha=0.3, axis="y")

correct_mask = y_test == y_pred
axes[1].pie(
    [correct_mask.sum(), (~correct_mask).sum()],
    labels=["Correct", "Incorrect"],
    autopct="%1.1f%%",
    colors=["#35b779", "#e63946"],
    startangle=90,
    explode=(0.05, 0.05),
)
axes[1].set_title(f"Prediction accuracy breakdown — {best_model_name}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Feature correlation with target (class separation)

# %%
fig, ax = plt.subplots(figsize=(10, 6))
feature_means_by_class = []
for species_idx, species_name in enumerate(iris_data.target_names):
    species_mask = iris_data.target == species_idx
    means = iris_data.features[species_mask].mean().values
    feature_means_by_class.append(means)

means_array = np.array(feature_means_by_class)
im = ax.imshow(means_array, cmap="viridis", aspect="auto")
ax.set_yticks(range(len(iris_data.target_names)))
ax.set_yticklabels(iris_data.target_names)
ax.set_xticks(range(len(feature_cols)))
ax.set_xticklabels([f.replace("_", " ").title() for f in feature_cols], rotation=45, ha="right")
ax.set_title("Mean feature values by species (heatmap)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Mean value")

for i in range(len(iris_data.target_names)):
    for j in range(len(feature_cols)):
        text = ax.text(
            j,
            i,
            f"{means_array[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if means_array[i, j] < means_array.mean() else "black",
            fontweight="bold",
        )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Notes
# - Iris is an easy dataset; you should expect high accuracy.
# - The visualizations show clear separation between species, especially in petal measurements.
# - PCA shows that most variance can be captured in 2 dimensions, explaining the high model performance.

