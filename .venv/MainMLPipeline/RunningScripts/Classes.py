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

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["ClinicalFeatureExtractor", "ColumnSelector"]


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
