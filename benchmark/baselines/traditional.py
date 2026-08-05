"""
Category ① traditional unsupervised baselines (`registry.py`). These are
meant as a cheap sanity floor, not competitors -- if a deep/federated
method can't beat PCA reconstruction error or IsolationForest on a given
fault type, that fault type's difficulty needs explaining before touting
any deep result on it (robo3er's own history: `stuck` sat near-chance for
every method tried until the physics-residual signal, see
`memory/robo3er-anomaly-detection-approaches.md`).

Both operate on FLATTENED windows [N, T*F] (no notion of a graph/sensor
structure), fit on normal-only fit-split data, score by distance/error on
held-out data -- same fit/calib/test_normal convention as everything else
in this project.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


class PCABaseline:
    def __init__(self, n_components: float = 0.95):
        self.n_components = n_components
        self.pca = None

    def fit(self, fit_windows):
        flat = fit_windows.reshape(len(fit_windows), -1)
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.pca.fit(flat)
        return self

    def score(self, windows):
        """Reconstruction error (higher = more anomalous)."""
        flat = windows.reshape(len(windows), -1)
        recon = self.pca.inverse_transform(self.pca.transform(flat))
        return np.mean((flat - recon) ** 2, axis=1)


class IsolationForestBaseline:
    def __init__(self, n_estimators: int = 200):
        self.n_estimators = n_estimators
        self.model = None

    def fit(self, fit_windows):
        flat = fit_windows.reshape(len(fit_windows), -1)
        self.model = IsolationForest(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)
        self.model.fit(flat)
        return self

    def score(self, windows):
        """Higher = more anomalous (IsolationForest's own score is
        negated so this matches every other baseline's convention)."""
        flat = windows.reshape(len(windows), -1)
        return -self.model.score_samples(flat)
