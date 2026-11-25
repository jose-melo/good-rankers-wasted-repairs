"""Reference-vs-window classifier with permutation importance, in the model-based explanation framework of
Hinder et al. (Neurocomputing 2023); our implementation.

Score of feature j = drop in out-of-fold AUC of a LightGBM classifier (reference = 0, window = 1) when j is permuted.
The score has no null: it is thresholded on no-change windows of other units (see score.py)."""
import numpy as np

KIND = "score"
MIN_W = 20


def run(ref, win, seed):
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); n = min(len(ref), 500); R = ref[rng.choice(len(ref), n, replace=False)]
    X = np.vstack([R, win]); y = np.r_[np.zeros(n), np.ones(len(win))]; d = X.shape[1]
    fold = rng.integers(2, size=len(y)); base = np.zeros(2); drop = np.zeros((2, d))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        if y[tr].sum() < 5 or y[te].sum() < 5:
            return np.full(d, np.nan)
        m = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=7, verbose=-1, random_state=seed, n_jobs=1,
                           class_weight="balanced").fit(X[tr], y[tr])
        base[f] = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        for j in range(d):
            Z = X[te].copy(); Z[:, j] = Z[rng.permutation(len(Z)), j]
            drop[f, j] = base[f] - roc_auc_score(y[te], m.predict_proba(Z)[:, 1])
    return drop.mean(0)
