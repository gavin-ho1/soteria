"""Patient-disjoint folds, rebuilt deterministically from the usable-seizure list

Folds are rebuilt every run from the filtered cohort and the condition seed, so they only contain seizures that pass the filters

patient_grouped() is the paper's patient-disjoint split (StratifiedGroupKFold grouped by subject)

Yields (train_idx, val_idx) index arrays into the seizures list and asserts patient-disjointness
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def patient_grouped(seizures, n_folds, seed):
    y = np.array([s.label for s in seizures])
    g = np.array([s.subject for s in seizures])
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(y)), y, groups=g))
    for tr, va in folds:
        assert not (set(g[tr]) & set(g[va])), "patient leakage in patient_grouped!"
    return folds
