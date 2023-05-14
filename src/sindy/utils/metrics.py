import numpy as np
from sklearn.metrics import jaccard_score, confusion_matrix


def iou_score(true_coef, coef, threshold=1e-8):
    return jaccard_score(true_coef.flatten(), (np.abs(coef) > threshold).astype(int).flatten())


def me_score(true_coef, coef, threshold=1e-8):
    conf = confusion_matrix(true_coef.flatten(), (np.abs(coef) > threshold).astype(int).flatten())
    return (conf[1, 0], conf[0, 1])  # (false negative, false positive)
