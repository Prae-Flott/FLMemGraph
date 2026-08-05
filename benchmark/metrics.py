"""
Evaluation metrics, per `mem_phys_prompt_zh.md` Sec 8.3's per-category
convention:
  - A/B (point/pixel anomaly detection): Precision/Recall/F1, AUROC, AUPRC
  - B (image anomaly localization): I-AUROC (image-level), P-AUROC + PRO
    (pixel-level) -- NOT implemented here, no image dataset wired up yet
    (see datasets/registry.py, category B is scaffolded, not implemented)
  - C (dynamic graph AD): AUROC, AUPRC
  - D (forecasting): MSE, MAE

Point-adjusted F1 (PA-F1) is intentionally NOT the default here. Xu et al.
2018 introduced it and it's widely used in the UAD literature (GDN,
OmniAnomaly, USAD, etc. all report it), but multiple recent papers (Kim
et al., "Towards a Rigorous Evaluation of Time-series Anomaly Detection,"
AAAI 2022; Doshi et al. 2022) showed PA-F1 is trivially inflatable -- a
detector that fires ONE correct point anywhere inside a long anomalous
segment gets full credit for the whole segment, so even random scoring
can score deceptively high PA-F1 on segment-heavy datasets like SWaT.
Sec 8.3 of the design doc explicitly flags this. Report both, but treat
plain F1 / AUC-PR as the trustworthy numbers and PA-F1 as a compatibility
number for comparing against papers that only report it.
"""
import numpy as np
from sklearn import metrics as sk_metrics


def auroc(y_true, y_score):
    return float(sk_metrics.roc_auc_score(y_true, y_score))


def auprc(y_true, y_score):
    return float(sk_metrics.average_precision_score(y_true, y_score))


def precision_recall_f1(y_true, y_pred):
    p = float(sk_metrics.precision_score(y_true, y_pred, zero_division=0))
    r = float(sk_metrics.recall_score(y_true, y_pred, zero_division=0))
    f1 = float(sk_metrics.f1_score(y_true, y_pred, zero_division=0))
    return {"precision": p, "recall": r, "f1": f1}


def point_adjusted(y_true, y_pred):
    """Xu et al. 2018's point-adjustment: within each contiguous true-
    anomaly segment, if ANY point is flagged, treat the WHOLE segment as
    correctly flagged. See module docstring for why this is reported
    separately from, not instead of, plain F1."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool).copy()

    in_segment = False
    seg_start = 0
    for i in range(len(y_true) + 1):
        is_true = y_true[i] if i < len(y_true) else False
        if is_true and not in_segment:
            in_segment = True
            seg_start = i
        elif not is_true and in_segment:
            in_segment = False
            if y_pred[seg_start:i].any():
                y_pred[seg_start:i] = True
    return y_pred


def point_adjusted_f1(y_true, y_pred):
    y_pred_pa = point_adjusted(y_true, y_pred)
    return precision_recall_f1(y_true, y_pred_pa)


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def full_report(y_true, y_score, threshold=None):
    """y_score: higher = more anomalous. threshold: if None, uses the
    score's own 95th percentile among y_true==0 (matches this project's
    convention throughout -- see dataset.py, train_gdn.py, etc)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if threshold is None:
        normal_scores = y_score[y_true == 0]
        threshold = float(np.percentile(normal_scores, 95)) if len(normal_scores) else float(np.median(y_score))
    y_pred = (y_score > threshold).astype(int)

    report = {
        "auroc": auroc(y_true, y_score),
        "auprc": auprc(y_true, y_score),
        "threshold": threshold,
        **precision_recall_f1(y_true, y_pred),
    }
    report["pa_f1"] = point_adjusted_f1(y_true, y_pred)["f1"]
    return report
