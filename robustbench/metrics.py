"""Detection metrics that answer the deployment question, not the benchmark one.

mAP is reported for comparability, but the numbers a deployed system is judged
on are precision and recall *at the operating threshold*, per class, because
that is the only point on the curve production actually runs at.
"""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def match_image(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    """Greedy match, highest score first. Returns (tp_mask, n_gt)."""
    tp = np.zeros(len(pred_boxes), dtype=bool)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return tp, len(gt_boxes)
    order = np.argsort(-pred_scores)
    used = np.zeros(len(gt_boxes), dtype=bool)
    ious = iou_matrix(pred_boxes[order], gt_boxes)
    for i in range(len(order)):
        best, best_j = 0.0, -1
        for j in range(len(gt_boxes)):
            if used[j]:
                continue
            if ious[i, j] > best:
                best, best_j = ious[i, j], j
        if best_j >= 0 and best >= iou_thr:
            used[best_j] = True
            tp[order[i]] = True
    return tp, len(gt_boxes)


def average_precision(scores, tp, n_gt, n_points=101):
    """101-point interpolated AP, the COCO convention."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-scores)
    tp = tp[order].astype(np.float32)
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    # make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    grid = np.linspace(0, 1, n_points)
    idx = np.searchsorted(recall, grid, side="left")
    out = np.zeros(n_points, dtype=np.float32)
    valid = idx < len(precision)
    out[valid] = precision[idx[valid]]
    return float(out.mean())


def evaluate(predictions, ground_truth, iou_thr=0.5, conf_thr=0.5, class_names=None):
    """predictions / ground_truth: list per image of (boxes, scores, classes).

    Returns aggregated metrics: mAP@iou, and per-class precision/recall at
    conf_thr, plus the confusion that matters — misses vs false alarms.
    """
    n_classes = 1 + max(
        [int(c.max()) for _, _, c in ground_truth if len(c)]
        + [int(c.max()) for _, _, c in predictions if len(c)]
        or [0]
    )
    names = class_names or {i: str(i) for i in range(n_classes)}

    per_class = {}
    for k in range(n_classes):
        scores_all, tp_all, n_gt_total = [], [], 0
        tp_at, fp_at, fn_at = 0, 0, 0
        for (pb, ps, pc), (gb, _, gc) in zip(predictions, ground_truth):
            pm, gm = pc == k, gc == k
            tp, n_gt = match_image(pb[pm], ps[pm], gb[gm], iou_thr)
            scores_all.append(ps[pm])
            tp_all.append(tp)
            n_gt_total += n_gt
            keep = ps[pm] >= conf_thr
            tp_at += int((tp & keep).sum())
            fp_at += int((~tp & keep).sum())
            fn_at += int(max(0, n_gt - (tp & keep).sum()))
        if n_gt_total == 0 and not any(len(s) for s in scores_all):
            continue
        s = np.concatenate(scores_all) if scores_all else np.array([])
        t = np.concatenate(tp_all) if tp_all else np.array([], dtype=bool)
        ap = average_precision(s, t, n_gt_total) if len(s) else 0.0
        prec = tp_at / max(tp_at + fp_at, 1)
        rec = tp_at / max(tp_at + fn_at, 1)
        per_class[names.get(k, str(k))] = {
            "ap": None if np.isnan(ap) else round(ap, 4),
            "precision_at_conf": round(prec, 4),
            "recall_at_conf": round(rec, 4),
            "tp": tp_at, "fp": fp_at, "fn": fn_at,
            "ground_truth_instances": int(n_gt_total),
        }

    maps = [v["ap"] for v in per_class.values() if v["ap"] is not None]
    return {
        "iou_threshold": iou_thr,
        "conf_threshold": conf_thr,
        "map": round(float(np.mean(maps)), 4) if maps else None,
        "per_class": per_class,
    }


def self_test(seed=0):
    """Exercise the metrics with synthetic data — no torch, no dataset, so CI
    (and a laptop) can verify the maths before a GPU is available."""
    rng = np.random.default_rng(seed)
    preds, gts = [], []
    for _ in range(12):
        gb = np.array([[10, 10, 60, 60], [100, 40, 160, 110]], dtype=np.float32)
        gts.append((gb, None, np.zeros(len(gb), dtype=int)))
        if rng.random() < 0.85:                        # 85% detection rate
            jitter = rng.normal(0, 2.0, gb.shape).astype(np.float32)
            pb = gb + jitter
            ps = rng.uniform(0.4, 0.95, len(pb)).astype(np.float32)
            if rng.random() < 0.3:                     # an extra false positive
                pb = np.vstack([pb, [[300, 200, 340, 240]]]).astype(np.float32)
                ps = np.append(ps, 0.55).astype(np.float32)
            preds.append((pb, ps, np.zeros(len(pb), dtype=int)))
        else:
            preds.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)))
    m = evaluate(preds, gts, iou_thr=0.5, conf_thr=0.5)
    assert m["map"] is not None and 0.3 < m["map"] <= 1.0, m["map"]
    pc = m["per_class"]["0"]
    assert pc["tp"] + pc["fn"] > 0
    print("metrics self-test ok:", m["map"], {k: (v["precision_at_conf"], v["recall_at_conf"]) for k, v in m["per_class"].items()})
    return m


if __name__ == "__main__":
    self_test()
