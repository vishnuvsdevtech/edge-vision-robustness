"""Command line: sweep a detector through deployment conditions.

    python -m robustbench run --weights yolov8n.pt --images data/images \
        --labels data/labels --device cuda --out out/

Without --weights it runs the corruption suite alone (no model), which is
useful for checking that your ingest pipeline and dataset survive the
transforms before you spend GPU time.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

from . import corruptions, metrics, report


def _load_pairs(images_dir, labels_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(images_dir, e))
    pairs = []
    for p in sorted(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(labels_dir, stem + ".txt") if labels_dir else None
        if lp and os.path.exists(lp):
            pairs.append((p, lp))
    return pairs


def _read_yolo_label(path):
    boxes, classes = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            c, x, y, w, h = parts[:5]
            boxes.append([float(x), float(y), float(w), float(h)])
            classes.append(int(float(c)))
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(classes, dtype=int)


def _to_xyxy(boxes, w, h):
    cx, cy, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def run(args):
    pairs = _load_pairs(args.images, args.labels)
    if not pairs:
        sys.exit(f"no image/label pairs found under {args.images}")

    model = None
    if args.weights:
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("ultralytics not installed: pip install ultralytics torch")
        model = YOLO(args.weights)
        if args.device == "auto":
            try:
                import torch
                args.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                args.device = "cpu"

    def predict(img_np_uint8):
        """Return (boxes xyxy, scores, classes) for one image."""
        if model is None:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        res = model.predict(img_np_uint8, device=args.device, conf=0.001, verbose=False)[0]
        b = res.boxes
        return (
            b.xyxy.cpu().numpy().astype(np.float32),
            b.conf.cpu().numpy().astype(np.float32),
            b.cls.cpu().numpy().astype(np.int32),
        )

    def sweep(tag, severity):
        preds, gts = [], []
        for img_path, lbl_path in pairs:
            im = np.asarray(Image.open(img_path).convert("RGB"))
            corrupted = corruptions.apply(im, tag, severity)
            arr = (corrupted * 255).round().astype(np.uint8)
            h, w = arr.shape[:2]
            yb, yc = _read_yolo_label(lbl_path)
            gts.append((_to_xyxy(yb, w, h) if len(yb) else np.zeros((0, 4), np.float32),
                        None, yc))
            preds.append(predict(arr))
        m = metrics.evaluate(preds, gts, iou_thr=args.iou, conf_thr=args.conf)
        print(f"  {tag:12} sev {severity}  mAP {m['map']}  "
              + " ".join(f"{k}:r={v['recall_at_conf']:.3f}" for k, v in m["per_class"].items()))
        return m

    print(f"images: {len(pairs)}  corruptions: {len(corruptions.SUITE)}  device: {args.device}")
    result = {"model": args.weights or "none (corruption sweep only)",
              "device": args.device, "iou_threshold": args.iou,
              "conf_threshold": args.conf, "conditions": {}}

    result["baseline"] = sweep("clean", 0)
    if args.weights:
        for tag in corruptions.SUITE:
            result["conditions"][tag] = {"condition": corruptions.CONDITION[tag]}
            for sev in args.severities:
                result["conditions"][tag][str(sev)] = sweep(tag, sev)

    jpath, hpath = report.write(result, args.out, title=args.title)
    print(f"\nwrote {jpath}\nwrote {hpath}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="robustbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="sweep corruptions against a detector")
    r.add_argument("--weights", default=None)
    r.add_argument("--images", required=True)
    r.add_argument("--labels", default=None)
    r.add_argument("--device", default="auto")
    r.add_argument("--iou", type=float, default=0.5)
    r.add_argument("--conf", type=float, default=0.5)
    r.add_argument("--severities", type=int, nargs="+", default=[1, 3, 5])
    r.add_argument("--out", default="out")
    r.add_argument("--title", default="Robustness report")
    r.set_defaults(func=run)
    st = sub.add_parser("selftest", help="verify the metrics without a GPU")
    st.set_defaults(func=lambda a: metrics.self_test())
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
