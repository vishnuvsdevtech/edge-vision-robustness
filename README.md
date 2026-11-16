# Edge Vision Robustness

**Live POC:** https://vishnuvsdevtech.github.io/edge-vision-robustness/

A detector's reported mAP is measured on clean images. It is deployed on rainy
gantries, in dim toll lanes, through 2 Mbps streams, at plates four pixels of
character tall. This harness measures the gap.

## What it does

Sweeps a detector through the conditions a deployed camera actually meets and
reports **where it breaks, by how much, and at which severity** — per class, at
the confidence threshold production runs at.

```
python -m robustbench run \
  --weights yolov8n.pt \
  --images data/images --labels data/labels \
  --device cuda --out out/
```

Output: `out/report.json` for tooling and `out/report.html` as a self-contained
sheet with a bar per condition and severity.

No weights? It still runs, validating that your ingest and dataset survive the
transforms before you spend GPU time.

## The metrics that matter

mAP is reported for comparability, but it is not the headline:

- mAP integrates every confidence threshold, including ones you never deploy.
- It averages classes you may not care about — a model can be excellent on cars
  and useless on pedestrians and still post a healthy number.
- It cannot express **confidence drift**: boxes still correct, scores sliding
  under the deployed cutoff, the system silently stops acting.

So the report leads with **precision and recall at your threshold, per class**,
plus the delta against the clean baseline.

## Conditions

Ten corruptions, each tagged with the deployment cause it represents, at five
severities:

| corruption | maps to |
|---|---|
| brightness, contrast | lighting |
| lowlight | night |
| blur | optics |
| motion_blur | speed |
| noise | sensor |
| jpeg | stream bitrate |
| downscale | distance |
| fog, rain | weather |

Pure numpy — no OpenCV, no GPU needed for the corruption layer.

## Verify without a GPU

```bash
python -m robustbench selftest        # exercises the metric maths on synthetic data
python -m robustbench run --images data/images --labels data/labels --out out
```

## Status

Working: corruption suite, 101-point AP, greedy IoU matching, per-class
precision/recall at threshold, JSON + HTML report, CLI, self-test.

Next, in order: FGSM/PGD against the detection loss; patch attack with a size
sweep; ANPR mode (exact-plate match and false-match rate instead of mAP);
synthetic plate generator so plate testing needs no customer data.

## Note on the live demo

The browser explorer runs the same corruption engine and reports what each
condition does to the **signal** — mean gradient magnitude, RMS contrast, mean
luma. It is not a model accuracy figure and is not presented as one; accuracy
numbers require your weights, which is what the Python harness is for.
