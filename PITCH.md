# For review: a robustness gate before edge deployment

## The gap

Our detectors are validated on clean sets and deployed against weather, night,
compression and distance. Nobody currently measures the second thing. A clean
mAP answers "does it work in the lab"; a customer asks "will it work at my gate
in the rain at 6pm".

## What this gives us

1. **A failure envelope per model** — precision/recall at our deployed threshold,
   per class, per condition, per severity. Replaces "it's 0.91" with "it holds
   until fog severity 3, where pedestrian recall drops 41%".
2. **A numbers-backed spec sheet** — "≥95% recall at ≤50 km/h, ≥24 px object
   height, dry, 2 Mbps" is a deployment constraint a customer and an installer
   can both work to. It is also a defence when a site underperforms.
3. **A regression gate** — re-run the sweep on every model change. A retrain
   that improves clean mAP but loses night recall is a regression we currently
   cannot see.
4. **Targeted data collection** — the sweep says which condition is worth
   filming, instead of collecting blind.

## Why a GPU

| task | CPU (this POC) | one GPU |
|---|---|---|
| 10 conditions × 5 severities × full val set | hours | minutes |
| PGD / patch optimisation (thousands of backward passes per image) | impractical | the point |
| Retrain against the discovered failure, then re-verify | impractical | one loop |

The harness already targets `--device cuda` and falls back to CPU. No rewrite.

## What is built and what is next

**Built:** corruption suite (10 × 5, numpy), COCO-style AP, per-class
precision/recall at threshold, JSON + HTML report, CLI, self-test. Pipeline
verified end to end without a GPU.

**Next, in order:**
1. Adversarial layer — FGSM/PGD, ε sweep, reported as the same degradation table.
2. Patch attack with a size sweep — "how many pixels of sticker hide a person".
3. ANPR mode — exact-plate match rate and false-match rate, because for plates a
   wrong character is worse than a miss.
4. Synthetic plate generator — plate robustness testing without customer data.

## Cost and timeline

| | |
|---|---|
| GPU | one mid-range card for sweeps; no training cluster needed |
| Timeline | adversarial layer ~1 week; ANPR mode ~2 weeks; plate generator ~3 days |
| Risk | low — nothing here touches customer data or live systems; publicly documented methods only |

## The ask

A GPU box and a slice of one engineer's time to take this from POC to a gate
that runs on every model we ship. The value is not the tool; it is that we stop
discovering our failure modes in the field.
