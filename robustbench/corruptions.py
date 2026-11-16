"""Corruption suite for detection robustness testing.

Pure numpy. Every function takes an HWC float image in [0,1] and a severity in
1..5, and returns a corrupted copy. Severity 0 is identity, so a sweep can
include the clean baseline without special-casing.

These are the conditions a deployed edge camera actually meets: a dim toll
lane, a rainy gantry, a 2 Mbps stream, a plate four pixels of character tall.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

SEVERITIES = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------- helpers
def _as_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img


def _clip(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 1.0)


def _pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray((_clip(img) * 255).round().astype(np.uint8))


def _np(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def _blur1d(img: np.ndarray, radius: float, axis: int) -> np.ndarray:
    """Box blur via cumulative sum — O(n), no scipy."""
    if radius < 1:
        return img
    r = int(round(radius))
    pad = [(0, 0), (0, 0), (0, 0)]
    pad[axis] = (r, r)
    padded = np.pad(img, pad, mode="edge")
    cs = np.cumsum(padded, axis=axis)
    zero_shape = list(cs.shape)
    zero_shape[axis] = 1
    cs = np.concatenate([np.zeros(zero_shape, dtype=cs.dtype), cs], axis=axis)
    n = 2 * r + 1
    hi = np.take(cs, range(n, cs.shape[axis]), axis=axis)
    lo = np.take(cs, range(0, cs.shape[axis] - n), axis=axis)
    return (hi - lo) / n


# ---------------------------------------------------------------- corruptions
def brightness(img, severity):
    if not severity:
        return img
    factor = 1.0 - 0.14 * severity          # s5 -> 30% of original
    return _clip(_as_float(img) * factor)


def contrast(img, severity):
    if not severity:
        return img
    x = _as_float(img)
    factor = 1.0 - 0.13 * severity
    return _clip((x - 0.5) * factor + 0.5)


def lowlight(img, severity):
    """Gamma darkening plus sensor noise — the night-time regime."""
    if not severity:
        return img
    x = _as_float(img) ** (1.0 + 0.55 * severity)
    noise = np.random.default_rng(severity).normal(0, 0.006 * severity, x.shape)
    return _clip(x + noise)


def blur(img, severity):
    if not severity:
        return img
    x = _as_float(img)
    r = severity * 0.9
    return _clip(_blur1d(_blur1d(x, r, axis=0), r, axis=1))


def motion_blur(img, severity):
    """Horizontal smear, as a fast-moving vehicle crosses the shutter."""
    if not severity:
        return img
    x = _as_float(img)
    length = severity * 5
    acc = np.zeros_like(x)
    for k in range(int(length)):
        acc += np.roll(x, k - int(length) // 2, axis=1)
    return _clip(acc / max(1, int(length)))


def noise(img, severity):
    if not severity:
        return img
    x = _as_float(img)
    rng = np.random.default_rng(1000 + severity)
    return _clip(x + rng.normal(0, 0.012 * severity, x.shape))


def jpeg(img, severity):
    """Compression at the bitrate a real stream actually uses."""
    if not severity:
        return img
    quality = max(12, 92 - severity * 16)
    buf = io.BytesIO()
    _pil(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return _np(Image.open(buf))


def downscale(img, severity):
    """Distance. Characters must survive this one to be read at all."""
    if not severity:
        return img
    x = _as_float(img)
    h, w = x.shape[:2]
    k = 1.0 - 0.11 * severity
    small = _pil(x).resize((max(8, int(w * k)), max(8, int(h * k))), Image.BILINEAR)
    return _np(small.resize((w, h), Image.BILINEAR))


def fog(img, severity):
    if not severity:
        return img
    x = _as_float(img)
    veil = 0.10 * severity
    return _clip(x * (1 - veil) + 0.72 * veil)


def rain(img, severity):
    """Diagonal streaks plus a darkening — crude, but it changes the statistics
    the detector keys on, which is all a corruption needs to do."""
    if not severity:
        return img
    x = _as_float(img).copy()
    h, w = x.shape[:2]
    rng = np.random.default_rng(77 + severity)
    n = severity * 90
    for _ in range(n):
        y = rng.integers(0, h - 12)
        xx = rng.integers(0, w - 6)
        length = int(rng.integers(5, 12))
        x[y:y + length, xx:xx + 1] = np.clip(x[y:y + length, xx:xx + 1] + 0.28, 0, 1)
    return _clip(x * (1 - 0.03 * severity))


SUITE = {
    "brightness": brightness,
    "contrast": contrast,
    "lowlight": lowlight,
    "blur": blur,
    "motion_blur": motion_blur,
    "noise": noise,
    "jpeg": jpeg,
    "downscale": downscale,
    "fog": fog,
    "rain": rain,
}

# Which corruptions map to which real deployment condition — the report groups
# by this so a reader sees "night", not "gamma 3.75".
CONDITION = {
    "brightness": "lighting",
    "contrast": "lighting",
    "lowlight": "night",
    "blur": "optics",
    "motion_blur": "speed",
    "noise": "sensor",
    "jpeg": "stream bitrate",
    "downscale": "distance",
    "fog": "weather",
    "rain": "weather",
}


def apply(img, name: str, severity: int):
    img = _as_float(img)
    if severity == 0:
        return img
    return SUITE[name](img, severity)
