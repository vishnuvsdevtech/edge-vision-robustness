/* Edge Vision Robustness — live corruption explorer.
 *
 * The corruptions here mirror robustbench/corruptions.py, severity for
 * severity, so the browser demo and the Python sweep degrade an image the same
 * way. What is measured is the SIGNAL — mean gradient magnitude (edge energy),
 * RMS contrast, mean luma — because that is what detectors and OCR actually
 * key on. It is not a model accuracy figure and is never presented as one.
 */

const W = 640, H = 360;
const SRC = 'assets/street.jpg';

const view = document.getElementById('c');
const vctx = view.getContext('2d', { willReadFrequently: true });
const work = document.createElement('canvas');
work.width = W; work.height = H;
const wctx = work.getContext('2d', { willReadFrequently: true });

let img = null;
let cleanEdges = 1;

const el = (id) => document.getElementById(id);

/* ------------------------------------------------------------ measurement */
function measure(ctx) {
  const { data } = ctx.getImageData(0, 0, W, H);
  const gray = new Float32Array(W * H);
  let sum = 0;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const g = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    gray[p] = g;
    sum += g;
  }
  const mean = sum / (W * H);
  let sq = 0;
  for (let p = 0; p < gray.length; p++) {
    const d = gray[p] - mean;
    sq += d * d;
  }
  const rms = Math.sqrt(sq / gray.length);

  // Sobel magnitude, summed — a stand-in for "how much structure is left"
  let edges = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const gx = -gray[i - W - 1] - 2 * gray[i - 1] - gray[i + W - 1]
                 + gray[i - W + 1] + 2 * gray[i + 1] + gray[i + W + 1];
      const gy = -gray[i - W - 1] - 2 * gray[i - W] - gray[i - W + 1]
                 + gray[i + W - 1] + 2 * gray[i + W] + gray[i + W + 1];
      edges += Math.abs(gx) + Math.abs(gy);
    }
  }
  const edgeEnergy = edges / ((W - 2) * (H - 2));
  return { mean, rms, edgeEnergy };
}

/* ------------------------------------------------------------ corruptions */
function drawClean(ctx) {
  ctx.filter = 'none';
  ctx.globalAlpha = 1;
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(img, 0, 0, W, H);
}

function addNoise(ctx, amount, seed) {
  const id = ctx.getImageData(0, 0, W, H);
  let s = seed || 12345;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return (s / 0x7fffffff) * 2 - 1;
  };
  const d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = rnd() * amount;
    d[i] = Math.max(0, Math.min(255, d[i] + n));
    d[i + 1] = Math.max(0, Math.min(255, d[i + 1] + n));
    d[i + 2] = Math.max(0, Math.min(255, d[i + 2] + n));
  }
  ctx.putImageData(id, 0, 0);
}

function jpegRoundTrip(ctx, quality) {
  return new Promise((resolve) => {
    const url = ctx.canvas.toDataURL('image/jpeg', quality);
    const im = new Image();
    im.onload = () => {
      ctx.filter = 'none';
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(im, 0, 0, W, H);
      resolve();
    };
    im.onerror = () => resolve();
    im.src = url;
  });
}

async function applyCorruption(ctx, cond, s) {
  drawClean(ctx);
  if (!s || cond === 'none') return;

  switch (cond) {
    case 'brightness':
      ctx.filter = `brightness(${(1 - 0.14 * s).toFixed(3)})`;
      drawClean(ctx);
      ctx.filter = 'none';
      break;

    case 'contrast':
      ctx.filter = `contrast(${(1 - 0.13 * s).toFixed(3)})`;
      drawClean(ctx);
      ctx.filter = 'none';
      break;

    case 'lowlight':
      ctx.filter = `brightness(${Math.pow(0.62, s / 2.2).toFixed(3)})`;
      drawClean(ctx);
      ctx.filter = 'none';
      addNoise(ctx, 2.2 * s, 7 + s);
      break;

    case 'blur':
      ctx.filter = `blur(${(s * 0.9).toFixed(2)}px)`;
      drawClean(ctx);
      ctx.filter = 'none';
      break;

    case 'motion_blur': {
      const len = s * 6;
      ctx.clearRect(0, 0, W, H);
      for (let k = 0; k < len; k++) {
        ctx.globalAlpha = 1 / len;
        ctx.drawImage(img, (k - len / 2) | 0, 0, W, H);
      }
      ctx.globalAlpha = 1;
      break;
    }

    case 'noise':
      addNoise(ctx, 3.2 * s, 900 + s);
      break;

    case 'jpeg':
      await jpegRoundTrip(ctx, Math.max(0.12, 0.92 - s * 0.16));
      break;

    case 'downscale': {
      const k = 1 - 0.11 * s;
      const small = document.createElement('canvas');
      small.width = Math.max(8, Math.round(W * k));
      small.height = Math.max(8, Math.round(H * k));
      const sctx = small.getContext('2d');
      sctx.drawImage(img, 0, 0, small.width, small.height);
      ctx.clearRect(0, 0, W, H);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(small, 0, 0, W, H);
      break;
    }

    case 'fog': {
      const veil = 0.10 * s;
      ctx.globalAlpha = 1;
      ctx.fillStyle = `rgba(186,188,182,${veil})`;
      ctx.fillRect(0, 0, W, H);
      break;
    }

    case 'rain': {
      ctx.globalAlpha = 1;
      let st = 4242 + s;
      const rnd = () => {
        st = (st * 1103515245 + 12345) & 0x7fffffff;
        return st / 0x7fffffff;
      };
      ctx.strokeStyle = 'rgba(226,232,240,0.30)';
      ctx.lineWidth = 1;
      for (let i = 0; i < s * 90; i++) {
        const x = rnd() * W, y = rnd() * H, l = 5 + rnd() * 7;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + l * 0.35, y + l);
        ctx.stroke();
      }
      ctx.fillStyle = `rgba(10,12,14,${0.03 * s})`;
      ctx.fillRect(0, 0, W, H);
      break;
    }
  }
}

/* ------------------------------------------------------------------- UI */
const CONDITION_LABEL = {
  none: 'clean', brightness: 'lighting', lowlight: 'night', contrast: 'lighting',
  blur: 'optics', motion_blur: 'speed', noise: 'sensor', jpeg: 'stream bitrate',
  downscale: 'distance', fog: 'weather', rain: 'weather',
};

function pct(v) { return (v * 100).toFixed(0) + '%'; }

function showMetrics(m) {
  const detail = m.edgeEnergy / cleanEdges;
  el('mDetail').textContent = pct(Math.min(1, detail));
  el('mDetail').style.color = detail < 0.5 ? 'var(--warn)' : 'var(--acc)';
  el('mContrast').textContent = m.rms.toFixed(1);
  el('mMean').textContent = m.mean.toFixed(1);
  return detail;
}

async function render() {
  const cond = el('cond').value;
  const sev = parseInt(el('sev').value, 10);
  el('sevVal').textContent = sev;
  el('condLabel').textContent = cond === 'none' ? 'clean' : cond.replace('_', ' ');
  el('sevLabel').textContent = 'severity ' + sev;
  await applyCorruption(vctx, cond, sev);
  showMetrics(measure(vctx));
}

const SEV_LABELS = [1, 3, 5];

function barFor(ratio) {
  const pctW = Math.max(2, Math.round(Math.min(1, ratio) * 64));
  const cls = ratio < 0.5 ? 'low' : '';
  return `<span class="bar"><i class="${cls}" style="width:${pctW}px"></i></span><span class="fill">${pct(ratio)}</span>`;
}

async function sweep() {
  const btn = el('sweep');
  btn.disabled = true;
  btn.textContent = 'sweeping…';
  const conds = Object.keys(CONDITION_LABEL).filter((c) => c !== 'none');
  const rows = [];
  for (const cond of conds) {
    const cells = [];
    for (const s of SEV_LABELS) {
      await applyCorruption(wctx, cond, s);
      const m = measure(wctx);
      cells.push(barFor(m.edgeEnergy / cleanEdges));
    }
    rows.push(
      `<tr><td>${cond.replace('_', ' ')}</td><td class="muted">${CONDITION_LABEL[cond]}</td>`
      + cells.map((c) => `<td>${c}</td>`).join('') + '</tr>'
    );
  }
  el('sweepTable').querySelector('tbody').innerHTML = rows.join('');
  el('sweepNote').textContent =
    'Detail retained vs clean, at severity 1 / 3 / 5. Amber below 50% — that is where a detector '
    + 'loses the structure it needs, and where OCR stops being able to separate 8 from B.';
  btn.disabled = false;
  btn.textContent = 'Re-run sweep →';
  render();
}

/* ------------------------------------------------------------------- boot */
img = new Image();
img.onload = () => {
  drawClean(wctx);
  cleanEdges = measure(wctx).edgeEnergy || 1;
  el('mContrast').textContent = '—';
  render();
};
img.onerror = () => {
  el('condLabel').textContent = 'image failed to load';
};
img.src = SRC;

el('cond').addEventListener('change', render);
el('sev').addEventListener('input', render);
el('sweep').addEventListener('click', sweep);
