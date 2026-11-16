"""Report generation: JSON for tooling, self-contained HTML for humans."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone


def _bar(value, worst=0.0, best=1.0, width=120):
    if value is None:
        return "<span class='muted'>n/a</span>"
    pct = max(0.0, min(1.0, (value - worst) / max(best - worst, 1e-9)))
    return (
        f"<span class='bar'><i style='width:{pct * width:.0f}px'></i></span>"
        f"<span class='num'>{value:.3f}</span>"
    )


def to_html(result: dict, title="Robustness report") -> str:
    rows = []
    base = result.get("baseline", {})
    base_rec = (base.get("per_class") or {})
    for cond, cond_res in sorted(result.get("conditions", {}).items()):
        sev_rows = []
        for sev, m in sorted(cond_res.items(), key=lambda kv: int(kv[0])):
            pc = m.get("per_class") or {}
            for cls, v in pc.items():
                b = (base_rec.get(cls) or {}).get("recall_at_conf")
                drop = None if (b is None or v["recall_at_conf"] is None) else v["recall_at_conf"] - b
                sev_rows.append(
                    f"<tr><td>{html.escape(str(sev))}</td><td>{html.escape(cls)}</td>"
                    f"<td>{_bar(v['recall_at_conf'])}</td>"
                    f"<td>{'—' if drop is None else f'{drop:+.3f}'}</td>"
                    f"<td>{v['fp']}</td><td>{v['fn']}</td></tr>"
                )
        rows.append(
            f"<tr class='cond'><td colspan='6'><strong>{html.escape(cond)}</strong>"
            f" <span class='muted'>· {html.escape(str(cond_res.get('condition', '')))}</span></td></tr>"
            + "".join(sev_rows)
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
 body{{background:#0e0f0d;color:#e8e5dd;font:14px/1.6 ui-sans-serif,system-ui,sans-serif;margin:0;padding:40px}}
 .wrap{{max-width:1000px;margin:0 auto}}
 h1{{font-size:22px;letter-spacing:-.02em;margin:0 0 4px}}
 .meta{{color:#8b877f;font:12px ui-monospace,monospace;margin-bottom:28px}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 th{{text-align:left;color:#8b877f;font:11px ui-monospace,monospace;text-transform:uppercase;
    letter-spacing:.1em;padding:6px 8px 6px 0;border-bottom:1px solid #2a2926}}
 td{{padding:6px 8px 6px 0;border-bottom:1px solid #1b1a17;vertical-align:middle}}
 tr.cond td{{padding-top:18px;border-bottom:1px solid #2a2926;font-size:13px}}
 .bar{{display:inline-block;width:120px;height:6px;background:#1b1a17;border-radius:3px;
   vertical-align:middle;margin-right:8px;overflow:hidden}}
 .bar i{{display:block;height:100%;background:#5fd08a}}
 .num{{font:12px ui-monospace,monospace;color:#c9c4ba}}
 .muted{{color:#6f6a62}}
</style></head><body><div class="wrap">
<h1>{html.escape(title)}</h1>
<div class="meta">generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ·
 model {html.escape(str(result.get('model', 'unknown')))} ·
 device {html.escape(str(result.get('device', 'unknown')))} ·
 threshold {result.get('conf_threshold', 0.5)}</div>
<table><thead><tr><th>severity</th><th>class</th><th>recall @ threshold</th><th>vs clean</th><th>FP</th><th>FN</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan=6 class="muted">no results</td></tr>'}</tbody></table>
</div></body></html>"""


def write(result: dict, out_dir: str, title="Robustness report", prefix="report"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, f"{prefix}.json")
    hpath = os.path.join(out_dir, f"{prefix}.html")
    with open(jpath, "w") as f:
        json.dump(result, f, indent=2)
    with open(hpath, "w") as f:
        f.write(to_html(result, title))
    return jpath, hpath
