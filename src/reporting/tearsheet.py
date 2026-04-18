"""
Tearsheet generator
====================

Produces a standalone HTML tearsheet from a ``BacktestResult`` or from a
validation-run artefact under ``fund_data/``. Purpose is a single, shareable
page summarising performance, risk, and trade quality — the artefact you send
to an allocator or pin to a PR.

Zero external dependencies beyond matplotlib (optional) and the stdlib. The
HTML embeds PNG equity / drawdown charts as base64 so the file is
self-contained.

Usage
-----

.. code-block:: bash

    # From the most recent validation JSON (auto-detected)
    python -m src.reporting.tearsheet --output fund_data/tearsheet.html

    # From a specific validation file
    python -m src.reporting.tearsheet \\
        --input fund_data/validation_v16.json \\
        --output fund_data/tearsheet.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _latest_validation(fund_dir: Path) -> Path | None:
    candidates = sorted(fund_dir.glob("validation_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _load_validation(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Metric extraction
# --------------------------------------------------------------------------
def _summarise(payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce a validation JSON to a flat set of headline metrics.

    The validation JSON schema is relatively free-form; we defensively pull
    any of the common keys we know about.
    """
    strategies = payload.get("strategies", []) or payload.get("results", [])
    if not isinstance(strategies, list):
        strategies = []

    keeps = [s for s in strategies if s.get("final_verdict") == "KEEP"]
    conditional = [s for s in strategies if s.get("final_verdict") == "CONDITIONAL"]
    rejected = [
        s
        for s in strategies
        if s.get("final_verdict") not in {"KEEP", "CONDITIONAL"}
    ]

    def _mean(key: str, rows: list[dict[str, Any]]) -> float | None:
        values = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        return sum(values) / len(values) if values else None

    return {
        "total_strategies": len(strategies),
        "keep": len(keeps),
        "conditional": len(conditional),
        "rejected": len(rejected),
        "avg_sharpe_keep": _mean("sharpe", keeps) or _mean("oos_sharpe", keeps),
        "avg_pf_keep": _mean("profit_factor", keeps) or _mean("pf", keeps),
        "avg_trades_keep": _mean("n_trades", keeps) or _mean("total_trades", keeps),
        "symbols": sorted({s.get("asset") for s in strategies if s.get("asset")}),
        "timeframe": payload.get("timeframe"),
        "days": payload.get("days"),
        "min_trades": payload.get("min_trades"),
        "oos_fraction": payload.get("oos_fraction"),
        "generated_at": payload.get("generated_at"),
        "rows_keep": keeps,
        "rows_conditional": conditional,
    }


# --------------------------------------------------------------------------
# Charting (optional — degrades gracefully without matplotlib)
# --------------------------------------------------------------------------
def _render_bar_chart(
    title: str, labels: list[str], values: list[float], ylabel: str
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        logger.info("matplotlib unavailable — skipping chart %s", title)
        return None

    if not labels or not values:
        return None

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(labels, values, color="#2563eb", edgecolor="#0f172a")
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------
_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SignalForge — Tearsheet</title>
<style>
  :root {{
    --bg: #0b1020; --card: #121a32; --fg: #e6eefc; --muted: #8aa0c6;
    --accent: #60a5fa; --good: #22c55e; --bad: #ef4444;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
    background: var(--bg); color: var(--fg); margin: 0; padding: 32px;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  h2 {{ font-size: 16px; margin: 28px 0 12px; color: var(--accent); }}
  .sub {{ color: var(--muted); margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .card {{
    background: var(--card); border: 1px solid #243055; border-radius: 10px;
    padding: 14px 16px;
  }}
  .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
  .v {{ font-size: 20px; font-weight: 600; margin-top: 4px; }}
  .good {{ color: var(--good); }}
  .bad  {{ color: var(--bad); }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
  th, td {{ padding: 8px 10px; text-align: right; border-bottom: 1px solid #1e2a4a; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--muted); font-weight: 500; font-size: 12px; text-transform: uppercase; }}
  img {{ max-width: 100%; border-radius: 8px; background: white; padding: 8px; }}
  footer {{ margin-top: 40px; color: var(--muted); font-size: 12px; }}
  code {{ background: #0f1730; padding: 1px 6px; border-radius: 4px; }}
</style>
</head>
<body>
  <h1>SignalForge — Tearsheet</h1>
  <div class="sub">
    Source: <code>{source}</code> &middot; Generated {generated}
  </div>

  <h2>Headline</h2>
  <div class="grid">
    <div class="card"><div class="k">Strategies tested</div><div class="v">{total}</div></div>
    <div class="card"><div class="k">KEEP</div><div class="v good">{keep}</div></div>
    <div class="card"><div class="k">Conditional</div><div class="v">{conditional}</div></div>
    <div class="card"><div class="k">Rejected</div><div class="v bad">{rejected}</div></div>
    <div class="card"><div class="k">Avg Sharpe (KEEP)</div><div class="v">{avg_sharpe}</div></div>
    <div class="card"><div class="k">Avg Profit Factor (KEEP)</div><div class="v">{avg_pf}</div></div>
    <div class="card"><div class="k">Avg trades (KEEP)</div><div class="v">{avg_trades}</div></div>
    <div class="card"><div class="k">Timeframe</div><div class="v">{timeframe}</div></div>
  </div>

  <h2>Run configuration</h2>
  <div class="card">
    <div><b>Symbols:</b> {symbols}</div>
    <div><b>Lookback:</b> {days} days</div>
    <div><b>OOS fraction:</b> {oos}</div>
    <div><b>Min trades gate:</b> {min_trades}</div>
  </div>

  {chart_section}

  <h2>KEEP strategies</h2>
  {keep_table}

  <h2>Conditional strategies</h2>
  {cond_table}

  <footer>
    SignalForge is research-grade software. This tearsheet is an internal
    performance summary and is not financial advice. See
    <a style="color:var(--accent)" href="https://github.com/varunteja0/SignalForge#-status--disclaimer">the disclaimer</a>.
  </footer>
</body>
</html>
"""


def _fmt(x: Any, precision: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if x != x:  # NaN
            return "—"
        return f"{x:,.{precision}f}"
    return str(x)


def _table(rows: list[dict[str, Any]], limit: int = 20) -> str:
    if not rows:
        return '<div class="card" style="color:var(--muted)">No rows.</div>'
    columns = [
        ("name", "Strategy"),
        ("asset", "Asset"),
        ("n_trades", "Trades"),
        ("sharpe", "Sharpe"),
        ("profit_factor", "PF"),
        ("max_drawdown", "MaxDD"),
        ("win_rate", "Win%"),
    ]
    head = "".join(f"<th>{title}</th>" for _, title in columns)
    body_rows: list[str] = []
    for r in rows[:limit]:
        cells: list[str] = []
        for key, _ in columns:
            val = r.get(key)
            if key == "max_drawdown" and isinstance(val, (int, float)):
                val = f"{val * 100:.1f}%"
            elif key == "win_rate" and isinstance(val, (int, float)):
                val = f"{val * 100:.1f}%"
            else:
                val = _fmt(val)
            cells.append(f"<td>{val}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _chart_section(summary: dict[str, Any]) -> str:
    keeps = summary["rows_keep"]
    if not keeps:
        return ""

    def _take(key: str) -> list[float]:
        return [float(r.get(key, 0) or 0) for r in keeps[:12]]

    labels = [str(r.get("name", "?"))[:24] for r in keeps[:12]]
    sharpe_png = _render_bar_chart("Sharpe — KEEP strategies", labels, _take("sharpe"), "Sharpe")
    pf_png = _render_bar_chart(
        "Profit Factor — KEEP strategies", labels, _take("profit_factor"), "PF"
    )

    imgs: list[str] = []
    if sharpe_png:
        imgs.append(f'<img alt="Sharpe" src="data:image/png;base64,{sharpe_png}"/>')
    if pf_png:
        imgs.append(f'<img alt="PF" src="data:image/png;base64,{pf_png}"/>')
    if not imgs:
        return ""
    return '<h2>Charts</h2><div class="card">' + "".join(imgs) + "</div>"


def render_html(summary: dict[str, Any], source: str) -> str:
    return _HTML.format(
        source=source,
        generated=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        total=summary["total_strategies"],
        keep=summary["keep"],
        conditional=summary["conditional"],
        rejected=summary["rejected"],
        avg_sharpe=_fmt(summary["avg_sharpe_keep"]),
        avg_pf=_fmt(summary["avg_pf_keep"]),
        avg_trades=_fmt(summary["avg_trades_keep"], precision=0),
        timeframe=summary.get("timeframe") or "—",
        symbols=", ".join(summary.get("symbols", [])) or "—",
        days=summary.get("days") or "—",
        oos=summary.get("oos_fraction") or "—",
        min_trades=summary.get("min_trades") or "—",
        chart_section=_chart_section(summary),
        keep_table=_table(summary["rows_keep"]),
        cond_table=_table(summary["rows_conditional"]),
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a SignalForge tearsheet HTML.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a validation JSON. Defaults to the most recent fund_data/validation_*.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fund_data/tearsheet.html"),
        help="Output HTML path. Default: fund_data/tearsheet.html",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    source = args.input or _latest_validation(Path("fund_data"))
    if source is None or not source.exists():
        print("No validation JSON found. Run `sf.py validate-all` first.", file=sys.stderr)
        return 2

    payload = _load_validation(source)
    summary = _summarise(payload)
    html = render_html(summary, source=str(source))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"✓ Tearsheet written to {args.output}")
    print(
        f"  {summary['keep']} KEEP · {summary['conditional']} CONDITIONAL · "
        f"{summary['rejected']} REJECTED (total {summary['total_strategies']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
