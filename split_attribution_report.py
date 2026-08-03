#!/usr/bin/env python3
"""
split_attribution_report.py
============================
Splits a combined attribution_report.html into one file per dataset.

Usage:
    python split_attribution_report.py [path/to/attribution_report.html]

Defaults to the most recent pipeline_runs/<run>/attribution_report.html.
Outputs:  attribution_report_<dataset>.html  in the same folder.
"""

import re
import sys
from pathlib import Path

# ── Resolve input path ────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    src = Path(sys.argv[1])
else:
    runs_dir = Path(__file__).resolve().parent / "pipeline_runs"
    candidates = sorted(runs_dir.glob("*/attribution_report.html"))
    if not candidates:
        sys.exit("No attribution_report.html found. Pass the path explicitly.")
    src = candidates[-1]

print(f"Splitting: {src}")
html = src.read_text(encoding="utf-8")
out_dir = src.parent

# ── Extract header (everything before the first ds-section) ──────────────────
first_section = html.index('<div class="ds-section">')
header = html[:first_section]

# ── Extract footer (everything after the last ds-section closing tag) ─────────
# Each ds-section ends with </div></div> — find the last one before </body>
footer_match = re.search(r'(</body></html>)$', html.strip())
footer = footer_match.group(0) if footer_match else "</body></html>"

# ── Split into ds-section blocks ─────────────────────────────────────────────
# Use a lookahead so we keep the opening tag with each block
sections = re.split(r'(?=<div class="ds-section">)', html[first_section:])
sections = [s for s in sections if s.strip()]

# Strip trailing </body></html> from the last section body
sections[-1] = re.sub(r'</body></html>\s*$', '', sections[-1])

# ── Extract dataset name from each section ────────────────────────────────────
name_re = re.compile(r'<h2>(.*?)</h2>')

written = []
for section in sections:
    m = name_re.search(section)
    if not m:
        continue
    display_name = m.group(1).strip()
    # Normalise to a safe filename slug
    slug = display_name.lower().replace(" ", "_").replace("/", "_")

    page = header + section + footer
    out_path = out_dir / f"attribution_report_{slug}.html"
    out_path.write_text(page, encoding="utf-8")
    kb = out_path.stat().st_size // 1024
    written.append((display_name, out_path.name, kb))
    print(f"  {display_name:20s} → {out_path.name}  ({kb} KB)")

print(f"\nDone — {len(written)} reports written to {out_dir}/")
