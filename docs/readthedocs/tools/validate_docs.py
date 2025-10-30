#!/usr/bin/env python3
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"

PROBLEMS = []

# 1) Forbid committing generated reports (docs/reports/*.md)
for md in (DOCS / "reports").glob("*.md"):
    PROBLEMS.append(f"Generated report should not be committed: {md.relative_to(ROOT)}")

# 2) Ensure MyST math fences use ```{math} not ```math
math_fence_bad = re.compile(r"^```\s*math\s*$", re.MULTILINE)
for md in DOCS.rglob("*.md"):
    if md.is_file():
        try:
            text = md.read_text(encoding="utf-8")
        except Exception:
            continue
        if math_fence_bad.search(text):
            msg = (
                "Use ```{math} fenced blocks, found ```math in "
                f"{md.relative_to(ROOT)}"
            )
            PROBLEMS.append(msg)

# 3) Basic hygiene: warn if any tabs in .rst files
for rst in DOCS.rglob("*.rst"):
    if rst.is_file():
        try:
            text = rst.read_text(encoding="utf-8")
        except Exception:
            continue
        if "\t" in text:
            PROBLEMS.append(
                f"Tabs found in RST file (use spaces): {rst.relative_to(ROOT)}"
            )

if PROBLEMS:
    print("Documentation validation issues found:")
    for p in PROBLEMS:
        print(f" - {p}")
    sys.exit(1)

print("Documentation validation passed.")
sys.exit(0)
