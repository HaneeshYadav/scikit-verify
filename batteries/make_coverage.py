"""Regenerate doc/coverage.md from the battery outputs.

Run each battery, save its stdout next to it as <name>.out, then run
this. The page is a tracking list: every function we tried, and what
happened -- nothing else.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
LIBS = [
    ("NumPy / SciPy", "scipy_full.out"),
    ("scikit-learn", "skl_full.out"),
    ("statsmodels", "sm_full.out"),
]

def parse(path):
    text = path.read_text()
    lifted = []
    m = re.search(r"^LIFT[^:]*: (.*)$", text, re.M)
    if m:
        lifted = [x.strip() for x in m.group(1).split(",") if x.strip()]
    refused = re.findall(r"^\s{2,}(\S+)\s+\|", text[text.find("REFUSED"):text.find("DIED")], re.M) if "REFUSED" in text else []
    died = re.findall(r"^\s{2,}(\S+)\s+\|", text[text.find("DIED"):], re.M) if "DIED" in text else []
    return lifted, refused, died

out = [
    "# Coverage",
    "",
    "Functions we have run through `to_sympy`, and what happened.",
    "**works** means the traced value matched the library exactly on",
    "that call; **refuses** means a one-sentence refusal (see",
    "[sharp bits](sharp-bits.md)). Regenerate with the batteries in",
    "`batteries/`.",
    "",
]
for title, fname in LIBS:
    p = HERE / fname
    if not p.exists():
        continue
    lifted, refused, died = parse(p)
    out.append(f"## {title}")
    out.append("")
    out.append(f"**Works ({len(lifted)}):** " + ", ".join(f"`{x}`" for x in lifted))
    out.append("")
    if refused:
        out.append(f"**Refuses ({len(refused)}):** " + ", ".join(f"`{x}`" for x in refused))
        out.append("")
    if died:
        out.append(f"**Known walls ({len(died)}):** " + ", ".join(f"`{x}`" for x in died))
        out.append("")
(HERE.parent / "doc" / "coverage.md").write_text("\n".join(out) + "\n")
print("doc/coverage.md written")
