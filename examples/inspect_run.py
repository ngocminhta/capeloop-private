"""Print the manifest and summary of a completed run."""

from pathlib import Path
import json
import sys


if len(sys.argv) != 2:
    raise SystemExit("usage: python examples/inspect_run.py RUN_DIR")

run = Path(sys.argv[1])
for relative in ("manifest.json", "metrics/summary.json"):
    print(f"\n## {relative}")
    print(json.dumps(json.loads((run / relative).read_text()), indent=2, sort_keys=True))

