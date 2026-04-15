"""Run every verify_*.py script and report a single pass/fail summary."""

import subprocess
import sys
from pathlib import Path

scripts = sorted(Path(__file__).parent.glob("verify_*.py"))
fails = []
for s in scripts:
    print(f"\n===== {s.name} =====")
    r = subprocess.run([sys.executable, str(s)])
    if r.returncode != 0:
        fails.append(s.name)

print("\n" + "=" * 50)
if fails:
    print(f"FAILED: {', '.join(fails)}")
    sys.exit(1)
print(f"All {len(scripts)} verify scripts passed.")
