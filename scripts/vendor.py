"""Fetch the upstream recogniser at a pinned revision and apply VoxLens patches.

The upstream repository does not run on Apple Silicon as published. Rather than
forking it, we vendor a pinned revision and keep our changes as reviewable
patches in patches/ — so an upstream bump fails loudly at `git apply` instead of
silently dropping a fix.

Idempotent: re-running restores the vendored tree to pinned + patched.

    uv run python scripts/vendor.py
    uv run python scripts/vendor.py --check   # verify, change nothing
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "usr2"
PATCHES = ROOT / "patches"

REPO = "https://github.com/ahaliassos/usr2.git"
REV = "df0c78b7a3807e625a0fcdadd14b1cf674d21c91"


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def die(message: str) -> None:
    print(f"vendor: {message}", file=sys.stderr)
    raise SystemExit(1)


def patches() -> list[Path]:
    found = sorted(PATCHES.glob("*.patch"))
    if not found:
        die(f"no patches found in {PATCHES}")
    return found


def check() -> int:
    if not (VENDOR / "espnet").is_dir():
        print("vendor: not vendored — run: uv run python scripts/vendor.py")
        return 1
    head = run(["git", "rev-parse", "HEAD"], cwd=VENDOR).stdout.strip()
    ok = True
    for patch in patches():
        # --reverse --check succeeds only when the patch is already applied.
        result = run(["git", "apply", "--reverse", "--check", str(patch)], cwd=VENDOR)
        state = "applied" if result.returncode == 0 else "MISSING"
        if result.returncode != 0:
            ok = False
        print(f"  {state:>7}  {patch.name}")
    print(f"vendor: {VENDOR} at {head[:12]} ({'pinned' if head == REV else 'UNPINNED'})")
    return 0 if ok and head == REV else 1


def vendor() -> int:
    if shutil.which("git") is None:
        die("git is required")

    if VENDOR.exists():
        shutil.rmtree(VENDOR)
    VENDOR.parent.mkdir(parents=True, exist_ok=True)

    print(f"vendor: cloning {REPO}")
    if run(["git", "clone", "--quiet", REPO, str(VENDOR)]).returncode != 0:
        die("clone failed")
    if run(["git", "checkout", "--quiet", REV], cwd=VENDOR).returncode != 0:
        die(f"pinned revision {REV[:12]} not found upstream")

    for patch in patches():
        result = run(["git", "apply", str(patch)], cwd=VENDOR)
        if result.returncode != 0:
            die(
                f"failed to apply {patch.name}:\n{result.stderr}\n"
                "The pinned revision and the patches have diverged. Re-cut the "
                "patch against the new revision rather than editing vendor/ in place."
            )
        print(f"  applied  {patch.name}")

    # The tree keeps its .git so --check can verify the pin and the patches.
    # vendor/ is gitignored, so this nested repository never reaches history.
    print(f"vendor: ready at {VENDOR}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify without changing anything")
    args = ap.parse_args()
    return check() if args.check else vendor()


if __name__ == "__main__":
    raise SystemExit(main())
