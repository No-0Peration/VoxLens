# Setup

VoxLens targets **Apple Silicon**. The recogniser it builds on does not run there
as published, so setup is a little more than `uv sync`.

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install -e . --group dev
uv run python scripts/vendor.py
uv run pytest
```

That should end with tests passing and one or two skips. The skips are expected —
they are the tests that need a checkpoint.

To run the full smoke test, point at a checkpoint you have obtained yourself:

```bash
VOXLENS_CHECKPOINT=/path/to/usr2_large.pth uv run pytest
```

## The checkpoint

**VoxLens never downloads the checkpoint.** It is roughly 4 GB, it is
CC BY-NC 4.0, and fetching it silently as a side effect of running a test would
be rude. Obtain USR 2.0 **Large** (high-resource, fine-tuned) yourself from
[`ahaliassos/usr2`](https://github.com/ahaliassos/usr2) and pass its path via
`VOXLENS_CHECKPOINT` or `--checkpoint`.

Large rather than Huge: Huge's parameter count is unpublished, and its training
data includes a corpus with more restrictive terms. See VoxLens issue #3.

## The vendored recogniser

`scripts/vendor.py` clones the upstream repository at a **pinned revision** into
`vendor/usr2` and applies the patches in `patches/`. `vendor/` is generated and
gitignored — never edit it in place, because the next vendor run discards your
changes.

```bash
uv run python scripts/vendor.py           # fetch and patch
uv run python scripts/vendor.py --check   # verify, change nothing
```

If a patch fails to apply, the pinned revision and the patch have diverged.
**Re-cut the patch against the new revision**; do not paper over it by editing
`vendor/` directly. Failing loudly is the point — a silently dropped patch here
produces confusing runtime errors much later.

### Why these patches exist

| Patch | Without it |
| --- | --- |
| `0001` CTC prefix scorer honours its input device | Upstream hardcodes cuda-or-cpu, so on Apple Silicon it resolves to **cpu** while the tensors are on the GPU, then fails mid-beam-search with an opaque device mismatch. |
| `0002` demo: MPS device and video loader | Upstream selects only `cuda` or `cpu`, so a Mac silently runs on CPU; and it calls `torchvision.io.read_video`, removed in torchvision 0.28, which crashes before inference. |

## Pinned dependencies you should not casually bump

**`mediapipe==0.10.35`** — an exact pin, not a range. Version 1.0.1 **aborts
fatally** on macOS 26 / Apple Silicon inside its Metal path:

```
F0000 graph_service.h:139] Check failed: service_ Service is unavailable
```

raised from `TensorsToDetectionsCalculator::Open()`. This is a process kill, not
a catchable exception, and **forcing the CPU delegate does not avoid it.** If you
bump it, re-run the smoke test end to end before believing it works.

## If you migrated this Mac from an Intel machine

This one costs hours if you meet it cold, because **no error message points at
the cause**.

Symptom: installing `torch` fails with *"requirements are unsatisfiable"*, or
`platform.mac_ver()` returns `('', ('', '', ''), '')`, or `pip` dies with
`ValueError: invalid literal for int() with base 10: ''`.

Three separate problems tend to arrive together:

1. **Intel Homebrew at `/usr/local` still owns `python3`.** It is an x86_64
   build, and torch ships no x86 macOS wheels above 2.4.1.
2. **`uv` itself may be an x86_64 binary**, in which case every Python it can see
   *or download* is `macos-x86_64`. Reinstall it from arm64 Homebrew:
   `brew install uv` (with `/opt/homebrew/bin/brew`).
3. **Homebrew's arm64 Python may be broken.** Its `pyexpat` links against the
   system `libexpat`, which lacks a symbol the bottle was built against. That
   breaks `plistlib`, which silently empties `platform.mac_ver()`, which breaks
   pip's and uv's platform detection. Diagnose with:

   ```bash
   python3 -c "import platform, plistlib; print(platform.mac_ver())"
   ```

   An empty tuple confirms it.

The reliable fix is a **self-contained Python that bundles its own expat**:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
```

Verify you landed somewhere sane:

```bash
uv run python -c "import platform, torch; print(platform.machine(), torch.backends.mps.is_available())"
# arm64 True
```

`arm64 True` is what you want. `x86_64` or `False` means you are still on the
wrong interpreter, and `tests/test_smoke.py::test_mps_backend_is_present` will
say so.

## Execution devices

`hybrid` — encoder on the GPU, beam search on CPU — is the default and the
fastest measured configuration (RTF 0.099, against 0.128 all-GPU and 0.232
all-CPU, at identical accuracy). See `src/voxlens/devices.py` and VoxLens
issue #7.
