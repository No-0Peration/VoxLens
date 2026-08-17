# Vendor the upstream recogniser at a pinned revision, with patches

**Status:** accepted — decided and implemented.

VoxLens does not fork the USR 2.0 repository, vendor a frozen copy, or depend on it as a package. It **clones a pinned revision into `vendor/` and applies patch files from `patches/`**, driven by `scripts/vendor.py`. `vendor/` is generated and never committed.

The upstream repository is a research script tree, not a library: it has no package metadata, no release cadence, and does not run on Apple Silicon as published. Something had to change, and the question was where those changes live.

## Consequences

- **Upstream divergence fails loudly.** When the pinned revision moves and a patch no longer applies, `git apply` errors during vendoring rather than silently dropping a fix. A dropped fix here surfaces much later as an incomprehensible runtime error, so failing at fetch time is the whole point.
- **Every local change is reviewable in isolation.** A patch file is a diff with a name and a reason; a fork is an ever-widening delta nobody reads.
- **`vendor/` must never be edited in place.** The next vendor run discards it. This is easy to forget and the failure is silent, so it is stated in `docs/setup.md` and in the script's own error text.
- Setup gains a step. `uv pip install` alone is not enough, which surprises people and is why the README points at the setup doc before anything else.

## Considered options

**Fork the repository.** Rejected: it moves the burden from "re-cut a patch" to "merge a research codebase", and makes the local delta invisible.

**Vendor a frozen copy, no upstream link.** Rejected: it discards provenance. Nobody could later tell what was changed, or why, or what upstream now says.

**Depend on it as a package.** Not available — upstream publishes nothing installable.
