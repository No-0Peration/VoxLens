# The CLI is the seam, and the evaluation harness goes through it

**Status:** accepted — decided and implemented.

VoxLens is tested at **two seams**: the CLI process boundary, and one pure function that turns a per-Frame detection mask into Occlusion spans. The evaluation harness drives the **CLI**, not internal Python APIs.

Most projects would reach for unit tests per module. That is rejected here deliberately.

## Consequences

- **Tests exercise the shipped path.** A regression in argument handling, output shape, or exit codes cannot hide behind a parallel code path that only tests use.
- **The measured numbers are the product's numbers.** When the harness reports WER, it is reporting what a user invoking the CLI would get — not what an internal function achieves under conditions the CLI never creates.
- **The output contract becomes load-bearing.** `--json` is not a convenience flag; it is the interface the harness depends on, so changing its shape breaks evaluation and must be treated as a breaking change.
- **Tests are slower and need fixtures.** Each CLI test loads a ~4 GB checkpoint. Tests requiring it skip with an actionable message when it is absent, so the suite stays runnable without it.
- **Module interfaces stay free to change.** With few seams, internal refactoring does not drag a test rewrite behind it.

The second seam exists because Occlusion span logic has real edge cases — runs at the exact threshold, runs touching either end of a Clip, adjacent runs merging — that are painful to provoke through a fixture video and absurd to verify through a 4 GB model. It is pure, so it costs microseconds.

## Considered options

**Unit tests per module.** Rejected: it multiplies interfaces that must stay stable, and tests the parts while leaving the assembled whole unverified — which is exactly where this pipeline's defects have shown up.

**A single seam, CLI only.** Tempting for purity, but every Occlusion edge case would then need a fixture video that actually produces that pattern. The second seam buys disproportionate coverage for one small pure function.
