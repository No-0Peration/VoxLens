# Clips first, streaming as the target

**Status:** accepted — decided and implemented.

VoxLens exists to read lips from a live Stream in real time, but the first implementations will process bounded Clips. Clips reach a working baseline faster and make published lip-reading benchmarks directly usable, so they are a development vehicle — not a second supported product mode.

## Consequences

- Conveniences a Clip offers and a Stream cannot — full-sequence bidirectional context, unbounded lookahead, knowing where the input ends before emitting its start — are borrowed, not free. A model that depends on them is a feasibility probe, not a step toward the product.
- Latency belongs in evaluation from the first baseline. A clip-only accuracy number can look excellent while saying nothing about whether the approach can stream at all.

## Considered options

**Streaming first.** Rejected: it front-loads bounded-latency and partial-revision problems before there is any evidence the core recognition works, and it gives up direct comparison against published clip-based results.
