# Hybrid execution: encoder on the GPU, beam search on the CPU

VoxLens runs the recogniser's encoder on Apple Silicon's GPU (MPS) and its decoder and beam search on the CPU, rather than putting the whole model on one device. Measured on an M4 Pro over 100 Clips at beam 1, this runs at **RTF 0.099** against 0.128 all-GPU and 0.232 all-CPU, at identical WER and lower memory.

The split is not a workaround. The encoder is dense 3D convolution — what a GPU is for — while beam search is hundreds of tiny sequential steps, where per-kernel launch overhead on MPS outweighs any throughput gain. Transfer cost between the two is ~0.004 RTF, small enough not to erode the benefit.

## Consequences

- The encoder must stay **separable** from the decoder. It is the streaming-critical component and the natural unit for any later Core ML conversion; fusing them would forfeit both.
- These figures are **throughput, not latency**. ADR-0001 asks for latency in evaluation, and beam search on the CPU is currently a whole-sequence step. A streaming design will have to measure the quantity this ADR does not.

## Considered options

**MLX.** Apple-native and likely better per watt, but every usable checkpoint ships as PyTorch and no Apple-Silicon-native port of this model lineage exists. Choosing it means hand-porting an architecture with nothing to validate against, to speed up a pipeline already running 10× faster than real time.

**Core ML.** The eventual on-device deployment path, deferred rather than dismissed. Converting late is the standard way this decision goes wrong, which is why the encoder stays separable now.

Detail and the full benchmark table: VoxLens issue #7.
