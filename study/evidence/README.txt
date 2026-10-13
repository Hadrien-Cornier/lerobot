LingBot-VLA 2.0 teacher profile

Purpose
Reduce teacher cost in LeRobot PR #4897 without a new pull request.
Original PR head: f58f7fb9dbb241304df96e967aea0f5b42ff9a4d.
Source patch: 8d2babb0e148d72396f8f0f43422a935cd80d720.

Selected change
Store only frozen depth Linear and Conv2d weights at their CUDA operation precision.
MoGe uses FP16. MoRGBD uses BF16. Convert on CPU before transfer to CUDA.
Keep normalization, positions, geometry and video code at their original precision.
The CPU path retains its original dtypes.

Actual source verification on L4
Four real OMX frame pairs cover current indices 30, 150, 300, 600 and future offset 49.
Each frame uses the actual processor resize and quantization contract.
All teacher files, dataset revision and source files have pinned SHA-256 checks.
The verification compares the actual changed constructor against the original class.
All 1,233 parameters and 6 buffers match their expected values, dtypes and strides.
All 25 target comparisons match exactly in the same process.
The study randomizes 24 paired target calls after 4 warmup calls per batch size.
The two teacher bundles stay on GPU for paired time measurements.
The separate memory phase retains only the active bundle on GPU.

Metric                              Original       Candidate
Constructor CUDA peak, GiB           2.0924         1.3479
Warmed target CUDA peak B1, GiB       2.9410         1.4333
Warmed target CUDA peak B4, GiB       3.1657         1.6439
Median complete targets B1, ms      83.0963        72.6545
Median complete targets B4, ms     237.9557       229.5506

These values measure teacher CUDA allocation and teacher time.
They do not measure host memory, initialization latency, or full-training lifetime peak.
See pretransfer/results.json and paired-summary.json.

Native training on H200
The native LeRobot CLI loads 6,375,906,359 trainable parameters from the real checkpoint.
The actual dataset, saved processors, forward, backward, clipping and AdamW run unchanged.
The candidate adapter uses the same selective precast after the original constructor.
This run precedes the CPU-before-transfer refinement.
Four fresh processes cover one diagnostic and 12 updates per variant.
After 3 warmup updates, 9 measured updates give these medians:
469.490 ms original,429.601 ms precast.
The update includes forward, backward, clipping, AdamW, scheduler and scalar logs.
It excludes data loading and preprocessing.
The maximum allocated memory across these measured updates is 59.667 and58.915 GiB.
All captured tensors and losses are finite.
Inputs and initial student parameters match exactly; all 3 video targets match exactly.
Two depth target hashes, gradients, updated parameters and AdamW states differ.
Do not interpret hashes as error magnitudes or claim exact native equivalence.
Native training enables cuDNN benchmarking; the L4 teacher study does not.
Fresh-process baseline repeats and benchmark-only controls remain pending.
See full-native/results.json and its four phase result files.

Other candidates
Joint depth batching changes targets; it is not selected.
A contiguous position layout preserves targets in the teacher study but is secondary.
See comparison/results.json.

Reproduction
Use the source patch branch for the L4 source study: bash study/run.sh.
Use commit 1a0d4aafde32e9d3ebad7588d02e8492b60490d3 for the native matrix.
Each GPU run uses a committed orx experiment and a pinned CUDA container.
The result configuration records exact assets, environment, inputs and timings.
The local workspace budget ledger contains provider duration and estimated GPU cost.
AI assistance: Codex prepares the patch, harness and evidence.
