Teacher stage baseline for LeRobot PR #4897

Purpose
Find one material improvement with actual teacher weights and measured time.
The PR source stays at f58f7fb9dbb241304df96e967aea0f5b42ff9a4d.
This experiment changes research files only.

Managed experiment
Project: 7ff15cc9-44cc-4017-8dce-f39efcfd0610
Baseline: bb853371-fbe6-4bd0-946a-e880c0352fc8
Branch: orx/teacher-stage-baseline
Fixed command: bash study/run.sh

Inputs and checks
The bootstrap downloads four pinned teacher files and one pinned video.
SHA-256 checks verify each file before the harness loads it.
The frame pairs use indices 30, 150, 300 and 600, with a future offset of 49.
All pairs stay inside episode zero, which contains 807 frames.
The processor uses float32 pixels, v2 resize with antialias, and uint8 quantization.
The effective FPS uses float32(30/49).

Measurements
The original teacher function supplies the primary synchronized wall time.
Separate runs supply diagnostic stages and a CUDA profiler trace.
Diagnostic stages overlap. Do not add their times.
CUDA event intervals include idle time during the CPU solver.
Inspect profiler kernels to measure GPU work.
Record allocated memory separately from allocator reservations.

Next experiment
Select a candidate after the baseline locates a material cost.
Create a child experiment. Keep this baseline unchanged after its run.
Compare target tensors before time, loss, gradient and full-policy checks.
Keep batch size, pixels, precision, FPS, device and weights equal.
Report the complete teacher time and the complete training step separately.

Authorization
The user authorizes a 10 USD total compute budget for Hugging Face Jobs.
Launch experiments through orx exp run within the recorded timeout reservations.
Do not create a PR or post a speculative performance comment.

Evidence
The wrapper prints complete result JSON with explicit delimiters.
It also uploads results to the existing private jobs-artifacts bucket.
Weights, raw video, package environments and caches stay out of that upload.
The workspace artifact folder is artifacts/lingbot-teacher-profile/.
