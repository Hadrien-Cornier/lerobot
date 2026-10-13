Teacher source patch verification for LeRobot PR #4897

Purpose
Find one material improvement with actual teacher weights and measured time.
The original source snapshot stays at f58f7fb9dbb241304df96e967aea0f5b42ff9a4d.
This child changes only the depth teacher constructor and research files.
It stores frozen Linear and Conv2d weights at their CUDA autocast precision.
It converts these weights on the CPU before the GPU transfer.

Managed experiment
Project: 7ff15cc9-44cc-4017-8dce-f39efcfd0610
Experiment: 91c3608c-5a4c-48b9-99dd-2bd20f737122
Parent: 093a3ebc-124c-4c9d-9267-b0c83fa85c9d
Branch: orx/teacher-precast-before-cuda-transfer
Fixed command: bash study/run.sh

Inputs and checks
The bootstrap downloads four pinned teacher files and one pinned video.
SHA-256 checks verify each file before the harness loads it.
The frame pairs use indices 30, 150, 300 and 600, with a future offset of 49.
All pairs stay inside episode zero, which contains 807 frames.
The processor uses float32 pixels, v2 resize with antialias, and uint8 quantization.
The effective FPS uses float32(30/49).

Measurements
The original teacher class supplies the control through its pinned source snapshot.
The changed constructor supplies the candidate.
Check all teacher parameters, buffers and targets before paired time measurements.
Measure constructor memory with only one teacher bundle on the GPU.
Record paired teacher time separately from profiler traces.
Record allocated memory separately from allocator reservations.

Next experiment
Select a candidate after the baseline locates a material cost.
Create a child experiment. Keep this baseline unchanged after its run.
Compare target tensors before time, loss, gradient and full-policy checks.
Keep batch size, pixels, precision, FPS, device and weights equal.
Report the complete teacher time and the complete training step separately.

Authorization
The user authorizes a total compute budget of 10 USD.
Reserve each run in the workspace budget ledger before its launch.
Launch experiments through orx exp run.
Do not create a PR or post a speculative performance comment.

Evidence
The wrapper prints complete result JSON with explicit delimiters.
It also uploads results to the existing private jobs-artifacts bucket.
Weights, raw video, package environments and caches stay out of that upload.
The workspace artifact folder is artifacts/lingbot-teacher-profile/.
