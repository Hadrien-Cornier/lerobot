# Reproducing LeRobot PR #4485

Verified September 21, 2026. This is a deterministic integration experiment using
real Gymnasium environments, real LeRobot rollout and real eval_policy. No rollout
or metric function is mocked. The environments have scripted rewards and the policy
outputs zero actions; this is not robot physics or a trained-policy benchmark.

## Exact revisions

- Before: `713a409faedd73bb5597481b8885f17fbee23330` (PR parent)
- After: `b7553d2323d5f0e429ad2be80dabbf2e3cb21eb3` (PR head)
- Gymnasium 1.3.0; PyTorch 2.11.0; CPU; existing Python 3.13 environment.
- Source: https://github.com/huggingface/lerobot/pull/4485

## Observed results

Robot A finishes its first episode after rewards -2 and -1. Robot B continues for
four steps, earning 1, 2, 3, 4 and succeeding. The vector environment uses SAME_STEP
autoreset. A's actual first-episode result is always total -3, maximum -1, failure.

1. Frozen tail, using LeRobot's FreezeAfterEpisodeEnd: A's rollout is [-2,-1,0,0].
   Before: total -3, maximum 0, failure. After: total -3, maximum -1, failure.
2. Negative next episode: A's rollout is [-2,-1,-2,-1].
   Before: total -5, maximum 0, failure. After: total -3, maximum -1, failure.
3. Successful next episode: A's rollout is [-2,-1,100,100].
   Before: total 97, maximum 100, success. After: total -3, maximum -1, failure.

B is unchanged in every case: total 10, maximum 4, success. In case 3 the aggregate
success rate changes from the incorrect 100% to the correct 50%.

## Follow the bug

All three cases have A's cumulative done flags [false,true,true,true]. The first
true is index 1. The old inclusive comparison against index+1 keeps [true,true,true,false].
It accidentally counts step index 2. The fix compares against index itself, keeping
[true,true,false,false], including the terminal transition but nothing afterward.

A correct boundary is necessary but insufficient. Multiplying excluded rewards by
zero leaves zeros in the array, which beat all-negative valid rewards during max.
The PR replaces excluded rewards with negative infinity for max. The report's
boundary_only_max field isolates this: it remains 0 for A in every scenario.

These tail entries are not always artificial zeros inserted by a padding routine.
The actual rollout keeps stepping vector slots. Depending on wrappers and autoreset,
the entries may be frozen zeros, reset-step zeros, or transitions from later episodes.
The +100 case demonstrates the unwrapped SAME_STEP path, not a claim that every
simulator/configuration leaks later successes.

## Run locally

From /Users/HCornier/Documents/Personal/LeRobot, use the existing dependency environment
and choose the source checkout explicitly. The script records the actual imported
source path and Git revision to guard against accidentally testing the wrong tree.

```sh
UV_CACHE_DIR=/private/tmp/lerobot-uv-cache \
PYTHONPATH="$PWD/lerobot-pr-4485-before/src" \
uv run --project lerobot --no-sync python eval-4485-evidence/reproduce.py \
  --output eval-4485-evidence/before.json --expect bug

UV_CACHE_DIR=/private/tmp/lerobot-uv-cache \
PYTHONPATH="$PWD/lerobot-pr-4485/src" \
uv run --project lerobot --no-sync python eval-4485-evidence/reproduce.py \
  --output eval-4485-evidence/after.json --expect fixed

UV_CACHE_DIR=/private/tmp/lerobot-uv-cache \
PYTHONPATH="$PWD/lerobot-pr-4485/src" \
uv run --project lerobot --no-sync pytest \
  lerobot-pr-4485/tests/scripts/test_lerobot_eval.py \
  lerobot-pr-4485/tests/envs/test_freeze_after_episode_end.py -q
```

Focused tests: 9 passed. This is additional verification after the real-loop reproduction;
the PR's own eval_policy regression test mocks rollout. No full suite was run.
Local imports emitted duplicate AVFoundation class warnings from OpenCV/PyAV; the
experiment and tests completed successfully without using either video backend.

The teaching policy subclasses PreTrainedPolicy and uses ACTConfig solely to satisfy
the real configuration contract. It does not instantiate ACT. Identity processors
are intentional no-ops because the scalar observation/action needs no transformation.

## Learning question

If an episode ends at index 1, why is `index <= 1` already inclusive of its final reward?
What different boundary would you use for an exclusive Python slice?
