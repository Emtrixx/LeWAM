# Validation record

Development checks performed on 2026-10-19. These establish implementation and
launch-path behavior; they are not trained-model quality or paper-result claims.

## Automated checks

- 34 pytest tests pass on the Mac CPU. They cover mode masks, past-action hiding,
  live encoder-target gradients, loss weights, Euler direction/determinism,
  encoder freezing, SIGReg collapse detection, BF16-input SIGReg loss/backward,
  fixed-batch learnability, full-size parameter count, causal LeWM behavior,
  episode/action alignment, train-only normalization, distractor determinism,
  foreground preservation, cache identity, planner budgets/bounds/imagined
  contexts, and the optional SWM cost contract.
- The interrupted/resumed CPU training test matches uninterrupted model weights
  and BatchNorm buffers exactly (zero numerical tolerance).
- Two-rank Linux/Gloo training completes with the full four-mode objective.
- The dedicated two-rank SIGReg check matches global-batch reference values and
  gradients. Its flattened gather also avoids PyTorch 2.8 Gloo's noncontiguous
  backward-output issue. The script is `tests/distributed_sigreg.py`.
- Ruff lint and formatting checks pass. Hydra smoke/full/policy/LeWM compositions,
  cluster YAML files, and shell launchers have been checked.
- The AMD64 Docker test stage passes all 34 tests with the locked CUDA-enabled
  dependencies. Real public PushT training, checkpoint loading, simulator rollout,
  and H.264 video encoding also complete inside that image on CPU. The installed
  runtime reports `2.8.0+cu128` and CUDA `12.8`.
- The built wheel contains all 22 Hydra YAML configs. A real Hydra multirun over
  two visual conditions completes with separate output/checkpoint directories.

## Actual data and model execution

- Downloaded the public `lerobot/pusht` revision
  `7628202a2180972f291ba1bc6723834921e72c19` and prepared six real episodes, checking
  metadata lengths and video/row timestamps. The final default config pins this SHA.
- M1 Pro native MPS/FP32: all four modes trained for 12 optimizer updates with
  finite gradients and offline W&B output, then resumed to step 14.
- Clean, Lissajous, and generated-video inputs each complete genuine-data training
  smoke runs. Native-resolution sample comparisons were visually inspected.
- Policy-only and causal forward-only LeWM controls each complete real-data
  training/validation smoke runs.
- The full-size **41,247,434**-parameter, 224px model completes a CPU optimizer step
  and validation at batch 2, with all modes and full 1024-projection SIGReg.
- PushT direct-policy and DS-CEM closed-loop runs produce metrics and playable
  MP4s. All six planner configurations also complete short real-simulator runs.
- Episode-held-out ridge probes and shuffled-target controls execute. Only agent
  position labels are available in the supplied public dataset; block-pose metrics
  are flagged unavailable. The tiny smoke checkpoint is not expected to solve PushT.

Local run artifacts are under `outputs/validation/` and are intentionally ignored
by Git. Commands, configs, and tests are tracked so checks can be rerun elsewhere.
The Docker smoke checkpoint and video are under `outputs/container-smoke/` and
`outputs/container-policy/`. The local `lewam:gpu` tag aliases the tested image,
which includes test dependencies; build the Dockerfile's `runtime` target for a
minimal training image on the GPU server. Intermediate images and download/build
caches were removed after validation to restore local disk space.

## Hardware limits

No H100 or GPU cluster was connected during development. The full global batch
128, CUDA/BF16 execution, CUDA SyncBatchNorm/NCCL, H100 peak memory/throughput,
multi-node networking, and site scheduler integration require the documented
server-side preflight. BF16-input SIGReg and Linux/Gloo tests do not substitute
for those hardware-specific measurements. Full 300-epoch/five-seed training and
paper success-rate reproduction have not been performed.
