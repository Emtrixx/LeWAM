# Code and data flow

Entry points in `lewam/cli.py` compose packaged Hydra groups and call ordinary
Python functions. The training loop is independent of Hydra and W&B: tests pass
plain dictionaries and disable logging without changing model behavior.

## Shapes

For global batch B, primitive action dimension D, chunk size F=5, context K=3:

| Value | Shape | Meaning |
| --- | --- | --- |
| dataset pixels | B x (K+1) x 3 x H x W | Native frames at offsets 0,5,10,15, resized once |
| dataset actions | B x K x (F*D) | Every primitive action between the selected frames |
| encoder latents | B x (K+1) x Z | One projected CLS vector per frame |
| token sequence | B x (2*K+1) x width | Alternating frame/action tokens |
| flow velocity | B x (F*D) | Prediction for the last action chunk |
| planner proposals | S x horizon x (F*D) | Noise or normalized action chunks |
| online context | 1 x k x Z | Last 1-3 real chunk-boundary observations |
| online action history | 1 x (k-1) x (F*D) | Executed, normalized chunks connecting context frames |

The model consumes pixels and action targets only. States and distractor labels
travel in the dataset batch for probes but are never read by the model.

## Dataset preparation

`lewam-prepare` reads LeRobot v3's episode metadata, Parquet rows, and video offsets.
It downloads a pinned public Hub revision with anonymous access by default. A
local LeRobot root with `meta/info.json` is also accepted. Set
`data.authenticated=true` only for a dataset that needs your Hub credentials.

Each complete prepared episode consists of uint8 native-resolution pixels,
float32 actions, and optional state arrays in `.npy` format. Training opens these
through memory maps. No entire 224px dataset is materialized in GPU or RAM.
Video timestamps are checked within .51 frame periods, and lengths must match
the episode records. A file lock prevents concurrent ranks from preparing the
same cache. The manifest is written only after all selected episodes finish.

Cache identity includes requested revision, camera, and selection. To intentionally
change local source data, remove its prepared cache or use a new root/revision.
Manifests and checkpoint data identities are compared on resume/evaluation.

The fixed split permutes episode IDs with `data.split_seed`, then applies the
train fraction. Action statistics have separate entries for every primitive
offset within a chunk and are fitted to training episodes only.

## Adding an environment

`environments/factory.py` defines PixelEnvironment: reset -> RGB, step -> RGB/done/info,
action bounds, and close. Register a new factory with `register_environment`.
The trainer does not instantiate environments; offline data and online simulation
can evolve independently. Metrics and primitive-action semantics belong in the
adapter. Existing generic Gymnasium registration works for environments whose
Box actions and rendered RGB satisfy this contract.

For OGBench or CarRacing, add a matching offline-data adapter/preparer, an
environment config with correct FPS/action bounds, and a target-image/goal
protocol appropriate to the task. New action dimension and chunk size are model
config values. Discrete ViZDoom/MiniWorld actions would additionally require an
explicit continuous encoding/decoder or a new action head; continuous flow
matching does not automatically become a categorical policy.

## Adding a model or objective

The public model surface is `encode`, `predict_next`, and `sample_actions`, with
`has_policy`/`has_dynamics` capability checks. `forward` is the joint training
objective. `tokens` makes masking auditable independently from attention. A new
encoder should preserve B,T,Z and a new planner should preserve executed history.

## Reproducibility and checkpoints

Every run writes resolved config, package/Git provenance, dataset manifest, episode
split, normalizer, metrics, and `last.pt`. A checkpoint includes model and BN
buffers, optimizer, scheduler, epoch/batch cursor, and RNG states per rank.
Validation restores the training RNG afterward. Dataset rendering is stateless,
so skipping already-consumed batches on resume does not draw new augmentations.

Exact resume requires the same architecture, world size, split, batch, schedule,
objective, and distractor. `train.max_steps` may be raised to continue a deliberately
capped smoke run without changing its epoch-based schedule. For a new experiment
use a fresh run; `init_from` initializes only the encoder and `freeze_encoder`
also freezes its BatchNorm statistics.

W&B mode does not control whether local metrics exist. Docker keeps data and
run outputs on mounted volumes; credentials remain environment variables.

Use `uv sync --frozen` for installation. Its explicit OpenCV dependency replacement
selects the headless cv2 provider instead of gym-pusht's GUI wheel (macOS wheels
can still include Cocoa video components). Do not install
both OpenCV distributions into the same environment or run an unrelated pip
resolver over the locked environment; they write the same Python module.
