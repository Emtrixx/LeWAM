# Official PH200 adapters

Select `experiment=lift`, `can`, `square`, `tool_hang` or `transport`. Each selects
the data source, environment, physical nuisance and primitive action dimension;
the paper architecture, batch 128, 300 epochs and optimizer schedule remain the
defaults. Use `hardware=h100` explicitly for the existing BF16 CUDA profile.
The default hardware remains the existing project default; it is not changed.

| Experiment | Simulator | Native camera | Native canvas | Actions | Horizon |
| --- | --- | --- | --- | --- | --- |
| lift | Lift | agentview | 84 × 84 | 7 | 100 |
| can | PickPlaceCan | agentview | 84 × 84 | 7 | 400 |
| square | NutAssemblySquare | agentview | 84 × 84 | 7 | 400 |
| tool_hang | ToolHang | sideview | 168 × 168 | 7 | 700 |
| transport | TwoArmTransport | shouldercamera0, shouldercamera1 | 512 × 256, vertical | 14 | 700 |

Every camera name and pose was checked against actual PH XML and the simulator.
Offline caches keep native uint8 pixels. `WindowDataset.image` resizes once to
the encoder size. Online rendering composes the same native images, then resizes
once. No depth, proprioception or simulator state enters model predictions.

The catalog pins public `robomimic/robomimic_datasets` at revision
`74fa018461f479cd9fd15b924a16103012096203`. Put the five files under
`LEWAM_ROBOMIMIC_DIR/v1.5/TASK/ph/demo_v15.hdf5`. Each is SHA-256 checked before
any simulator use, including prepared-cache reads. Raw actions/states retain
their original precision in the cache. Training windows/normalization follow
the original reader's precision and sample order. The 90/10 split is at the
episode level, with split seed 0; no validation action fits the normalizer.

The cache identity includes render configuration, selected episodes/frame cap,
source hashes for the renderer/preparer/catalog and simulator/image dependency
versions. Checkpoint dataset consumers check these fields, raw hash, cameras and
episode records. Model, optimizer, scheduler, cursor and per-rank RNG fields are
unchanged. Evaluation relocates the raw root for both data and environment.

## Isolated CPU runtime

Use Python **3.12.12**, robosuite **1.5.0**, MuJoCo **3.2.3**, robomimic **0.3.0**,
h5py **3.14.0** and CPU torch **2.8.0+cpu**. The complete tested package pins are
in `requirements/robomimic-cpu.lock`. The preparation handoff also supplies a
hash-expanded lock. This runtime is independent of the original `uv.lock`;
the existing PushT lock, numerical sources and configurations remain unchanged.

From a new isolated LeWAM source directory, with an already available 3.12.12
interpreter, use a workspace-local disk TMPDIR:

```sh
export TMPDIR="$PWD/tmp"
export UV_CACHE_DIR="$TMPDIR/uv-cache" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR"
uv venv --python /path/to/python3.12 .venv-robomimic
uv pip install --no-cache --python .venv-robomimic/bin/python cmake==3.31.6
export PATH="$PWD/.venv-robomimic/bin:$PATH"
uv pip sync --no-cache --python .venv-robomimic/bin/python \
  --index-strategy unsafe-best-match -r requirements/robomimic-cpu.lock
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMBA_CACHE_DIR="$TMPDIR/numba"
```

The CMake bootstrap is needed for egl-probe 1.0.2's older build policy. Actual
simulation calls robosuite directly, using the official robomimic model/state
restore sequence; it does not run egl-probe's GPU selection. The existing
project's `opencv-python` exclusion is intentional: headless 4.11.0.86 supplies
cv2. `uv pip check` reports the two textual GUI-provider requirements from
gym-pusht and robosuite; there is one actual cv2 provider, verified in production.

For this machine's Mesa software renderer:

```sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export MUJOCO_EGL_DEVICE_ID=0
```

The CPU proof must report llvmpipe/softpipe, never a CUDA/GPU EGL renderer.
Paths and Mesa availability need checking on another host. No OSMesa library
or Docker is required by the validated local path. Mesa's software-device warning
is retained; actual GL_RENDERER was llvmpipe. Do not unset the empty CUDA mask
to work around robosuite's device parser.

## Bounded real-data checks

```sh
export LEWAM_ROBOMIMIC_DIR=/absolute/raw
export LEWAM_PROOF_DIR="$PWD/proofs"
timeout 300 .venv-robomimic/bin/python -m pytest tests -q \
  --basetemp="$TMPDIR/pytest"
timeout 300 .venv-robomimic/bin/python -c 'from lewam.cli import prepare; prepare()' \
  experiment=lift hardware=cpu data.max_episodes=2 data.max_frames=20
```

The seven integration cases render only two 20-frame prefixes per task and a
few diagnostic views. Two cases test tiny-model exact CPU continuation, all
checkpoint state, pixel privacy and actual online rollout for 7/14 actions.
Without raw data these cases skip; the existing 34 tests remain independent.

For production preparation use `data.max_episodes=null data.max_frames=null`.
This renders all 200 episodes on the preparation host and is intentionally not
done locally. A completed episode receipt is written only after its three arrays
are saved. Repeating the same bounded command reuses complete episodes and
rerenders an unfinished episode. The aggregate manifest appears only when all
requested episodes are complete. Never train on a partial proof cache and call
it PH200. Retain partial/error evidence on failure and stop if bounded attempts
make no progress. The handoff provides finite storage and whole-phase bounds.

For a differently located raw download, the pinned runtime CLI uses:

```sh
hf download robomimic/robomimic_datasets --repo-type dataset \
  --revision 74fa018461f479cd9fd15b924a16103012096203 \
  --include 'v1.5/*/ph/demo_v15.hdf5' --local-dir /absolute/raw
```

Use the catalog's five exact paths rather than broadening the download if the
upstream tree gains more matches. Rehash locally; Hub metadata alone is not the
raw-file identity check. Downloading or CPU qualification never authorizes GPU
reservation, campaign activation or scientific completion.
