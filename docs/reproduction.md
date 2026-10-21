# Reproduction contract and assumptions

This is an **unofficial, independently maintained implementation**. The original
LeWAM research is by Shashank Hegde, Alexander Popov, Elie Aljalbout and Nikolai
Smolyanskiy; see the [paper credit and citation](../README.md#original-paper-and-citation).

The target is [arXiv:2610.12407v1](https://arxiv.org/abs/2610.12407v1), the NVIDIA
paper by Hegde et al., included in this directory. There are other unrelated
papers named LeWAM. This code targets the four-mode JEPA/flow model in this PDF.

## Specified behavior

| Paper reference | Implementation |
| --- | --- |
| Section 3; Figures 1-2 | Four frame positions, three intervening action chunks; shared encoder and backbone |
| Section 3 modes | Forward masks the last latent; backward masks the first; inverse sees all frames and a noisy final action; policy masks the future and hides past actions |
| Section 3 modes | Every active mode runs on every batch, encoding all four frames once |
| Section 3 encoder; Appendix C | Scratch ViT width 192, depth 12, heads 3, patch 14; normalized CLS; BN projector to latent width 384 |
| Section 3 targets | Live encoder outputs, no stop-gradient and no target EMA |
| Table 5 | Forward weight 2; other mode weights 1; SIGReg weight .36, 17 knots, 1024 projections |
| Section 3 action generation | `(1-sigma)*action + sigma*noise`, target `noise-action`; eight Euler updates from sigma 1 to 0 |
| Sections 4.1-4.4 | Final imagined-latent L2 cost; each noise chunk is decoded at the current imagined context |
| Section 4.4 | Project each chunk's proposals onto the radius-sqrt(chunk_dim) sphere, not the entire multi-chunk plan |
| Algorithm 1; Appendix D | CEM/MPPI/CMA-ES; shared 64x3 candidate budget, horizon 5 chunks, CEM 8 elites |
| Section 5.1 | 206 human PushT demonstrations, 10Hz absolute target positions, 5-step chunks, pixels only |
| Appendix B | Random context length 1-3; newest frame retained; no missing-context mask tokens |
| Appendix D | Episode-held-out ridge probes, grouped cross-validation, shuffled-target control |

Policy loss is action-flow MSE plus latent MSE. The policy-only reference removes
the latter and SIGReg. The forward-only baseline uses a causal action-conditioned
AdaLN-zero predictor; it is distinct from merely disabling modes in LeWAM.

## Explicit interpretations

1. **Token count:** Section 3 says three frame latents in one sentence, but the
   mode descriptions and both figures show four positions including the target.
   We use four, interleaved `z0,a0,z1,a1,z2,a2,z3`.
2. **Attention width:** the residual width is 512, with 16 heads of dimension 64.
   QKV therefore projects to a 1024-dimensional attention interior. Standard
   `MultiheadAttention(embed_dim=512,num_heads=16)` would instead give 32-D heads.
3. **Projector:** the paper specifies hidden width and BatchNorm but not all layer
   details. We use `Linear(192,2048) -> BN -> GELU -> Linear(2048,384)`, consistent
   with the released LeWM MLP. The resulting LeWAM has **41,247,434** parameters,
   rather than an artificial adjustment to exactly 41.0M.
4. **Encoder:** DINOv2-style pre-LN GELU blocks, qkv bias, LayerScale 1.0, learnable
   CLS/position embeddings, ImageNet input normalization, no pretrained weights,
   no register tokens, no pixel reconstruction. Full-size resolution is fixed.
5. **Unspecified transformer details:** pre-LN, learned frame/modality embeddings,
   sinusoidal sigma embedding plus MLP, linear output heads, zero dropout. Sigma
   is zero for clean chunks and sampled only for the target noisy action.
6. **Schedule:** Table 5 enumerates robomimic settings, not every PushT setting.
   We apply its AdamW lr 5e-5, weight decay .001, batch 128, BF16, clip 1.0, and
   300 epochs to PushT. Linear warmup lasts 5% of scheduled optimizer steps,
   followed by cosine decay to zero. AdamW otherwise uses PyTorch defaults.
7. **Context truncation:** one length is sampled per batch for efficient dense
   attention; the oldest frames/actions are removed. Encoded frame statistics
   still use all four frames. Incomplete windows at episode boundaries are
   discarded; online control starts with one real reset observation and grows
   its history. No future frame or invented past action is input to the policy.
8. **Planning:** each decision starts from the prior, without warm starts. CEM
   and CMA-ES return the best evaluated candidate across iterations; MPPI returns
   its weighted mean. MPPI adapts diagonal spread; temperature 1 and minimum
   standard deviation .05 are explicit defaults. CMA-ES uses positive rank-mu
   covariance and cumulative step-size adaptation on CPU for MPS portability.
   The paper does not enumerate these optimizer constants.
9. **Goals:** the default bank uses five training demonstrations (configurable to
   one for the paper's 'given demonstration'). Retrieve the closest current
   latent, then the frame 25 primitive steps later, clamped to that episode's end.
   No held-out demonstration enters the training split or normalizer.
10. **Action constraints:** inverse z-scoring and environment bounds are applied
    before execution. MPC also bounds decoded actions before model scoring, so
    the forward model scores the same action that can actually be executed.
11. **PushT evaluation:** use gym-pusht's 95% block-coverage success and 300-step
    horizon. The PDF does not specify a separate PushT horizon or exact success
    threshold. Log coverage and episode length alongside success.
12. **Distractors:** exact paper phases, amplitudes, colors, and bar dimensions
    are not specified. Our seeded renderer documents these in code/config. The
    procedural video comparison is an extension, not a paper reproduction claim.
    Dot/bar time is in 10Hz PushT steps. Video preserves nonwhite foreground pixels.
13. **Probes:** negative R2 is retained instead of clipped to zero; targets are
    standardized using the training fold. Public PushT's state has only agent
    xy. We do not invent block-pose ground truth or substitute image estimates.
14. **LeWM control:** follows the released causal, action-conditioned AdaLN
    predictor pattern with shared encoder/projector and its own prediction
    projector. It is a documented independent reimplementation, not byte-for-byte
    upstream parity or a claim of exact parameter matching for every baseline.

Changing one of these choices is a config/code experiment and must be named in
reports. Mac smoke runs deliberately change scale, data size, precision, SIGReg
projection count, and planner budget; they establish plumbing, not paper metrics.

## Robomimic extension: explicit fidelity limits

The five adapters use the official PH200 raw v1.5 HDF5 files pinned in
`src/lewam/data/robomimic-catalog.json`. These contain simulator states, actions
and episode XML, not the paper's augmented pixels. See [the executable contract
and runtime recipe](robomimic.md). Every frame is rendered from its corresponding
raw state, with the original action and episode order retained. Transport's two
native shoulder images are stacked vertically before the sole encoder resize.
Vertical stacking is an interpretation of the paper's unspecified stacking axis.

Two grey, noncolliding MuJoCo mocap boxes are rendered by the same recipe offline
and online. Their sizes, placement, sinusoidal motion, phase seeds and Transport
placement are explicit assumptions in `distractor/sim_boxes.yaml` and its task
override. Their relationship to each task object's size is not paper-validated.
They are physical scene geometry, with occlusion and perspective, rather than
2D overlays. These data qualify as **an adapted physical-box condition**, never
paper-equivalent augmented data. No proprietary augmentation was obtained.

The native success predicate is robosuite's `_check_success()`. Lift uses the
paper's 100-step horizon despite the official registry's longer horizon. These
adapters use 20 Hz metadata and seven OSC pose/gripper actions per arm; Transport
has fourteen. The optimization recipe, episode split and train-only normalizer
are unchanged. Simulator state is retained for diagnostics only; model inputs
remain pixels and actions. Raw simulator coordinates are not relabelled as the
paper's semantic end-effector/object probe targets.

The bounded CPU checks show exact direct-state replay/cache alignment, online
reset/step, successful recorded final states and matching offline/online camera
poses for every task. They do not establish exact integrated demonstration
replay. The inspected Can first-step raw-state discrepancy includes qpos about
0.030 and qvel about 0.784 on unused Milk geometry; the Can joint also differs
(qpos about 0.000486, qvel about 0.233). The physical-box and undecorated native
simulators agree on that same action to numerical precision. The cause of the
recorded-versus-integrated mismatch is unresolved; paper-equivalent simulator
qualification remains blocked. The other tasks have smaller nonzero integrated
raw-state discrepancies recorded in the preparation evidence. This does not
change the direct raw-state pixel replay into an integrated-action replay claim.

## Framework decision

[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) provides
useful planning/environment/data APIs, but its `scripts/train/lewm.py` trains LeWM,
not this four-mode model, and introduces Lightning/stable-pretraining. Its default
PushT uses relative actions and state-distance success, unlike LeRobot PushT.
The core therefore remains plain PyTorch. `integrations/stable_worldmodel.py`
provides an independently tested `get_cost`/`criterion` adapter for the documented
Costable protocol, accepting normalized action chunks and frozen executed history.
Full SWM package/solver interoperability is optional and not a required dependency.

Architecture references: [DINOv2](https://github.com/facebookresearch/dinov2),
[LeWM components](https://github.com/galilai-group/stable-worldmodel/tree/main/stable_worldmodel/wm/lewm),
[SIGReg reference](https://github.com/galilai-group/stable-worldmodel/blob/main/stable_worldmodel/wm/loss.py),
[SWM planning contract](https://galilai-group.github.io/stable-worldmodel/api/planning/).
