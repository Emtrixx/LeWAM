from copy import deepcopy

from lewam.data.windows import ActionNormalizer
from lewam.training.checkpoint import load_checkpoint
from lewam.training.runtime import initialize
from lewam.training.trainer import build_model


def load_evaluation(cfg):
    if not cfg["eval"]["checkpoint"]:
        raise ValueError("Set eval.checkpoint=/path/to/last.pt")
    checkpoint = load_checkpoint(cfg["eval"]["checkpoint"])
    saved = deepcopy(checkpoint["config"])
    saved.update(
        hardware=cfg["hardware"],
        output_dir=cfg["output_dir"],
        eval=cfg["eval"],
        logging=cfg["logging"],
        seed=cfg["seed"],
        resume=None,
        init_from=None,
    )
    # Relocate data without changing its identity, e.g. when moving a checkpoint to a cluster.
    saved["data"]["root"] = cfg["data"]["root"]
    if saved["environment"]["backend"] == "robomimic":
        saved["environment"]["source"] = deepcopy(saved["data"])
    condition = cfg["eval"]["visual_condition"]
    if condition != "checkpoint":
        if condition not in {"clean", "lissajous", "video"}:
            raise ValueError("eval.visual_condition must be checkpoint, clean, lissajous, or video")
        saved["distractor"]["kind"] = condition
    runtime = initialize(saved["hardware"], saved["seed"])
    if runtime.world_size > 1:
        runtime.close()
        raise ValueError("Evaluation is single-process; use separate jobs for evaluation seeds")
    model = build_model(saved).to(runtime.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, ActionNormalizer.from_dict(checkpoint["normalizer"]), saved, checkpoint, runtime
