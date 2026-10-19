import json
import subprocess
from importlib.metadata import distributions
from pathlib import Path

from lewam.data.prepare import write_json


class RunLogger:
    def __init__(self, output, cfg, enabled=True):
        self.output, self.run = Path(output), None
        self.enabled = enabled
        if not enabled:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        write_json(
            self.output / "provenance.json",
            {
                "git_commit": commit,
                "git_dirty": dirty,
                "packages": {d.metadata["Name"]: d.version for d in distributions()},
            },
        )
        if cfg["logging"]["mode"] != "disabled":
            import wandb

            options = cfg["logging"]
            self.run = wandb.init(
                project=options["project"],
                entity=options["entity"],
                mode=options["mode"],
                name=options["name"],
                group=options["group"],
                dir=str(self.output),
                config=cfg,
            )
            self.run.define_metric("step")
            self.run.define_metric("*", step_metric="step")

    def log(self, metrics, step):
        if not self.enabled:
            return
        row = {"step": step, **metrics}
        with (self.output / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        if self.run:
            self.run.log(row)

    def video(self, path, key="eval/video"):
        if self.run:
            import wandb

            self.run.log({key: wandb.Video(str(path), format="mp4")})

    def close(self):
        if self.run:
            self.run.finish()
