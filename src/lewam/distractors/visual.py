"""One pure, episode/time-indexed renderer shared by datasets and environments."""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class Distractor:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.kind = self.cfg["kind"]
        if self.kind not in {"clean", "lissajous", "video"}:
            raise ValueError(f"Unknown distractor: {self.kind}")
        self.video = None
        if self.kind == "video" and self.cfg.get("video_path"):
            import av

            with av.open(str(Path(self.cfg["video_path"]).expanduser())) as container:
                self.video = np.stack(
                    [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
                )
            if len(self.video) == 0:
                raise ValueError("Video distractor contains no frames")

    def parameters(self, episode):
        # No environment/action RNG is read or modified here. Stable across workers and runs.
        return np.random.default_rng(np.random.SeedSequence([self.cfg["seed"], int(episode)]))

    def state(self, episode, step):
        if self.kind == "video":
            if self.video is not None:
                offset = int(self.parameters(episode).integers(len(self.video)))
                phase = 2 * math.pi * ((step + offset) % len(self.video)) / len(self.video)
                return np.array([math.sin(phase), math.cos(phase)], dtype=np.float32)
            phase = self.parameters(episode).uniform(0, 2 * math.pi, 3)
            phase = phase + step * np.array([0.09, 0.11, 0.13])
            return np.concatenate((np.sin(phase), np.cos(phase))).astype(np.float32)
        rng = self.parameters(episode)
        phase = rng.uniform(0, 2 * math.pi, 3)
        t = step / 10.0
        return np.array(
            [
                0.5 + 0.38 * math.sin(1.1 * t + phase[0]),
                0.5 + 0.38 * math.sin(1.7 * t + phase[1]),
                0.5 + 0.4 * math.sin(0.7 * t + phase[2]),
            ],
            dtype=np.float32,
        )

    @property
    def state_names(self):
        if self.kind == "video":
            return (
                ["clip_phase_sin", "clip_phase_cos"]
                if self.video is not None
                else [f"wave_{j}_{component}" for component in ("sin", "cos") for j in range(3)]
            )
        prefix = "unrendered_" if self.kind == "clean" else ""
        return [prefix + name for name in ("dot_x", "dot_y", "bar_x")]

    def __call__(self, frame, episode, step):
        """RGB uint8 in/out, on the native canvas before the sole encoder resize."""
        frame = np.asarray(frame, dtype=np.uint8)
        if self.kind == "clean":
            return frame.copy()
        h, w = frame.shape[:2]
        s = self.state(episode, step)
        if self.kind == "lissajous":
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            r = self.cfg["dot_radius"] * w / 96
            x, y = s[0] * w, s[1] * h
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(220, 55, 55))
            bw = self.cfg["bar_width"] * w / 96
            bx = s[2] * (w - bw)
            draw.rectangle((bx, 0, bx + bw, max(2, h // 24)), fill=(60, 60, 60))
            return np.asarray(image).copy()
        if self.video is not None:
            offset = int(self.parameters(episode).integers(len(self.video)))
            background = np.asarray(
                Image.fromarray(self.video[(step + offset) % len(self.video)]).resize(
                    (w, h), Image.Resampling.BILINEAR
                )
            )
        else:
            # Procedural video: a continuous, independently phased moving RGB wave field.
            y, x = np.mgrid[0:h, 0:w].astype(np.float32)
            phase = self.parameters(episode).uniform(0, 2 * math.pi, 3)
            background = np.stack(
                [
                    128
                    + 95 * np.sin(x / w * 7 + y / h * (j + 1) + step * (0.09 + j * 0.02) + phase[j])
                    for j in range(3)
                ],
                -1,
            ).astype(np.uint8)
        # Preserve the agent, block, goal, and walls. This mask is specific to white-table PushT.
        mask = np.all(frame >= 245, axis=-1, keepdims=True)
        alpha = self.cfg["video_strength"]
        mixed = np.clip((1 - alpha) * frame + alpha * background, 0, 255).astype(np.uint8)
        return np.where(mask, mixed, frame)
