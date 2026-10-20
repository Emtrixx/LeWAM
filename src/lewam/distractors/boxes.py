"""Assumed physical nuisance recipe, shared by raw-state replay and online rendering.

These are MuJoCo mocap bodies with noncolliding box geoms, not image overlays.
The paper does not release their placement, sizes or trajectories. This recipe
is an explicitly labelled adaptation and does not qualify paper-equivalent data.
"""

import xml.etree.ElementTree as ET

import numpy as np


class SimBoxes:
    state_names = [f"box_{i}_{axis}" for i in range(2) for axis in "xyz"]

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        if self.cfg["kind"] != "sim_boxes":
            raise ValueError("Robomimic requires the physical sim_boxes condition")

    def state(self, episode, step):
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg["seed"], int(episode)]))
        phase = rng.uniform(0, 2 * np.pi, (2, 3))
        centers = np.asarray(self.cfg["centers"], dtype=np.float64)
        t = step / self.cfg["fps"]
        return (
            (
                centers
                + np.asarray(self.cfg["amplitude"])
                * np.sin(t * np.asarray(self.cfg["frequency"]) + phase)
            )
            .ravel()
            .astype("float32")
        )

    def __call__(self, frame, episode, step):
        # Offline frames already include these geoms. Never draw a second nuisance.
        return np.array(frame, dtype=np.uint8, copy=True)

    def add_to_xml(self, xml):
        root = ET.fromstring(xml)
        world = root.find("worldbody")
        if world is None or root.find(".//body[@name='lewam_box_0']") is not None:
            raise ValueError("Expected an unaugmented MuJoCo model")
        for i, center in enumerate(self.cfg["centers"]):
            body = ET.SubElement(
                world, "body", name=f"lewam_box_{i}", mocap="true", pos=" ".join(map(str, center))
            )
            ET.SubElement(
                body,
                "geom",
                name=f"lewam_box_geom_{i}",
                type="box",
                size=" ".join(map(str, self.cfg["half_size"])),
                rgba="0.5 0.5 0.5 1",
                contype="0",
                conaffinity="0",
                group="1",
            )
        return ET.tostring(root, encoding="unicode")

    def move(self, sim, episode, step):
        for i, position in enumerate(self.state(episode, step).reshape(2, 3)):
            sim.data.set_mocap_pos(f"lewam_box_{i}", position)
        sim.forward()
