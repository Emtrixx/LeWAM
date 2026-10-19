from pathlib import Path

import numpy as np
import torch
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from lewam.data import make_datasets
from lewam.data.prepare import write_json
from lewam.training.logging import RunLogger

from .common import load_evaluation


def fit_probe(train_x, train_y, groups, val_x, val_y, alphas, folds):
    n_splits = min(folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Probe cross-validation needs at least two training episodes")
    estimator = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), Ridge()), transformer=StandardScaler()
    )

    def standardized_score(estimator, features, target):
        error = (estimator.predict(features) - target) / np.maximum(
            estimator.transformer_.scale_, 1e-8
        )
        return -float(np.mean(error**2))

    search = GridSearchCV(
        estimator,
        {"regressor__ridge__alpha": alphas},
        cv=GroupKFold(n_splits),
        scoring=standardized_score,
        n_jobs=1,
    )
    search.fit(train_x, train_y, groups=groups)
    predicted = search.predict(val_x)
    mean, std = train_y.mean(0), np.maximum(train_y.std(0), 1e-8)
    error = (predicted - val_y) / std
    mse = np.mean(error**2, axis=0)
    variance = np.var((val_y - mean) / std, axis=0)
    # Preserve negative R2, unlike the paper's clipped display table.
    r2 = 1 - mse / np.maximum(variance, 1e-8)
    correlations = []
    for p, y in zip(predicted.T, val_y.T, strict=True):
        correlations.append(float(np.corrcoef(p, y)[0, 1]) if min(p.std(), y.std()) > 1e-8 else 0.0)
    return {
        "mse": mse.tolist(),
        "r2": r2.tolist(),
        "pearson": correlations,
        "mean_r2": float(r2.mean()),
        "alpha": search.best_params_["regressor__ridge__alpha"],
    }


@torch.no_grad()
def probe(cfg):
    model, normalizer, saved, checkpoint, runtime = load_evaluation(cfg)
    logger = None
    try:
        train, val = make_datasets(
            saved, normalizer=normalizer, expected_manifest=checkpoint["manifest"]
        )

        def extract(dataset):
            features, states, distractors, groups = [], [], [], []
            for batch in DataLoader(dataset, batch_size=32, shuffle=False):
                with runtime.autocast():
                    latent = model.encode(batch["pixels"][:, -1:].to(runtime.device))[:, 0]
                features.append(latent.float().cpu().numpy())
                states.append(batch["state"][:, -1].numpy())
                distractors.append(batch["distractor_state"][:, -1].numpy())
                groups.append(batch["episode"].numpy())
            return tuple(
                np.concatenate(values) for values in (features, states, distractors, groups)
            )

        x, y, d, groups = extract(train)
        vx, vy, vd, _ = extract(val)
        kwargs = dict(
            train_x=x,
            groups=groups,
            val_x=vx,
            alphas=saved["eval"]["probe_alphas"],
            folds=saved["eval"]["probe_folds"],
        )
        metrics = {
            "state_names": train.manifest["state_names"],
            "distractor_names": train.distractor.state_names,
            "state": fit_probe(train_y=y, val_y=vy, **kwargs),
            "distractor": fit_probe(train_y=d, val_y=vd, **kwargs),
        }
        rng = np.random.default_rng(saved["seed"])
        metrics["shuffled_state_control"] = fit_probe(
            train_y=y[rng.permutation(len(y))], val_y=vy[rng.permutation(len(vy))], **kwargs
        )
        metrics["block_pose_available"] = "block_x" in metrics["state_names"]
        write_json(Path(saved["output_dir"]) / "probes.json", metrics)
        logger = RunLogger(saved["output_dir"], saved)
        logger.log(
            {
                f"probe/{name}/mean_r2": metrics[name]["mean_r2"]
                for name in ("state", "distractor", "shuffled_state_control")
            },
            0,
        )
        return metrics
    finally:
        if logger:
            logger.close()
        runtime.close()
