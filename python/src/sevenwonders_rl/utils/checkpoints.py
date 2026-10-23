"""One checkpoint contract for training, evaluation and snapshot inference."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

import torch

from sevenwonders_rl.features.action_feature_schema import FEATURE_NAMES
from sevenwonders_rl.utils.ppo_model import ActorCritic, MODEL_ARCHITECTURE_VERSION
from sevenwonders_rl.utils.seeding import SEED_VERSION

FEATURE_SEMANTICS_VERSION = 3  # shared science/jokers, chains, discounts and effective costs
TRAINING_VERSION = 3          # neural-only self-play, audited opponents and fixed neural evaluation


def schema_hash():
    return hashlib.sha256("\n".join(FEATURE_NAMES).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(repo_root, config):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()
    return {
        "config": config, "git_commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "engine_sha256": file_hash(Path(repo_root) / "build/7Wonders"),
        "protocol_version": 2, "seed_version": SEED_VERSION,
        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
        "training_version": TRAINING_VERSION, "feature_schema_hash": schema_hash(),
        "feature_names": FEATURE_NAMES, "python": platform.python_version(),
        "platform": platform.platform(), "processor": platform.processor(),
        "versions": {n: importlib.metadata.version(n) for n in ("numpy", "torch", "gymnasium")},
        "torch_threads": torch.get_num_threads(),
        "outcome": "strict_vp_win; reward=(agent-best_opponent)/reward_scale",
    }


def save_checkpoint(path, model, optimizer, timestep, config, run_manifest, metrics=None):
    checkpoint = {
        "state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "timestep": timestep,
        "config": {**config, "model_architecture_version": model.architecture_version},
        "feature_schema_version": FEATURE_SEMANTICS_VERSION,
        "meta": {**run_manifest, "model_architecture_version": model.architecture_version,
                 "feature_schema_hash": schema_hash(), "eval_metrics": metrics},
    }
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def load_model(path, device="cpu", allow_legacy=False):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    meta = checkpoint.get("meta", {})
    if not allow_legacy and (meta.get("feature_schema_hash") != schema_hash()
                            or meta.get("feature_semantics_version") != FEATURE_SEMANTICS_VERSION):
        raise ValueError("Checkpoint feature semantics/schema mismatch; start a fresh training run with current features")
    cfg = checkpoint["config"]
    # Older checkpoints have no architecture metadata. Their linear head has
    # different state-dict keys; never reinterpret those weights as the new head.
    weights_version = 1 if "policy_head.weight" in checkpoint["state_dict"] else MODEL_ARCHITECTURE_VERSION
    versions = (cfg.get("model_architecture_version", weights_version),
                meta.get("model_architecture_version", weights_version))
    if any(version != weights_version for version in versions):
        raise ValueError("Checkpoint model architecture metadata does not match its weights")
    if not allow_legacy and weights_version != MODEL_ARCHITECTURE_VERSION:
        raise ValueError("Checkpoint model architecture mismatch; use explicit legacy override for historical inference")
    # Loading weights must not consume the training process's global random stream.
    with torch.random.fork_rng(devices=[]):
        model = ActorCritic(int(cfg["global_dim"]), int(cfg["action_feat_dim"]),
                            hidden=int(cfg.get("hidden", cfg.get("hidden_dim", 128))),
                            architecture_version=weights_version).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
