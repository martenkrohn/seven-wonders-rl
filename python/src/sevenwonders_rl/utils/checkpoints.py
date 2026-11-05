"""One checkpoint contract for training, evaluation and snapshot inference."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import torch

from sevenwonders_rl.features.action_feature_schema import FEATURE_NAMES
from sevenwonders_rl.utils.ppo_model import (ActorCritic, MultiHeadActorCritic,
    MemoryActorCritic, MemoryMultiHeadActorCritic, MODEL_ARCHITECTURE_VERSION,
    MULTIHEAD_ARCHITECTURE_VERSION, MEMORY_ARCHITECTURE_VERSION, MEMORY_MULTIHEAD_ARCHITECTURE_VERSION)
from sevenwonders_rl.utils.seeding import SEED_VERSION
from sevenwonders_rl.rewards import RewardSpec

FEATURE_SEMANTICS_VERSION = 3  # shared science/jokers, chains, discounts and effective costs
TRAINING_VERSION = 6          # optional seat-local observation history shared by actor and critic


def schema_hash():
    return hashlib.sha256("\n".join(FEATURE_NAMES).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(repo_root, config):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()
    engine_root = Path(repo_root) / "engine"
    engine_contract = json.loads((engine_root / "engine-contract.json").read_text())
    engine_commit = subprocess.check_output(
        ["git", "-C", str(engine_root), "rev-parse", "HEAD"], text=True).strip()
    engine_dirty = bool(subprocess.check_output(
        ["git", "-C", str(engine_root), "status", "--porcelain"], text=True).strip())
    return {
        "config": config, "git_commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "engine_sha256": file_hash(Path(repo_root) / "build/7Wonders"),
        "engine_source_commit": engine_commit, "engine_source_dirty": engine_dirty,
        "engine_ruleset": engine_contract["ruleset"], "engine_release": engine_contract["release"],
        "protocol_version": 2, "seed_version": SEED_VERSION,
        "model_architecture_version": config.get('model_architecture_version',
            (MEMORY_MULTIHEAD_ARCHITECTURE_VERSION if config.get('num_heads', 1) > 1 else MEMORY_ARCHITECTURE_VERSION)
            if config.get('observation_memory', False) else
            MULTIHEAD_ARCHITECTURE_VERSION if config.get('num_heads', 1) > 1 else MODEL_ARCHITECTURE_VERSION),
        "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
        "training_version": TRAINING_VERSION, "feature_schema_hash": schema_hash(),
        "public_state_schema_hash": _public_hash(config.get('public_information', False)),
        "observation_memory_schema_hash": _memory_hash(config.get('observation_memory', False)),
        "feature_names": FEATURE_NAMES, "python": platform.python_version(),
        "platform": platform.platform(), "processor": platform.processor(),
        "versions": {n: importlib.metadata.version(n) for n in ("numpy", "torch", "gymnasium")},
        "torch_threads": torch.get_num_threads(),
        "outcome": "strict_vp_win; evaluation uses unshaped VP; training objective recorded in reward_spec",
        'reward_spec': RewardSpec(config.get('reward_profile', 'standard'),
                                  config.get('specialist_weight', 1.0),
                                  config.get('reward_scale', 1.0)).describe(),
    }


def save_checkpoint(path, model, optimizer, timestep, config, run_manifest, metrics=None, *, runtime_state=None):
    checkpoint = {
        "state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "timestep": timestep,
        "runtime_state": runtime_state,
        "config": {**config, "model_architecture_version": model.architecture_version,
                   "num_heads": model.num_heads,
                   "prior_strength": getattr(model, 'prior_strength', 0.0),
                   "prior_hidden": getattr(model, 'prior_hidden', 32),
                   "public_information": getattr(model, 'public_information', False),
                   "observation_memory": getattr(model, 'memory_enabled', False),
                   "prior_global_dim": getattr(model, 'prior_global_dim', None),
                   "prior_action_dim": getattr(model, 'prior_action_dim', None)},
        "feature_schema_version": FEATURE_SEMANTICS_VERSION,
        "meta": {**run_manifest, "model_architecture_version": model.architecture_version,
                 "feature_schema_hash": schema_hash(), "eval_metrics": metrics,
                 "public_state_schema_hash": _public_hash(getattr(model, 'public_information', False)),
                 "observation_memory_schema_hash": _memory_hash(getattr(model, 'memory_enabled', False))},
    }
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def _public_hash(enabled):
    if enabled:
        from sevenwonders_rl.features.public_state import public_schema_hash
        return public_schema_hash()
    return None


def _memory_hash(enabled):
    if enabled:
        from sevenwonders_rl.features.observation_history import memory_schema_hash
        return memory_schema_hash()
    return None


def load_model(path, device="cpu", allow_legacy=False):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    meta = checkpoint.get("meta", {})
    if not allow_legacy and (meta.get("feature_schema_hash") != schema_hash()
                            or meta.get("feature_semantics_version") != FEATURE_SEMANTICS_VERSION):
        raise ValueError("Checkpoint feature semantics/schema mismatch; start a fresh training run with current features")
    cfg = checkpoint["config"]
    public = cfg.get('public_information', False)
    memory = cfg.get('observation_memory', False)
    base_global_dim = cfg['global_dim']
    if memory:
        from sevenwonders_rl.features.observation_history import MEMORY_DIM
        if not public or meta.get('observation_memory_schema_hash') != _memory_hash(True):
            raise ValueError('Checkpoint observation-memory schema mismatch')
        base_global_dim -= MEMORY_DIM
    if public:
        from sevenwonders_rl.features.public_state import PUBLIC_GLOBAL_DIM, PUBLIC_ACTION_DIM
        if (meta.get('public_state_schema_hash') != _public_hash(True) or
                (base_global_dim, cfg['action_feat_dim']) != (102 + PUBLIC_GLOBAL_DIM, 99 + PUBLIC_ACTION_DIM)):
            raise ValueError('Checkpoint public observation schema/dimensions mismatch')
    # Older checkpoints have no architecture metadata. Their linear head has
    # different state-dict keys; never reinterpret those weights as the new head.
    has_memory_weights = any(key.startswith('information_encoder.') for key in checkpoint['state_dict'])
    if has_memory_weights != memory:
        raise ValueError('Checkpoint memory metadata does not match its weights')
    weights_version = (MULTIHEAD_ARCHITECTURE_VERSION if any(
        key.startswith('policy_heads.') for key in checkpoint['state_dict']) else
        1 if "policy_head.weight" in checkpoint["state_dict"] else MODEL_ARCHITECTURE_VERSION)
    if memory:
        weights_version = (MEMORY_MULTIHEAD_ARCHITECTURE_VERSION if weights_version == MULTIHEAD_ARCHITECTURE_VERSION
                           else MEMORY_ARCHITECTURE_VERSION)
    versions = (cfg.get("model_architecture_version", weights_version),
                meta.get("model_architecture_version", weights_version))
    if any(version != weights_version for version in versions):
        raise ValueError("Checkpoint model architecture metadata does not match its weights")
    if not allow_legacy and weights_version not in (MODEL_ARCHITECTURE_VERSION, MULTIHEAD_ARCHITECTURE_VERSION,
                                                   MEMORY_ARCHITECTURE_VERSION, MEMORY_MULTIHEAD_ARCHITECTURE_VERSION):
        raise ValueError("Checkpoint model architecture mismatch; use explicit legacy override for historical inference")
    # Loading weights must not consume the training process's global random stream.
    with torch.random.fork_rng(devices=[]):
        dimensions = (int(cfg['global_dim']), int(cfg['action_feat_dim']))
        hidden = int(cfg.get('hidden', cfg.get('hidden_dim', 128)))
        if weights_version in (MULTIHEAD_ARCHITECTURE_VERSION, MEMORY_MULTIHEAD_ARCHITECTURE_VERSION):
            cls = MemoryMultiHeadActorCritic if memory else MultiHeadActorCritic
            model = cls(*dimensions, hidden=hidden,
                num_heads=int(cfg['num_heads']), prior_strength=float(cfg['prior_strength']),
                prior_hidden=int(cfg.get('prior_hidden', 32)), public_information=public,
                prior_global_dim=cfg.get('prior_global_dim'), prior_action_dim=cfg.get('prior_action_dim')).to(device)
        else:
            model = (MemoryActorCritic(*dimensions, hidden=hidden, public_information=public) if memory else
                     ActorCritic(*dimensions, hidden=hidden,
                                architecture_version=weights_version, public_information=public)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
