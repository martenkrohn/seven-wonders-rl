"""PPO for offered-option observations, with an incompatible explicit contract."""

from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from sevenwonders_rl.features.decision_features import K, N_OPTIONS, feature_contract
from sevenwonders_rl.features.decision_memory import C, H, J, PUBLIC_SHAPES
from sevenwonders_rl.selfplay.decision_league import (
    DecisionLeague,
    LeagueConfig,
    atomic_json,
)
from sevenwonders_rl.utils.decision_workers import DecisionWorkers, LocalDecisionWorker
from sevenwonders_rl.utils.gae import compute_gae, normalize_advantages


def tensors(observations):
    if len(observations) == 1:
        return {
            key: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
            for key, value in observations[0].items()
        }
    return {
        key: torch.as_tensor(
            np.stack([o[key] for o in observations]), dtype=torch.float32
        )
        for key in observations[0]
    }


class DecisionActorCritic(nn.Module):
    def __init__(self, hidden_dim=64, memory_dim=32):
        super().__init__()
        if hidden_dim < 1 or memory_dim < 1:
            raise ValueError("Positive encoder dimensions required")
        self.architecture = {
            "name": "ordered-visible-memory-v1",
            "hidden_dim": hidden_dim,
            "memory_dim": memory_dim,
        }
        public_dim = sum(math.prod(s) for s in PUBLIC_SHAPES.values())
        self.register_buffer(
            "time_scale",
            torch.tensor(
                [3, 7, 21, 1, 1, 1, 1, 1, 1, 1, 1, J, 12, H], dtype=torch.float32
            ),
        )
        # Nonpersistent helpers leave existing checkpoint parameter/buffer keys intact.
        self.register_buffer(
            "_decision_positions", torch.arange(1, H + 1), persistent=False
        )
        self.register_buffer(
            "_public_positions", torch.arange(1, J + 1), persistent=False
        )
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, hidden_dim), nn.Tanh()
        )
        self.hand_encoder = nn.Sequential(nn.Linear(2 * C + 17, memory_dim), nn.Tanh())
        self.decision_encoder = nn.Sequential(
            nn.Linear(hidden_dim + memory_dim + K + C + 1, memory_dim), nn.Tanh()
        )
        self.public_history_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 8, memory_dim), nn.Tanh()
        )
        # Ordered flattening gives every valid time position its own learned weights.
        self.decision_history_encoder = nn.Sequential(
            nn.Linear(H * memory_dim, hidden_dim), nn.Tanh()
        )
        self.public_sequence_encoder = nn.Sequential(
            nn.Linear(J * memory_dim, hidden_dim), nn.Tanh()
        )
        self.state = nn.Sequential(
            nn.Linear(3 * hidden_dim + memory_dim, hidden_dim), nn.Tanh()
        )
        self.candidate = nn.Sequential(nn.Linear(K, hidden_dim), nn.Tanh())
        self.actor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def _public(self, obs, prefix, history=False):
        start = 2 if history else 1
        return self.public_encoder(
            torch.cat([obs[prefix + k].flatten(start) for k in PUBLIC_SHAPES], dim=-1)
        )

    def _hand(self, obs, prefix):
        return self.hand_encoder(
            torch.cat(
                (
                    obs[prefix + "hand_counts"],
                    obs[prefix + "discard_offer_counts"],
                    obs[prefix + "hand_visible"].unsqueeze(-1),
                    obs[prefix + "discard_offer_visible"].unsqueeze(-1),
                    obs[prefix + "time"] / self.time_scale,
                    obs[prefix + "discard_count"].unsqueeze(-1) / 63,
                ),
                dim=-1,
            )
        )

    def _history_length(self, valid, positions):
        return max(1, int((valid.bool() * positions).amax()))

    def forward(self, obs):
        # Use the last valid position, rather than the number of valid rows:
        # sparse masks and mixed-length batches retain their original time slots.
        nd = self._history_length(
            obs["decision_history.valid"], self._decision_positions
        )
        npub = self._history_length(obs["public_history.valid"], self._public_positions)
        obs = {
            k: (
                v[:, :nd]
                if k.startswith("decision_history.")
                else v[:, :npub]
                if k.startswith("public_history.")
                else v
            )
            for k, v in obs.items()
        }
        mask = obs["action_mask"].bool()
        if not mask.any(dim=-1).all():
            raise ValueError("Policy received an observation without legal options")
        valid = obs["decision_history.valid"].unsqueeze(-1)
        # Zero masked rows before biased encoders as well as after them. Padding
        # cannot affect gradients or outputs even when a caller fills it with junk.
        decision_obs = {
            k: (v * valid if v.ndim == 3 else v * valid.squeeze(-1))
            for k, v in obs.items()
            if k.startswith("decision_history.") and ".public." not in k
        }
        public_valid = obs["public_history.valid"].unsqueeze(-1)
        dpublic = self._public(
            {
                k: v * valid.reshape(*valid.shape[:2], *([1] * (v.ndim - 2)))
                for k, v in obs.items()
                if k.startswith("decision_history.public.")
            },
            "decision_history.public.",
            True,
        )
        dhand = self._hand(decision_obs, "decision_history.")
        dh = (
            self.decision_encoder(
                torch.cat(
                    (
                        dpublic,
                        dhand,
                        decision_obs["decision_history.applied_action_feats"],
                        decision_obs["decision_history.applied_card"],
                        decision_obs["decision_history.applied_mask"].unsqueeze(-1),
                    ),
                    dim=-1,
                )
            )
            * valid
        )
        ppublic = self._public(
            {
                k: v
                * public_valid.reshape(*public_valid.shape[:2], *([1] * (v.ndim - 2)))
                for k, v in obs.items()
                if k.startswith("public_history.public.")
            },
            "public_history.public.",
            True,
        )
        ptime = (
            torch.cat(
                (
                    obs["public_history.source_turn"].unsqueeze(-1) / 21,
                    obs["public_history.followup_seq"].unsqueeze(-1) / 12,
                    obs["public_history.phase_flags"],
                    obs["public_history.discard_count"].unsqueeze(-1) / 63,
                ),
                dim=-1,
            )
            * public_valid
        )
        ph = (
            self.public_history_encoder(torch.cat((ppublic, ptime), dim=-1))
            * public_valid
        )
        state = self.state(
            torch.cat(
                (
                    self._public(obs, "public."),
                    self._hand(obs, ""),
                    self.decision_history_encoder(
                        nn.functional.pad(
                            dh.flatten(1),
                            (0, (H - dh.shape[1]) * self.architecture["memory_dim"]),
                        )
                    ),
                    self.public_sequence_encoder(
                        nn.functional.pad(
                            ph.flatten(1),
                            (0, (J - ph.shape[1]) * self.architecture["memory_dim"]),
                        )
                    ),
                ),
                dim=-1,
            )
        )
        candidates = self.candidate(obs["action_feats"])
        logits = self.actor(
            torch.cat((state[:, None, :].expand(-1, N_OPTIONS, -1), candidates), dim=-1)
        ).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        pooled = (candidates * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True)
        value = self.critic(torch.cat((state, pooled), dim=-1)).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def select_action(self, obs, rng=None, deterministic=False):
        logits, _ = self(tensors([obs]))
        if deterministic:
            return int(logits.argmax(-1).item())
        probabilities = logits.softmax(-1)[0].cpu().numpy().astype(np.float64)
        probabilities /= probabilities.sum()
        return int((rng or np.random.default_rng()).choice(N_OPTIONS, p=probabilities))


def save_checkpoint(
    path, model, optimizer, steps, contract, metrics, league_state=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if contract != feature_contract(contract.get("ruleset")):
        raise ValueError("Checkpoint feature contract mismatch")
    data = {
        "format": "decision-ppo-v2",
        "contract": contract,
        "architecture": model.architecture,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": steps,
        "metrics": metrics,
        "torch_rng": torch.get_rng_state(),
        "league": league_state,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def load_checkpoint(path, contract, optimizer=None):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (
        data.get("format") != "decision-ppo-v2"
        or data.get("contract") != contract
        or contract != feature_contract(contract.get("ruleset"))
    ):
        raise ValueError(
            "Checkpoint decision/data/rules/feature contract mismatch; legacy models cannot be reused"
        )
    arch = data.get("architecture", {})
    if (
        set(arch) != {"name", "hidden_dim", "memory_dim"}
        or arch["name"] != "ordered-visible-memory-v1"
    ):
        raise ValueError("Checkpoint architecture contract mismatch")
    model = DecisionActorCritic(arch["hidden_dim"], arch["memory_dim"])
    model.load_state_dict(data["model"], strict=True)
    if optimizer is not None:
        raise ValueError(
            "Create an optimizer for the returned model, then restore optimizer state"
        )
    return model, data


def provenance(env):
    root = Path(__file__).resolve().parents[4]

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    lock_path = root / "engine.lock.json"
    expected = (
        json.loads(lock_path.read_text())["commit"] if lock_path.exists() else None
    )
    versions = {}
    for package in (
        "numpy",
        "torch",
        "gymnasium",
        "sevenwonders-engine-client",
        "sevenwonders-rl",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "rl_commit": git("rev-parse", "HEAD"),
        "rl_worktree_status": git("status", "--porcelain"),
        "rl_diff_sha256": hashlib.sha256(
            (git("diff", "HEAD") or "").encode()
        ).hexdigest(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                root / "python/src/sevenwonders_rl/features/decision_memory.py",
                root / "python/src/sevenwonders_rl/features/decision_features.py",
                root / "python/src/sevenwonders_rl/envs/decision_env.py",
                root / "python/src/sevenwonders_rl/selfplay/decision_league.py",
                root / "python/src/sevenwonders_rl/selfplay/pair_pfsp.py",
                root / "python/src/sevenwonders_rl/selfplay/opponent_pool.py",
                root / "python/src/sevenwonders_rl/utils/decision_workers.py",
                root / "python/src/sevenwonders_rl/utils/gae.py",
                root / "python/src/sevenwonders_rl/evaluation/decision_intermediate.py",
                root / "python/src/sevenwonders_rl/evaluation/decision_evaluation.py",
                root / "python/scripts/train_decision_ppo.py",
            ]
            if p.is_file()
        },
        "engine_commit": env.engine_metadata["source_commit"],
        "expected_engine_commit": expected,
        "engine_matches_pin": env.engine_metadata["source_commit"] == expected,
        "engine_binary": str(env.engine_path),
        "engine_binary_sha256": hashlib.sha256(
            env.engine_path.read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "versions": versions,
    }


def train(
    env,
    steps=128,
    seed=42,
    output=None,
    resume=None,
    rollout_size=None,
    epochs=None,
    league_config=None,
    progress=None,
    workers=None,
    evaluation_every_games=0,
):
    """Train against supplied callbacks, or an own-snapshot league.

    league_config=None preserves the API's callback mode for fresh runs and
    restores a saved league on resume. False explicitly requires non-league mode.
    The CLI opts into LeagueConfig by default.
    """
    from sevenwonders_rl.selfplay.decision_league import training_directory
    from sevenwonders_rl.utils.decision_control import control_run

    if steps is None and output is None:
        raise ValueError("Continuous training requires an output directory")
    if type(evaluation_every_games) is not int or evaluation_every_games < 0:
        raise ValueError("Nonnegative intermediate evaluation interval required")
    with training_directory(output, resume):
        saved = (
            (
                torch.load(resume, map_location="cpu", weights_only=True)
                .get("metrics", {})
                .get("configuration", {})
            )
            if resume
            else {}
        )
        if rollout_size is None:
            rollout_size = saved.get("rollout_size", 64)
        if epochs is None:
            epochs = saved.get("epochs", 4)
        if workers is None:
            workers = saved.get("workers", 1)
        if type(workers) is not int or workers < 1:
            raise ValueError("Positive worker count required")
        if output is not None:
            with control_run(output) as control:
                return _train(
                    env,
                    steps,
                    seed,
                    output,
                    resume,
                    rollout_size,
                    epochs,
                    league_config,
                    progress,
                    control,
                    workers=workers,
                    evaluation_every_games=evaluation_every_games,
                )
        return _train(
            env,
            steps,
            seed,
            output,
            resume,
            rollout_size,
            epochs,
            league_config,
            progress,
            None,
            workers=workers,
            evaluation_every_games=evaluation_every_games,
        )


def _update(model, optimizer, batch, actions, old, returns, adv, epochs):
    losses = []
    size = len(actions)
    for _ in range(epochs):
        for indices in torch.randperm(size).split(min(16, size)):
            logits, value = model({k: v[indices] for k, v in batch.items()})
            distribution = Categorical(logits=logits)
            ratio = (distribution.log_prob(actions[indices]) - old[indices]).exp()
            policy = -torch.minimum(
                ratio * adv[indices], ratio.clamp(0.8, 1.2) * adv[indices]
            ).mean()
            loss = (
                policy
                + 0.5 * (value - returns[indices]).square().mean()
                - 0.01 * distribution.entropy().mean()
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite PPO gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
    return losses


def _train(
    env,
    steps,
    seed,
    output,
    resume,
    rollout_size,
    epochs,
    league_config,
    progress,
    control,
    workers,
    evaluation_every_games=0,
):
    if (steps is not None and steps < 1) or rollout_size < 1 or epochs < 1:
        raise ValueError("Positive training sizes required")
    if workers > 1 and env.opponents and not hasattr(env, "_decision_league_base_seat"):
        raise ValueError(
            "Parallel collection does not support custom opponent callbacks; use workers=1"
        )
    base_seat = getattr(env, "_decision_league_base_seat", env.agent_seat)
    env.agent_seat = base_seat
    callbacks = (
        dict(env.opponents) if not hasattr(env, "_decision_league_base_seat") else {}
    )
    env.opponents = callbacks
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    _, info = env.reset(seed=seed)
    contract = info["feature_contract"]
    env.close()
    model = DecisionActorCritic()
    checkpoint = None
    offset = 0
    if resume:
        model, checkpoint = load_checkpoint(resume, contract)
        offset = checkpoint["steps"]
        if checkpoint.get("role") == "frozen-league-snapshot":
            raise ValueError("Frozen opponent snapshots cannot resume a learner")
    saved_league = checkpoint.get("league") if checkpoint else None
    if saved_league is not None and league_config is False:
        raise ValueError("Cannot resume a league checkpoint without its league")
    if saved_league is not None and league_config is None:
        league_config = LeagueConfig(**saved_league["config"])
    if isinstance(league_config, dict):
        league_config = LeagueConfig(**league_config)
    if league_config not in (None, False) and not isinstance(
        league_config, LeagueConfig
    ):
        raise ValueError("Expected a LeagueConfig or non-league mode")
    league_enabled = isinstance(league_config, LeagueConfig)
    if evaluation_every_games and not league_enabled:
        raise ValueError("Intermediate evaluation requires league training")
    if league_enabled and output is None:
        raise ValueError("League training requires an output directory")
    if league_enabled and resume and saved_league is None:
        raise ValueError(
            "Resume cannot silently add a new league to a non-league learner"
        )
    if (
        league_enabled
        and resume
        and Path(resume).resolve() != Path(output).resolve() / "checkpoint.pt"
    ):
        raise ValueError("Resume the durable checkpoint.pt in its league run directory")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        saved = checkpoint["metrics"]["configuration"]
        # Non-league checkpoints have no league contract to enforce these values.
        if not league_enabled and any(
            saved.get(k, v) != v
            for k, v in [
                ("rollout_size", rollout_size),
                ("epochs", epochs),
                ("reward_scale", env.reward_scale),
                ("agent_seat", base_seat),
            ]
        ):
            raise ValueError("Training resume configuration mismatch")
    training_contract = {
        "rollout_size": rollout_size,
        "epochs": epochs,
        "reward_scale": env.reward_scale,
        "base_seat": base_seat,
        "engine_commit": info["engine"]["source_commit"],
        "engine_dirty": info["engine"].get("source_dirty"),
        "opponent_action_mode": "stochastic-seat-local-rng",
        "collector_workers": workers,
    }
    league = None
    if league_enabled:
        league = DecisionLeague(
            output,
            contract,
            league_config,
            seed,
            base_seat,
            saved_league,
            offset,
            training_contract,
        )
        env._decision_league_base_seat = base_seat
        if not saved_league:
            league.publish(model, 0)
    old_metrics = checkpoint["metrics"] if checkpoint else {}
    episode_seeds = list(old_metrics.get("episode_seeds", []))
    next_seed = max(
        seed,
        max(episode_seeds, default=seed - 1) + 1,
        league.reserved_seed_max + 1 if league else seed,
    )
    previous_games = old_metrics.get(
        "completed_games_total", old_metrics.get("completed_games", 0)
    )
    previous_seconds = old_metrics.get("training_seconds_total", 0.0)
    duration_complete = (
        old_metrics.get("training_duration_complete", False) if checkpoint else True
    )
    run_provenance = provenance(env)
    completed = 0
    applied = 0
    losses = []
    status = "running"
    session_started = None
    evaluation_seconds = 0.0
    from sevenwonders_rl.evaluation.decision_intermediate import IntermediateEvaluations

    intermediate = IntermediateEvaluations(
        output, evaluation_every_games, workers, old_metrics
    )
    done = [True] * workers

    def metrics_at(step):
        elapsed = (
            max(0.0, time.monotonic() - session_started - evaluation_seconds)
            if session_started is not None
            else 0.0
        )
        return {
            "status": status,
            "continuous": steps is None,
            "steps": step,
            "new_steps": applied,
            "actually_applied": applied,
            "training_seconds_session": elapsed,
            "training_seconds_total": previous_seconds + elapsed,
            "training_duration_complete": duration_complete,
            "completed_games": completed,
            "intermediate_evaluation": intermediate.state_dict(),
            "completed_games_total": league.completed
            if league
            else previous_games + completed,
            "updates": len(losses),
            "loss_mean": float(np.mean(losses)) if losses else None,
            "contract": contract,
            "engine": info["engine"],
            "seed": seed,
            "episode_seeds": list(episode_seeds),
            "architecture": model.architecture,
            "configuration": {
                "steps": steps,
                "rollout_size": rollout_size,
                "epochs": epochs,
                "workers": workers,
                "evaluation_every_games": evaluation_every_games,
                "agent_seat": base_seat,
                "reward_scale": env.reward_scale,
                "allow_development": env.allow_development,
                "resume": str(resume) if resume else None,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip": 0.2,
                "learning_rate": 3e-4,
                "value_weight": 0.5,
                "entropy_weight": 0.01,
                "max_grad_norm": 0.5,
                "minibatch_size": min(16, rollout_size),
                "league": asdict(league_config) if league else None,
            },
            "league": {
                "completed_games": league.completed,
                "selections": league.selections,
                "active_snapshots": league.pool.pool_size(step),
                "abandoned_games": league.selections
                - league.completed
                - len(league.active_by_slot),
                "active_games": len(league.active_by_slot),
            }
            if league
            else None,
            "provenance": run_provenance,
            "resume_semantics": "Fresh engine games; model/optimizer/Torch/league sampler restored; unfinished games unscored",
        }

    def persist(step):
        metrics = metrics_at(step)
        if output:
            directory = Path(output)
            save_checkpoint(
                directory / "checkpoint.pt",
                model,
                optimizer,
                step,
                contract,
                metrics,
                league.state_dict() if league else None,
            )
            atomic_json(directory / "metrics.json", metrics)
            atomic_json(directory / "manifest.json", metrics)
            atomic_json(
                directory / "training_status.json",
                {
                    k: metrics[k]
                    for k in (
                        "status",
                        "continuous",
                        "steps",
                        "completed_games_total",
                        "training_seconds_total",
                        "training_seconds_session",
                        "training_duration_complete",
                        "intermediate_evaluation",
                    )
                }
                | {
                    "completed_games_session": completed,
                    "checkpoint": str(directory / "checkpoint.pt"),
                },
            )
            if league:
                atomic_json(directory / "league_status.json", league.report(step))
        return metrics

    # Keep a fresh run resumable even if a child cannot start. Seed reservations
    # written after this checkpoint are recovered by the normal journal repair.
    if output:
        persist(offset)
    collector = (
        (lambda: LocalDecisionWorker(env, league, callbacks))
        if workers == 1
        else (
            lambda: DecisionWorkers(
                workers,
                env,
                league.snapshot_dir if league else None,
                league_config.pool_max if league else 1,
            )
        )
    )
    with collector() as pool:

        def reset(slot, step):
            nonlocal next_seed
            if league:
                assignment = league.reserve_episode(next_seed, step, slot)
                payload = {**assignment, "contract": contract}
            else:
                payload = {"seed": next_seed, "agent_seat": base_seat}
            episode_seeds.append(next_seed)
            next_seed += 1
            pool.send(slot, "reset", payload)

        def reset_received(slot):
            result = pool.receive(slot)
            if (
                result["schema_hash"] != contract["schema_hash"]
                or result["engine"] != info["engine"]
            ):
                raise RuntimeError(
                    "Worker engine/observation contract differs from learner"
                )
            done[slot] = False

        # Even a run shorter than the worker count records/reserves every open
        # game's seed before its engine starts. No unfinished game is scored.
        for slot in range(workers):
            reset(slot, offset)
        for slot in range(workers):
            reset_received(slot)
        session_started = time.monotonic()
        if output:
            persist(offset)
        if progress:
            progress(metrics_at(offset))
        while steps is None or applied < steps:
            if control and control.should_stop():
                break
            update_start = offset + applied
            size = rollout_size if steps is None else min(rollout_size, steps - applied)
            if league:
                size = min(
                    size,
                    league_config.snapshot_every
                    - update_start % league_config.snapshot_every,
                )
            observations = []
            actions = []
            old_logs = []
            values = []
            rewards = []
            dones = []
            trajectories = {slot: [] for slot in range(workers)}
            while len(actions) < size:
                count = min(workers, size - len(actions))
                cursor = (update_start + len(actions)) % workers
                slots = [(cursor + i) % workers for i in range(count)]
                restarting = [slot for slot in slots if done[slot]]
                for slot in restarting:
                    reset(slot, update_start)
                for slot in restarting:
                    reset_received(slot)
                current = [pool.observation(slot) for slot in slots]
                with torch.no_grad():
                    logits, value = model(tensors(current))
                    distribution = Categorical(logits=logits)
                    action = distribution.sample()
                    logs = distribution.log_prob(action)
                for i, slot in enumerate(slots):
                    pool.send(slot, "step", int(action[i]))
                for i, slot in enumerate(slots):
                    result = pool.receive(slot)
                    trajectories[slot].append(len(actions))
                    observations.append(current[i])
                    actions.append(int(action[i]))
                    old_logs.append(float(logs[i]))
                    values.append(float(value[i]))
                    rewards.append(result["reward"])
                    dones.append(result["done"])
                    done[slot] = result["done"]
                    applied += 1
                    if done[slot]:
                        completed += 1
                        if league:
                            league.finish_episode(result["totals"], update_start, slot)
            nonterminal = [
                slot
                for slot, indices in trajectories.items()
                if indices and not done[slot]
            ]
            bootstrap = {slot: 0.0 for slot in trajectories}
            if nonterminal:
                with torch.no_grad():
                    next_values = model(
                        tensors([pool.observation(s) for s in nonterminal])
                    )[1]
                bootstrap.update(
                    {slot: float(next_values[i]) for i, slot in enumerate(nonterminal)}
                )
            advantages = torch.zeros(size)
            for slot, indices in trajectories.items():
                if not indices:
                    continue
                vals = torch.tensor([values[i] for i in indices])[:, None]
                ends = torch.tensor([dones[i] for i in indices])[:, None]
                next_vals = torch.cat((vals[1:], torch.tensor([[bootstrap[slot]]])))
                estimate, _ = compute_gae(
                    torch.tensor([rewards[i] for i in indices])[:, None],
                    vals,
                    ends,
                    torch.zeros_like(ends),
                    next_vals,
                    gamma=0.99,
                    gae_lambda=0.95,
                )
                advantages[indices] = estimate[:, 0]
            returns = advantages + torch.tensor(values)
            adv = normalize_advantages(advantages)
            losses.extend(
                _update(
                    model,
                    optimizer,
                    tensors(observations),
                    torch.tensor(actions),
                    torch.tensor(old_logs),
                    returns,
                    adv,
                    epochs,
                )
            )
            current_step = offset + applied
            if league and current_step % league_config.snapshot_every == 0:
                league.publish(model, current_step)
            if output or progress:
                update_metrics = persist(current_step)
                if progress:
                    progress(update_metrics)
                if not (control and control.should_stop()):
                    started = time.monotonic()
                    if intermediate.run_due(
                        update_metrics,
                        env.engine_path,
                        emit=print if progress else None,
                    ):
                        evaluation_seconds += time.monotonic() - started
                        persist(current_step)
        status = "stopped" if control and control.should_stop() else "completed"
        return model, persist(offset + applied)


def evaluate(env, model, games=3, seed=1000):
    results = []
    for episode in range(games):
        obs, info = env.reset(seed=seed + episode)
        while True:
            action = model.select_action(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if truncated:
                raise RuntimeError("Evaluation exchange failed")
            if done:
                results.append(
                    {
                        "seed": seed + episode,
                        "totals": info["final_totals"],
                        "margin": reward,
                    }
                )
                break
    return results
