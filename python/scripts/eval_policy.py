from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import hashlib
import torch

from sevenwonders_rl.envs import SevenWondersEnv, SeatPolicy
from sevenwonders_rl.policies import (
    HeuristicPolicy,
    MilitaryTempoPolicy,
    ScienceSetPolicy,
    EconomyPolicy,
    WonderTempoPolicy,
    VPFinisherPolicy,
)
from sevenwonders_rl.features.action_feature_schema import FEATURE_NAMES, K as K_ACTION_FEATS
from sevenwonders_rl.utils.ppo_model import ActorCritic
from sevenwonders_rl.utils.ppo_utils import select_action_from_obs


def _select_agent_action(
    agent_policy: str,
    obs: Dict[str, Any],
    rng: np.random.Generator,
    policy_obj: Any | None,
    model: ActorCritic | None,
    device: torch.device,
    deterministic: bool,
) -> int:
    mask = obs["action_mask"]
    legal = np.flatnonzero(mask == 1)
    if legal.size == 0:
        return 0

    if model is not None:
        return int(select_action_from_obs(obs, model, device, deterministic=deterministic))

    if policy_obj is not None:
        return int(policy_obj.select_action(obs, rng))

    return int(rng.choice(legal))


def _make_opponent_policies(
    opponents: str,
    agent_seat: int,
    rng: np.random.Generator,
) -> List[SeatPolicy]:
    policies = [SeatPolicy("random"), SeatPolicy("random"), SeatPolicy("random")]

    opp_seats = [s for s in (0, 1, 2) if s != agent_seat]
    if opponents == "random":
        return policies

    if opponents == "heuristic":
        for s in opp_seats:
            policies[s] = SeatPolicy("heuristic")
        return policies

    if opponents == "mixed":
        rng.shuffle(opp_seats)
        policies[opp_seats[0]] = SeatPolicy("heuristic")
        policies[opp_seats[1]] = SeatPolicy("random")
        return policies

    raise ValueError(f"Unknown opponents config: {opponents}")


def evaluate(
    games: int,
    seed: int,
    agent_seat: int,
    opponents: str,
    agent_policy: str,
    out_path: Path | None,
    jsonl_path: Path | None,
    model: ActorCritic | None,
    device: torch.device,
    deterministic: bool,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]

    results = []
    
    # Instantiate policy based on agent_policy string
    policy_obj = None
    if agent_policy == "heuristic":
        policy_obj = HeuristicPolicy()
    elif agent_policy == "military":
        policy_obj = MilitaryTempoPolicy()
    elif agent_policy == "science":
        policy_obj = ScienceSetPolicy()
    elif agent_policy == "economy":
        policy_obj = EconomyPolicy()
    elif agent_policy == "wonder":
        policy_obj = WonderTempoPolicy()
    elif agent_policy == "vp_finisher":
        policy_obj = VPFinisherPolicy()
    elif agent_policy != "random":
        raise ValueError(f"Unknown agent_policy: {agent_policy}")

    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("", encoding="utf-8")

    for game_idx in range(games):
        game_seed = seed + game_idx
        rng = np.random.default_rng(game_seed)

        opponent_policies = _make_opponent_policies(opponents, agent_seat, rng)
        env = SevenWondersEnv(
            repo_root=str(repo_root),
            agent_seat=agent_seat,
            opponent_policies=opponent_policies,
        )

        try:
            obs, info = env.reset(seed=game_seed)
            done = False
            action_counts = {
                "discard": 0,
                "build_wonder": 0,
                "prod_build": 0,
                "other": 0,
            }
            invalid_ct = 0
            while not done:
                action = _select_agent_action(
                    agent_policy,
                    obs,
                    rng,
                    policy_obj,
                    model,
                    device,
                    deterministic,
                )
                
                # Count action type (for debug)
                if policy_obj is not None or (model is not None):
                    action_feats = obs.get("action_feats")
                    if action_feats is not None and action < len(action_feats):
                        from sevenwonders_rl.features.action_feature_schema import IDX
                        row = action_feats[action]
                        if row[IDX["action_type_3"]] > 0.5:
                            action_counts["discard"] += 1
                        elif row[IDX["action_type_1"]] > 0.5:
                            action_counts["build_wonder"] += 1
                        elif sum(row[IDX[f"delta_prod_{r}"]] for r in ["wood", "ore", "clay", "stone", "loom", "glass", "papyrus"]) > 0 or row[IDX["delta_prod_choice_raw"]] > 0 or row[IDX["delta_prod_choice_manuf"]] > 0:
                            action_counts["prod_build"] += 1
                        else:
                            action_counts["other"] += 1
                
                obs, reward, terminated, truncated, info = env.step(action)
                if info.get("agent_action_invalid_requested", False):
                    invalid_ct += 1
                done = terminated or truncated

            status = info.get("status", {})
            players = status.get("players", {})
            totals = []
            for sid in (0, 1, 2):
                totals.append(int(players[str(sid)]["points"]["total"]))

            agent_total = totals[agent_seat]
            opp_best = max(t for i, t in enumerate(totals) if i != agent_seat)
            opp_mean = float(np.mean([t for i, t in enumerate(totals) if i != agent_seat]))
            vp_diff = agent_total - opp_best
            win = 1 if vp_diff > 0 else 0

            points = players[str(agent_seat)]["points"]
            wonder_stage = int(players[str(agent_seat)].get("wonder_stage", 0))
            civilian = int(points.get("civilian", 0))
            
            # Extract final production for validation
            resources = players[str(agent_seat)].get("resources", {})
            final_prod_self = [
                int(resources.get("wood", 0)),
                int(resources.get("ore", 0)),
                int(resources.get("clay", 0)),
                int(resources.get("stone", 0)),
                int(resources.get("loom", 0)),
                int(resources.get("glass", 0)),
                int(resources.get("papyrus", 0)),
            ]
            
            # Extract termination info
            turn_end = int(status.get("game", {}).get("turn", -1))
            
            game_record = {
                "seed": game_seed,
                "win": win,
                "agent_total": agent_total,
                "opp_best": opp_best,
                "opp_mean": opp_mean,
                "totals": totals,
                "vp_diff": vp_diff,
                "vp_diff_from_mean": agent_total - opp_mean,
                "points": points,
                "wonder_stage": wonder_stage,
                "civilian": civilian,
                "final_prod_self": final_prod_self,
                "discard_count_agent": action_counts["discard"],
                "build_wonder_count_agent": action_counts["build_wonder"],
                "prod_build_count_agent": action_counts["prod_build"],
                "invalid_requested_count_agent": invalid_ct,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "turn_end": turn_end,
                "truncated_reason": info.get("truncated_reason", "") if truncated else "",
            }
            results.append(
                {
                    **game_record,
                }
            )

            if jsonl_path is not None:
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(game_record) + "\n")
        finally:
            env.close()

    vp_diffs = np.array([r["vp_diff"] for r in results], dtype=np.float32)
    agent_totals = np.array([r["agent_total"] for r in results], dtype=np.float32)
    wins = np.array([r["win"] for r in results], dtype=np.float32)

    summary = {
        "games": games,
        "seed": seed,
        "opponents": opponents,
        "agent_policy": "model" if model is not None else agent_policy,
        "deterministic": deterministic,
        "win_rate": float(wins.mean()),
        "vp_diff_mean": float(vp_diffs.mean()),
        "vp_diff_median": float(np.median(vp_diffs)),
        "vp_diff_std": float(vp_diffs.std(ddof=0)),
        "agent_vp_mean": float(agent_totals.mean()),
    }

    payload = {"summary": summary, "games_detail": results}

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy in SevenWondersEnv")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent-seat", type=int, default=0)
    parser.add_argument("--opponents", choices=["random", "heuristic", "mixed"], default="random")
    parser.add_argument(
        "--agent-policy",
        choices=["random", "heuristic", "military", "science", "economy", "wonder", "vp_finisher"],
        default="random"
    )
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--opp-model", type=str, default="")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--jsonl", type=str, default="")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else None
    jsonl_path = Path(args.jsonl) if args.jsonl else None

    device = torch.device(args.device)
    model = None
    if args.model:
        ckpt = torch.load(args.model, map_location=device)
        cfg = ckpt.get("config", {})
        global_dim = int(cfg.get("global_dim", 0))
        action_feat_dim = int(cfg.get("action_feat_dim", 0))
        hidden = int(cfg.get("hidden", 128))
        if global_dim <= 0 or action_feat_dim <= 0:
            raise ValueError("Checkpoint config missing global_dim or action_feat_dim")

        model = ActorCritic(global_dim, action_feat_dim, hidden=hidden).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        meta = ckpt.get("meta", {})
        current_hash = hashlib.sha256("\n".join(FEATURE_NAMES).encode("utf-8")).hexdigest()
        ckpt_hash = meta.get("feature_schema_hash")
        if ckpt_hash and ckpt_hash != current_hash:
            print(
                "Warning: feature_schema_hash mismatch; eval results may be invalid. "
                f"ckpt={ckpt_hash} current={current_hash}"
            )
        if meta.get("k_action_feats") not in (None, int(K_ACTION_FEATS)):
            print(
                "Warning: k_action_feats mismatch; eval results may be invalid. "
                f"ckpt={meta.get('k_action_feats')} current={int(K_ACTION_FEATS)}"
            )
    payload = evaluate(
        games=args.games,
        seed=args.seed,
        agent_seat=args.agent_seat,
        opponents=args.opponents,
        agent_policy=args.agent_policy,
        out_path=out_path,
        jsonl_path=jsonl_path,
        model=model,
        device=device,
        deterministic=args.deterministic,
    )

    summary = payload["summary"]
    print(
        f"games={summary['games']} seed={summary['seed']} opponents={summary['opponents']} agent={summary['agent_policy']}"
    )
    print(
        "win_rate={win_rate:.3f} vp_diff_mean={vp_diff_mean:.2f} "
        "vp_diff_std={vp_diff_std:.2f} agent_vp_mean={agent_vp_mean:.2f}".format(**summary)
    )

    if args.render:
        print("Render is not supported in this environment.")


if __name__ == "__main__":
    main()
 