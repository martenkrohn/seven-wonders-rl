"""Validate opponent sampling distribution and determinism.

Runs 200 resets and verifies:
1. Sampling frequency matches expected weights
2. Deterministic behavior (same seed -> same sequence)
3. Per-env diversity (different env_id -> different sequences)
"""

from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "python" / "src"))

from sevenwonders_rl.envs import SevenWondersEnv


# Expected weights from train_ppo.py
OPP_WEIGHTS = {
    "baseline": 0.40,
    "mad_scientist": 0.30,
    "military": 0.20,
    "random": 0.10,
}

N_GAMES = 200
SEED = 42


def validate_sampling_distribution():
    """Test that sampling frequency matches expected weights."""
    print("=" * 80)
    print("VALIDATING OPPONENT SAMPLING DISTRIBUTION")
    print("=" * 80)
    print(f"Expected weights: {OPP_WEIGHTS}")
    print(f"Games: {N_GAMES}, Seed: {SEED}\n")
    
    io_dir = repo_root / "io_validate" / "io"
    io_dir.mkdir(parents=True, exist_ok=True)
    
    env = SevenWondersEnv(
        repo_root=str(repo_root),
        agent_seat=0,
        io_dir=str(io_dir),
        opponent_weights=OPP_WEIGHTS,
        seed=SEED,
        env_id=0,
    )
    
    # Collect opponent names from 200 resets (no stepping)
    left_counts = Counter()
    right_counts = Counter()
    total_counts = Counter()
    
    for i in range(N_GAMES):
        obs, info = env.reset(seed=SEED + i)
        
        opponents = info.get("opponents", {})
        left = opponents.get("left", "")
        right = opponents.get("right", "")
        
        left_counts[left] += 1
        right_counts[right] += 1
        total_counts[left] += 1
        total_counts[right] += 1
        
        if i < 5:
            print(f"  Game {i}: left={left}, right={right}")
    
    env.close()
    
    # Total opponent slots = 2 * N_GAMES = 400
    total_slots = 2 * N_GAMES
    
    print(f"\n{'Policy':<15} {'Expected':>10} {'Actual':>10} {'Deviation':>10}")
    print("-" * 50)
    
    all_pass = True
    for policy_name in sorted(OPP_WEIGHTS.keys()):
        expected = OPP_WEIGHTS[policy_name] * total_slots
        actual = total_counts[policy_name]
        deviation = abs(actual - expected)
        
        # Allow binomial variance: std = sqrt(n * p * (1-p))
        # For n=400, p=0.4: std ≈ 9.8
        # Accept if deviation < 3*std (99.7% confidence)
        p = OPP_WEIGHTS[policy_name]
        std = (total_slots * p * (1 - p)) ** 0.5
        threshold = 3 * std
        
        passed = deviation <= threshold
        status = "✓" if passed else "✗"
        
        print(f"{policy_name:<15} {expected:>10.1f} {actual:>10} {deviation:>10.1f} {status}")
        
        if not passed:
            all_pass = False
            print(f"  WARNING: Deviation {deviation:.1f} exceeds 3σ threshold {threshold:.1f}")
    
    print()
    if all_pass:
        print("✅ Sampling distribution PASSED")
    else:
        print("❌ Sampling distribution FAILED")
    
    return all_pass


def validate_determinism():
    """Test that same seed produces identical sequences."""
    print("\n" + "=" * 80)
    print("VALIDATING DETERMINISM")
    print("=" * 80)
    
    io_dir = repo_root / "io_validate_det" / "io"
    io_dir.mkdir(parents=True, exist_ok=True)
    
    sequences = []
    
    for run in range(2):
        env = SevenWondersEnv(
            repo_root=str(repo_root),
            agent_seat=0,
            io_dir=str(io_dir),
            opponent_weights=OPP_WEIGHTS,
            seed=SEED,
            env_id=0,
        )
        
        seq = []
        for i in range(10):
            obs, info = env.reset(seed=SEED + i)
            opponents = info.get("opponents", {})
            seq.append((opponents.get("left", ""), opponents.get("right", "")))
        
        sequences.append(seq)
        env.close()
    
    # Check if both runs produced identical sequences
    if sequences[0] == sequences[1]:
        print("✅ Determinism PASSED: Identical sequences across runs")
        for i, (left, right) in enumerate(sequences[0][:5]):
            print(f"  Episode {i}: ({left}, {right})")
        return True
    else:
        print("❌ Determinism FAILED: Sequences differ")
        for i in range(min(5, len(sequences[0]))):
            left0, right0 = sequences[0][i]
            left1, right1 = sequences[1][i]
            match = "✓" if (left0, right0) == (left1, right1) else "✗"
            print(f"  Episode {i}: ({left0}, {right0}) vs ({left1}, {right1}) {match}")
        return False


def validate_env_diversity():
    """Test that different env_id produces different sequences."""
    print("\n" + "=" * 80)
    print("VALIDATING ENV DIVERSITY")
    print("=" * 80)
    
    sequences = []
    
    for env_id in range(3):
        io_dir = repo_root / f"io_validate_env{env_id}" / "io"
        io_dir.mkdir(parents=True, exist_ok=True)
        
        env = SevenWondersEnv(
            repo_root=str(repo_root),
            agent_seat=0,
            io_dir=str(io_dir),
            opponent_weights=OPP_WEIGHTS,
            seed=SEED,
            env_id=env_id,
        )
        
        seq = []
        for i in range(10):
            obs, info = env.reset(seed=SEED + i)
            opponents = info.get("opponents", {})
            seq.append((opponents.get("left", ""), opponents.get("right", "")))
        
        sequences.append(seq)
        env.close()
        
        print(f"Env {env_id} first 3 episodes:")
        for i in range(3):
            left, right = seq[i]
            print(f"  Episode {i}: ({left}, {right})")
    
    # Check that at least some episodes differ across env_ids
    diff_count = 0
    for i in range(10):
        if len(set(sequences[env_id][i] for env_id in range(3))) > 1:
            diff_count += 1
    
    if diff_count >= 5:  # At least 50% different
        print(f"\n✅ Env diversity PASSED: {diff_count}/10 episodes differ across env_ids")
        return True
    else:
        print(f"\n❌ Env diversity FAILED: Only {diff_count}/10 episodes differ")
        return False


def main():
    print("Starting opponent sampling validation...\n")
    
    results = []
    results.append(("Distribution", validate_sampling_distribution()))
    results.append(("Determinism", validate_determinism()))
    results.append(("Env Diversity", validate_env_diversity()))
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<20} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print()
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED")
        return 0
    else:
        print("⚠️  SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
