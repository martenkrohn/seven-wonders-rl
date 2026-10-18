#!/usr/bin/env python3
"""Validation script for action features (Step 4).

Runs 50-200 random games and validates:
- No unknown card names in cards_played
- No NaNs/Infs in action features
- No crashes during feature computation
- Action features are deterministic for identical states
"""

import sys
import numpy as np
from pathlib import Path
from typing import Set
import warnings

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / 'src'))
sys.path.insert(0, str(repo_root / 'python' / 'src'))

from sevenwonders_rl.envs import SevenWondersEnv
from sevenwonders_rl.data.cards import by_name_strict, load_card_db
from sevenwonders_rl.features.action_feature_schema import K, FEATURE_NAMES


def validate_cards_played(status: dict) -> Set[str]:
    """Check that all cards_played names resolve to valid card IDs.
    
    Returns:
        Set of unknown card names (should be empty)
    """
    unknown_names = set()
    
    for player_id, player in status['players'].items():
        for card_name in player.get('cards_played', []):
            try:
                by_name_strict(card_name)
            except ValueError as e:
                unknown_names.add(card_name)
                print(f"ERROR: Unknown card name: {card_name}")
                print(f"  {e}")
    
    return unknown_names


def validate_action_features(action_feats: np.ndarray) -> tuple[bool, list[str]]:
    """Check that action features are valid.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Check shape
    if action_feats.shape != (28, K):
        errors.append(f"Wrong shape: {action_feats.shape} vs (28, {K})")
    
    # Check dtype
    if action_feats.dtype != np.float32:
        errors.append(f"Wrong dtype: {action_feats.dtype} vs float32")
    
    # Check for NaNs/Infs
    nan_count = np.isnan(action_feats).sum()
    if nan_count > 0:
        errors.append(f"Found {nan_count} NaN values")
    
    inf_count = np.isinf(action_feats).sum()
    if inf_count > 0:
        errors.append(f"Found {inf_count} Inf values")
    
    # Check ranges (all features should be roughly in [-1e9, 1e9])
    too_large = np.abs(action_feats) > 1e10
    if too_large.sum() > 0:
        errors.append(f"Found {too_large.sum()} values with magnitude > 1e10")
    
    return len(errors) == 0, errors


def run_validation_games(num_games: int = 50, max_steps: int = 100) -> dict:
    """Run games and validate action features.
    
    Args:
        num_games: Number of games to run
        max_steps: Maximum steps per game
    
    Returns:
        Validation report dict
    """
    report = {
        'total_games': num_games,
        'successful_games': 0,
        'total_steps': 0,
        'crashes': [],
        'unknown_cards': set(),
        'feature_errors': [],
        'determinism_checks': [],
    }
    
    try:
        env = SevenWondersEnv(
            repo_root=str(repo_root),
            agent_seat=0,
            max_wait_seconds=10.0
        )
        
        for game_idx in range(num_games):
            try:
                print(f"Running game {game_idx + 1}/{num_games}...", end=' ')
                sys.stdout.flush()
                
                obs, info = env.reset()
                
                # Check initial state action features
                action_feats = obs['action_feats']
                is_valid, errors = validate_action_features(action_feats)
                if not is_valid:
                    report['feature_errors'].append({
                        'game': game_idx,
                        'step': 0,
                        'errors': errors,
                    })
                
                # Validate cards_played
                unknown = validate_cards_played(info.get('status', {}))
                report['unknown_cards'].update(unknown)
                
                # Run game
                step_count = 0
                while step_count < max_steps:
                    action_mask = obs['action_mask']
                    
                    if action_mask.sum() == 0:
                        break  # Game over
                    
                    # Take a random legal action
                    legal_actions = np.where(action_mask == 1)[0]
                    action = int(legal_actions[np.random.randint(0, len(legal_actions))])
                    
                    # Check determinism at this state
                    if step_count == 0 and game_idx == 0:
                        # Save first state's features for determinism check
                        first_feats = action_feats.copy()
                        obs_repeat, _ = env.reset()
                        second_feats = obs_repeat['action_feats']
                        
                        if not np.allclose(first_feats, second_feats, atol=1e-5):
                            report['determinism_checks'].append({
                                'game': game_idx,
                                'step': 0,
                                'issue': 'Same state produced different features'
                            })
                        
                        # Re-step with same action
                        obs, info = env.step(action)
                    else:
                        obs, info = env.step(action)
                    
                    # Validate action features at this step
                    action_feats = obs['action_feats']
                    is_valid, errors = validate_action_features(action_feats)
                    if not is_valid:
                        report['feature_errors'].append({
                            'game': game_idx,
                            'step': step_count + 1,
                            'errors': errors,
                        })
                    
                    # Validate cards_played
                    unknown = validate_cards_played(info.get('status', {}))
                    report['unknown_cards'].update(unknown)
                    
                    step_count += 1
                
                report['successful_games'] += 1
                report['total_steps'] += step_count
                print(f"OK ({step_count} steps)")
                
            except Exception as e:
                report['crashes'].append({
                    'game': game_idx,
                    'error': str(e),
                    'type': type(e).__name__,
                })
                print(f"CRASH: {e}")
        
        env.close()
    
    except Exception as e:
        print(f"FATAL: Could not initialize environment: {e}")
        report['crashes'].append({
            'context': 'environment_init',
            'error': str(e),
        })
    
    return report


def print_report(report: dict) -> None:
    """Print validation report."""
    print("\n" + "=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)
    
    print(f"\nGames run: {report['successful_games']} / {report['total_games']}")
    print(f"Total steps: {report['total_steps']}")
    
    if report['crashes']:
        print(f"\n❌ CRASHES: {len(report['crashes'])}")
        for crash in report['crashes']:
            if 'game' in crash:
                print(f"  Game {crash['game']}: {crash['type']}: {crash['error'][:100]}")
            else:
                print(f"  {crash['context']}: {crash['error'][:100]}")
    else:
        print(f"\n✅ No crashes")
    
    if report['unknown_cards']:
        print(f"\n❌ UNKNOWN CARD NAMES: {len(report['unknown_cards'])}")
        for name in sorted(report['unknown_cards'])[:10]:
            print(f"  - {name}")
        if len(report['unknown_cards']) > 10:
            print(f"  ... and {len(report['unknown_cards']) - 10} more")
    else:
        print(f"\n✅ All card names resolved successfully")
    
    if report['feature_errors']:
        print(f"\n❌ FEATURE ERRORS: {len(report['feature_errors'])}")
        for err in report['feature_errors'][:5]:
            print(f"  Game {err['game']}, step {err['step']}:")
            for e in err['errors'][:2]:
                print(f"    - {e}")
        if len(report['feature_errors']) > 5:
            print(f"  ... {len(report['feature_errors']) - 5} more")
    else:
        print(f"\n✅ All action features valid")
    
    if report['determinism_checks']:
        print(f"\n⚠️  DETERMINISM ISSUES: {len(report['determinism_checks'])}")
        for check in report['determinism_checks']:
            print(f"  Game {check['game']}: {check['issue']}")
    else:
        print(f"\n✅ Features are deterministic")
    
    print("\n" + "=" * 80)
    
    # Summary
    all_passed = (
        len(report['crashes']) == 0 and
        len(report['unknown_cards']) == 0 and
        len(report['feature_errors']) == 0 and
        len(report['determinism_checks']) == 0
    )
    
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        return 1


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate Step 4 action features across multiple games'
    )
    parser.add_argument(
        '--games', type=int, default=50,
        help='Number of games to run (default: 50)'
    )
    parser.add_argument(
        '--steps', type=int, default=100,
        help='Maximum steps per game (default: 100)'
    )
    
    args = parser.parse_args()
    
    print(f"Starting validation: {args.games} games, up to {args.steps} steps each")
    print(f"Feature schema: K={K} features")
    print(f"Database: {len(load_card_db())} cards\n")
    
    report = run_validation_games(num_games=args.games, max_steps=args.steps)
    exit_code = print_report(report)
    sys.exit(exit_code)
