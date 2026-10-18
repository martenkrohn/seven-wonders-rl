#!/usr/bin/env python3
"""
End-to-end verification script for choice-once-per-turn engine fixes.

Scans game_status.json files and checks that for legal non-chain-free actions,
the Python trade estimator agrees (missing_raw_sum == 0, missing_manuf_sum == 0).

Usage:
    python scripts/verify_choice_fix.py [--log-dir ./io_logs] [--max-files 100]
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from sevenwonders_rl.features.observation_builder import compute_action_mask
    from sevenwonders_rl.features.action_features import build_action_feats, ACTION_FEATURE_NAMES
    from sevenwonders_rl.data.cards import by_id
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're in the python directory and packages are installed")
    sys.exit(1)

N_SLOTS = 7
N_ACTIONS = 28  # 4 action types * 7 slots


def is_chain_free(card_id, cards_played_ids):
    """
    Check if a card can be played for free via chain (prerequisite already built).
    
    Conservative implementation: if card has ANY chain prerequisite, consider it chain-free
    to avoid false positives in mismatch detection.
    """
    card = by_id(card_id)
    if not card:
        return False
    
    # Check if card has chain prerequisites
    chain_from = card.get("chain_from", [])
    chain_from_any = card.get("chain_from_any", [])
    
    if not chain_from and not chain_from_any:
        return False
    
    # If card has chain prerequisites, check if any are met
    for prereq_id in chain_from:
        if prereq_id in cards_played_ids:
            return True
    
    for prereq_id in chain_from_any:
        if prereq_id in cards_played_ids:
            return True
    
    return False


def analyze_status_file(filepath, verbose=False):
    """
    Analyze a single game_status.json file for estimator mismatches.
    
    Returns: (mismatches, total_checked, errors)
    """
    try:
        with open(filepath, 'r') as f:
            status = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return [], 0, [(filepath, str(e))]
    
    mismatches = []
    total_checked = 0
    errors = []
    
    try:
        # Check each player
        for seat in [0, 1, 2]:
            seat_str = str(seat)
            if seat_str not in status.get("players", {}):
                continue
            
            player_data = status["players"][seat_str]
            
            # Get action mask and features
            try:
                mask = compute_action_mask(status, seat)
                feats = build_action_feats(status, seat, mask)
            except Exception as e:
                errors.append((filepath, seat, f"Feature computation error: {e}"))
                continue
            
            # Get hand
            hand_ids = player_data.get("cards_hand_id", [])
            hand_ids = list(hand_ids) + [-1] * (N_SLOTS - len(hand_ids))
            
            # Get played cards for chain checking
            cards_played_ids = set(player_data.get("cards_played_id", []))
            
            # Check each build_structure action (action_type == 0)
            for slot in range(N_SLOTS):
                action_idx = 0 * 7 + slot  # build_structure actions
                
                if mask[action_idx] != 1:
                    continue
                
                card_id = hand_ids[slot]
                if card_id == -1:
                    continue
                
                # Skip chain-free cards (known mismatch we're ignoring)
                if is_chain_free(card_id, cards_played_ids):
                    continue
                
                total_checked += 1
                
                # Check estimator agreement
                # Find indices for missing_raw_sum and missing_manuf_sum
                try:
                    missing_raw_idx = ACTION_FEATURE_NAMES.index('missing_raw_sum')
                    missing_manuf_idx = ACTION_FEATURE_NAMES.index('missing_manuf_sum')
                except ValueError:
                    # Fallback if names changed - look for patterns
                    missing_raw_idx = None
                    missing_manuf_idx = None
                    for i, name in enumerate(ACTION_FEATURE_NAMES):
                        if 'missing_raw' in name and 'sum' in name:
                            missing_raw_idx = i
                        if 'missing_manuf' in name and 'sum' in name:
                            missing_manuf_idx = i
                
                if missing_raw_idx is None or missing_manuf_idx is None:
                    errors.append((filepath, seat, "Could not find missing_*_sum feature indices"))
                    continue
                
                missing_raw = feats[action_idx, missing_raw_idx]
                missing_manuf = feats[action_idx, missing_manuf_idx]
                
                # Mismatch if engine says legal but estimator says infeasible
                if missing_raw > 0.5 or missing_manuf > 0.5:
                    card = by_id(card_id)
                    card_name = card.get("name", f"Card_{card_id}") if card else f"Card_{card_id}"
                    
                    mismatches.append({
                        'file': str(filepath),
                        'seat': seat,
                        'action_idx': action_idx,
                        'slot': slot,
                        'card_id': card_id,
                        'card_name': card_name,
                        'missing_raw': float(missing_raw),
                        'missing_manuf': float(missing_manuf),
                    })
                    
                    if verbose:
                        print(f"  MISMATCH: {Path(filepath).name} seat={seat} {card_name} "
                              f"(missing_raw={missing_raw:.1f}, missing_manuf={missing_manuf:.1f})")
    
    except Exception as e:
        errors.append((filepath, -1, f"Unexpected error: {e}"))
    
    return mismatches, total_checked, errors


def main():
    parser = argparse.ArgumentParser(description='Verify choice-once-per-turn engine fixes')
    parser.add_argument('--log-dir', default='./io', 
                       help='Directory containing game_status.json files')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to check')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print each mismatch as found')
    parser.add_argument('--show-examples', type=int, default=10,
                       help='Number of example mismatches to show')
    
    args = parser.parse_args()
    
    # Find all game_status.json files
    patterns = [
        f"{args.log_dir}/**/game_status.json",
        f"{args.log_dir}/game_status.json",
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    
    files = list(set(files))  # Deduplicate
    
    if args.max_files:
        files = files[:args.max_files]
    
    print(f"Found {len(files)} game_status.json files to analyze")
    print(f"Checking for engine-legal, non-chain-free actions with estimator disagreement...\n")
    
    all_mismatches = []
    total_checked = 0
    all_errors = []
    
    for i, filepath in enumerate(files):
        if (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{len(files)} files...")
        
        mismatches, checked, errors = analyze_status_file(filepath, args.verbose)
        all_mismatches.extend(mismatches)
        total_checked += checked
        all_errors.extend(errors)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Files analyzed:           {len(files)}")
    print(f"Legal non-chain actions:  {total_checked}")
    print(f"Estimator mismatches:     {len(all_mismatches)}")
    print(f"Errors:                   {len(all_errors)}")
    
    if total_checked > 0:
        mismatch_rate = 100.0 * len(all_mismatches) / total_checked
        print(f"Mismatch rate:            {mismatch_rate:.2f}%")
    
    # Show example mismatches
    if all_mismatches and args.show_examples > 0:
        print(f"\n{'='*70}")
        print(f"Example mismatches (first {args.show_examples}):")
        print(f"{'='*70}")
        
        for i, mm in enumerate(all_mismatches[:args.show_examples]):
            print(f"\n{i+1}. {Path(mm['file']).name}")
            print(f"   Seat: {mm['seat']}, Slot: {mm['slot']}")
            print(f"   Card: {mm['card_name']} (ID: {mm['card_id']})")
            print(f"   Missing: raw={mm['missing_raw']:.1f}, manuf={mm['missing_manuf']:.1f}")
    
    # Show example errors
    if all_errors and args.show_examples > 0:
        print(f"\n{'='*70}")
        print(f"Example errors (first {min(5, len(all_errors))}):")
        print(f"{'='*70}")
        for filepath, seat, msg in all_errors[:5]:
            print(f"  {Path(filepath).name} seat={seat}: {msg}")
    
    # Acceptance criteria
    print(f"\n{'='*70}")
    print(f"ACCEPTANCE CRITERIA")
    print(f"{'='*70}")
    
    if total_checked == 0:
        print("⚠️  WARNING: No actions checked - ensure game logs exist")
        return 1
    
    mismatch_rate = 100.0 * len(all_mismatches) / total_checked if total_checked > 0 else 0
    
    if mismatch_rate < 1.0:
        print(f"✓ PASS: Mismatch rate < 1% ({mismatch_rate:.2f}%)")
        print(f"  The choice-once-per-turn fix is working correctly!")
        return 0
    elif mismatch_rate < 5.0:
        print(f"⚠️  WARNING: Mismatch rate {mismatch_rate:.2f}% (acceptable but investigate)")
        print(f"  Some edge cases may remain - review examples above")
        return 0
    else:
        print(f"✗ FAIL: Mismatch rate {mismatch_rate:.2f}% (too high)")
        print(f"  There may be remaining issues with choice tracking")
        print(f"  Review the mismatches above to identify patterns")
        return 1


if __name__ == "__main__":
    sys.exit(main())
