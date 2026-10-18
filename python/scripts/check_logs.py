#!/usr/bin/env python3
"""
Quick verification that game_status.json files can be loaded.
Run this to ensure the E2E verification script will work once the Python package is set up.
"""

import json
import glob
import sys
from pathlib import Path

def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else './io'
    
    patterns = [
        f"{log_dir}/**/game_status.json",
        f"{log_dir}/game_status.json",
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    
    files = list(set(files))
    
    print(f"Found {len(files)} game_status.json files in {log_dir}")
    
    if not files:
        print("\n⚠️  No game logs found. Run the engine to generate logs first:")
        print("   cd build && ./7Wonders")
        return 1
    
    # Try loading first file
    try:
        with open(files[0], 'r') as f:
            status = json.load(f)
        
        print(f"\n✓ Successfully loaded: {Path(files[0]).name}")
        print(f"  Players: {len(status.get('players', {}))}")
        print(f"  Era: {status.get('era', 'N/A')}")
        print(f"  Turn: {status.get('turn', 'N/A')}")
        
        # Check structure
        for seat in range(3):
            seat_str = str(seat)
            if seat_str in status.get("players", {}):
                p = status["players"][seat_str]
                hand_size = len(p.get("cards_hand_id", []))
                played_size = len(p.get("cards_played_id", []))
                print(f"  Player {seat}: {hand_size} in hand, {played_size} played")
        
        print(f"\n✓ Game logs are ready for verification")
        print(f"\nNext step:")
        print(f"  1. Set up Python environment (if not done):")
        print(f"     cd python && pip install -e .")
        print(f"  2. Run full verification:")
        print(f"     python scripts/verify_choice_fix.py --log-dir {log_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error loading game log: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
