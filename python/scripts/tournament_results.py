#!/usr/bin/env python3
"""Display tournament results from the round-robin."""

print("=" * 80)
print("ROUND-ROBIN TOURNAMENT RESULTS (50 games per matchup)")
print("=" * 80)
print()

# Results from the tournament
results = {
    "Baseline": {
        "vs Military": (90.0, 38.5),
        "vs MadScientist": (88.0, 40.4),
        "overall": (89.0, 39.5),
    },
    "Military": {
        "vs Baseline": (78.0, 30.6),
        "vs MadScientist": (74.0, 28.7),
        "overall": (76.0, 29.6),
    },
    "MadScientist": {
        "vs Baseline": (96.0, 37.2),
        "vs Military": (90.0, 37.4),
        "overall": (93.0, 37.3),
    },
}

print("DETAILED MATCHUP MATRIX")
print("-" * 80)
print(f"{'Policy':<15} | {'vs Baseline':<15} | {'vs Military':<15} | {'vs MadScientist':<15}")
print("-" * 80)

for policy in ["Baseline", "Military", "MadScientist"]:
    row = f"{policy:<15} | "
    
    for opponent in ["Baseline", "Military", "MadScientist"]:
        if policy == opponent:
            row += f"{'---':<15} | "
        else:
            matchup_key = f"vs {opponent}"
            if matchup_key in results[policy]:
                win_rate, avg_vp = results[policy][matchup_key]
                row += f"{win_rate:5.1f}% ({avg_vp:4.1f}VP) | "
            else:
                row += f"{'N/A':<15} | "
    
    print(row.rstrip(" | "))

print()
print("=" * 80)
print("FINAL RANKING")
print("=" * 80)

ranking = [
    ("MadScientist", 93.0, 37.3),
    ("Baseline", 89.0, 39.5),
    ("Military", 76.0, 29.6),
]

for rank, (name, win_rate, avg_vp) in enumerate(ranking, 1):
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
    print(f"{medal} {rank}. {name:<15}  Win Rate: {win_rate:5.1f}%  Avg VP: {avg_vp:4.1f}")

print()
print("=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("✓ MadScientist DOMINATES with 93% overall win rate")
print("  - 96% vs Baseline (hyper-aggressive science beats balanced VP)")
print("  - 90% vs Military (science scales better than military)")
print()
print("✓ Baseline is second with 89% overall win rate")
print("  - Strong production + VP-equivalent scoring")
print("  - Highest avg VP (39.5) but loses to MadScientist")
print()
print("✓ Military is weakest at 76% overall")
print("  - Future military value not enough vs strong opponents")
print("  - Lowest avg VP (29.6)")
print()
print("STRATEGY DIVERSITY: ✓ All 3 policies play distinctly different styles")
print("=" * 80)
