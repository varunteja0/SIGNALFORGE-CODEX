"""
SignalForge Strategy Factory
==============================
Industrialized hypothesis testing. The core engine.

Pipeline:
    Scanner → Validator → Deployer → Monitor → Killer

    1. Scanner:   Generate + test signal hypotheses against raw data
    2. Validator: OOS split + walk-forward to filter survivors
    3. Deployer:  Paper trade the validated strategies
    4. Monitor:   Track live performance, detect decay
    5. Killer:    Remove dead strategies, trigger new scan
"""
