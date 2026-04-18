"""Autonomous research utilities under :mod:`src.research`."""
from .autoloop import (
    AutoLoopResult,
    Candidate,
    Gates,
    Hypothesis,
    compile_signal,
    deflated_sharpe,
    generate_hypotheses,
    run_auto_loop,
    synthesize_features,
)

__all__ = [
    "Hypothesis",
    "Candidate",
    "Gates",
    "AutoLoopResult",
    "synthesize_features",
    "generate_hypotheses",
    "compile_signal",
    "deflated_sharpe",
    "run_auto_loop",
]
