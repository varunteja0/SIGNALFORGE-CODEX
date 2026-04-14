"""
Autonomous Quant Engine
========================
Self-improving trading system that discovers, validates, deploys,
and monitors strategies continuously.

Components:
    StrategyFactory   — Generate strategy candidates from templates
    StrategyRanker    — Backtest, score, and rank candidates
    StrategyAllocator — Portfolio-level capital allocation
    AutonomousEngine  — Orchestrator that runs the full loop
"""

from src.engine.strategy_factory import StrategyFactory, StrategyCandidate
from src.engine.ranker import StrategyRanker, ScoredStrategy
from src.engine.allocator import StrategyAllocator
from src.engine.autonomous import AutonomousEngine
