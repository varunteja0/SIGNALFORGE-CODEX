"""
Liquidation Oracle — DeFi Liquidation Prediction Engine
========================================================
Predicts forced-seller cascades before they happen by mapping
all visible leveraged positions across DeFi protocols.
"""

from src.liquidation.protocols import (
    LeveragedPosition,
    ProtocolSnapshot,
    ProtocolAdapter,
    AaveV3Adapter,
    CompoundV3Adapter,
    SyntheticPositionGenerator,
)
from src.liquidation.cascade import (
    CascadeSimulator,
    CascadeResult,
    LiquidationWave,
)
from src.liquidation.oracle import (
    LiquidationOracle,
    LiquidationRiskScore,
    LiquidationSignal,
)

__all__ = [
    "LeveragedPosition", "ProtocolSnapshot", "ProtocolAdapter",
    "AaveV3Adapter", "CompoundV3Adapter", "SyntheticPositionGenerator",
    "CascadeSimulator", "CascadeResult", "LiquidationWave",
    "LiquidationOracle", "LiquidationRiskScore", "LiquidationSignal",
]
