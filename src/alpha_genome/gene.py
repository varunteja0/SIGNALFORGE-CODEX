"""
Alpha Genome — Expression Tree (Gene) Representation
=====================================================
The DNA of trading strategies. Each gene is a mathematical expression
tree that transforms market features into trading signals.

Every tree evaluates to a pd.Series of floats. The sign determines
the signal direction: positive = buy, negative = sell, zero = neutral.

Design principles:
    1. Protected operations — no div-by-zero, no NaN/Inf propagation
    2. Vectorized evaluation — full DataFrame operations, no row loops
    3. Depth-limited — prevents bloat (the #1 GP problem)
    4. Serializable — save/load evolved strategies as JSON
"""

import copy
import json
import hashlib
import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Available market features (produced by DataEngine.compute_features)
# Legacy list kept for backward compatibility
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "open", "high", "low", "close", "volume",
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_50",
    "vol_10", "vol_20", "vol_50",
    "vol_ratio_10", "vol_ratio_20", "vol_ratio_50",
    "price_vs_ma_10", "price_vs_ma_20", "price_vs_ma_50",
    "price_vs_ma_100", "price_vs_ma_200",
    "rsi_7", "rsi_14", "rsi_21",
    "macd", "macd_signal", "macd_hist",
    "bb_pct_20",
    "atr_pct_14", "atr_pct_21",
    "bar_position",
]

# Structural features produced by StructuralDataFetcher.fetch_all()
# These use the actual prefixed column names from structural.py
STRUCTURAL_FEATURE_NAMES = [
    # Funding (prefix: fund_)
    "fund_funding_rate", "fund_funding_zscore",
    "fund_funding_annualized", "fund_funding_ma_7d", "fund_funding_ma_30d",
    # Open Interest (prefix: oi_)
    "oi_oi_change_1h", "oi_oi_change_4h", "oi_oi_change_24h", "oi_oi_zscore",
    # Long/Short Ratio (prefix: lsr_)
    "lsr_long_short_ratio", "lsr_lsr_zscore",
    # Taker Volume (prefix: taker_)
    "taker_buy_sell_ratio", "taker_taker_imbalance",
    # Composite (no prefix)
    "leverage_heat", "liq_pressure", "smart_money_divergence",
]

# Multi-venue features produced by MultiVenueFetcher (optional — evaluate to 0 if missing)
MULTI_VENUE_FEATURE_NAMES = [
    # Top trader vs retail divergence
    "top_trader_ls_ratio", "top_retail_divergence",
    # Cross-venue funding
    "cross_venue_funding_spread", "cross_venue_funding_zscore",
    # Crowding + cascade scores (from intelligence layer)
    "crowding_score", "cascade_probability",
]

# Advanced features from compute_all_features() — 120+ features
# Import dynamically to avoid circular imports
try:
    from src.data.features import ADVANCED_FEATURE_NAMES
except ImportError:
    ADVANCED_FEATURE_NAMES = FEATURE_NAMES

ALL_FEATURE_NAMES = list(dict.fromkeys(
    ADVANCED_FEATURE_NAMES + STRUCTURAL_FEATURE_NAMES + MULTI_VENUE_FEATURE_NAMES
))

# Operations
UNARY_OPS = ["neg", "abs", "sin", "tanh", "sqrt_p", "log_p", "sigmoid"]
BINARY_OPS = ["add", "sub", "mul", "div_p", "max", "min"]
COMPARISON_OPS = ["gt", "lt"]
TS_OPS = ["ts_mean", "ts_std", "ts_max", "ts_min", "delay"]
TS_WINDOWS = [3, 5, 10, 20]

# ---------------------------------------------------------------------------
# Protected numeric helpers
# ---------------------------------------------------------------------------

def _safe(s: pd.Series) -> pd.Series:
    """Replace inf/nan with 0 — prevents NaN propagation through the tree."""
    return s.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _protected_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return _safe(a / b.where(b.abs() > 1e-10, 1.0))


def _protected_log(a: pd.Series) -> pd.Series:
    return _safe(np.log(a.clip(lower=1e-10)))


def _protected_sqrt(a: pd.Series) -> pd.Series:
    return _safe(np.sqrt(a.abs()))


def _sigmoid(a: pd.Series) -> pd.Series:
    clipped = a.clip(-20, 20)
    return _safe(1.0 / (1.0 + np.exp(-clipped)))


# ---------------------------------------------------------------------------
# Node base class
# ---------------------------------------------------------------------------

class Node(ABC):
    """Abstract base for all expression tree nodes."""

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate this node on a DataFrame, returning a Series."""

    @abstractmethod
    def depth(self) -> int:
        """Maximum depth of this subtree."""

    @abstractmethod
    def size(self) -> int:
        """Total number of nodes in this subtree."""

    @abstractmethod
    def children(self) -> list["Node"]:
        """Direct child nodes."""

    @abstractmethod
    def clone(self) -> "Node":
        """Deep copy."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""

    def __repr__(self):
        return tree_to_formula(self)


# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------

class FeatureNode(Node):
    """Leaf: references a column in the market data DataFrame."""

    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.feature_name in df.columns:
            return _safe(df[self.feature_name].astype(float))
        return pd.Series(0.0, index=df.index)

    def depth(self) -> int:
        return 0

    def size(self) -> int:
        return 1

    def children(self) -> list[Node]:
        return []

    def clone(self) -> "FeatureNode":
        return FeatureNode(self.feature_name)

    def to_dict(self) -> dict:
        return {"type": "feature", "name": self.feature_name}


class ConstantNode(Node):
    """Leaf: a fixed numeric constant in [-1, 1]."""

    def __init__(self, value: float):
        self.value = round(float(value), 4)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index)

    def depth(self) -> int:
        return 0

    def size(self) -> int:
        return 1

    def children(self) -> list[Node]:
        return []

    def clone(self) -> "ConstantNode":
        return ConstantNode(self.value)

    def to_dict(self) -> dict:
        return {"type": "constant", "value": self.value}


# ---------------------------------------------------------------------------
# Internal nodes
# ---------------------------------------------------------------------------

class UnaryNode(Node):
    """Applies a unary operation to a single child."""

    def __init__(self, op: str, child: Node):
        assert op in UNARY_OPS, f"Unknown unary op: {op}"
        self.op = op
        self.child = child

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        a = self.child.evaluate(df)
        if self.op == "neg":
            return _safe(-a)
        elif self.op == "abs":
            return _safe(a.abs())
        elif self.op == "sin":
            return _safe(np.sin(a.clip(-100, 100)))
        elif self.op == "tanh":
            return _safe(np.tanh(a))
        elif self.op == "sqrt_p":
            return _protected_sqrt(a)
        elif self.op == "log_p":
            return _protected_log(a)
        elif self.op == "sigmoid":
            return _sigmoid(a)
        return a

    def depth(self) -> int:
        return 1 + self.child.depth()

    def size(self) -> int:
        return 1 + self.child.size()

    def children(self) -> list[Node]:
        return [self.child]

    def clone(self) -> "UnaryNode":
        return UnaryNode(self.op, self.child.clone())

    def to_dict(self) -> dict:
        return {"type": "unary", "op": self.op, "child": self.child.to_dict()}


class BinaryNode(Node):
    """Applies a binary operation to two children."""

    def __init__(self, op: str, left: Node, right: Node):
        assert op in BINARY_OPS, f"Unknown binary op: {op}"
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        a = self.left.evaluate(df)
        b = self.right.evaluate(df)
        if self.op == "add":
            return _safe(a + b)
        elif self.op == "sub":
            return _safe(a - b)
        elif self.op == "mul":
            return _safe(a * b)
        elif self.op == "div_p":
            return _protected_div(a, b)
        elif self.op == "max":
            return _safe(pd.concat([a, b], axis=1).max(axis=1))
        elif self.op == "min":
            return _safe(pd.concat([a, b], axis=1).min(axis=1))
        return a

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def children(self) -> list[Node]:
        return [self.left, self.right]

    def clone(self) -> "BinaryNode":
        return BinaryNode(self.op, self.left.clone(), self.right.clone())

    def to_dict(self) -> dict:
        return {
            "type": "binary", "op": self.op,
            "left": self.left.to_dict(), "right": self.right.to_dict(),
        }


class ComparisonNode(Node):
    """Compares two children, outputting 1.0 (true) or -1.0 (false).
    This is how expression trees produce directional signals."""

    def __init__(self, op: str, left: Node, right: Node):
        assert op in COMPARISON_OPS, f"Unknown comparison op: {op}"
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        a = self.left.evaluate(df)
        b = self.right.evaluate(df)
        if self.op == "gt":
            return pd.Series(np.where(a > b, 1.0, -1.0), index=df.index)
        elif self.op == "lt":
            return pd.Series(np.where(a < b, 1.0, -1.0), index=df.index)
        return pd.Series(0.0, index=df.index)

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def children(self) -> list[Node]:
        return [self.left, self.right]

    def clone(self) -> "ComparisonNode":
        return ComparisonNode(self.op, self.left.clone(), self.right.clone())

    def to_dict(self) -> dict:
        return {
            "type": "comparison", "op": self.op,
            "left": self.left.to_dict(), "right": self.right.to_dict(),
        }


class TimeSeriesNode(Node):
    """Applies a rolling time-series operation to a child.
    This is key — it lets the tree reason about *temporal* patterns."""

    def __init__(self, op: str, child: Node, window: int):
        assert op in TS_OPS, f"Unknown ts op: {op}"
        self.op = op
        self.child = child
        self.window = max(2, window)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        a = self.child.evaluate(df)
        if self.op == "ts_mean":
            return _safe(a.rolling(self.window, min_periods=1).mean())
        elif self.op == "ts_std":
            return _safe(a.rolling(self.window, min_periods=2).std())
        elif self.op == "ts_max":
            return _safe(a.rolling(self.window, min_periods=1).max())
        elif self.op == "ts_min":
            return _safe(a.rolling(self.window, min_periods=1).min())
        elif self.op == "delay":
            return _safe(a.shift(self.window))
        return a

    def depth(self) -> int:
        return 1 + self.child.depth()

    def size(self) -> int:
        return 1 + self.child.size()

    def children(self) -> list[Node]:
        return [self.child]

    def clone(self) -> "TimeSeriesNode":
        return TimeSeriesNode(self.op, self.child.clone(), self.window)

    def to_dict(self) -> dict:
        return {
            "type": "timeseries", "op": self.op,
            "window": self.window, "child": self.child.to_dict(),
        }


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def _random_leaf(features: list[str], rng: random.Random) -> Node:
    """Create a random leaf node."""
    if rng.random() < 0.7:
        return FeatureNode(rng.choice(features))
    return ConstantNode(round(rng.uniform(-1, 1), 4))


def _random_expression(
    max_depth: int, features: list[str], rng: random.Random,
) -> Node:
    """Recursively build a random expression subtree (no comparison at root)."""
    if max_depth <= 0:
        return _random_leaf(features, rng)

    roll = rng.random()

    if roll < 0.25:
        # Leaf (early termination prevents bloat)
        return _random_leaf(features, rng)
    elif roll < 0.45:
        # Unary
        op = rng.choice(UNARY_OPS)
        child = _random_expression(max_depth - 1, features, rng)
        return UnaryNode(op, child)
    elif roll < 0.75:
        # Binary
        op = rng.choice(BINARY_OPS)
        left = _random_expression(max_depth - 1, features, rng)
        right = _random_expression(max_depth - 1, features, rng)
        return BinaryNode(op, left, right)
    else:
        # Time series
        op = rng.choice(TS_OPS)
        window = rng.choice(TS_WINDOWS)
        child = _random_expression(max_depth - 1, features, rng)
        return TimeSeriesNode(op, child, window)


def random_tree(
    max_depth: int = 5,
    features: Optional[list[str]] = None,
    seed: Optional[int] = None,
) -> Node:
    """Generate a complete random signal tree.

    70% of trees use a ComparisonNode root producing directional
    signals (+1 / -1).  30% use a continuous expression root
    (the FitnessEvaluator discretises via z-score thresholds).
    This diversity lets evolution explore both crisp rule-based
    and smooth continuous alpha signals.
    """
    features = features or FEATURE_NAMES
    rng = random.Random(seed)

    if rng.random() < 0.7:
        # Comparison root → discrete {-1, +1}
        op = rng.choice(COMPARISON_OPS)
        left = _random_expression(max_depth - 1, features, rng)
        right = _random_expression(max_depth - 1, features, rng)
        return ComparisonNode(op, left, right)
    else:
        # Continuous expression root (tanh keeps output bounded)
        inner = _random_expression(max_depth - 1, features, rng)
        return UnaryNode("tanh", inner)


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _collect_nodes(tree: Node) -> list[tuple[Node, Optional[Node], str]]:
    """Collect all (node, parent, attr_name) triples via BFS."""
    result = [(tree, None, "")]
    queue = [(tree, None, "")]
    while queue:
        node, parent, attr = queue.pop(0)
        if isinstance(node, UnaryNode):
            result.append((node.child, node, "child"))
            queue.append((node.child, node, "child"))
        elif isinstance(node, TimeSeriesNode):
            result.append((node.child, node, "child"))
            queue.append((node.child, node, "child"))
        elif isinstance(node, (BinaryNode, ComparisonNode)):
            result.append((node.left, node, "left"))
            result.append((node.right, node, "right"))
            queue.append((node.left, node, "left"))
            queue.append((node.right, node, "right"))
    return result


def crossover(parent1: Node, parent2: Node) -> tuple[Node, Node]:
    """Subtree crossover: swap random subtrees between two parents.

    Returns two new offspring without mutating parents.
    """
    child1 = parent1.clone()
    child2 = parent2.clone()

    nodes1 = _collect_nodes(child1)
    nodes2 = _collect_nodes(child2)

    # Pick random non-root nodes to swap
    candidates1 = [(n, p, a) for n, p, a in nodes1 if p is not None]
    candidates2 = [(n, p, a) for n, p, a in nodes2 if p is not None]

    if not candidates1 or not candidates2:
        return child1, child2

    _, p1, a1 = random.choice(candidates1)
    n2, p2, a2 = random.choice(candidates2)

    # Get the subtree from child1 at the swap point
    swapped_from_1 = getattr(p1, a1).clone()

    # Replace in child1 with subtree from child2
    setattr(p1, a1, n2.clone())
    # Replace in child2 with subtree from child1
    setattr(p2, a2, swapped_from_1)

    # Enforce max depth — if too deep, return parents unchanged
    if child1.depth() > 8 or child2.depth() > 8:
        return parent1.clone(), parent2.clone()

    return child1, child2


def mutate(
    tree: Node,
    features: Optional[list[str]] = None,
    mutation_rate: float = 0.15,
) -> Node:
    """Apply random mutations to a tree.

    Mutation types:
    - Replace a subtree with a new random subtree
    - Change a feature reference
    - Change a constant value
    - Change an operation
    - Insert/remove a unary wrapper
    """
    features = features or FEATURE_NAMES
    rng = random.Random()
    result = tree.clone()

    nodes = _collect_nodes(result)

    for node, parent, attr in nodes:
        if rng.random() > mutation_rate:
            continue

        if parent is None:
            # Don't mutate root comparison — it defines signal direction
            continue

        roll = rng.random()

        if roll < 0.2 and isinstance(node, FeatureNode):
            # Change feature
            new_node = FeatureNode(rng.choice(features))
            setattr(parent, attr, new_node)

        elif roll < 0.35 and isinstance(node, ConstantNode):
            # Perturb constant
            new_val = node.value + rng.gauss(0, 0.2)
            new_val = max(-2, min(2, new_val))
            setattr(parent, attr, ConstantNode(new_val))

        elif roll < 0.5 and isinstance(node, (UnaryNode, TimeSeriesNode)):
            # Change operation
            if isinstance(node, UnaryNode):
                node.op = rng.choice(UNARY_OPS)
            else:
                node.op = rng.choice(TS_OPS)
                node.window = rng.choice(TS_WINDOWS)

        elif roll < 0.65 and isinstance(node, BinaryNode):
            # Change binary op
            node.op = rng.choice(BINARY_OPS)

        elif roll < 0.8:
            # Wrap in a unary or time-series node
            if node.depth() < 6:
                if rng.random() < 0.6:
                    wrapped = UnaryNode(rng.choice(UNARY_OPS), node.clone())
                else:
                    wrapped = TimeSeriesNode(
                        rng.choice(TS_OPS), node.clone(), rng.choice(TS_WINDOWS)
                    )
                setattr(parent, attr, wrapped)

        else:
            # Replace with new random subtree
            max_d = max(1, 3 - node.depth())
            new_subtree = _random_expression(max_d, features, rng)
            setattr(parent, attr, new_subtree)

    # Enforce max depth
    if result.depth() > 8:
        return tree.clone()

    return result


# ---------------------------------------------------------------------------
# Human-readable formula
# ---------------------------------------------------------------------------

def tree_to_formula(node: Node) -> str:
    """Convert a tree to a human-readable mathematical formula."""
    if isinstance(node, FeatureNode):
        return node.feature_name
    elif isinstance(node, ConstantNode):
        return str(node.value)
    elif isinstance(node, UnaryNode):
        child_str = tree_to_formula(node.child)
        return f"{node.op}({child_str})"
    elif isinstance(node, BinaryNode):
        l = tree_to_formula(node.left)
        r = tree_to_formula(node.right)
        op_sym = {
            "add": "+", "sub": "-", "mul": "*", "div_p": "/",
            "max": "max", "min": "min",
        }
        sym = op_sym.get(node.op, node.op)
        if sym in ("+", "-", "*", "/"):
            return f"({l} {sym} {r})"
        return f"{sym}({l}, {r})"
    elif isinstance(node, ComparisonNode):
        l = tree_to_formula(node.left)
        r = tree_to_formula(node.right)
        sym = ">" if node.op == "gt" else "<"
        return f"({l} {sym} {r})"
    elif isinstance(node, TimeSeriesNode):
        child_str = tree_to_formula(node.child)
        return f"{node.op}({child_str}, {node.window})"
    return "?"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def tree_to_dict(node: Node) -> dict:
    """Serialize a tree to a JSON-safe dictionary."""
    return node.to_dict()


def tree_from_dict(d: dict) -> Node:
    """Deserialize a tree from a dictionary."""
    t = d["type"]
    if t == "feature":
        return FeatureNode(d["name"])
    elif t == "constant":
        return ConstantNode(d["value"])
    elif t == "unary":
        child = tree_from_dict(d["child"])
        return UnaryNode(d["op"], child)
    elif t == "binary":
        left = tree_from_dict(d["left"])
        right = tree_from_dict(d["right"])
        return BinaryNode(d["op"], left, right)
    elif t == "comparison":
        left = tree_from_dict(d["left"])
        right = tree_from_dict(d["right"])
        return ComparisonNode(d["op"], left, right)
    elif t == "timeseries":
        child = tree_from_dict(d["child"])
        return TimeSeriesNode(d["op"], child, d["window"])
    raise ValueError(f"Unknown node type: {t}")


def tree_hash(node: Node) -> str:
    """Deterministic hash for a tree structure."""
    formula = tree_to_formula(node)
    return hashlib.sha256(formula.encode()).hexdigest()[:16]
