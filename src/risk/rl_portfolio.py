"""
Reinforcement Learning Portfolio Manager
===========================================
Static Kelly criterion = optimal for IID bets.
Real markets are NOT IID. Correlations shift, volatility clusters,
regime transitions change the game.

RL learns optimal position sizing from experience:
  - State: portfolio composition, recent returns, regime, volatility
  - Action: position size adjustments
  - Reward: risk-adjusted returns (Sharpe/Sortino)

This replaces static Kelly with a learned policy that adapts to:
  - Serial correlation in returns
  - Time-varying volatility
  - Regime transitions
  - Drawdown penalties

Implementation: Proximal Policy Optimization (PPO) - the most
reliable RL algorithm, used by OpenAI for everything from robotics
to language models. Pure numpy implementation (no PyTorch needed).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RLState:
    """State representation for the RL agent."""
    # Portfolio state
    current_weights: np.ndarray       # Current allocation weights
    portfolio_value: float = 1.0
    drawdown: float = 0

    # Market features
    recent_returns: np.ndarray = None     # Last N returns per asset
    volatilities: np.ndarray = None       # Current vol estimates
    regime: int = 0                       # 0=normal, 1=trending, 2=volatile, 3=crisis
    regime_confidence: float = 0.5

    # Risk features
    var_95: float = 0
    correlation_mean: float = 0
    kelly_fractions: np.ndarray = None    # Static Kelly for comparison

    def to_vector(self) -> np.ndarray:
        """Flatten state to feature vector."""
        parts = [
            self.current_weights,
            [self.portfolio_value, self.drawdown],
            self.recent_returns.flatten() if self.recent_returns is not None else [0],
            self.volatilities if self.volatilities is not None else [0],
            [self.regime / 3, self.regime_confidence],
            [self.var_95, self.correlation_mean],
            self.kelly_fractions if self.kelly_fractions is not None else [0],
        ]
        return np.concatenate([np.atleast_1d(p).astype(float) for p in parts])


class PPOAgent:
    """Proximal Policy Optimization agent for portfolio management.

    Pure numpy implementation. Uses linear function approximation
    with softmax policy for robustness.
    """

    def __init__(
        self,
        state_dim: int,
        n_assets: int,
        hidden_dim: int = 64,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.2,        # PPO clip ratio
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.state_dim = state_dim
        self.n_assets = n_assets
        self.hidden_dim = hidden_dim
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Policy network: state -> action (portfolio weights)
        # Two-layer network: state -> hidden -> weights
        scale = np.sqrt(2 / state_dim)
        self.W1_policy = np.random.randn(state_dim, hidden_dim) * scale
        self.b1_policy = np.zeros(hidden_dim)
        self.W2_policy = np.random.randn(hidden_dim, n_assets) * np.sqrt(2 / hidden_dim)
        self.b2_policy = np.zeros(n_assets)

        # Log-std for exploration (per-asset)
        self.log_std = np.zeros(n_assets) - 0.5  # Start with moderate exploration

        # Value network: state -> scalar (estimated return)
        self.W1_value = np.random.randn(state_dim, hidden_dim) * scale
        self.b1_value = np.zeros(hidden_dim)
        self.W2_value = np.random.randn(hidden_dim, 1) * np.sqrt(2 / hidden_dim)
        self.b2_value = np.zeros(1)

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        """Numerically stable softmax."""
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward_policy(self, state: np.ndarray) -> tuple:
        """Forward pass through policy network.

        Returns (mean_weights, std) for Gaussian policy.
        """
        h = self._relu(state @ self.W1_policy + self.b1_policy)
        logits = h @ self.W2_policy + self.b2_policy

        # Softmax to ensure weights sum to 1
        mean_weights = self._softmax(logits)
        std = np.exp(self.log_std)

        return mean_weights, std, h  # h for backward pass

    def forward_value(self, state: np.ndarray) -> float:
        """Forward pass through value network."""
        h = self._relu(state @ self.W1_value + self.b1_value)
        value = (h @ self.W2_value + self.b2_value)[0]
        return value

    def act(self, state: np.ndarray) -> tuple:
        """Select action (portfolio weights) from policy.

        Returns (weights, log_prob, value_estimate)
        """
        mean_weights, std, _ = self.forward_policy(state)
        value = self.forward_value(state)

        # Sample from Gaussian, then project to valid weights
        noise = np.random.randn(self.n_assets) * std
        raw_action = mean_weights + noise

        # Project to valid portfolio weights (positive, sum to ≤1)
        weights = np.clip(raw_action, 0, 1)
        weight_sum = weights.sum()
        if weight_sum > 1:
            weights = weights / weight_sum

        # Log probability
        diff = weights - mean_weights
        log_prob = -0.5 * np.sum((diff / (std + 1e-8)) ** 2) - np.sum(np.log(std + 1e-8))

        return weights, log_prob, value

    def act_deterministic(self, state: np.ndarray) -> np.ndarray:
        """Select action without exploration (for deployment)."""
        mean_weights, _, _ = self.forward_policy(state)
        weights = np.clip(mean_weights, 0, 1)
        weight_sum = weights.sum()
        if weight_sum > 1:
            weights = weights / weight_sum
        return weights

    def store_transition(
        self, state, action, reward, log_prob, value, done=False
    ):
        """Store a transition in the experience buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value: float, lam: float = 0.95) -> tuple:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros(len(self.rewards))
        returns = np.zeros(len(self.rewards))

        gae = 0
        next_value = last_value

        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * mask - self.values[t]
            gae = delta + self.gamma * lam * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
            next_value = self.values[t]

        return advantages, returns

    def update(self, n_epochs: int = 4, batch_size: int = 32):
        """PPO update step using SPSA gradient approximation."""
        if len(self.states) < batch_size:
            return {}

        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.log_probs)

        # Compute advantages
        last_value = self.forward_value(states[-1])
        advantages, returns = self.compute_gae(last_value)

        # Normalize advantages
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        total_policy_loss = 0
        total_value_loss = 0

        for epoch in range(n_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), batch_size):
                idx = indices[start:start + batch_size]
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                # SPSA update for policy
                perturbation_scale = 0.01
                delta_policy = {}

                for param_name, param in [
                    ("W1_policy", self.W1_policy),
                    ("b1_policy", self.b1_policy),
                    ("W2_policy", self.W2_policy),
                    ("b2_policy", self.b2_policy),
                    ("log_std", self.log_std),
                ]:
                    delta = np.random.choice([-1, 1], size=param.shape)
                    delta_policy[param_name] = delta

                    # Perturb +
                    param += perturbation_scale * delta
                    loss_plus = self._compute_policy_loss(
                        mb_states, mb_actions, mb_old_log_probs, mb_advantages
                    )
                    param -= perturbation_scale * delta

                    # Perturb -
                    param -= perturbation_scale * delta
                    loss_minus = self._compute_policy_loss(
                        mb_states, mb_actions, mb_old_log_probs, mb_advantages
                    )
                    param += perturbation_scale * delta

                    # SPSA gradient
                    grad = (loss_plus - loss_minus) / (2 * perturbation_scale) * delta
                    grad = np.clip(grad, -self.max_grad_norm, self.max_grad_norm)
                    param -= self.lr_policy * grad

                # SPSA update for value network
                for param_name, param in [
                    ("W1_value", self.W1_value),
                    ("b1_value", self.b1_value),
                    ("W2_value", self.W2_value),
                    ("b2_value", self.b2_value),
                ]:
                    delta = np.random.choice([-1, 1], size=param.shape)

                    param += perturbation_scale * delta
                    vl_plus = self._compute_value_loss(mb_states, mb_returns)
                    param -= perturbation_scale * delta

                    param -= perturbation_scale * delta
                    vl_minus = self._compute_value_loss(mb_states, mb_returns)
                    param += perturbation_scale * delta

                    grad = (vl_plus - vl_minus) / (2 * perturbation_scale) * delta
                    grad = np.clip(grad, -self.max_grad_norm, self.max_grad_norm)
                    param -= self.lr_value * grad

                total_policy_loss += self._compute_policy_loss(
                    mb_states, mb_actions, mb_old_log_probs, mb_advantages
                )
                total_value_loss += self._compute_value_loss(mb_states, mb_returns)

        n_updates = n_epochs * max(1, len(states) // batch_size)
        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
        }

    def _compute_policy_loss(self, states, actions, old_log_probs, advantages):
        """Compute clipped PPO policy loss."""
        total_loss = 0
        for i in range(len(states)):
            mean_w, std, _ = self.forward_policy(states[i])
            diff = actions[i] - mean_w
            new_log_prob = -0.5 * np.sum((diff / (std + 1e-8)) ** 2) - np.sum(np.log(std + 1e-8))

            ratio = np.exp(new_log_prob - old_log_probs[i])
            clipped_ratio = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)

            surr1 = ratio * advantages[i]
            surr2 = clipped_ratio * advantages[i]
            policy_loss = -min(surr1, surr2)

            # Entropy bonus
            entropy = 0.5 * np.sum(np.log(2 * np.pi * np.e * std**2))
            policy_loss -= self.entropy_coef * entropy

            total_loss += policy_loss

        return total_loss / max(1, len(states))

    def _compute_value_loss(self, states, returns):
        """Compute value network MSE loss."""
        total_loss = 0
        for i in range(len(states)):
            value = self.forward_value(states[i])
            total_loss += (value - returns[i]) ** 2
        return total_loss / max(1, len(states))


class RLPortfolioManager:
    """RL-based portfolio manager that replaces static Kelly sizing.

    Wraps the PPO agent with portfolio-specific logic:
    - State construction from market data
    - Reward shaping (Sharpe with drawdown penalty)
    - Training from historical trade sequences
    - Graceful fallback to Kelly when untrained
    """

    def __init__(
        self,
        n_assets: int = 5,
        lookback: int = 20,
        max_drawdown_penalty: float = 2.0,
        min_training_episodes: int = 50,
    ):
        self.n_assets = n_assets
        self.lookback = lookback
        self.dd_penalty = max_drawdown_penalty
        self.min_episodes = min_training_episodes

        self._state_dim = None
        self.agent = None
        self._trained = False
        self._training_episodes = 0

        # Tracking
        self._peak_value = 1.0
        self._portfolio_value = 1.0
        self._current_weights = np.ones(n_assets) / n_assets
        self._return_history = []

    def _build_agent(self, state_dim: int):
        """Lazy initialization of PPO agent."""
        self._state_dim = state_dim
        self.agent = PPOAgent(
            state_dim=state_dim,
            n_assets=self.n_assets,
        )

    def build_state(
        self,
        returns: np.ndarray,          # (lookback, n_assets) recent returns
        volatilities: np.ndarray,     # (n_assets,) current vols
        regime: int = 0,
        regime_confidence: float = 0.5,
        kelly_fractions: np.ndarray = None,
    ) -> RLState:
        """Build RL state from market data."""
        if kelly_fractions is None:
            kelly_fractions = np.ones(self.n_assets) * 0.1

        state = RLState(
            current_weights=self._current_weights.copy(),
            portfolio_value=self._portfolio_value,
            drawdown=(self._peak_value - self._portfolio_value) / max(1e-8, self._peak_value),
            recent_returns=returns,
            volatilities=volatilities,
            regime=regime,
            regime_confidence=regime_confidence,
            var_95=np.percentile(returns.flatten(), 5) if returns.size > 0 else 0,
            correlation_mean=np.mean(np.corrcoef(returns.T)) if returns.shape[0] > 1 else 0,
            kelly_fractions=kelly_fractions,
        )

        return state

    def get_weights(
        self,
        returns: np.ndarray,
        volatilities: np.ndarray,
        regime: int = 0,
        regime_confidence: float = 0.5,
        kelly_fractions: np.ndarray = None,
        explore: bool = False,
    ) -> np.ndarray:
        """Get optimal portfolio weights from RL agent.

        Falls back to Kelly fractions if agent isn't trained enough.
        """
        if kelly_fractions is None:
            kelly_fractions = np.ones(self.n_assets) * 0.1

        state = self.build_state(
            returns, volatilities, regime, regime_confidence, kelly_fractions
        )
        state_vec = state.to_vector()

        # Lazy init
        if self.agent is None:
            self._build_agent(len(state_vec))

        # If not enough training, blend with Kelly
        if not self._trained or self._training_episodes < self.min_episodes:
            if explore and self.agent:
                rl_weights, _, _ = self.agent.act(state_vec)
                # Blend: 80% Kelly, 20% RL during early training
                blend_ratio = min(0.5, self._training_episodes / self.min_episodes)
                weights = (1 - blend_ratio) * kelly_fractions + blend_ratio * rl_weights
            else:
                weights = kelly_fractions
        else:
            if explore:
                weights, _, _ = self.agent.act(state_vec)
            else:
                weights = self.agent.act_deterministic(state_vec)

        # Ensure valid weights
        weights = np.clip(weights, 0, 1)
        if weights.sum() > 1:
            weights = weights / weights.sum()

        self._current_weights = weights
        return weights

    def compute_reward(self, portfolio_return: float) -> float:
        """Compute shaped reward from portfolio return.

        Reward = return - drawdown_penalty * max(0, new_drawdown)
        """
        # Update portfolio value
        self._portfolio_value *= (1 + portfolio_return)
        self._peak_value = max(self._peak_value, self._portfolio_value)
        drawdown = (self._peak_value - self._portfolio_value) / self._peak_value

        # Shaped reward
        reward = portfolio_return

        # Drawdown penalty
        if drawdown > 0.05:  # Penalize drawdowns > 5%
            reward -= self.dd_penalty * drawdown

        # Sharpe-like bonus for consistent returns
        self._return_history.append(portfolio_return)
        if len(self._return_history) >= 20:
            recent = self._return_history[-20:]
            if np.std(recent) > 0:
                sharpe_signal = np.mean(recent) / np.std(recent)
                reward += 0.01 * sharpe_signal

        return reward

    def train_step(
        self,
        returns: np.ndarray,
        volatilities: np.ndarray,
        portfolio_return: float,
        regime: int = 0,
        kelly_fractions: np.ndarray = None,
    ):
        """One training step: observe, act, get reward, store."""
        state = self.build_state(returns, volatilities, regime, kelly_fractions=kelly_fractions)
        state_vec = state.to_vector()

        if self.agent is None:
            self._build_agent(len(state_vec))

        weights, log_prob, value = self.agent.act(state_vec)
        reward = self.compute_reward(portfolio_return)

        self.agent.store_transition(
            state_vec, weights, reward, log_prob, value
        )

        self._current_weights = weights
        self._training_episodes += 1

    def update(self) -> dict:
        """Run PPO update on collected experience."""
        if self.agent is None:
            return {}

        result = self.agent.update()
        if self._training_episodes >= self.min_episodes:
            self._trained = True
            logger.info(f"RL agent trained: {self._training_episodes} episodes")

        return result

    def train_from_history(
        self,
        price_history: pd.DataFrame,
        kelly_fractions: np.ndarray = None,
    ) -> dict:
        """Train on historical price data.

        Args:
            price_history: DataFrame with columns for each asset, rows = periods
            kelly_fractions: Static Kelly fractions for blending
        """
        if price_history.empty or len(price_history) < self.lookback + 10:
            return {"error": "insufficient_data"}

        # Compute returns
        returns_df = price_history.pct_change().dropna()

        n_assets = min(self.n_assets, returns_df.shape[1])
        returns_matrix = returns_df.iloc[:, :n_assets].values

        if kelly_fractions is None:
            kelly_fractions = np.ones(n_assets) * 0.1

        # Reset tracking
        self._portfolio_value = 1.0
        self._peak_value = 1.0
        self._return_history = []
        self.n_assets = n_assets
        self._current_weights = np.ones(n_assets) / n_assets

        # Walk through history
        for t in range(self.lookback, len(returns_matrix)):
            window = returns_matrix[t - self.lookback:t]
            vols = np.std(window, axis=0)

            # Portfolio return from current weights
            portfolio_return = np.dot(self._current_weights[:n_assets], returns_matrix[t, :n_assets])

            self.train_step(
                returns=window,
                volatilities=vols,
                portfolio_return=portfolio_return,
                kelly_fractions=kelly_fractions[:n_assets],
            )

            # Periodically update
            if t % 64 == 0 and len(self.agent.states) >= 32:
                self.agent.update()

        # Final update
        result = self.update()
        result["training_episodes"] = self._training_episodes
        result["final_portfolio_value"] = self._portfolio_value
        result["trained"] = self._trained

        return result

    def get_summary(self) -> dict:
        """Get RL portfolio manager status."""
        return {
            "trained": self._trained,
            "training_episodes": self._training_episodes,
            "portfolio_value": self._portfolio_value,
            "current_weights": self._current_weights.tolist(),
            "peak_value": self._peak_value,
            "drawdown": (self._peak_value - self._portfolio_value) / max(1e-8, self._peak_value),
            "agent_initialized": self.agent is not None,
        }
