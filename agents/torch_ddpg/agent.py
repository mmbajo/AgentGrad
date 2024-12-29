from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray

from agents.ddpg.actor import Actor
from agents.ddpg.critic import Critic
from agents.utils.replay_buffer import ReplayBuffer


class DDPGAgent:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.gamma = config.gamma
        self.device = config.device

        # Create networks
        self.actor = Actor(config)
        self.critic = Critic(config)
        self.replay_buffer = ReplayBuffer(config)

    def _update_target_networks(self) -> None:
        self.actor.update()
        self.critic.update()

    def _compute_target_value(
        self, next_state: torch.Tensor, reward: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        next_action = self.actor.actor_target_model(next_state)
        target_q = self.critic.critic_target_model(next_state, next_action)
        target_value = reward + self.gamma * (1 - done) * target_q
        return target_value.detach()

    def _compute_critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        target_value = self._compute_target_value(next_state, reward, done)
        current_value = self.critic.critic_model(state, action)
        critic_loss = nn.MSELoss()(current_value, target_value)
        return critic_loss

    def _compute_actor_loss(self, state: torch.Tensor) -> torch.Tensor:
        action = self.actor.actor_model(state)
        actor_loss = -self.critic.critic_model(state, action).mean()
        return actor_loss

    def store_experience(
        self,
        state: NDArray[np.float32],
        action: NDArray[np.float32],
        reward: float,
        next_state: NDArray[np.float32],
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def get_action(self, state: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.actor.get_action(state)

    def get_exploration_action(self, state: NDArray[np.float32]) -> NDArray[np.float32]:
        action = self.get_action(state)
        noise = np.random.normal(
            0, self.config.exploration_noise, size=self.config.action_dim
        )
        return np.clip(
            action + noise, self.config.action_low, self.config.action_high
        )  # Actions are typically normalized to [-1, 1]

    def train(self) -> Tuple[float, float]:
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.config["batch_size"]
        )

        # Update critic
        self.critic.critic_optimizer.zero_grad()
        critic_loss = self._compute_critic_loss(state, action, reward, next_state, done)
        critic_loss.backward()
        self.critic.critic_optimizer.step()

        # Update actor
        self.actor.actor_optimizer.zero_grad()
        actor_loss = self._compute_actor_loss(state)
        actor_loss.backward()
        self.actor.actor_optimizer.step()

        # Update target networks
        self._update_target_networks()
        return critic_loss.item(), actor_loss.item()