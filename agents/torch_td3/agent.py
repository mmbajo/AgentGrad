from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray

from agents.torch_td3.actor import Actor
from agents.torch_td3.critic import Critic
from agents.utils.replay_buffer import ReplayBuffer


class DDPGAgent:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.gamma = config.gamma
        self.device = config.device
        self.target_policy_noise = config.target_policy_noise
        self.target_policy_clip = config.target_policy_clip
        self.action_dim = config.action_dim
        self.action_low = config.action_low
        self.action_high = config.action_high

        # Create networks
        self.actor = Actor(config)
        self.critic = Critic(config)
        self.replay_buffer = ReplayBuffer(config)
        
        self.loss_fn = nn.MSELoss()

    def _update_target_networks(self) -> None:
        self.actor.update()
        self.critic.update()

    def _compute_target_value(
        self, next_state: torch.Tensor, reward: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        
        def target_policy_smoothing(action: torch.Tensor) -> torch.Tensor:
            noise = torch.normal(0, self.target_policy_noise, size=self.action_dim)
            clipped_noise = torch.clamp(noise, -self.target_policy_clip, self.target_policy_clip)
            noised_action = action + clipped_noise
            clipped_noised_action = torch.clamp(noised_action, self.action_low, self.action_high)
            return clipped_noised_action
        
        next_action = target_policy_smoothing(self.actor.actor_target_model(next_state))
        target_q1, target_q2 = self.critic.critic_target_model(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
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
        current_value1, current_value2 = self.critic.critic_model(state, action)
        critic_loss = self.loss_fn(current_value1, target_value) + self.loss_fn(current_value2, target_value)
        return critic_loss

    def _compute_actor_loss(self, state: torch.Tensor) -> torch.Tensor:
        action = self.actor.actor_model(state)
        q1, q2 = self.critic.critic_model(state, action)
        actor_loss = -torch.min(q1, q2).mean()
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

    def get_save_dict(self) -> Dict[str, Any]:
        """Get complete state dict for saving.
        
        Returns:
            Dictionary containing all states needed to restore the agent
        """
        return {
            "actor_model_state": self.actor.actor_model.state_dict(),
            "actor_target_model_state": self.actor.actor_target_model.state_dict(),
            "actor_optimizer_state": self.actor.actor_optimizer.state_dict(),
            "critic_model_state": self.critic.critic_model.state_dict(),
            "critic_target_model_state": self.critic.critic_target_model.state_dict(),
            "critic_optimizer_state": self.critic.critic_optimizer.state_dict(),
        }
    
    def load_save_dict(self, save_dict: Dict[str, Any]) -> None:
        """Load complete state from saved dictionary.
        
        Args:
            save_dict: Dictionary containing saved states
        """
        self.actor.actor_model.load_state_dict(save_dict["actor_model_state"])
        self.actor.actor_target_model.load_state_dict(save_dict["actor_target_model_state"])
        self.actor.actor_optimizer.load_state_dict(save_dict["actor_optimizer_state"])
        self.critic.critic_model.load_state_dict(save_dict["critic_model_state"])
        self.critic.critic_target_model.load_state_dict(save_dict["critic_target_model_state"])
        self.critic.critic_optimizer.load_state_dict(save_dict["critic_optimizer_state"])