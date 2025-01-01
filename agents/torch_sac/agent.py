from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray

from agents.torch_sac.actor import Actor
from agents.torch_sac.critic import Critic
from agents.utils.replay_buffer import ReplayBuffer


class SACAgent:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.gamma = config.gamma
        self.device = config.device
        self.action_dim = config.action_dim
        self.action_low = config.action_low
        self.action_high = config.action_high

        # SAC specific parameters
        self.auto_tune_alpha = config.get("auto_tune_alpha", True)
        self.initial_alpha = config.get("initial_alpha", 1.0)
        
        # Set target entropy to -action_dim if not specified
        target_entropy = config.get("target_entropy", None)
        self.target_entropy = float(-self.action_dim) if target_entropy is None else float(target_entropy)

        # Ablation study parameters
        self.use_double_q = config.get("use_double_q", True)

        # Create networks
        self.actor = Actor(config)
        self.critic = Critic(config)
        self.replay_buffer = ReplayBuffer(config)

        # Initialize temperature parameter
        self.log_alpha = torch.tensor(np.log(self.initial_alpha), requires_grad=True, device=self.device)
        if self.auto_tune_alpha:
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        
        self.loss_fn = nn.MSELoss()
        self.global_step = 0

    @property
    def alpha(self) -> torch.Tensor:
        """Get current temperature parameter."""
        return self.log_alpha.exp()

    def _update_target_networks(self) -> None:
        self.critic.update()

    def _compute_target_value(
        self, next_state: torch.Tensor, reward: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            # Sample action from policy
            next_action, next_log_prob, _ = self.actor.get_action(next_state)

            # Get Q values from target critics
            target_q1, target_q2 = self.critic.critic_target_model(next_state, next_action)

            # Use minimum Q value if double Q is enabled, otherwise use Q1
            if self.use_double_q:
                target_q = torch.min(target_q1, target_q2)
            else:
                target_q = target_q1

            # Add entropy term
            target_value = reward + self.gamma * (1 - done) * (target_q - self.alpha * next_log_prob)

        return target_value

    def _compute_critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        target_value = self._compute_target_value(next_state, reward, done)
        current_q1, current_q2 = self.critic.critic_model(state, action)

        # Compute loss based on whether double Q is enabled
        if self.use_double_q:
            critic_loss = self.loss_fn(current_q1, target_value) + self.loss_fn(current_q2, target_value)
        else:
            critic_loss = self.loss_fn(current_q1, target_value)

        return critic_loss

    def _compute_actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action, log_prob, _ = self.actor.get_action(state)
        q1, q2 = self.critic.critic_model(state, action)
        
        # Use minimum Q value for policy update
        min_q = torch.min(q1, q2) if self.use_double_q else q1
        
        # Policy loss is expectation of Q-value minus entropy
        actor_loss = (self.alpha * log_prob - min_q).mean()
        
        return actor_loss, log_prob

    def _compute_temperature_loss(self, log_prob: torch.Tensor) -> torch.Tensor:
        """Compute loss for temperature parameter."""
        # Ensure target entropy is a tensor on the correct device
        target_entropy = torch.tensor(self.target_entropy, device=self.device)
        return -(self.log_alpha * (log_prob.detach() + target_entropy)).mean()

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
        """Get deterministic action for evaluation."""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            action, _, _ = self.actor.get_action(state)
            return action.cpu().numpy()

    def get_exploration_action(self, state: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get stochastic action for training."""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            action, _, _ = self.actor.get_action(state)
            return action.cpu().numpy()

    def train(self) -> Tuple[float, float]:
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.config.batch_size
        )

        # Update critic
        self.critic.critic_optimizer.zero_grad()
        critic_loss = self._compute_critic_loss(state, action, reward, next_state, done)
        critic_loss.backward()
        self.critic.critic_optimizer.step()

        # Update actor
        self.actor.actor_optimizer.zero_grad()
        actor_loss, log_prob = self._compute_actor_loss(state)
        actor_loss.backward()
        self.actor.actor_optimizer.step()

        # Update temperature
        if self.auto_tune_alpha:
            self.alpha_optimizer.zero_grad()
            temp_loss = self._compute_temperature_loss(log_prob)
            temp_loss.backward()
            self.alpha_optimizer.step()

        # Update target networks
        self._update_target_networks()

        self.global_step += 1
        return critic_loss.item(), actor_loss.item()

    def get_save_dict(self) -> Dict[str, Any]:
        """Get complete state dict for saving."""
        save_dict = {
            "actor_model_state": self.actor.actor_model.state_dict(),
            "critic_model_state": self.critic.critic_model.state_dict(),
            "critic_target_model_state": self.critic.critic_target_model.state_dict(),
            "actor_optimizer_state": self.actor.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.critic.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.data,
            "global_step": self.global_step,
        }
        
        if self.auto_tune_alpha:
            save_dict["alpha_optimizer_state"] = self.alpha_optimizer.state_dict()
            
        return save_dict

    def load_save_dict(self, save_dict: Dict[str, Any]) -> None:
        """Load complete state from saved dictionary."""
        self.actor.actor_model.load_state_dict(save_dict["actor_model_state"])
        self.critic.critic_model.load_state_dict(save_dict["critic_model_state"])
        self.critic.critic_target_model.load_state_dict(save_dict["critic_target_model_state"])
        self.actor.actor_optimizer.load_state_dict(save_dict["actor_optimizer_state"])
        self.critic.critic_optimizer.load_state_dict(save_dict["critic_optimizer_state"])
        self.log_alpha.data = save_dict["log_alpha"]
        self.global_step = save_dict.get("global_step", 0)
        
        if self.auto_tune_alpha and "alpha_optimizer_state" in save_dict:
            self.alpha_optimizer.load_state_dict(save_dict["alpha_optimizer_state"])
