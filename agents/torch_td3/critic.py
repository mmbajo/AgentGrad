import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from numpy.typing import NDArray
from typing import Tuple
import numpy as np


class CriticModel(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super(CriticModel, self).__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_dim
        self.device = config.device
        self.use_double_q = config.get(
            "use_double_q", True
        )  # Default to True for backward compatibility

        # First Q-network
        self.net1 = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        # Second Q-network (only if double Q is enabled)
        if self.use_double_q:
            self.net2 = nn.Sequential(
                nn.Linear(self.state_dim + self.action_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )

        self.to(self.device)

    def _format(
        self,
        state: torch.Tensor | NDArray[np.float32],
        action: torch.Tensor | NDArray[np.float32],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)

        state = state.to(self.device)
        action = action.to(self.device)
        return state, action

    def forward(
        self,
        state: torch.Tensor | NDArray[np.float32],
        action: torch.Tensor | NDArray[np.float32],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state, action = self._format(state, action)
        state_action = torch.cat([state, action], dim=-1)
        q1 = self.net1(state_action)

        if self.use_double_q:
            q2 = self.net2(state_action)
            return q1, q2
        else:
            # Return same Q value twice when double Q is disabled
            return q1, q1

    def get_value(
        self,
        state: torch.Tensor | NDArray[np.float32],
        action: torch.Tensor | NDArray[np.float32],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state, action = self._format(state, action)
        return self.forward(state, action)


class Critic:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.device = config.device
        self.tau = config.tau
        self.use_double_q = config.get(
            "use_double_q", True
        )  # Default to True for backward compatibility

        self.critic_model = CriticModel(config)
        self.critic_target_model = CriticModel(config)
        self.critic_target_model.load_state_dict(self.critic_model.state_dict())

        # Only optimize parameters of active networks
        if self.use_double_q:
            params = self.critic_model.parameters()
        else:
            params = self.critic_model.net1.parameters()

        self.critic_optimizer = optim.Adam(params, lr=config.critic_lr)

    def update(self) -> None:
        for param, target_param in zip(
            self.critic_model.parameters(), self.critic_target_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
