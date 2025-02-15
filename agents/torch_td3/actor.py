from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig


class ActorModel(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super(ActorModel, self).__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_dim  # Default hidden dimension
        self.device = config.device
        self.action_high = config.action_high
        self.action_low = config.action_low

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Tanh(),
        )

        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        x = self.net(state)
        # Scale from [-1, 1] to [low, high]
        return 0.5 * (x + 1.0) * (self.action_high - self.action_low) + self.action_low

    def get_action(self, state: NDArray[np.float32]) -> NDArray[np.float32]:
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            action = self.forward(state)
            return action.cpu().numpy()


class Actor:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.device = config.device
        self.actor_model = ActorModel(config)
        self.actor_target_model = ActorModel(config)
        self.actor_target_model.load_state_dict(self.actor_model.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor_model.parameters(), lr=config.actor_lr
        )
        self.tau = config.tau

    def get_action(self, state: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.actor_model.get_action(state)

    def update(self) -> None:
        for param, target_param in zip(
            self.actor_model.parameters(), self.actor_target_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
