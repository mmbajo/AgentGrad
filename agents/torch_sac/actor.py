from typing import Dict, Any, Tuple
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
        self.hidden_dim = config.hidden_dim
        self.device = config.device
        self.action_high = config.action_high
        self.action_low = config.action_low
        self.epsilon = 1e-6

        self.mean_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.log_std_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        mean = self.mean_net(state)
        log_std = self.log_std_net(state)
        return mean, log_std
    
    def _full_forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        def _rescale_action(action: torch.Tensor) -> torch.Tensor:
            tanh_low, tanh_high = -1, 1
            ret = (action - tanh_low) / (tanh_high - tanh_low) * (self.action_high - self.action_low) + self.action_low
            return ret
        
        mean, log_std = self.forward(state)
        curr_dist = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh_action = curr_dist.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = _rescale_action(tanh_action)
        
        # Original formula for change of variable:
        # log p(y) = log p(x) - log|det(dy/dx)|
        # y = tanh(x) -> dy/dx = 1 - tanh(x)^2
        log_px = curr_dist.log_prob(pre_tanh_action)
        log_jacobian = torch.log(1 - tanh_action.pow(2) + self.epsilon)
        log_prob = log_px - log_jacobian
        log_prob = log_prob.sum(dim=-1, keepdim=True) # determinant of Jacobian
        
        return action, log_prob, _rescale_action(mean)

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
