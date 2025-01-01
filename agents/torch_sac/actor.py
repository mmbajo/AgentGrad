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
        self.epsilon = 1e-6  # For numerical stability
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        # Mean and log_std heads
        self.mean_net = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std_net = nn.Linear(self.hidden_dim, self.action_dim)
        
        # Initialize output layers with small weights
        nn.init.uniform_(self.mean_net.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_net.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_net.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_net.bias, -3e-3, 3e-3)

        self.to(self.device)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std."""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
            
        x = self.encoder(state)
        mean = self.mean_net(x)
        log_std = self.log_std_net(x)
        
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from the Gaussian policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        # https://pytorch.org/docs/stable/_modules/torch/distributions/distribution.html#Distribution.rsample
        x = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        y = torch.tanh(x)
        
        # Scale action from [-1, 1] to [low, high]
        action = self._rescale_action(y)
        
        # Compute log probability
        # log p(a) = log p(x) - log|det(da/dx)|
        # = log p(x) - log|det(dy/dx) * det(da/dy)|
        log_prob = (
            normal.log_prob(x) - 
            torch.log(1 - y.pow(2) + self.epsilon) # Account for tanh squashing: log|det(dy/dx)| = log(1 - tanh(x)^2)
        )
        
        log_prob = log_prob.sum(-1, keepdim=True)  # Sum over action dimensions
        
        return action, log_prob, mean
    
    def _rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Rescale action from [-1, 1] to [low, high]."""
        return 0.5 * (action + 1.0) * (self.action_high - self.action_low) + self.action_low


class Actor:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.device = config.device
        self.actor_model = ActorModel(config)
        self.actor_optimizer = optim.Adam(
            self.actor_model.parameters(), lr=config.actor_lr
        )

    def get_action(
        self, state: torch.Tensor | NDArray[np.float32]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, and mean from policy."""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        return self.actor_model.sample(state)
