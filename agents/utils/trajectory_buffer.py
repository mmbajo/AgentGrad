from typing import Dict, Any, Tuple, List
import torch
import numpy as np
from numpy.typing import NDArray


class TrajectoryBuffer:
    """Buffer for storing trajectories for on-policy algorithms.
    
    Unlike the replay buffer which stores individual transitions,
    this buffer stores complete trajectories (sequences of state-action-reward tuples).
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize trajectory buffer.
        
        Args:
            config: Configuration dictionary containing:
                - buffer_size: Maximum number of trajectories to store
                - device: Device to store tensors on
                - state_dim: Dimension of state space
                - action_dim: Dimension of action space
                - num_timesteps: Maximum length of trajectories
        """
        self.config = config
        self.buffer_size = config["buffer_size"]
        self.device = config["device"]
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.num_timesteps = config["num_timesteps"]

        # Initialize buffers for complete trajectories
        # Shape: (buffer_size, num_timesteps, feature_dim)
        self.state_buffer = torch.zeros(
            (self.buffer_size, self.num_timesteps, self.state_dim),
            device=self.device
        )
        self.action_buffer = torch.zeros(
            (self.buffer_size, self.num_timesteps, self.action_dim),
            device=self.device
        )
        self.reward_buffer = torch.zeros(
            (self.buffer_size, self.num_timesteps),
            device=self.device
        )
        self.next_state_buffer = torch.zeros(
            (self.buffer_size, self.num_timesteps, self.state_dim),
            device=self.device
        )
        self.done_buffer = torch.zeros(
            (self.buffer_size, self.num_timesteps),
            device=self.device
        )
        self.log_prob_buffer = torch.zeros(
            (self.buffer_size, self.num_timesteps),
            device=self.device
        )
        self.value_buffer = torch.zeros(
            (self.buffer_size, self.num_timesteps),
            device=self.device
        )
        self.advantage_buffer = torch.zeros(
            (self.buffer_size, self.num_timesteps),
            device=self.device
        )
        self.return_buffer = torch.zeros(
            (self.buffer_size, self.num_timesteps),
            device=self.device
        )
        self.mask_buffer = torch.zeros(
            (self.buffer_size, self.num_timesteps),
            device=self.device,
            dtype=torch.bool
        )

        # Current trajectory being built
        self.current_traj: Dict[str, List] = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "log_probs": [],
            "values": [],
        }
        
        self.index = 0  # Current trajectory index
        self.size = 0   # Number of complete trajectories stored

    def add(
        self,
        state: NDArray[np.float32],
        action: NDArray[np.float32],
        reward: float,
        next_state: NDArray[np.float32],
        done: bool,
        log_prob: float | None = None,
        value: float | None = None,
    ) -> None:
        """Add a transition to the current trajectory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            log_prob: Log probability of action (for PPO)
            value: Value estimate (for PPO/A2C)
        """
        # Add to current trajectory
        self.current_traj["states"].append(torch.FloatTensor(state))
        self.current_traj["actions"].append(torch.FloatTensor(action))
        self.current_traj["rewards"].append(torch.FloatTensor([reward]))
        self.current_traj["next_states"].append(torch.FloatTensor(next_state))
        self.current_traj["dones"].append(torch.FloatTensor([float(done)]))
        
        if log_prob is not None:
            self.current_traj["log_probs"].append(torch.FloatTensor([log_prob]))
        if value is not None:
            self.current_traj["values"].append(torch.FloatTensor([value]))

        # If episode ended or trajectory is full, store it
        if done or len(self.current_traj["states"]) >= self.num_timesteps:
            self._store_trajectory()

    def _store_trajectory(self) -> None:
        """Store the current trajectory in the buffer."""
        traj_len = len(self.current_traj["states"])
        
        # Convert lists to tensors and pad if necessary
        states = torch.stack(self.current_traj["states"]).to(self.device)
        actions = torch.stack(self.current_traj["actions"]).to(self.device)
        rewards = torch.cat(self.current_traj["rewards"]).to(self.device)
        next_states = torch.stack(self.current_traj["next_states"]).to(self.device)
        dones = torch.cat(self.current_traj["dones"]).to(self.device)
        
        # Store in buffer with padding
        self.state_buffer[self.index, :traj_len] = states
        self.action_buffer[self.index, :traj_len] = actions
        self.reward_buffer[self.index, :traj_len] = rewards
        self.next_state_buffer[self.index, :traj_len] = next_states
        self.done_buffer[self.index, :traj_len] = dones
        self.mask_buffer[self.index, :traj_len] = True  # Mark valid timesteps
        
        # Store optional data if available
        if self.current_traj["log_probs"]:
            log_probs = torch.cat(self.current_traj["log_probs"]).to(self.device)
            self.log_prob_buffer[self.index, :traj_len] = log_probs
            
        if self.current_traj["values"]:
            values = torch.cat(self.current_traj["values"]).to(self.device)
            self.value_buffer[self.index, :traj_len] = values

        # Update buffer state
        self.index = (self.index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
        # Clear current trajectory
        self.current_traj = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "log_probs": [],
            "values": [],
        }

    def get_all_trajectories(
        self, batch_size: int | None = None
    ) -> Tuple[torch.Tensor, ...]:
        """Get all stored trajectories.
        
        Args:
            batch_size: If provided, randomly sample this many trajectories
            
        Returns:
            Tuple of tensors containing trajectory data
        """
        if batch_size is None or batch_size >= self.size:
            indices = torch.arange(self.size)
        else:
            indices = torch.randperm(self.size)[:batch_size]
            
        return (
            self.state_buffer[indices],
            self.action_buffer[indices],
            self.reward_buffer[indices],
            self.next_state_buffer[indices],
            self.done_buffer[indices],
            self.log_prob_buffer[indices],
            self.value_buffer[indices],
            self.advantage_buffer[indices],
            self.return_buffer[indices],
            self.mask_buffer[indices],
        )

    def compute_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        normalize: bool = True,
    ) -> None:
        """Compute advantages and returns for all trajectories using GAE.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            normalize: Whether to normalize advantages
        """
        # TODO: Vectorize this
        for i in range(self.size):
            # Get valid timesteps for this trajectory
            mask = self.mask_buffer[i]
            if not mask.any():
                continue
                
            rewards = self.reward_buffer[i][mask]
            values = self.value_buffer[i][mask]
            dones = self.done_buffer[i][mask]
            
            # Get next values (0 for terminal states)
            next_values = torch.zeros_like(values)
            next_values[:-1] = values[1:]
            
            # Compute TD errors
            deltas = rewards + gamma * next_values * (1 - dones) - values
            
            # Compute GAE advantages
            advantages = torch.zeros_like(deltas)
            lastgae = 0
            for t in reversed(range(len(deltas))):
                advantages[t] = lastgae = deltas[t] + gamma * gae_lambda * (1 - dones[t]) * lastgae
                
            # Store advantages and returns
            self.advantage_buffer[i, mask] = advantages
            self.return_buffer[i, mask] = advantages + values
            
        # Normalize advantages
        if normalize and self.size > 0:
            valid_advantages = self.advantage_buffer[self.mask_buffer]
            adv_mean = valid_advantages.mean()
            adv_std = valid_advantages.std()
            self.advantage_buffer[self.mask_buffer] = (valid_advantages - adv_mean) / (adv_std + 1e-8)

    def clear(self) -> None:
        """Clear the buffer."""
        self.index = 0
        self.size = 0
        self.current_traj = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "log_probs": [],
            "values": [],
        }
        
        # Clear all buffers
        self.state_buffer.zero_()
        self.action_buffer.zero_()
        self.reward_buffer.zero_()
        self.next_state_buffer.zero_()
        self.done_buffer.zero_()
        self.log_prob_buffer.zero_()
        self.value_buffer.zero_()
        self.advantage_buffer.zero_()
        self.return_buffer.zero_()
        self.mask_buffer.zero_()

    def __len__(self) -> int:
        """Get number of complete trajectories stored."""
        return self.size
