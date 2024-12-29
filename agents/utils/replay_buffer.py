from typing import Dict, Any, Tuple
import torch
import numpy as np
from numpy.typing import NDArray


class ReplayBuffer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.buffer_size = config["buffer_size"]
        self.device = config["device"]
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]

        self.state_buffer = torch.zeros((self.buffer_size, self.state_dim))
        self.action_buffer = torch.zeros((self.buffer_size, self.action_dim))
        self.reward_buffer = torch.zeros((self.buffer_size, 1))
        self.next_state_buffer = torch.zeros((self.buffer_size, self.state_dim))
        self.done_buffer = torch.zeros((self.buffer_size, 1))

        self.index = 0
        self.size = 0

    def add(
        self,
        state: NDArray[np.float32],
        action: NDArray[np.float32],
        reward: float,
        next_state: NDArray[np.float32],
        done: bool,
    ) -> None:
        # Convert numpy arrays to tensors if necessary
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)
        if isinstance(reward, (float, int)):
            reward = torch.FloatTensor([reward])
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state)
        if isinstance(done, bool):
            done = torch.FloatTensor([float(done)])

        self.state_buffer[self.index] = state
        self.action_buffer[self.index] = action
        self.reward_buffer[self.index] = reward
        self.next_state_buffer[self.index] = next_state
        self.done_buffer[self.index] = done

        self.index = (self.index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.size, (batch_size,))
        return (
            self.state_buffer[indices].to(self.device),
            self.action_buffer[indices].to(self.device),
            self.reward_buffer[indices].to(self.device),
            self.next_state_buffer[indices].to(self.device),
            self.done_buffer[indices].to(self.device),
        )

    def __len__(self) -> int:
        return self.size
