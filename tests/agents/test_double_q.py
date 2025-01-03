import pytest
import torch
import numpy as np
from typing import Dict, Any
from omegaconf import DictConfig


from agents.torch_td3.agent import TD3Agent
from agents.torch_sac.agent import SACAgent


@pytest.fixture
def base_config() -> DictConfig:
    """Base configuration for testing."""
    config = {
        "state_dim": 3,
        "action_dim": 1,
        "hidden_dim": 64,
        "action_high": 1.0,
        "action_low": -1.0,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 1000,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "tau": 0.005,
        "device": "cpu"
    }
    return DictConfig(config)


def test_td3_double_q_initialization(base_config):
    """Test that TD3 agent properly initializes with double Q flag."""
    config = DictConfig({
        **base_config,
        "target_policy_noise": 0.2,
        "target_policy_clip": 0.5,
        "policy_freq": 2,
        "use_double_q": True,
        "exploration_noise": 0.1,
    })
    
    agent = TD3Agent(config)
    assert agent.use_double_q is True
    
    # Test critic outputs with random input
    state = torch.randn(1, config.state_dim)
    action = torch.randn(1, config.action_dim)
    state_action = torch.cat([state, action], dim=1)
    
    # Both critics should output different values
    q1 = agent.critic.critic_model.net1(state_action)
    q2 = agent.critic.critic_model.net2(state_action)
    assert not torch.allclose(q1, q2)


def test_td3_single_q_initialization(base_config):
    """Test that TD3 agent properly initializes without double Q."""
    config = DictConfig({
        **base_config,
        "target_policy_noise": 0.2,
        "target_policy_clip": 0.5,
        "policy_freq": 2,
        "use_double_q": False,
        "exploration_noise": 0.1,
    })
    
    agent = TD3Agent(config)
    assert agent.use_double_q is False


def test_sac_double_q_initialization(base_config):
    """Test that SAC agent properly initializes with double Q flag."""
    config = DictConfig({
        **base_config,
        "initial_alpha": 1.0,
        "auto_tune_alpha": True,
        "alpha_lr": 3e-4,
        "use_double_q": True,
    })
    
    agent = SACAgent(config)
    assert agent.use_double_q is True
    
    # Test critic outputs with random input
    state = torch.randn(1, config.state_dim)
    action = torch.randn(1, config.action_dim)
    state_action = torch.cat([state, action], dim=1)
    
    # Both critics should output different values
    q1 = agent.critic.critic_model.net1(state_action)
    q2 = agent.critic.critic_model.net2(state_action)
    assert not torch.allclose(q1, q2)


def test_sac_single_q_initialization(base_config):
    """Test that SAC agent properly initializes without double Q."""
    config = DictConfig({
        **base_config,
        "initial_alpha": 1.0,
        "auto_tune_alpha": True,
        "alpha_lr": 3e-4,
        "use_double_q": False,
    })
    
    agent = SACAgent(config)
    assert agent.use_double_q is False


def test_td3_double_q_target_computation(base_config):
    """Test that TD3 computes target values differently with double Q."""
    config = DictConfig({
        **base_config,
        "target_policy_noise": 0.2,
        "target_policy_clip": 0.5,
        "policy_freq": 2,
        "use_double_q": True,
        "exploration_noise": 0.1,
    })
    
    agent = TD3Agent(config)
    
    # Create sample batch with correct shapes
    next_state = torch.randn(config.batch_size, config.state_dim)
    reward = torch.randn(config.batch_size, 1)
    done = torch.zeros(config.batch_size, 1)
    
    # Get target value with double Q
    target_double_q = agent._compute_target_value(next_state, reward, done)
    
    # Change to single Q and recompute
    agent.use_double_q = False
    target_single_q = agent._compute_target_value(next_state, reward, done)
    
    # Values should be different
    assert not torch.allclose(target_double_q, target_single_q)


def test_sac_double_q_target_computation(base_config):
    """Test that SAC computes target values differently with double Q."""
    config = DictConfig({
        **base_config,
        "initial_alpha": 1.0,
        "auto_tune_alpha": True,
        "alpha_lr": 3e-4,
        "use_double_q": True,
    })
    
    agent = SACAgent(config)
    
    # Create sample batch with correct shapes
    next_state = torch.randn(config.batch_size, config.state_dim)
    reward = torch.randn(config.batch_size, 1)
    done = torch.zeros(config.batch_size, 1)
    
    # Get target value with double Q
    target_double_q = agent._compute_target_value(next_state, reward, done)
    
    # Change to single Q and recompute
    agent.use_double_q = False
    target_single_q = agent._compute_target_value(next_state, reward, done)
    
    # Values should be different
    assert not torch.allclose(target_double_q, target_single_q)


def test_td3_critic_loss_shape(base_config):
    """Test that critic loss computation works with both single and double Q."""
    config = DictConfig({
        **base_config,
        "target_policy_noise": 0.2,
        "target_policy_clip": 0.5,
        "policy_freq": 2,
        "use_double_q": True,
        "exploration_noise": 0.1,
    })
    
    agent = TD3Agent(config)
    
    # Create sample batch with correct shapes
    state = torch.randn(config.batch_size, config.state_dim)  # Shape: (32, 3)
    action = torch.randn(config.batch_size, config.action_dim)  # Shape: (32, 1)
    next_state = torch.randn(config.batch_size, config.state_dim)  # Shape: (32, 3)
    reward = torch.randn(config.batch_size, 1)  # Shape: (32, 1)
    done = torch.zeros(config.batch_size, 1)  # Shape: (32, 1)
    
    # Test with double Q
    critic_loss_double = agent._compute_critic_loss(state, action, reward, next_state, done)
    assert isinstance(critic_loss_double, torch.Tensor)
    assert critic_loss_double.shape == torch.Size([])
    
    # Test with single Q
    agent.use_double_q = False
    critic_loss_single = agent._compute_critic_loss(state, action, reward, next_state, done)
    assert isinstance(critic_loss_single, torch.Tensor)
    assert critic_loss_single.shape == torch.Size([])


def test_sac_critic_loss_shape(base_config):
    """Test that critic loss computation works with both single and double Q."""
    config = DictConfig({
        **base_config,
        "initial_alpha": 1.0,
        "auto_tune_alpha": True,
        "alpha_lr": 3e-4,
        "use_double_q": True,
    })
    
    agent = SACAgent(config)
    
    # Create sample batch with correct shapes
    state = torch.randn(config.batch_size, config.state_dim)  # Shape: (32, 3)
    action = torch.randn(config.batch_size, config.action_dim)  # Shape: (32, 1)
    next_state = torch.randn(config.batch_size, config.state_dim)  # Shape: (32, 3)
    reward = torch.randn(config.batch_size, 1)  # Shape: (32, 1)
    done = torch.zeros(config.batch_size, 1)  # Shape: (32, 1)
    
    # Test with double Q
    critic_loss_double = agent._compute_critic_loss(state, action, reward, next_state, done)
    assert isinstance(critic_loss_double, torch.Tensor)
    assert critic_loss_double.shape == torch.Size([])
    
    # Test with single Q
    agent.use_double_q = False
    critic_loss_single = agent._compute_critic_loss(state, action, reward, next_state, done)
    assert isinstance(critic_loss_single, torch.Tensor)
    assert critic_loss_single.shape == torch.Size([])