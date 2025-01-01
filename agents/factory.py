from typing import Dict, Any, Type
from omegaconf import DictConfig
from loguru import logger

from agents.torch_ddpg.agent import DDPGAgent
from agents.torch_td3.agent import TD3Agent
from agents.torch_sac.agent import SACAgent


class AgentFactory:
    """Factory class to create appropriate agents."""

    _registry = {
        "ddpg": DDPGAgent,
        "td3": TD3Agent,
        "sac": SACAgent,
    }

    @classmethod
    def register(cls, name: str, agent_class: Type) -> None:
        """Register a new agent class.

        Args:
            name: Name of the agent (e.g., "ddpg", "td3", "sac")
            agent_class: Agent class to register
        """
        cls._registry[name] = agent_class
        logger.info(f"Registered agent: {name}")

    @classmethod
    def create(cls, config: DictConfig) -> Any:
        """Create agent instance based on configuration.

        Args:
            config: Agent configuration including name and parameters

        Returns:
            Appropriate agent instance

        Raises:
            ValueError: If agent type is not registered
        """
        agent_name = config.name.lower()

        if agent_name not in cls._registry:
            raise ValueError(
                f"Agent '{agent_name}' not found in registry. "
                f"Available agents: {list(cls._registry.keys())}"
            )

        agent_class = cls._registry[agent_name]
        logger.info(f"Creating agent: {agent_name}")
        return agent_class(config)

    @classmethod
    def list_available(cls) -> list[str]:
        """Get list of registered agent names."""
        return list(cls._registry.keys())
