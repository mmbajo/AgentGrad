from pathlib import Path
from typing import Dict, Any
import torch
from loguru import logger


class SaveManager:
    """Manages saving operations for models and metrics."""

    def __init__(self, save_dir: Path, metrics_freq: int = 200, model_freq: int = 200):
        """Initialize save manager.
        
        Args:
            save_dir: Base directory for saving
            metrics_freq: Frequency of metrics saves
            model_freq: Frequency of model saves
        """
        self.save_dir = save_dir
        self.metrics_freq = metrics_freq
        self.model_freq = model_freq
        
        # Create directories
        self.metrics_dir = save_dir / "metrics"
        self.model_dir = save_dir / "models"
        self.metrics_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        logger.info(f"Created save directories: {self.metrics_dir}, {self.model_dir}")
    
    def should_save_metrics(self, episode: int) -> bool:
        """Check if metrics should be saved."""
        return (episode + 1) % self.metrics_freq == 0
    
    def should_save_model(self, episode: int) -> bool:
        """Check if model should be saved."""
        return (episode + 1) % self.model_freq == 0
    
    def save_metrics(self, metrics: Any, episode: int, is_training: bool = True) -> None:
        """Save metrics and clear accumulated data.
        
        Args:
            metrics: Metrics object to save
            episode: Current episode number
            is_training: Whether these are training metrics
        """
        # Save metrics for this period
        period = episode + 1
        metrics_file = self.metrics_dir / f"metrics_{period}.json"
        summary_file = self.metrics_dir / f"summary_{period}.json"
        
        metrics.save(metrics_file, is_training=is_training)
        metrics.save_summary(summary_file, is_training=is_training)
        metrics.clear_metrics()  # Clear accumulated data after saving
        logger.info(f"Saved and cleared metrics for period {period}")
    
    def save_model(
        self,
        agent: Any,
        episode: int,
        reward: float,
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        """Save model checkpoint.
        
        Args:
            agent: Agent to save
            episode: Current episode number
            reward: Current episode reward
            is_best: Whether this is the best model so far
            is_final: Whether this is the final model
        """
        checkpoint = {
            "episode": episode,
            "reward": reward,
            "agent_state": agent.get_save_dict(),
        }
        
        if is_best:
            path = self.model_dir / "best_model.pt"
            checkpoint["best_reward"] = reward
            logger.info(f"Saved best model with reward {reward:.2f}")
        elif is_final:
            path = self.model_dir / "final_model.pt"
            checkpoint["final_reward"] = reward
            logger.info(f"Saved final model with reward {reward:.2f}")
        else:
            path = self.model_dir / f"model_episode_{episode + 1}.pt"
            logger.info(f"Saved model checkpoint at episode {episode + 1}")
        
        torch.save(checkpoint, path)
    
    def load_model(self, agent: Any, path: Path) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            agent: Agent to load state into
            path: Path to checkpoint file
            
        Returns:
            Dictionary containing checkpoint metadata
        """
        checkpoint = torch.load(path)
        agent.load_save_dict(checkpoint["agent_state"])
        return {
            "episode": checkpoint["episode"],
            "reward": checkpoint["reward"],
            "best_reward": checkpoint.get("best_reward"),
            "final_reward": checkpoint.get("final_reward"),
        } 