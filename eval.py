import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import wandb
from pathlib import Path
from loguru import logger
import numpy as np
import random
import os
from dotenv import load_dotenv

from agents.factory import AgentFactory
from utils.logger import setup_logger
from utils.metrics.factory import MetricsFactory
from utils.save_manager import SaveManager

load_dotenv()


@hydra.main(version_base=None, config_path="config", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained agent."""

    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    OmegaConf.set_struct(cfg, False)  # Allow config modification
    cfg.device = device
    cfg.agent.device = device

    # Setup logging
    exp_dir = Path(cfg.eval.exp_dir)
    eval_dir = exp_dir / "eval"
    setup_logger(eval_dir)
    logger.info(f"Using device: {device}")

    # Create environment
    env = gym.make(
        cfg.env.name,
        render_mode="rgb_array",
    )
    logger.info(f"Created environment: {cfg.env.name}")

    # Set random seed
    torch.manual_seed(cfg.eval.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.eval.seed)
    np.random.seed(cfg.eval.seed)
    random.seed(cfg.eval.seed)
    env.reset(seed=cfg.eval.seed)
    logger.info(f"Set random seed to {cfg.eval.seed}")

    # Update config with environment info
    cfg.agent.state_dim = env.observation_space.shape[0]
    cfg.agent.action_dim = env.action_space.shape[0]
    cfg.agent.action_high = float(env.action_space.high[0])
    cfg.agent.action_low = float(env.action_space.low[0])
    cfg.agent.device = device

    # Create agent and load model
    exp_config.agent.device = device
    agent = AgentFactory.create(exp_config.agent)
    save_manager = SaveManager(exp_dir)
    model_path = exp_dir / cfg.save.model_dir / cfg.eval.model_name
    metadata = save_manager.load_model(agent, model_path)
    logger.info(
        f"Loaded {exp_config.agent.name} model from episode {metadata['episode']} with reward {metadata['reward']:.2f}"
    )

    # Setup metrics
    metrics = MetricsFactory.create(
        env_id=cfg.env.name,
        writer=None,  # No tensorboard for eval
        use_wandb=False,  # No wandb for eval
    )
    logger.info(f"Created metrics for environment: {cfg.env.name}")

    # Evaluation loop
    logger.info(f"Starting evaluation for {cfg.eval.episodes} episodes...")
    for episode in range(cfg.eval.episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        total_critic_loss = 0
        total_actor_loss = 0

        while not done:
            episode_steps += 1
            action = agent.get_action(state)  # No exploration during eval
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            if truncated:
                done = True

        # Update metrics
        metrics.push_back(
            reward=episode_reward,
            critic_loss=total_critic_loss,
            actor_loss=total_actor_loss,
            length=episode_steps,
            action=action,
            state=state,
            info=info,
        )

        logger.info(
            f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}"
        )

    # Save evaluation metrics
    eval_dir.mkdir(exist_ok=True)
    metrics_file = eval_dir / f"metrics_{cfg.eval.model_name.split('.')[0]}.json"
    summary_file = eval_dir / f"summary_{cfg.eval.model_name.split('.')[0]}.json"
    metrics.save(metrics_file, is_training=False)
    metrics.save_summary(summary_file, is_training=False)
    logger.info("\nEvaluation metrics summary:")
    metrics.print_summary()

    # Record video
    if cfg.eval.record_video:
        video_dir = eval_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        env_video = RecordVideo(
            env,
            str(video_dir),
            episode_trigger=lambda x: True,
            name_prefix=f"eval_{cfg.eval.model_name.split('.')[0]}",
        )

        state, _ = env_video.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, _, terminated, truncated, _ = env_video.step(action)
            state = next_state
            done = terminated or truncated

        env_video.close()
        logger.info(f"Saved evaluation video to {video_dir}")

    env.close()
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    evaluate()
