import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import torch
import wandb
from pathlib import Path
from loguru import logger
import numpy as np
import random
import os
from dotenv import load_dotenv

from agents.torch_ddpg.agent import DDPGAgent
from utils.logger import setup_logger
from utils.metrics.factory import MetricsFactory
from utils.save_manager import SaveManager

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
wandb.login(key=os.getenv("WANDB_API_KEY"))

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    cfg.device = device

    # Setup logging
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    setup_logger(log_dir)
    logger.info(f"Using device: {device}")

    # Create environment
    env = gym.make(cfg.env.name)
    logger.info(f"Created environment: {cfg.env.name}")
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    env.reset(seed=cfg.seed)
    logger.info(f"Set random seed to {cfg.seed}")

    # Update config with environment info
    OmegaConf.set_struct(cfg, False)  # Allow config modification
    cfg.agent.state_dim = env.observation_space.shape[0]
    cfg.agent.action_dim = env.action_space.shape[0]
    cfg.agent.action_high = float(env.action_space.high[0])
    cfg.agent.action_low = float(env.action_space.low[0])

    # Log agent configuration
    logger.info("\nAgent configuration:")
    logger.info(OmegaConf.to_yaml(cfg.agent))

    # Setup metrics
    if cfg.tensorboard.enabled:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))
    else:
        writer = None
    metrics = MetricsFactory.create(
        env_id=cfg.env.name,
        writer=writer,
        use_wandb=cfg.wandb.mode != "disabled"
    )
    logger.info(f"Created metrics for environment: {cfg.env.name}")

    # Initialize wandb if enabled
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            entity=cfg.wandb.entity,
            config=dict(cfg),
            mode=cfg.wandb.mode,
            dir=str(log_dir),
        )
        logger.info("Initialized W&B logging")

    # Create agent
    cfg.agent.device = device
    agent = DDPGAgent(cfg.agent)
    logger.info("Created DDPG agent")

    # Setup save manager
    save_manager = SaveManager(
        save_dir=log_dir,
        metrics_freq=cfg.save.metrics_freq,
        model_freq=cfg.save.model_freq,
    )

    # Training loop
    total_steps = 0
    best_reward = float("-inf")

    logger.info("Starting training...")
    for episode in range(cfg.env.max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        total_critic_loss = 0
        total_actor_loss = 0

        while not done:
            episode_steps += 1
            total_steps += 1

            if total_steps < cfg.env.max_steps:
                action = env.action_space.sample()
            else:
                action = agent.get_exploration_action(state)

            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            agent.store_experience(state, action, reward, next_state, done)

            if (
                len(agent.replay_buffer) > cfg.agent.batch_size
                and total_steps > cfg.env.max_steps
            ):
                critic_loss, actor_loss = agent.train()
                total_critic_loss += critic_loss
                total_actor_loss += actor_loss

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

        # Update best reward and save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            logger.info(f"New best reward: {best_reward:.2f}")
            save_manager.save_model(
                agent=agent,
                episode=episode,
                reward=episode_reward,
                is_best=True,
            )

        # Periodic saves
        if save_manager.should_save_metrics(episode):
            save_manager.save_metrics(metrics, episode)

        if save_manager.should_save_model(episode):
            save_manager.save_model(
                agent=agent,
                episode=episode,
                reward=episode_reward,
            )

        logger.info(
            f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}, Best = {best_reward:.2f}"
        )

    # Save final metrics and model
    save_manager.save_metrics(metrics, episode)
    save_manager.save_model(
        agent=agent,
        episode=episode,
        reward=episode_reward,
        is_final=True,
    )
    logger.info("\nFinal metrics summary:")
    metrics.print_summary()

    if cfg.wandb.mode != "disabled":
        wandb.finish()

    if writer:
        writer.close()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
