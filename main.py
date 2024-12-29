import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import torch
import wandb
from pathlib import Path
from loguru import logger
import sys
from datetime import datetime

from agents.ddpg.agent import DDPGAgent

def setup_logger(log_dir: Path) -> None:
    """Setup loguru logger with file and console outputs."""
    # Remove default handler
    logger.remove()
    
    # Add console handler with color
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler
    log_file = log_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger.add(
        str(log_file),
        rotation="100 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )

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
    
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=dict(cfg),
            mode=cfg.wandb.mode,
            dir=str(log_dir),
        )
        logger.info("Initialized W&B logging")
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    
    # Create environment
    env = gym.make(cfg.env.name)
    logger.info(f"Created environment: {cfg.env.name}")
    
    # Update config with environment info
    OmegaConf.set_struct(cfg, False)  # Allow config modification
    cfg.agent.state_dim = env.observation_space.shape[0]
    cfg.agent.action_dim = env.action_space.shape[0]
    cfg.agent.action_high = float(env.action_space.high[0])
    cfg.agent.action_low = float(env.action_space.low[0])
    
    # Log agent configuration
    logger.info("\nAgent configuration:")
    logger.info(OmegaConf.to_yaml(cfg.agent))
    
    # Create agent directly
    cfg.agent.device = device
    agent = DDPGAgent(cfg.agent)
    logger.info("Created DDPG agent")
    
    # Training loop
    total_steps = 0
    best_reward = float('-inf')
    
    logger.info("Starting training...")
    for episode in range(cfg.env.max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
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
            
            if len(agent.replay_buffer) > cfg.agent.batch_size and total_steps > cfg.env.max_steps:
                agent.train()
                
            state = next_state
            
            if truncated:
                done = True
        
        metrics = {
            "episode": episode,
            "reward": episode_reward,
            "steps": episode_steps,
            "total_steps": total_steps,
        }
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            logger.info(f"New best reward: {best_reward:.2f}")
        
        logger.info(
            f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}, Best = {best_reward:.2f}"
        )
        
        if cfg.wandb.mode != "disabled":
            wandb.log(metrics)
        
        if episode_reward >= cfg.env.max_episode_reward:
            logger.success(f"Solved in {episode} episodes!")
            break
    
    if cfg.wandb.mode != "disabled":
        wandb.finish()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
