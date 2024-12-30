from pathlib import Path
from loguru import logger
import sys
from datetime import datetime


def setup_logger(log_dir: Path) -> None:
    """Setup loguru logger with file and console outputs.

    Args:
        log_dir: Directory to save log files
    """
    # Remove default handler
    logger.remove()

    # Add console handler with color
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Add file handler
    log_file = log_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger.add(
        str(log_file),
        rotation="100 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
    )

    return logger
