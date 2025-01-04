import pytest
import os
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="session")
def config_dir(project_root):
    """Return the config directory."""
    return project_root / "config"


@pytest.fixture(scope="session")
def agent_config_dir(config_dir):
    """Return the agent config directory."""
    return config_dir / "agent" 