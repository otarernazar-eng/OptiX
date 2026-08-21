import pytest
from pathlib import Path
import sys

# Add root to python path to import src
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from src.utils.config import load_config

def test_load_config():
    """Test that config loads successfully and contains required keys."""
    config = load_config()
    
    assert "project" in config
    assert config["project"] == "OptiX"
    
    assert "model" in config
    assert "architecture" in config["model"]
    assert config["model"]["architecture"] == "EfficientNet-B0"
    
    assert "training" in config
    assert config["training"]["image_size"] == 384
    assert config["training"]["batch_size"] == 16
    assert config["training"]["learning_rate"] == 0.001
    
    assert "data" in config
    assert "raw_dir" in config["data"]
