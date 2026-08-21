import yaml
from pathlib import Path

def load_config(config_path="configs/base.yaml"):
    """
    Load YAML configuration file.
    """
    # Get the project root directory (two levels up from this file)
    project_root = Path(__file__).resolve().parents[2]
    
    # Resolve the full path to the config file
    full_config_path = project_root / config_path
    
    if not full_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_config_path}")
        
    with open(full_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config
