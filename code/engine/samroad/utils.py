import yaml
from addict import Dict
from datetime import datetime
import os
from pathlib import Path

from package_paths import PROJECT_ROOT, SAM_CKPT_PATH, resolve_path

def load_config(path):
    config_path = resolve_path(path)
    with open(config_path, encoding='utf-8-sig') as file:
        config_dict = yaml.safe_load(file)
    config = Dict(config_dict)
    configured_sam = Path(str(config.get('SAM_CKPT_PATH') or ''))
    if not configured_sam.is_absolute() and SAM_CKPT_PATH.is_file():
        config.SAM_CKPT_PATH = str(SAM_CKPT_PATH.resolve())
    if not config.get('CONFIG_PATH'):
        config.CONFIG_PATH = str(config_path)
    if not config.get('PROJECT_ROOT'):
        config.PROJECT_ROOT = str(PROJECT_ROOT)
    return config

def create_output_dir_and_save_config(output_dir_prefix, config, specified_dir=None):
    if specified_dir:
        output_dir = Path(specified_dir)
    else:
        # Generate the output directory name with the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{output_dir_prefix}_{timestamp}")
    
    # Create the directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the path for the config file
    config_path = output_dir / "config.yaml"
    
    # Save the config as a YAML file
    with open(config_path, 'w', encoding='utf-8') as file:
        yaml.dump(config.to_dict(), file, allow_unicode=True)
    
    return str(output_dir)
