import os
import yaml


# Define the base directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_config_path(config_name):
    # Define a mapping of config names to their relative paths
    config_paths = {
        'camera_config.yaml': os.path.join(BASE_DIR,'Config', 'camera_config.yaml'),
        'db_config.yaml': os.path.join(BASE_DIR,'Config', 'db_config.yaml'),
        'vision_ai_config.yaml': os.path.join(BASE_DIR, 'Config', 'vision_ai_config.yaml'),
    }
    return config_paths.get(config_name)

def load_config(config_name):
    config_path = get_config_path(config_name)
    if config_path:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        raise ValueError(f"Configuration file {config_name} not found.")
