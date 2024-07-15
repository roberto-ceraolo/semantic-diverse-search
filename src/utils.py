import os
import json
import logging
from datetime import datetime
import yaml

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def __getattr__(self, name):
        return self.config.get(name)

def setup_logging(config: Config) -> str:
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    log_filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    logging.basicConfig(filename=log_filepath, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Configuration: {json.dumps(config.config, indent=2)}")

    return log_filepath

