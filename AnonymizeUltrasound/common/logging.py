import os
import logging
import sys
from datetime import datetime

def setup_logging(log_dir='logs', process_name='', log_level='INFO'):
    """Set up logging configuration for error tracking."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{process_name}_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) # WANDB will capture this if it's been initialized
        ]
    )
    
    return logging.getLogger(__name__), log_file

def get_or_warn(d, key, default, logger=logging.getLogger(__name__)):
    if key in d and d[key] is not None:
        return d[key]
    logger.warning("Missing %s; using default %r", key, default)
    return default