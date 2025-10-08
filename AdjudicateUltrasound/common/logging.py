import os
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

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

# Simple registry to avoid duplicate handlers per module
_module_file_handlers = {}

def _map_level(level_str: str) -> int:
    level_str = (level_str or 'INFO').upper()
    return {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }.get(level_str, logging.INFO)

def _default_logs_dir_for_module(module_name: str) -> str:
    # Place logs under the module folder by default: <module_dir>/logs
    # This file lives at .../AnonymizeUltrasound/common/logging.py → go up one
    module_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(module_dir, 'logs')

def start_file_logging(module_name: str, level: str = 'INFO', directory: str = '') -> str:
    """Attach a rotating file handler to the root logger.

    Returns the path of the log file in use. Subsequent calls for the same
    module_name will be no-ops and return the current file path.
    """
    # If already started, do nothing
    if module_name in _module_file_handlers:
        handler, log_file_path = _module_file_handlers[module_name]
        try:
            # Keep level updated if user changes it
            handler.setLevel(_map_level(level))
        except Exception:
            pass
        return log_file_path

    logs_dir = directory if directory else _default_logs_dir_for_module(module_name)
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(logs_dir, f"{module_name}_{timestamp}.log")

    handler = RotatingFileHandler(log_file_path, maxBytes=10 * 1024 * 1024, backupCount=3)
    handler.setLevel(_map_level(level))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    _module_file_handlers[module_name] = (handler, log_file_path)
    return log_file_path

def stop_file_logging(module_name: str) -> None:
    """Detach and close the rotating file handler for the given module if present."""
    if module_name not in _module_file_handlers:
        return
    handler, _ = _module_file_handlers.pop(module_name)
    root_logger = logging.getLogger()
    try:
        root_logger.removeHandler(handler)
    except Exception:
        pass
    try:
        handler.close()
    except Exception:
        pass