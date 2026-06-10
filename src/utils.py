# src/utils.py
import os
import logging
import yaml

def load_config(config_path="config.yaml"):
    """
    Loads project configurations from a YAML file.
    
    Parameters:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Parsed configurations.
    """
    # Handle absolute/relative path resolving if needed
    if not os.path.exists(config_path):
        # Check if one level up (if running from notebooks/ or src/)
        parent_config = os.path.join("..", config_path)
        if os.path.exists(parent_config):
            config_path = parent_config
        else:
            raise FileNotFoundError(f"Configuration file not found at '{config_path}' or '{parent_config}'.")
            
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_logger(name, config=None):
    """
    Configures and returns a professional logger.
    
    Parameters:
        name (str): Name of the logger.
        config (dict, optional): Logger configurations dict loaded from load_config().
        
    Returns:
        logging.Logger: Pre-configured logger object.
    """
    logger = logging.getLogger(name)
    
    # If logger already has handlers, don't duplicate them
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Defaults if config is missing
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file = "reports/engine.log"
    
    if config and 'logging' in config:
        log_cfg = config['logging']
        log_format = log_cfg.get('format', log_format)
        log_file = log_cfg.get('log_file', log_file)
        level_str = log_cfg.get('level', 'INFO')
        logger.setLevel(getattr(logging, level_str.upper(), logging.INFO))
        
    # Ensure log file directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    formatter = logging.Formatter(log_format)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file handler for '{log_file}': {e}")
        
    return logger
