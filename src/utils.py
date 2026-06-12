# src/utils.py
import os
import logging
import yaml

# Resolve project root relative to this file's location (src/ → project root)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads project configurations from a YAML file.

    Resolves paths relative to the project root so the config can be loaded
    correctly regardless of the current working directory.

    Parameters:
        config_path (str): Path to the YAML configuration file.
            Accepts an absolute path or a path relative to the project root.

    Returns:
        dict: Parsed configurations.

    Raises:
        FileNotFoundError: If the config file cannot be located.
    """
    # If not absolute, resolve relative to the project root
    if not os.path.isabs(config_path):
        resolved = os.path.join(PROJECT_ROOT, config_path)
    else:
        resolved = config_path

    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"Configuration file not found at '{resolved}'. "
            "Ensure 'config.yaml' exists at the project root."
        )

    with open(resolved, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config


def get_logger(name: str, config: dict = None) -> logging.Logger:
    """
    Configures and returns a structured logger with console and file handlers.

    Parameters:
        name (str): Logger name (typically __name__ of the calling module).
        config (dict, optional): Parsed config dict from load_config().

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated imports
    if logger.handlers:
        return logger

    # Defaults
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file = os.path.join(PROJECT_ROOT, "reports", "engine.log")
    log_level = logging.INFO

    if config and "logging" in config:
        log_cfg = config["logging"]
        log_format = log_cfg.get("format", log_format)
        raw_file = log_cfg.get("log_file", "reports/engine.log")
        # Resolve log file path relative to project root
        log_file = os.path.join(PROJECT_ROOT, raw_file) if not os.path.isabs(raw_file) else raw_file
        log_level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    logger.setLevel(log_level)

    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning(f"Could not create log file handler for '{log_file}': {exc}")

    return logger
