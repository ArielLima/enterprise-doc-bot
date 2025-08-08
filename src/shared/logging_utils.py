import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rich_console: bool = True
) -> logging.Logger:
    logger = logging.getLogger("docbot")
    logger.setLevel(getattr(logging, level.upper()))
    
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if rich_console:
        console = Console()
        rich_handler = RichHandler(
            console=console,
            show_path=False,
            show_time=True,
            rich_tracebacks=True
        )
        rich_handler.setFormatter(formatter)
        logger.addHandler(rich_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "docbot") -> logging.Logger:
    return logging.getLogger(name)