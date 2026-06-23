"""
Módulo para configuração centralizada de logging.

Este módulo facilita a configuração consistente de logging em toda a aplicação.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configura e retorna um logger.

    Args:
        name (str): Nome do logger.
        log_file (str, optional): Caminho do arquivo de log.
        level (int): Nível de logging. Default: INFO.
        format_string (str, optional): Formato das mensagens de log.

    Returns:
        logging.Logger: Logger configurado.
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    formatter = logging.Formatter(format_string)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para arquivo (se especificado)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Obtém um logger existente ou cria um novo.

    Args:
        name (str): Nome do logger.

    Returns:
        logging.Logger: Logger.
    """
    return logging.getLogger(name)
