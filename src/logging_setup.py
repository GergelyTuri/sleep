# logging_setup.py

import logging
import sys


class LoggingSetup:
    """Configures a logger that writes to both a file and stdout."""

    @staticmethod
    def configure_logger(log_file: str = "parsing_log.log", level=logging.INFO):
        """
        Configures logging with a specified log file and level.
        
        Args:
            log_file (str): The name of the log file.
            level (int): The logging level.
        """
        # Basic logging configuration
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename=log_file,
            filemode="a",  # Append to the log file
        )

        # StreamHandler to output log messages to the console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        
        # Add handler to root logger
        logging.getLogger().addHandler(console_handler)
        
        return logging.getLogger(__name__)
