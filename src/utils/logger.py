import logging
import sys
import os

def get_logger(name: str, level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Creates and configures a logger.
    Optionally logs to a file in addition to stdout.
    """
    logger_instance = logging.getLogger(name)
    
    # Check if logger already has handlers to prevent duplicate logs
    if not logger_instance.handlers:
        logger_instance.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"
        )

        # StreamHandler for stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger_instance.addHandler(stdout_handler)

        # FileHandler if log_file is specified
        if log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, mode='a') # Append mode
            file_handler.setFormatter(formatter)
            logger_instance.addHandler(file_handler)
            
    return logger_instance

if __name__ == "__main__":
    # Example usage:
    # Basic logger to console
    console_logger = get_logger("ConsoleLoggerExample")
    console_logger.info("This is an info message to console.")
    console_logger.warning("This is a warning message to console.")

    # Logger to console and file
    # Ensure 'logs' directory exists or adjust path
    if not os.path.exists("logs"): os.makedirs("logs")
    file_logger = get_logger("FileLoggerExample", log_file="logs/example.log", level=logging.DEBUG)
    file_logger.debug("This is a debug message to console and file.")
    file_logger.info("This is an info message to console and file.")
    print("Check logs/example.log for file output.")