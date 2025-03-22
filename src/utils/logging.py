import logging
import os
import sys
from datetime import datetime

def setup_logger(name, log_dir="logs", level=logging.INFO):
    """Set up a logger with file and console handlers.
    
    Args:
        name: Name of the logger
        log_dir: Directory to store logs
        level: Logging level
        
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create handlers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set log level for handlers
    file_handler.setLevel(level)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Example usage
if __name__ == "__main__":
    logger = setup_logger("benchmark_test")
    logger.info("Logging setup complete")
    logger.warning("This is a warning message")
    logger.error("This is an error message")