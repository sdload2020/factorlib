from loguru import logger
import os, sys


_LOGGER_INITIALIZED = False

def setup_execution_logger(logs_path):
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return
    log_folder = os.path.join(logs_path, "execution_log")
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "{time:YYYY-MM-DD}.log")
    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=True)
    logger.add(log_file, rotation="00:00", retention="14 days", enqueue=True)
    _LOGGER_INITIALIZED = True
