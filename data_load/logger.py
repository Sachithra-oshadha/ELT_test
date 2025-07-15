import logging
from datetime import datetime

_logger = None

def setup_logger():
    global _logger
    if _logger is not None:
        return _logger

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"data_insertion_{now}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)