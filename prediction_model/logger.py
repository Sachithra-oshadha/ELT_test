from imports import *

def setup_logger():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f'customer_behavior_bilstm_{now}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)