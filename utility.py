 
import logging

def setup_logger():
    FORMAT = r"[%(asctime)-15s] %(levelname)s - %(message)s"
    DATE_FORMAT = r"%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level = logging.DEBUG, format = FORMAT, datefmt = DATE_FORMAT)
