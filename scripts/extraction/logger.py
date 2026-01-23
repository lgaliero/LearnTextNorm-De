# scripts/extraction/logger.py
import logging
from configs import Paths

# Setup debug logging to file (not to terminal)
logging.basicConfig(
    filename=Paths.EXT_LOG_FILE,
    filemode='w',
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def debug(msg):
    """Debug logging helper used throughout extraction pipeline."""
    logging.debug(msg)