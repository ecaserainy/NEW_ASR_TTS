import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.now().strftime("%Y-%m-%d") + ".log"
log_path = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("asr_tts_server")