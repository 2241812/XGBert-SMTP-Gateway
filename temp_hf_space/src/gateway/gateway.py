import asyncio
import logging
from aiosmtpd.controller import Controller

from .config import SMTP_HOST, SMTP_PORT
from .database import init_db
from .handler import PhishingSMTPHandler
from .monitoring import get_metrics_tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("gateway")


def run_gateway():
    init_db()
    metrics_tracker = get_metrics_tracker()
    logger.info("Gateway metrics file: %s", metrics_tracker.metrics_file)
    handler = PhishingSMTPHandler()
    controller = Controller(handler, hostname=SMTP_HOST, port=SMTP_PORT)
    controller.start()
    logger.info("SMTP Gateway listening on %s:%d", SMTP_HOST, SMTP_PORT)

    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down gateway...")
        controller.stop()


if __name__ == "__main__":
    run_gateway()
