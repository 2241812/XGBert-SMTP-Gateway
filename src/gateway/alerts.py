import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText

from .config import (
    MAILGUN_SMTP_HOST,
    MAILGUN_SMTP_PORT,
    MAILGUN_LOGIN,
    MAILGUN_PASSWORD,
    MAILGUN_FROM,
    ALERT_TO,
)

logger = logging.getLogger("gateway")


def send_phishing_alert(url: str, from_email: str, to_email: str) -> None:
    if not MAILGUN_LOGIN or not MAILGUN_PASSWORD or not ALERT_TO:
        logger.warning("Mailgun not configured, skipping alert")
        return
    try:
        msg = MIMEText(
            f"PHISHING ALERT\n\n"
            f"Malicious URL detected and blocked:\n{url}\n\n"
            f"From: {from_email}\n"
            f"To: {to_email}\n"
            f"Time: {datetime.now().isoformat()}"
        )
        msg["Subject"] = "smtpBERT: Phishing URL Blocked"
        msg["From"] = MAILGUN_FROM
        msg["To"] = ALERT_TO

        with smtplib.SMTP(MAILGUN_SMTP_HOST, MAILGUN_SMTP_PORT) as server:
            server.starttls()
            server.login(MAILGUN_LOGIN, MAILGUN_PASSWORD)
            server.sendmail(MAILGUN_FROM, [ALERT_TO], msg.as_string())
        logger.info("Alert sent to %s", ALERT_TO)
    except Exception as e:
        logger.error("Failed to send alert: %s", str(e))