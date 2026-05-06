import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "phishing_model"))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "phishing_data.db"))
SMTP_HOST = os.environ.get("SMTP_HOST", "0.0.0.0")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "25"))

MAILGUN_SMTP_HOST = os.environ.get("MAILGUN_SMTP_HOST", "smtp.mailgun.org")
MAILGUN_SMTP_PORT = int(os.environ.get("MAILGUN_SMTP_PORT", "587"))
MAILGUN_LOGIN = os.environ.get("MAILGUN_LOGIN", "")
MAILGUN_PASSWORD = os.environ.get("MAILGUN_PASSWORD", "")
MAILGUN_FROM = os.environ.get("MAILGUN_FROM", "smtpbert@smtpbert.com")
ALERT_TO = os.environ.get("ALERT_TO", "")

GATEWAY_CONFIG = os.path.join(BASE_DIR, "gateway_config.json")