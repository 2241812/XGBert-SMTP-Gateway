import sqlite3
import logging

from .config import DB_PATH

logger = logging.getLogger("gateway")


def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS malicious_urls
           (url TEXT PRIMARY KEY, label INTEGER DEFAULT 1, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized at %s with WAL mode enabled", DB_PATH)


def check_url_cache(url):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    cursor.execute("SELECT label FROM malicious_urls WHERE url = ?", (url,))
    result = cursor.fetchone()
    conn.close()
    if result:
        logger.info("Cache hit for URL: %s (label=%d)", url, result[0])
        return result[0]
    return None


def insert_malicious_url(url, label=1):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO malicious_urls (url, label) VALUES (?, ?)", (url, int(label)))
        conn.commit()
        logger.info("Inserted malicious URL into cache: %s (label=%d)", url, int(label))
    except sqlite3.IntegrityError:
        logger.info("Malicious URL already exists in cache: %s", url)
    finally:
        conn.close()