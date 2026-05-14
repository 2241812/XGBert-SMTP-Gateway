import logging
import time
from email.message import Message
from email.parser import BytesParser

from .alerts import send_phishing_alert
from .database import check_url_cache, insert_malicious_url
from .detector import PhishingDetector
from .monitoring import get_metrics_tracker
from .policy import extract_urls, label_name

logger = logging.getLogger("gateway")


class PhishingSMTPHandler:
    def __init__(self):
        logger.info("Initializing PhishingDetector (loading model once)...")
        self.detector = PhishingDetector()
        self.metrics = get_metrics_tracker()
        logger.info("PhishingDetector initialized successfully")

    async def handle_CONNECT(self, server, session, envelope):
        logger.info(f"CONNECT request from {session.remote_host}")
        return "200 SMTP ready"

    async def handle_DATA(self, server, session, envelope):
        started = time.perf_counter()
        try:
            if self.detector.check_reload_needed():
                logger.info("Updated model detected on disk; restart gateway process to load new weights.")

            email_data = envelope.content
            parser = BytesParser()
            msg: Message = parser.parsebytes(email_data)

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type in ("text/plain", "text/html"):
                        payload = part.get_payload(decode=True)
                        if payload:
                            body += payload.decode("utf-8", errors="ignore") + "\n"
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="ignore")

            urls = extract_urls(body)
            self.metrics.record_email(len(urls))
            logger.info(
                "Email from %s to %s: extracted %d URLs",
                envelope.mail_from,
                envelope.rcpt_tos,
                len(urls),
            )

            if not urls:
                logger.info("No URLs found, accepting email")
                elapsed_ms = (time.perf_counter() - started) * 1000
                self.metrics.record_decision(accepted=True, latency_ms=elapsed_ms)
                return "250 OK: Email accepted"

            for url in urls:
                cached_result = check_url_cache(url)
                self.metrics.record_cache_result(cache_hit=cached_result is not None)
                if cached_result is not None and int(cached_result) in self.detector.block_labels:
                    label = label_name(int(cached_result))
                    logger.warning("BLOCKED: Known malicious URL detected: %s (label=%s)", url, label)
                    self.metrics.record_prediction(int(cached_result), malicious_probability=1.0, blocked=True)
                    send_phishing_alert(
                        url, envelope.mail_from, ",".join(envelope.rcpt_tos)
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    self.metrics.record_decision(accepted=False, latency_ms=elapsed_ms)
                    return f"550 Rejected: Cached malicious URL ({label})"

            for url in urls:
                result = self.detector.predict(url)
                self.metrics.record_prediction(
                    prediction=result["prediction"],
                    malicious_probability=result["malicious_probability"],
                    blocked=result["blocked"],
                )
                if result["blocked"]:
                    label = label_name(result["prediction"])
                    logger.warning(
                        "BLOCKED: AI detected malicious URL: %s (label=%s, confidence=%.3f, malicious=%.3f)",
                        url,
                        label,
                        result["probability"],
                        result["malicious_probability"],
                    )
                    insert_malicious_url(url, label=int(result["prediction"]))
                    send_phishing_alert(
                        url, envelope.mail_from, ",".join(envelope.rcpt_tos)
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    self.metrics.record_decision(accepted=False, latency_ms=elapsed_ms)
                    return (
                        "550 Rejected: Malicious URL detected "
                        f"({label}, confidence={result['probability']:.2f}, "
                        f"threshold_reason={result['decision_reason']})"
                    )
                else:
                    logger.info(
                        "SAFE: URL classified safe: %s (label=%s, confidence=%.3f)",
                        url,
                        label_name(result["prediction"]),
                        result["probability"],
                    )

            logger.info("All URLs safe, accepting email")
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.metrics.record_decision(accepted=True, latency_ms=elapsed_ms)
            return "250 OK: Email accepted"

        except Exception as e:
            logger.error("Error processing email: %s", str(e))
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.metrics.record_decision(accepted=False, latency_ms=elapsed_ms, had_error=True)
            return "451 Temporary error, please try again later"
