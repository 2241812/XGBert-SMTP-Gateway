import logging
import os
from typing import Any, Dict
from urllib.parse import urlparse

logger = logging.getLogger("gateway")

LABEL_MAP = {0: "Benign", 1: "Phishing", 2: "Malware", 3: "Defacement"}

USE_RULE_BASED = os.environ.get("USE_RULE_BASED", "false").lower() == "true"
USE_DISTILBERT = os.environ.get("USE_DISTILBERT", "false").lower() == "true"

if USE_RULE_BASED:
    from src.gateway.url_classifier import classify_url as _ml_predict
elif USE_DISTILBERT:
    from src.models.model_loader import predict_url as _ml_predict
else:
    from src.models.url_model_loader import predict_url as _ml_predict


# =============================================================================
# WHITELIST: Known safe domains — checked BEFORE ML model to prevent false positives
# =============================================================================
# These domains are verified legitimate and will NEVER be blocked by the model.
# This prevents the ML model from misclassifying well-known sites as phishing.
#
# IMPORTANT: _get_domain() strips "www." and normalizes to lowercase,
# so only the base domain (e.g., "google.com") is needed here.
WHITELISTED_DOMAINS = {
    # Search engines
    "google.com",
    "bing.com",
    "yahoo.com",
    "duckduckgo.com",

    # Social media
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "linkedin.com",
    "youtube.com",

    # E-commerce & tech
    "amazon.com",
    "ebay.com",
    "paypal.com",
    "microsoft.com",
    "apple.com",

    # Developer & knowledge
    "github.com",
    "stackoverflow.com",
    "wikipedia.org",
    "reddit.com",
    "netflix.com",
    "spotify.com",

    # Email providers
    "gmail.com",
    "outlook.com",
    "hotmail.com",
}


def _get_domain(url: str) -> str:
    """
    Extract the base domain (registered domain) from a URL.

    Uses urlparse for robust parsing, then extracts the registered domain
    (the last two parts for common TLDs like .com, .org, country codes).

    Examples:
        "https://www.google.com/path?q=1" → "google.com"
        "https://co.uk.google.com/path" → "google.com"
        "http://google.com" → "google.com"
        "google.com" → "google.com"

    Args:
        url: The URL to parse

    Returns:
        Registered domain string (lowercase)
    """
    try:
        parsed = urlparse(url if url.startswith(("http://", "https://")) else "https://" + url)
        hostname = parsed.hostname or parsed.path.split("/")[0]
    except Exception:
        return ""

    if not hostname:
        return ""

    # Normalize to lowercase
    hostname = hostname.lower()

    # Remove leading www.
    if hostname.startswith("www."):
        hostname = hostname[4:]

    # For country-code TLDs like co.uk, com.au, extract the registered domain
    # e.g., www.google.co.uk → google.co.uk
    parts = hostname.split(".")
    if len(parts) >= 2:
        # Common two-part TLDs (country codes + commercial)
        if len(parts) >= 3 and parts[-2] in ("co", "com", "net", "org", "gov", "ac"):
            return ".".join(parts[-3:])
        # Standard domain: take last two parts
        return ".".join(parts[-2:])
    return hostname


def _is_whitelisted(url: str) -> bool:
    """
    Check if a URL belongs to a known safe domain.

    Uses _get_domain() to normalize the URL before checking whitelist.
    This handles google.com, www.google.com, google.com/path, etc.

    Args:
        url: The URL to check

    Returns:
        True if the URL's domain is in the whitelist
    """
    domain = _get_domain(url)

    # Check exact match
    if domain in WHITELISTED_DOMAINS:
        return True

    # Check if domain ends with .edu or .gov (common TLDs for legitimate sites)
    if domain.endswith(".edu") or domain.endswith(".gov"):
        return True

    return False


class PhishingDetector:

    def __init__(self):
        self.block_labels = {1, 2, 3}
        self.malicious_threshold = 0.65
        self.class_thresholds = {1: 0.60, 2: 0.55, 3: 0.55}
        self.model_file_candidates = (
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "phishing_model",
                "model.safetensors",
            ),
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "phishing_model",
                "pytorch_model.bin",
            ),
        )
        self._last_model_time = self._resolve_model_mtime()
        logger.info("PhishingDetector initialized")

    def predict(self, url: str, model_name: str = "DistilBERT (Fine-tuned)") -> Dict[str, Any]:
        result = _ml_predict(url)

        prediction = result["prediction"]
        malicious_prob = result.get("malicious_probability", 0.0)
        confidence = result.get("probability", 0.0)
        is_trusted = result.get("is_trusted", False)
        heuristic_score = result.get("heuristic_score", 0.0)
        heuristic_reasons = result.get("heuristic_reasons", [])
        domain = _get_domain(url)

        blocked = (
            prediction in self.block_labels
            and (confidence > 0.7 or malicious_prob > 0.5)
        )

        if is_trusted and heuristic_score < 0.2:
            return {
                **result,
                "blocked": False,
                "decision_reason": f"Trusted domain ({domain}) - no suspicious patterns detected",
                "whitelisted": True,
                "whitelisted_domain": domain,
                "model_used": model_name,
            }

        if heuristic_score >= 0.3:
            blocked = True
            result["blocked"] = True

        return {
            **result,
            "blocked": blocked,
            "whitelisted": is_trusted and heuristic_score < 0.2,
            "whitelisted_domain": domain if is_trusted else None,
            "model_used": model_name,
        }

    def _should_block(
        self, prediction: int, confidence: float, malicious_probability: float
    ) -> tuple[bool, str]:
        if prediction not in self.block_labels:
            return False, "predicted-benign"

        class_threshold = self.class_thresholds.get(
            prediction, self.malicious_threshold
        )
        confidence_gate = confidence >= class_threshold
        malicious_gate = malicious_probability >= self.malicious_threshold
        blocked = confidence_gate or malicious_gate

        if blocked:
            return (
                True,
                f"blocked(label={prediction}, confidence={confidence:.3f}, "
                f"malicious={malicious_probability:.3f})",
            )
        return (
            False,
            f"below-threshold(label={prediction}, confidence={confidence:.3f}, "
            f"malicious={malicious_probability:.3f})",
        )

    def _resolve_model_mtime(self) -> float:
        for candidate in self.model_file_candidates:
            if os.path.exists(candidate):
                return os.path.getmtime(candidate)
        return 0.0

    def check_reload_needed(self) -> bool:
        """
        Check if a newer model file exists on disk.

        The model loader is process-singleton and lazy; once this returns True, the
        current process should be restarted to pick up the new model cleanly.
        """
        current_time = self._resolve_model_mtime()
        if current_time > self._last_model_time:
            logger.info("New model artifact detected on disk.")
            self._last_model_time = current_time
            return True
        return False
