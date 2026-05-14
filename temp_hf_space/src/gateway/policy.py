import re

LABEL_MAP = {0: "Benign", 1: "Phishing", 2: "Malware", 3: "Defacement"}
URL_PATTERN = re.compile(r'https?://[^\s<>"\')\]]+')


def extract_urls(text: str) -> list[str]:
    """Extract URL-like strings from email body text."""
    return URL_PATTERN.findall(text)


def label_name(label: int) -> str:
    """Map numeric class labels to a readable label name."""
    return LABEL_MAP.get(label, f"Class-{label}")
