"""
Rule-Based URL Classifier for Demo Purposes
============================================
A simple heuristic classifier that works logically for demonstration,
without requiring ML model inference. Can be swapped for real ML model later.

Usage:
    - Detects phishing/malware patterns in URL structure
    - Uses domain reputation and URL patterns
    - Returns proper probability distributions
"""

import re
from typing import Dict, Any, List, Tuple
from urllib.parse import urlparse

LABEL_MAP = {0: "Benign", 1: "Phishing", 2: "Malware", 3: "Defacement"}

# Suspicious TLDs commonly used in phishing/malware campaigns
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "pw", "club", "zip", "loan",
    "work", "racing", "win", "review", "country", "stream", "download", "trade"
}

# Legitimate domains that should almost always be benign
TRUSTED_DOMAINS = {
    "google.com", "microsoft.com", "apple.com", "amazon.com", "facebook.com",
    "twitter.com", "instagram.com", "linkedin.com", "github.com", "youtube.com",
    "wikipedia.org", "reddit.com", "netflix.com", "spotify.com", "ebay.com",
    "paypal.com", "yahoo.com", "bing.com", "duckduckgo.com", "amazon.co.uk",
    "google.co.uk", "microsoft.com", "apple.com", "bbc.co.uk", "cnn.com",
    "nytimes.com", "washingtonpost.com", "forbes.com", "reuters.com", "bbc.com"
}

# Suspicious keywords in paths that often indicate phishing/malware
SUSPICIOUS_PATH_KEYWORDS = {
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "password", "credential", "authenticate", "alert", "suspended",
    "unusual", "suspicious", "paypal", "appleid", "microsoft", "wallet"
}

# Legitimate path keywords
LEGIT_PATH_KEYWORDS = {
    "search", "query", "help", "support", "about", "contact", "faq", "terms",
    "privacy", "policy", "blog", "news", "shop", "cart", "checkout", "product"
}

# IP address pattern
IP_PATTERN = re.compile(r'http[s]?://(?:\d{1,3}\.){3}\d{1,3}')

# Dots in subdomain count (> 3 is suspicious)
SUBDOMAIN_DOT_THRESHOLD = 4

# Long URL threshold (relative to domain length)
LENGTH_RATIO_THRESHOLD = 5.0


def extract_domain_parts(url: str) -> Tuple[str, str, List[str]]:
    """Extract registered domain, TLD, and subdomains."""
    try:
        parsed = urlparse(url if url.startswith(("http://", "https://")) else "https://" + url)
        hostname = parsed.hostname or parsed.path.split("/")[0]
    except Exception:
        return "", "", []

    hostname = hostname.lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    parts = hostname.split(".")
    if len(parts) >= 2:
        if len(parts) >= 3 and parts[-2] in ("co", "com", "net", "org", "gov", "ac"):
            tld = ".".join(parts[-2:])
            domain = parts[-3] if len(parts) >= 3 else parts[0]
            subdomains = parts[:-3] if len(parts) > 3 else []
        else:
            tld = ".".join(parts[-2:])
            domain = parts[-2] if len(parts) >= 2 else parts[0]
            subdomains = parts[:-2] if len(parts) > 2 else []
        return domain, tld, subdomains
    return hostname, "", []


def calculate_suspicion_score(url: str) -> Tuple[float, List[str]]:
    """
    Calculate suspicion score (0-1) based on URL features.
    Returns (score, list of reasons)
    """
    reasons = []
    score = 0.0

    try:
        parsed = urlparse(url if url.startswith(("http://", "https://")) else "https://" + url)
        full_hostname = parsed.hostname or parsed.path.split("/")[0]
        path = parsed.path
        query = parsed.query
    except Exception:
        return 1.0, ["Invalid URL format"]

    hostname = full_hostname.lower() if full_hostname else ""
    if hostname.startswith("www."):
        hostname = hostname[4:]

    domain, tld, subdomains = extract_domain_parts(url)

    # Check 1: Trusted domain (very low score)
    if hostname in TRUSTED_DOMAINS:
        return 0.05, ["Trusted domain (Google, Microsoft, etc.)"]

    # Check 2: IP address as domain (high suspicion)
    if IP_PATTERN.match(url.lower()):
        score += 0.5
        reasons.append("IP address instead of domain name")

    # Check 3: Suspicious TLD
    if tld in SUSPICIOUS_TLDS:
        score += 0.35
        reasons.append(f"Suspicious TLD: .{tld}")

    # Check 4: Too many subdomains
    if len(subdomains) >= SUBDOMAIN_DOT_THRESHOLD:
        score += 0.25
        reasons.append(f"Excessive subdomains ({len(subdomains)})")

    # Check 5: Long URL relative to domain
    if hostname:
        url_length = len(url)
        domain_length = len(hostname)
        if domain_length > 0 and url_length / domain_length > LENGTH_RATIO_THRESHOLD:
            score += 0.2
            reasons.append("Unusually long URL for the domain")

    # Check 6: Suspicious keywords in path
    path_lower = (path + "?" + query).lower()
    suspicious_count = sum(1 for kw in SUSPICIOUS_PATH_KEYWORDS if kw in path_lower)
    if suspicious_count >= 2:
        score += 0.25
        reasons.append(f"Multiple suspicious keywords ({suspicious_count})")
    elif suspicious_count == 1:
        score += 0.1
        reasons.append("Contains suspicious keyword")

    # Check 7: URL-encoded characters (often used in obfuscation)
    if "%" in url or "%20" in url.lower():
        score += 0.1
        reasons.append("Contains URL encoding")

    # Check 8: @ symbol (often used in phishing)
    if "@" in url:
        score += 0.4
        reasons.append("Contains @ symbol (possible phishing obfuscation)")

    # Check 9: HTTPS vs HTTP (HTTP is slightly more suspicious)
    if url.lower().startswith("http://"):
        score += 0.05
        reasons.append("Uses insecure HTTP")

    # Check 10: Numbers in domain (some legitimate, but can indicate phishing)
    domain_part = domain if domain else hostname
    if domain_part and any(c.isdigit() for c in domain_part):
        # Count digit sequences
        digit_sequences = re.findall(r'\d+', domain_part)
        for seq in digit_sequences:
            if len(seq) >= 4:  # Long number sequences are suspicious
                score += 0.15
                reasons.append(f"Contains long digit sequence: {seq}")
                break

    # Check 11: Hyphen count in hostname
    if hostname.count('-') >= 3:
        score += 0.2
        reasons.append(f"Multiple hyphens in domain ({hostname.count('-')})")

    # Check 12: Look-alike characters (e.g., g00gle instead of google)
    lookalike_patterns = [
        (r'g00gle', 'google'),
        (r'm1crosoft', 'microsoft'),
        (r'app1e', 'apple'),
        (r'amaz0n', 'amazon'),
        (r'paypa1', 'paypal'),
    ]
    for pattern, brand in lookalike_patterns:
        if re.search(pattern, hostname or ''):
            score += 0.5
            reasons.append(f"Look-alike domain impersonating {brand}")
            break

    # Normalize score to 0-1 range
    score = min(1.0, score)

    return score, reasons


def classify_url(url: str) -> Dict[str, Any]:
    """
    Classify a URL using rule-based heuristics.

    Returns:
        dict with keys: prediction, label, probability, class_probabilities,
                       malicious_probability, suspicion_score, reasons
    """
    suspicion_score, reasons = calculate_suspicion_score(url)

    # Convert suspicion score to class probabilities
    # Low suspicion -> Benign, High suspicion -> Phishing/Malware/Defacement

    if suspicion_score < 0.2:
        # Very likely benign
        class_probs = [
            0.95 + (0.05 * (1 - suspicion_score / 0.2)),  # Benign: 95-100%
            0.03 * (suspicion_score / 0.2),               # Phishing: 0-3%
            0.01 * (suspicion_score / 0.2),               # Malware: 0-1%
            0.01 * (suspicion_score / 0.2),               # Defacement: 0-1%
        ]
        prediction = 0
    elif suspicion_score < 0.4:
        # Probably benign with some suspicion
        base = suspicion_score / 0.4
        class_probs = [
            0.7 - (0.3 * base),   # Benign: 70-40%
            0.2 + (0.4 * base),   # Phishing: 20-60%
            0.05 + (0.1 * base),  # Malware: 5-15%
            0.05 + (0.1 * base),  # Defacement: 5-15%
        ]
        prediction = 0 if suspicion_score < 0.3 else 1
    elif suspicion_score < 0.6:
        # Suspicious - likely phishing
        base = (suspicion_score - 0.4) / 0.2
        class_probs = [
            0.2 - (0.1 * base),   # Benign: 20-10%
            0.5 + (0.3 * base),   # Phishing: 50-80%
            0.15 + (0.1 * base),  # Malware: 15-25%
            0.15 - (0.05 * base), # Defacement: 15-10%
        ]
        prediction = 1
    elif suspicion_score < 0.8:
        # Likely malicious
        base = (suspicion_score - 0.6) / 0.2
        class_probs = [
            0.05,                                          # Benign: 5%
            0.5 + (0.2 * base),                           # Phishing: 50-70%
            0.25 + (0.15 * base),                         # Malware: 25-40%
            0.2 - (0.1 * base),                           # Defacement: 20-10%
        ]
        prediction = 1
    else:
        # Highly likely malicious
        base = (suspicion_score - 0.8) / 0.2
        class_probs = [
            0.02,                                          # Benign: 2%
            0.4 + (0.3 * base),                           # Phishing: 40-70%
            0.35 + (0.2 * base),                           # Malware: 35-55%
            0.25 - (0.15 * base),                          # Defacement: 25-10%
        ]
        prediction = 1 if suspicion_score < 0.9 else 2

    # Normalize probabilities to sum to 1
    total = sum(class_probs)
    class_probs = [p / total for p in class_probs]

    # Find confidence (highest probability)
    max_prob = max(class_probs)
    confidence = max_prob

    # Determine blocking
    malicious_prob = class_probs[1] + class_probs[2] + class_probs[3]

    # Blocking logic: block if malicious_prob > 0.5 or if high suspicion
    blocked = malicious_prob > 0.5 or suspicion_score >= 0.6

    reason_str = "; ".join(reasons) if reasons else "No obvious suspicious indicators"

    return {
        "prediction": prediction,
        "label": LABEL_MAP[prediction],
        "probability": float(confidence),
        "class_probabilities": [float(p) for p in class_probs],
        "malicious_probability": float(malicious_prob),
        "suspicion_score": float(suspicion_score),
        "reasons": reasons,
        "decision_reason": reason_str,
        "blocked": blocked,
        "whitelisted": False,
        "whitelisted_domain": None,
    }


def demo_classifier():
    """Demo the classifier with various URLs."""
    test_urls = [
        ("https://google.com/search?q=test", "Benign"),
        ("https://google.com/signup-login0pswwd", "Should be Benign"),
        ("https://accounts.google.com/signin", "Should be Benign"),
        ("http://malicious-phishing.example.com/login", "Phishing"),
        ("https://safe-bank.com/portal", "Benign"),
        ("http://malware-download.evil.org/payload.exe", "Malware"),
        ("http://fake-paypal.scam.net/login", "Phishing"),
        ("http://defaced-site.vandal.com/index", "Defacement"),
        ("https://github.com/search", "Benign"),
        ("http://123.45.67.89/login.php", "Phishing (IP)"),
        ("https://www.google.com.xyz/phishing", "Phishing (suspicious TLD)"),
        ("https://microsoft.com.evil.phishing.com/secure", "Phishing (lookalike)"),
    ]

    print("=" * 70)
    print("RULE-BASED URL CLASSIFIER DEMO")
    print("=" * 70)

    for url, expected in test_urls:
        result = classify_url(url)
        print(f"\nURL: {url}")
        print(f"  Expected: {expected}")
        print(f"  Prediction: {result['label']} (confidence: {result['probability']:.1%})")
        print(f"  Suspicion Score: {result['suspicion_score']:.2f}")
        print(f"  Reasons: {result['reasons']}")
        print(f"  Blocked: {result['blocked']}")


if __name__ == "__main__":
    demo_classifier()