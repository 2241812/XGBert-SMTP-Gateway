import json
import logging
import os
import threading
import time
from datetime import datetime

from .config import BASE_DIR

logger = logging.getLogger("gateway")

METRICS_FILE = os.environ.get("GATEWAY_METRICS_FILE", os.path.join(BASE_DIR, "logs", "gateway_metrics.json"))
METRICS_FLUSH_INTERVAL_SEC = int(os.environ.get("METRICS_FLUSH_INTERVAL_SEC", "60"))


class GatewayMetricsTracker:
    """Collect and persist SMTP gateway runtime metrics."""
    def __init__(self, metrics_file: str = METRICS_FILE, flush_interval_sec: int = METRICS_FLUSH_INTERVAL_SEC):
        self.metrics_file = metrics_file
        self.flush_interval_sec = flush_interval_sec
        self._lock = threading.Lock()
        self._last_flush = 0.0
        self._metrics = self._initial_metrics()
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        self.flush(force=True)

    def record_email(self, url_count: int) -> None:
        """Record email counters and URL volume."""
        with self._lock:
            self._metrics["emails_total"] += 1
            self._metrics["urls_total"] += int(url_count)
            if url_count > 0:
                self._metrics["emails_with_urls"] += 1
            self._flush_if_needed()

    def record_cache_result(self, cache_hit: bool) -> None:
        """Record cache hit or miss for a URL lookup."""
        with self._lock:
            if cache_hit:
                self._metrics["cache_hits"] += 1
            else:
                self._metrics["cache_misses"] += 1
            self._flush_if_needed()

    def record_prediction(self, prediction: int, malicious_probability: float, blocked: bool) -> None:
        """Record model prediction distribution and block outcomes."""
        with self._lock:
            self._metrics["class_counts"][str(int(prediction))] += 1
            self._metrics["malicious_probability_sum"] += float(malicious_probability)
            if blocked:
                self._metrics["blocked_urls"] += 1
            self._flush_if_needed()

    def record_decision(self, accepted: bool, latency_ms: float, had_error: bool = False) -> None:
        """Record final SMTP accept/reject decision and latency."""
        with self._lock:
            if accepted:
                self._metrics["accepted_emails"] += 1
            else:
                self._metrics["rejected_emails"] += 1
            if had_error:
                self._metrics["errors"] += 1
            self._metrics["latency_ms_sum"] += float(latency_ms)
            self._metrics["latency_samples"] += 1
            self._flush_if_needed()

    def flush(self, force: bool = False) -> None:
        """Persist current metrics snapshot to disk."""
        with self._lock:
            now = time.time()
            if not force and now - self._last_flush < self.flush_interval_sec:
                return
            snapshot = self._snapshot_unlocked()
            with open(self.metrics_file, "w", encoding="utf-8") as file:
                json.dump(snapshot, file, indent=2)
            self._last_flush = now

    def _flush_if_needed(self) -> None:
        now = time.time()
        if now - self._last_flush >= self.flush_interval_sec:
            snapshot = self._snapshot_unlocked()
            with open(self.metrics_file, "w", encoding="utf-8") as file:
                json.dump(snapshot, file, indent=2)
            self._last_flush = now

    def _snapshot_unlocked(self) -> dict:
        emails_total = self._metrics["emails_total"]
        decisions_total = self._metrics["accepted_emails"] + self._metrics["rejected_emails"]
        reject_rate = (self._metrics["rejected_emails"] / max(decisions_total, 1)) * 100
        cache_total = self._metrics["cache_hits"] + self._metrics["cache_misses"]
        cache_hit_rate = (self._metrics["cache_hits"] / max(cache_total, 1)) * 100
        avg_latency_ms = self._metrics["latency_ms_sum"] / max(self._metrics["latency_samples"], 1)
        avg_malicious_probability = self._metrics["malicious_probability_sum"] / max(
            sum(self._metrics["class_counts"].values()), 1
        )
        return {
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "metrics": dict(self._metrics),
            "derived": {
                "reject_rate_percent": round(reject_rate, 3),
                "cache_hit_rate_percent": round(cache_hit_rate, 3),
                "avg_latency_ms": round(avg_latency_ms, 3),
                "avg_malicious_probability": round(avg_malicious_probability, 5),
                "emails_total": emails_total,
            },
        }

    @staticmethod
    def _initial_metrics() -> dict:
        return {
            "emails_total": 0,
            "emails_with_urls": 0,
            "urls_total": 0,
            "accepted_emails": 0,
            "rejected_emails": 0,
            "blocked_urls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "latency_ms_sum": 0.0,
            "latency_samples": 0,
            "malicious_probability_sum": 0.0,
            "class_counts": {"0": 0, "1": 0, "2": 0, "3": 0},
        }


_tracker: GatewayMetricsTracker | None = None


def get_metrics_tracker() -> GatewayMetricsTracker:
    """Return a process-wide singleton gateway metrics tracker."""
    global _tracker
    if _tracker is None:
        _tracker = GatewayMetricsTracker()
    return _tracker
