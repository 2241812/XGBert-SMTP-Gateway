import logging
import os
from typing import Any, Dict

from src.models.model_loader import predict_url

logger = logging.getLogger("gateway")

LABEL_MAP = {0: "Benign", 1: "Phishing", 2: "Malware", 3: "Defacement"}


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

    def predict(self, url: str) -> Dict[str, Any]:
        result = predict_url(url)
        prediction = result["prediction"]
        confidence = result["probability"]
        malicious_probability = result["malicious_probability"]

        blocked, reason = self._should_block(
            prediction, confidence, malicious_probability
        )

        return {
            **result,
            "blocked": blocked,
            "decision_reason": reason,
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
