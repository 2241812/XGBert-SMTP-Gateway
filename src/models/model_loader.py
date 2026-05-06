"""DistilBERT model loader for URL classification."""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_MAP = {0: "Benign", 1: "Phishing", 2: "Malware", 3: "Defacement"}


class DistilBERTLoader:
    """Loads and manages DistilBERT model for inference."""

    def __init__(self, model_dir=None):
        if model_dir is None:
            # Default to phishing_model in project root (two levels up from src/models)
            self.model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "phishing_model"
            )
        else:
            self.model_dir = model_dir
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the DistilBERT model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, url: str) -> dict:
        """
        Predict the class of a URL.

        Args:
            url (str): The URL to classify

        Returns:
            dict: Prediction results including class, probabilities, etc.
        """
        inputs = self.tokenizer(
            url,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            class_probs = probs[0].detach().cpu().tolist()

        return {
            "prediction": prediction,
            "label": LABEL_MAP[prediction],
            "probability": float(class_probs[prediction]),
            "class_probabilities": class_probs,
            "malicious_probability": float(sum(class_probs[1:])),
        }


# Singleton instance for convenience
_model_loader = None


def get_model_loader():
    """Get or create the DistilBERT model loader singleton."""
    global _model_loader
    if _model_loader is None:
        _model_loader = DistilBERTLoader()
    return _model_loader


def predict_url(url: str) -> dict:
    """
    Predict the class of a URL using the DistilBERT model.

    Args:
        url (str): The URL to classify

    Returns:
        dict: Prediction results
    """
    loader = get_model_loader()
    return loader.predict(url)