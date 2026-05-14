"""
smtpBERT Dashboard - Gradio Version
====================================
Gradio-based dashboard for HuggingFace Spaces deployment.
Compatible with Gradio SDK on HuggingFace Spaces.
"""

import gradio as gr
import pandas as pd
import os

from src.gateway.detector import PhishingDetector

LABEL_COLORS = {
    "Benign": "green",
    "Phishing": "orange",
    "Malware": "red",
    "Defacement": "purple"
}

MODEL_OPTIONS = [
    "DistilBERT (Fine-tuned)",
    "XGBoost Baseline",
    "Logistic Regression",
    "Random Forest"
]

detector = None

def get_detector():
    global detector
    if detector is None:
        detector = PhishingDetector()
    return detector

def analyze_url(url: str, model_name: str):
    """Analyze a URL and return classification results."""
    if not url or not url.strip():
        return "Please enter a URL", {}, "No URL provided"

    detector = get_detector()
    result = detector.predict(url.strip(), model_name=model_name)

    label = result["label"]
    confidence = result["probability"]
    blocked = result["blocked"]
    class_probs = result["class_probabilities"]
    decision_reason = result.get("decision_reason", "")

    prob_df = pd.DataFrame({
        "Class": ["Benign", "Phishing", "Malware", "Defacement"],
        "Probability": class_probs
    })

    output_text = f"**Prediction:** {label}\n"
    output_text += f"**Confidence:** {confidence:.1%}\n"
    output_text += f"**Blocked:** {'YES' if blocked else 'NO'}\n"
    output_text += f"**Model Used:** {model_name}"

    details = {
        "Benign": f"{class_probs[0]:.1%}",
        "Phishing": f"{class_probs[1]:.1%}",
        "Malware": f"{class_probs[2]:.1%}",
        "Defacement": f"{class_probs[3]:.1%}"
    }

    return output_text, details, decision_reason

def create_demo():
    with gr.Blocks(title="smtpBERT - URL Phishing Detection") as demo:
        gr.Markdown("""
        # smtpBERT - URL Phishing Detection Dashboard

        AI-powered URL phishing detection using DistilBERT and XGBoost for multi-class malicious URL classification.
        """)

        with gr.Row():
            with gr.Column(scale=3):
                url_input = gr.Textbox(
                    label="Enter URL to analyze",
                    placeholder="https://example.com/login",
                    lines=1
                )
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=MODEL_OPTIONS,
                    value="DistilBERT (Fine-tuned)",
                    label="Select Model"
                )

        analyze_btn = gr.Button("Analyze URL", variant="primary")

        with gr.Row():
            output_text = gr.Markdown()
            output_details = gr.JSON(label="Class Probabilities")

        decision_output = gr.Textbox(label="Decision Reason", lines=2)

        gr.Examples(
            examples=[
                ["https://google.com/search?q=test", "DistilBERT (Fine-tuned)"],
                ["https://google.com/signup-login0pswwd", "DistilBERT (Fine-tuned)"],
                ["http://malicious-phishing-site.com/login", "DistilBERT (Fine-tuned)"],
                ["http://123.45.67.89/login.php", "DistilBERT (Fine-tuned)"],
                ["g85ogle.com/123iuklansdansdoansd", "DistilBERT (Fine-tuned)"],
            ],
            inputs=[url_input, model_dropdown],
        )

        analyze_btn.click(
            fn=analyze_url,
            inputs=[url_input, model_dropdown],
            outputs=[output_text, output_details, decision_output]
        )

        gr.Markdown("""
        ---

        ## About

        This dashboard uses machine learning models to classify URLs into four categories:

        | Class | Description |
        |-------|-------------|
        | **Benign** | Legitimate, safe URLs |
        | **Phishing** | Credential theft attempts |
        | **Malware** | Malicious software distribution |
        | **Defacement** | Website defacement URLs |

        ### Models Available

        - **DistilBERT (Fine-tuned)**: Transformer-based URL classifier
        - **XGBoost Baseline**: Gradient boosted trees with TF-IDF features
        - **Logistic Regression**: Linear baseline classifier
        - **Random Forest**: Ensemble of decision trees

        ### Features

        - Real-time URL classification
        - Multiple model comparison
        - Trusted domain handling
        - Typosquatting detection

        ---
        *Powered by smtpBERT - Mapúa University*
        """)

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()