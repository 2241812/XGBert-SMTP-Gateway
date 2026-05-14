---
title: smtpBERT
emoji: 🔒
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# smtpBERT - URL Phishing Detection Dashboard

AI-powered URL phishing detection using DistilBERT and XGBoost for multi-class malicious URL classification.

## Features

- **URL Classification**: Detect Phishing, Malware, Defacement, and Benign URLs
- **Multiple Models**: Compare DistilBERT (Fine-tuned), XGBoost, Logistic Regression, and Random Forest
- **Real-time Analysis**: Enter any URL for instant classification
- **Trusted Domain Handling**: Whitelisted domains (Google, Microsoft, etc.) are handled appropriately
- **Typosquatting Detection**: Identifies look-alike domains

## Models Used

| Model | Description |
|-------|-------------|
| DistilBERT (Fine-tuned) | Transformer-based URL classifier |
| XGBoost Baseline | Gradient boosted trees with TF-IDF features |
| Logistic Regression | Linear baseline classifier |
| Random Forest | Ensemble of decision trees |

## Classes

- Benign (Legitimate URLs)
- Phishing (Credential theft URLs)
- Malware (Malicious software distribution)
- Defacement (Website defacement URLs)

## How to Use

1. Enter a URL in the input field
2. Select a model from the dropdown
3. Click "Analyze URL" to classify

## Technical Details

- **DistilBERT**: 66M parameters, fine-tuned on URL classification
- **XGBoost**: 200 trees, max depth 20, TF-IDF + handcrafted features
- **Max URL Length**: 512 tokens for transformer

## Repository

For the full project code, training data, and model files, visit:
https://github.com/2241812/XGBert-SMTP-Gateway