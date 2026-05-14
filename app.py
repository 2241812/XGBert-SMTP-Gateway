"""
smtpBERT Dashboard
==================
Streamlit-based dashboard for URL phishing detection.
Compatible with Streamlit Cloud deployment.

Usage:
    streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app import main

if __name__ == "__main__":
    main()