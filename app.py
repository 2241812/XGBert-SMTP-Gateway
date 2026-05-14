"""
smtpBERT Dashboard
==================
Streamlit Cloud entry point for the smtpBERT URL phishing detection dashboard.

This file serves as the main entry point when deployed on Streamlit Cloud.
The actual application logic is in src/app.py

Usage:
    streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app import main

if __name__ == "__main__":
    main()