# Deploying smtpBERT Dashboard on Streamlit Cloud

## Quick Deployment

1. **Go to [streamlit.io/cloud](https://streamlit.io/cloud)**

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Configure the deployment:**
   - Repository: `2241812/XGBert-SMTP-Gateway`
   - Branch: `main`
   - Main file path: `app.py`

5. **Click "Deploy!"**

The dashboard will be available at a permanent URL like: `https://your-app-name.streamlit.app`

---

## How It Works

- `app.py` in the root is the Streamlit Cloud entry point
- It imports and runs `src/app.py` which contains the actual dashboard
- Dependencies are installed from `requirements.txt`

---

## Accessing the Deployed Dashboard

Once deployed, anyone with the URL can:
- Test URLs against the phishing classifier
- Select different models (DistilBERT, XGBoost, Logistic Regression, Random Forest)
- View model information and class probabilities

---

## Note on Model Downloads

The first time the dashboard loads, it will download the HuggingFace model (~15MB) from the internet. This may take a moment on first load but will be cached afterward.