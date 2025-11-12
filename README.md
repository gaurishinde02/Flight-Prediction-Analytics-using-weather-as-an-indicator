# ✈️ Flight Delay Prediction Using Weather as an Indicator

🎯 **Live Demo:** https://flight-prediction-analytics-using-weather-as-an-indicator-abgd.streamlit.app/

A machine-learning project that predicts **flight departure delays** using a combination of **flight schedule data** and **weather indicators**.  
It demonstrates the complete ML lifecycle — from **data collection and preprocessing**, through **feature selection and model training**, to **an interactive Streamlit web app** for prediction.

---

## 🧭 Overview

Airline delays impact both passengers and operations.  
This project leverages meteorological and flight-related variables to **predict flight delays** and identify how different weather factors contribute to disruptions.  
The trained model is deployed via **Streamlit Cloud**, allowing users to interactively estimate delay probabilities.

---

## 🗂️ Project Structure

```text
data/
  raw/                        # Original weather and flight datasets
  processed/                  # Cleaned and transformed dataset for modeling
notebooks/
  01_eda.ipynb                # Exploratory Data Analysis and visualization
  02_model_training.ipynb     # Feature selection, encoding, model training
models/
  rfc_smote.pkl               # Trained Random Forest model (SMOTE-balanced)
  encoders.pkl                # Label encoders for categorical features
app/
  streamlit_app.py            # Streamlit user interface for predictions
requirements.txt
README.md
