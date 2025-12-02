# Diabetes Prediction Web Application  
A full-stack machine learning application that predicts the likelihood of diabetes using four clinically relevant biomarkers.  
Built with **Python, Flask, Scikit-Learn, HTML, CSS, and JavaScript** — and deployed on **Render**.

<p align="left">
  <img src="https://img.shields.io/badge/ML-Random%20Forest-blueviolet" />
  <img src="https://img.shields.io/badge/Backend-Flask-black" />
  <img src="https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20JS-green" />
  <img src="https://img.shields.io/badge/Data-Synthetic%20500%20Patients-orange" />
  <img src="https://img.shields.io/badge/Deployed%20On-Render.com-blue" />
</p>

---

## 📌 Overview
This project simulates a real clinical workflow: the user enters four medical measurements, the backend machine learning model evaluates the inputs, and the system returns a prediction with confidence scores — all without refreshing the page.

To avoid external datasets, the backend generates a **synthetic dataset of 500 patients** on startup using realistic medical ranges and controlled noise. A Random Forest classifier is trained on this dataset, and the app exposes a `/predict` API endpoint for real-time inference.

The frontend is built as a clean, responsive medical dashboard with animated probability bars and interactive tooltips.

---

## 🚀 Features

### 🔬 Machine Learning
- Synthetic dataset of **500 medically realistic patient profiles**
- Values follow known diabetes diagnostic thresholds (FPG, OGTT, HbA1c, etc.)
- Controlled noise and overlapping distributions
- Random Forest classifier with probability output
- Train/test split + preprocessing pipeline

### 🧩 Backend (Flask)
- `/predict` REST endpoint returning structured JSON predictions  
- Confidence scores for both classes  
- Input validation and graceful error responses  
- ML training executed automatically on startup  

### 🎨 Frontend (HTML/CSS/JS)
- Modern, professional **medical dashboard UI**
- Animated probability bars  
- Input validation and user-friendly error states  
- Smooth transitions and responsive layout  
- Helpful tooltips explaining each medical parameter  
- Keyboard shortcuts for quick testing (Ctrl+D, Ctrl+N)

### ☁️ Deployment
- Backend deployed on **Render.com**  
- Lightweight frontend, compatible with any hosting setup  

---

## 📂 Project Structure
