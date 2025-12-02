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

To avoid external datasets, the backend generates a **synthetic dataset of 500 patients** on startup using realistic medical ranges and controlled noise. A Random Forest classifier is trained on this data, and the app exposes a `/predict` API endpoint for real-time inference.

The frontend is built as a clean, responsive medical dashboard with animated probability bars and interactive tooltips.

---

## 🚀 Features  

### 🔬 Machine Learning  
- Automatic generation of **500 medically realistic patient profiles**  
- Realistic diagnostic ranges with controlled data noise  
- Random Forest classifier with probability output  
- Train/test split + preprocessing pipeline  

### 🧩 Backend (Flask)  
- `/predict` REST endpoint returning structured JSON  
- Confidence scores for both classes  
- Robust input validation and error handling  
- ML pipeline trained automatically on startup  

### 🎨 Frontend (HTML/CSS/JS)  
- Professional medical dashboard UI  
- Animations for probability bars and result transitions  
- Tooltips explaining each medical parameter  
- Client-side validation + user-friendly errors  
- Keyboard shortcuts (e.g., Ctrl+D and Ctrl+N for test values)  
- Fully responsive design for desktop and mobile  

### ☁️ Deployment  
- Backend hosted on **Render.com**  
- Minimal, fast-loading frontend  

---

## 📂 Project Structure  
project/
├── app.py # Flask backend, ML training, synthetic data generation
├── templates/
│   └── index.html # User interface
└── static/
    ├── style.css # Dashboard styling + animations
    └── script.js # Frontend logic, AJAX, probability visualization

---

## 🧠 How It Works  

1. **Synthetic Data Generation**  
   Generates realistic diabetic and non-diabetic patient distributions (~40% diabetic), with overlapping noise to mimic real clinical uncertainty.

2. **Model Training**  
   Trains a Random Forest classifier using four biomarkers:  
   - FPG  
   - OGTT  
   - Random Plasma Glucose  
   - HbA1c  

3. **Prediction Pipeline**  
   - User submits values via the frontend  
   - JavaScript sends data asynchronously to the backend  
   - Flask returns prediction + class probabilities  
   - UI updates with animated visualizations  

---

## 🛠️ Installation & Running Locally  

### **1. Clone the repository**
```bash
git clone https://github.com/rpolishchuk3/diabetes-prediction-app.git
cd diabetes-prediction-app
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### **3. Start the application**
```bash
python app.py
```

### **4. Open in browser**
```bash
http://localhost:5000
```

---

## 📘 Skills Demonstrated
- Full-stack machine learning development  
- Synthetic data generation  
- REST API design using Flask  
- JavaScript async workflows (fetch, JSON parsing)  
- UI/UX design for medical applications  
- End-to-end ML inference pipeline design  
- Deployment and hosting using Render  

---

## 📈 Future Enhancements
- Persist the trained model using `joblib`  
- Add feature importance visualization  
- Implement multiple ML model options (LogReg, XGBoost, etc.)  
- Add user accounts and prediction history tracking  
- Replace synthetic data with a cloud-hosted database
