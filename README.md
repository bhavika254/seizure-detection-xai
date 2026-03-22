# 🧠 EEG-Based Epileptic Seizure Detection with Explainable AI (XAI)

A machine learning project for detecting and classifying epileptic seizures from EEG data with SHAP-based explainability.

---

## 📌 Current Status
- [x] Data Preprocessing
- [ ] Model Training
- [ ] XAI with SHAP

---

## 🗂️ Project Structure

```
seizure-detection-xai/
├── data/                        # ← Place dataset here (see below)
├── outputs/                     # Generated plots
│   ├── class_distribution.png
│   └── correlation_heatmap.png
├── preprocessing.py             
├── .gitignore
└── README.md
```

---

## 📊 Dataset

> ⚠️ Dataset not included due to size limits.

👉 **Download here:** [[ADD GOOGLE DRIVE LINK](https://drive.google.com/file/d/1wqFaYATOI8-SJ05W_2tES2uwcl8lVRat/view?usp=drive_link)]

Place it at: `data/epilepsy_federated_dataset.csv`

---

## ⚙️ Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 🚀 Run Preprocessing

```bash
python preprocessing.py
```

---

## 🌿 Branch
Currently on: `feature/preprocessing`
