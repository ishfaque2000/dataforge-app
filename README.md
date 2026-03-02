# ⚗️ DataForge — ML Data Pipeline

> **Upload dirty data → Full Analysis → Smart Cleaning → Model-Ready Dataset**

A powerful 3-phase Machine Learning data preparation app built with Streamlit. Simply upload your raw/dirty dataset and DataForge will analyze every issue, clean it intelligently, and transform it into a fully model-ready dataset — all with a beautiful dark UI.

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rawdatatransformer.streamlit.app/)

---

## 📸 Features at a Glance

| Phase | What it Does |
|-------|-------------|
| 🔍 **Phase 1 · Analysis** | Full EDA on raw/dirty data — health score, missing values, outliers, distributions, correlations |
| 🧹 **Phase 2 · Cleaning** | Smart cleaning — duplicates, imputation, outlier handling, type fixing |
| 🚀 **Phase 3 · Model Ready** | Encoding, scaling, skewness fix, feature selection — fully ML-ready |

---

## 🔍 Phase 1 — Full Data Analysis

- 📊 **Dataset Health Score** (0–100) with visual indicator
- 📋 **Column-by-Column Summary** — types, missing %, unique values, min/max/mean
- 🕳️ **Missing Values** — heatmap pattern + bar chart with severity levels
- ♊ **Duplicate Detection** — count + preview of duplicate rows
- 📈 **Distribution Plots** — histogram + KDE for every numeric column
- 📦 **Box Plots** — outlier visualization per feature
- 🎯 **Outlier Summary Table** — IQR fences, count, severity (🔴🟡🟢)
- 🧪 **Normality Tests** — Shapiro-Wilk test with pass/fail verdict
- 🏷️ **Categorical Analysis** — bar charts + pie charts per column
- 🔗 **Correlation Heatmap** — with top correlated pairs chart
- 📝 **Smart Recommendations** — auto-detects issues & suggests fixes

---

## 🧹 Phase 2 — Smart Cleaning

- ✂️ Strip whitespace from all text columns
- 🔤 Normalize column names (lowercase, no special characters)
- 🔄 Replace all placeholder nulls (`NA`, `NULL`, `?`, `-`, `none`, etc.)
- 🔢 Auto-detect and convert numeric/datetime columns
- ♊ Remove duplicate rows
- 🗑️ Drop columns/rows with too many missing values (configurable threshold)
- 🩹 **Imputation options:** Median · Mean · KNN
- 📐 **Outlier handling:** IQR Cap 1.5× · IQR Cap 3× · Z-score Remove · Winsorize
- 📊 Before vs After distribution comparison charts

---

## 🚀 Phase 3 — Model Ready

- 📈 **Skewness Correction** — log1p for right-skewed, cbrt for symmetric skewed
- 🏷️ **Encoding options:** Label Encoding · One-Hot Encoding · Frequency Encoding
- 📏 **Scaling options:** StandardScaler · MinMaxScaler · RobustScaler
- 🧹 Remove low-variance features (near-zero std)
- 🔗 Remove highly correlated features (|r| > 0.95)
- 📅 Extract datetime features (year, month, day, day-of-week)
- 🎯 Target column analysis (regression & classification)
- ✅ **ML Readiness Checklist** — 5-point verification before training
- ⬇️ Download both cleaned + model-ready CSVs

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/dataforge-app.git
cd dataforge-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run data_pipeline_app.py
```

---

## 🗂️ File Structure

```
dataforge-app/
│
├── data_pipeline_app.py     # Main Streamlit application
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 📋 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
openpyxl
```

> Python 3.8+ recommended

---

## 📁 Supported File Formats

| Format | Extension |
|--------|-----------|
| CSV | `.csv` |
| Excel | `.xlsx` / `.xls` |
| JSON | `.json` |

---

## ⚙️ Configurable Options

### Cleaning Options
| Option | Choices |
|--------|---------|
| Missing Value Imputation | Median, Mean, KNN |
| Outlier Handling | IQR 1.5×, IQR 3×, Z-score Remove, Winsorize |
| Drop high-missing columns | Configurable threshold (30–90%) |
| Drop high-missing rows | Configurable threshold (30–90%) |

### Model Prep Options
| Option | Choices |
|--------|---------|
| Encoding | Label, One-Hot, Frequency |
| Scaling | StandardScaler, MinMaxScaler, RobustScaler |
| Skewness Fix | log1p / cbrt auto-applied |
| Feature Selection | Remove low-variance & high-correlation |

---

## 🖥️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)

---

## 👨‍💻 Author

Ishfaque Ahmed

---

## ⭐ Support

If you find this useful, please **star the repository** ⭐ — it means a lot!
