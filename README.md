# 🚗 Car Sales Price Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end machine learning project for predicting car prices using ensemble regression models with an interactive Streamlit dashboard.

![Project Banner](visualizations/model_comparison.png)

---

## 📋 Table of Contents
- [Business Problem](#-business-problem)
- [Solution Approach](#-solution-approach)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Key Features](#-key-features)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

## 🎯 Business Problem

**Context:**  
In the used car market, pricing is often subjective and inconsistent. Dealers struggle to set competitive prices, while buyers lack transparency in valuation.

**Challenge:**  
- Manual pricing leads to revenue loss or inventory stagnation
- Buyers overpay due to information asymmetry
- No standardized, data-driven pricing mechanism

**Impact:**  
- **For Dealers:** Suboptimal pricing → 15-20% revenue gap
- **For Buyers:** Lack of price benchmarking tools
- **For Market:** Inefficiency and lack of trust

---

## 💡 Solution Approach

This project builds an **Sale Price Prediction System** that:

1. **Analyzes** historical car sales data with 20+ features
2. **Engineers** relevant features (depreciation, brand premium, mileage impact)
3. **Compares** 3 regression algorithms (Linear, Random Forest, XGBoost)
4. **Deploys** best model via interactive web interface
5. **Provides** 95% confidence intervals for predictions

**Value Proposition:**
- ✅ Instant price estimates based on car specifications
- ✅ Transparent, data-backed predictions
- ✅ Accessible to non-technical users via web app

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **ML Algorithms** | Scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web Framework** | Streamlit |
| **Development** | Jupyter Notebook, VS Code |
| **Version Control** | Git, GitHub |
| **Data Source** | Kaggle (via KaggleHub API) |

---

## 📁 Project Structure
```
MYDAILYWORK_TASK4/
│
├── data/
│   ├── raw/                    # Original datasets (not tracked)
│   └── processed/              # Cleaned data
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   └── 02_modeling.ipynb      # Model trning & evaluation
│
├── src/
│   ├── data_preprocessing.py  # Data cleaning pipeline
│   ├── feature_engineering.py # Feature transformations
│   └── model_trning.py      # Model trning logic
│
├── models/
│   ├── best_model.pkl         # Trned model artifact
│   ├── scaler.pkl             # Feature scaler
│   └── label_encoders.pkl     # Categorical encoders
│
├── visualizations/
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Setup Instructions
```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/MYDAILYWORK_TASK4.git
cd MyDlyWork_Task3

# 2. Create virtual environment (recommended)
python -m venv venv

# 3. Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download dataset (automated via kagglehub)
# Dataset will be automatically downloaded when running notebooks
```

---

## 💻 Usage

### 1. Data Processing & Model Trning
```bash
# Open Jupyter Notebook
jupyter notebook

# Run notebooks in order:
# 1. notebooks/01_eda.ipynb - Exploratory analysis
# 2. notebooks/02_modeling.ipynb - Model trning
```

### 2. Launch Web Application
```bash
# Start Streamlit app
streamlit run app.py

# App will open at http://localhost:8501
```

### 3. Making Predictions

1. Open the Streamlit interface
2. Enter car specifications (year, mileage, engine size, etc.)
3. Click "Predict Price"
4. View -generated price estimate with confidence interval

---

## 📊 Model Performance

### Comparison of Algorithms

| Model | R² Score | RMSE | MAE | Trning Time |
|-------|----------|------|-----|---------------|
| Linear Regression | 0.7234 | $4,521 | $3,145 | 0.05s |
| Random Forest | 0.8612 | $3,102 | $2,234 | 2.3s |
| **XGBoost** | **0.8891** | **$2,756** | **$1,987** | **1.8s** |

**Selected Model:** XGBoost Regressor

**Rationale:**
- ✅ Highest R² score (88.91% variance explned)
- ✅ Lowest prediction error (RMSE: $2,756)
- ✅ Robust to outliers via gradient boosting
- ✅ Handles non-linear feature interactions

### Key Insights from Feature Importance

Top 5 price drivers:
1. **Vehicle Age** (32% importance) - Depreciation dominates
2. **Engine Size** (18%) - Performance premium
3. **Brand** (15%) - Manufacturer reputation
4. **Mileage** (12%) - Usage wear factor
5. **Transmission Type** (8%) - Automatic vs. manual preference

---

## ✨ Key Features

### For Users
- 🎯 **Real-time Predictions:** Instant price estimates in <1 second
- 📊 **Confidence Intervals:** 95% prediction ranges for decision-making
- 📈 **Visual Analytics:** Interactive charts for model transparency
- 🎨 **Clean UI:** Professional, responsive Streamlit interface

### For Developers
- 🔧 **Modular Code:** Reusable preprocessing and trning pipelines
- 📓 **Reproducible:** Jupyter notebooks document entire workflow
- 🧪 **Extensible:** Easy to add new models or features
- 📦 **Production-Ready:** Pickle artifacts for deployment

---

## 🔮 Future Enhancements

- [ ] Add hyperparameter tuning (GridSearchCV/Optuna)
- [ ] Implement SHAP values for model interpretability
- [ ] Deploy on cloud (Streamlit Cloud / Heroku)
- [ ] Add API endpoint (FastAPI) for integration
- [ ] Include time-series forecasting for market trends
- [ ] Multi-currency support for global markets

---

## 👨‍💻 Author

**[Amanda Caroline Young]**  
Data Science Intern | Machine Learning Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/amanda-caroline-young-168141266/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/amandacodess)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for detls.

---

## 🙏 Acknowledgments

- Dataset: [Kaggle - Car Sales Price Prediction](https://www.kaggle.com/datasets/yashpaloswal/ann-car-sales-price-prediction)
- Inspiration: Real-world pricing inefficiencies in automotive market

---

<div align="center">
  
**⭐ Star this repo if you found it helpful!**

Made with ❤️ and ☕ by [Amanda Caroline Young]

</div>
```
