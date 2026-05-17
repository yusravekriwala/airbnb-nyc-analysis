# NYC Airbnb Price Analysis & ML

**Clustering market segments and predicting nightly prices across ~102k NYC Airbnb listings.**

This project was built as part of my Master's in Data Analytics at Ca' Foscari University of Venice. It covers the full data science pipeline: exploratory analysis, cleaning, unsupervised clustering, and supervised regression with hyperparameter tuning.

---

## What This Project Does

Two questions drive the analysis:

1. **Can we cluster listings into meaningful market segments?** → K-Means with elbow + silhouette validation
2. **Can we predict nightly price from listing features?** → Three models compared: Linear Regression, Decision Tree, and Random Forest

---

## Results at a Glance

| Model | R² | Notes |
|---|---|---|
| Linear Regression | Low | Price is not linearly separable |
| Decision Tree | Medium | Overfit without tuning |
| **Random Forest (tuned)** | **Best** | Clear winner; neighbourhood + room type are top drivers |

**Clusters found (k=3):** Budget shared rooms · Mid-range private rooms · Premium entire apartments

Top price drivers: neighbourhood, room type, geographic coordinates (lat/long).

---

## Project Structure

```
├── airbnb_analysis.ipynb   # Main notebook — run top to bottom
├── README.md
└── requirements.txt
```

> **Dataset not included** (100MB+). Download `Airbnb_Open_Data.csv` from [Kaggle](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata) and place it in this folder before running.

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yusravekriwala/airbnb-nyc-analysis
cd airbnb-nyc-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset (see above)

# 4. Launch the notebook
jupyter notebook airbnb_analysis.ipynb
```

---

## Tech Stack

- **Python 3.12** — pandas, NumPy, Matplotlib, Seaborn
- **scikit-learn** — KMeans, MinMaxScaler, OneHotEncoder, RandomForestRegressor, RandomizedSearchCV

---

## Key Design Decisions

- **Log-transform on price** the target has a strong right skew; `log(1+price)` stabilises variance and helps all three regressors.
- **Train/test split before preprocessing**  the scaler is fit only on training data to prevent data leakage.
- **Service fee excluded** it's directly derived from price, so including it would be leaking the answer.
- **Random sampling for K-Means** 10k rows sampled randomly (not `iloc[:10000]`) to avoid any row-ordering bias.

---

## What Could Be Improved

- NLP features from listing descriptions (TF-IDF or sentence embeddings)
- XGBoost / LightGBM for better gradient boosting performance
- Spatial features: distance to subway, landmarks, or CBDs
- Time-aware train/test split to better simulate production conditions

---

## Author

**Yusra Imran Vekriwala**  
MSc Data Analytics for Business and Society; Università Ca' Foscari, Italia 
[LinkedIn](https://www.linkedin.com/in/yusra-vekriwala/) · [GitHub](https://github.com/yusravekriwala)
