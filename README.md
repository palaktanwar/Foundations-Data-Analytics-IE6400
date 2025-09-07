### Foundations for Data Analytics Engineering (IE 6400)

This repository contains my projects for IE 6400: Foundations for Data Analytics Engineering. The work spans probability and linear algebra fundamentals, data wrangling, unsupervised learning, text mining, and time series analysis. Each project is implemented in Jupyter notebooks with a focus on clear methodology, reproducibility, and effective communication of results.

### About Me
I am pursuing data analytics engineering and created these projects to strengthen practical, end‑to‑end skills: from data preparation and exploratory analysis to modeling, evaluation, and visualization.

### Course Overview
- Probability and linear algebra essentials (eigenvalues/eigenvectors)
- Data cleaning, feature engineering, and wrangling with modern data structures
- Unsupervised learning (clustering) and model selection
- Text mining and representation learning for unstructured data
- Time series analysis and baseline forecasting

### Projects
- Project 1 — Unsupervised Learning and EDA
  - Notebooks: `Project 1/Code_Kmeans.ipynb`, `Project 1/Code_Hierarchical.ipynb`, `Project 1/Code_PairPlots.ipynb`
  - Focus: K‑Means and hierarchical clustering, feature scaling, pairwise exploration, and cluster evaluation.
- Project 2 — Text Mining
  - Notebook: `Project 2/Code.ipynb`
  - Focus: turning text into numeric features (e.g., tokenization and TF‑IDF), dimensionality reduction, and baseline modeling.
- Project 3 — Time Series Analysis
  - Notebook: `Project 3/Code.ipynb`
  - Focus: trend/seasonality decomposition, stationarity checks, ACF/PACF diagnostics, and simple forecasting.

### Libraries and Tools
- Python: `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`
- Visualization: `matplotlib`, `seaborn`
- NLP (as needed): `scikit-learn` feature extraction, optionally `nltk`/`spaCy`
- Environment: Jupyter notebooks

### Techniques and Methods
- Data preparation: cleaning, handling missing values, encoding, scaling/normalization
- Exploratory analysis: descriptive statistics, pair plots, correlation analysis
- Clustering: K‑Means (elbow and silhouette methods), hierarchical clustering (linkage, dendrograms)
- Text mining: tokenization, stop‑word removal, n‑grams, TF‑IDF, dimensionality reduction (e.g., Truncated SVD)
- Time series: decomposition, differencing, stationarity testing, ACF/PACF, ARIMA/SARIMA baselines

### Visualizations
- Pair plots and scatter/line charts
- Correlation heatmaps
- Elbow and silhouette score plots
- Dendrograms
- ACF/PACF plots

### What I Learnt
- Building analysis‑ready datasets from messy inputs and documenting assumptions
- Selecting and justifying unsupervised methods and hyperparameters
- Representing text data effectively and evaluating models on sparse features
- Diagnosing time series properties and creating simple, defensible forecasts
- Communicating findings with clear visuals and concise narratives
- Writing reproducible analyses in notebooks

### Repository Structure
- `Project 1/` — Clustering and exploratory analysis notebooks
- `Project 2/` — Text mining notebook
- `Project 3/` — Time series notebook
- `README.md` — Overview and guidance

### Getting Started
1. Clone the repository and open the notebooks in Jupyter.
2. Use Python 3.10+ and install common data packages (pandas, numpy, scipy, scikit‑learn, matplotlib, seaborn, statsmodels).
3. Run notebooks cell‑by‑cell; adjust paths as needed for your environment.
