# Detecting Demographic Bias in AI-Based Insurance Claim Denial Predictions

Predicting whether denied medical claims will be overturned during California's Independent Medical Review (IMR) process — and evaluating whether model performance is consistent across demographic groups.

## Overview

This project investigates whether machine learning and NLP models can predict the outcome of an Independent Medical Review (IMR) for a denied medical claim. The task is a binary classification problem with clear practical importance in the healthcare domain: understanding which denied claims are likely to be overturned has implications for patients, insurers, and regulators alike.

Beyond overall prediction performance, the project examines whether model errors are distributed consistently across age and gender groups. Rather than treating demographic bias as the prediction target, this study treats it as a core part of model evaluation — a model that performs well in aggregate but poorly for specific subgroups is not a reliable decision-support tool.

## Hypothesis

Ensemble and boosting methods will outperform simpler baseline models on the IMR outcome prediction task overall. However, higher aggregate accuracy does not guarantee consistent behavior across demographic groups. If performance varies meaningfully by age or gender, this may indicate that either the underlying data distribution or the model family influences subgroup-level outcomes.

## Dataset

**Source:** California Independent Medical Review (IMR) Determinations Trend dataset, published by the California Department of Managed Health Care (DMHC) through the [California Health and Human Services Open Data Portal](https://data.chhs.ca.gov/).

The dataset contains real IMR case decisions, making it a realistic source for studying healthcare claim outcome prediction.

**Target variable (binary):**
- `1` = Overturned
- `0` = Not Overturned (includes upheld, modified, and other non-overturned outcomes)

**Input features:**
- *Structured:* patient age range, patient gender, year of decision, treatment/service category, and other categorical case fields
- *Text-based:* review or case text transformed into features via TF-IDF (when usable text is available after cleaning)

Since the dataset includes labeled outcomes, no manual annotation is required.

## Methodology

**Preprocessing**
- Cleaning missing and inconsistent values
- Defining the binary target class
- Encoding categorical variables
- Transforming text features with TF-IDF
- Train/test splitting

**Models compared**
- Logistic Regression (baseline)
- Naive Bayes (baseline)
- Random Forest
- AdaBoost
- XGBoost
- LightGBM

**Evaluation**
- *Overall performance:* accuracy, precision, recall, F1, ROC-AUC
- *Subgroup analysis:* performance metrics broken down by age range and gender to surface disparities in model behavior across demographic groups

## Tech Stack

- Python 3.x
- pandas, NumPy — data manipulation
- scikit-learn — preprocessing, TF-IDF, Logistic Regression, Naive Bayes, Random Forest, AdaBoost
- XGBoost, LightGBM — gradient boosting models
- matplotlib, seaborn — visualization
- Google Colab — development environment

## Repository Structure
```
├── notebooks/
│   └── bias_analysis.ipynb
├── data/
│   └── README.md
├── results/
│   └── figures/
└── README.md
```

## How to Run
1. Open `notebooks/bias_analysis.ipynb` in Colab
2. Download the IMR dataset from the [CHHS Open Data Portal](https://data.chhs.ca.gov/) and upload it to your Google Drive
3. Update the `DATA_PATH` variable to point to your file
4. Run all cells
```

## Collaborators

Harshita Divakarreddy
Anosha Rehan
