# Vehicle Insurance Prediction - Machine Learning

This project predicts whether existing health-insurance customers are likely to respond positively to a vehicle-insurance cross-sell offer. It is based on the public health insurance cross-sell dataset and compares several machine learning approaches for an imbalanced binary classification problem.

## Project Summary

The response variable is `Response`, where `1` indicates customer interest in vehicle insurance and `0` indicates no interest. Because positive responses are much less common than negative responses, the project focuses on class imbalance, validation performance, and F1 score rather than accuracy alone.

The analysis includes:

- exploratory review of class imbalance and predictor distributions
- categorical and numeric feature preparation
- multiple resampling strategies for imbalanced data
- Naive Bayes and logistic regression baselines
- random forest and gradient boosting models
- linear SVM comparison
- neural network experiments
- confusion-matrix and F1-score based evaluation

## Data Files

The notebook reads data directly from the files in this repository:

- `train.csv`
- `test.csv`

The original data source referenced in the notebook is the Kaggle Health Insurance Cross Sell Prediction dataset.

## Main Notebook

- `Project.ipynb`

The notebook has been updated to load local repository files directly:

```python
from pathlib import Path
import pandas as pd

DATA_DIR = Path.cwd()
train_full = pd.read_csv(DATA_DIR / "train.csv", index_col=0)
test = pd.read_csv(DATA_DIR / "test.csv", index_col=0)

validation = train_full.sample(frac=0.33, random_state=2021)
train = train_full.drop(validation.index, axis=0)
```

## Methodology

1. Split the original training data into training and validation sets.
2. Examine class imbalance in both sets.
3. Review numeric distributions and categorical correlations.
4. Apply feature preparation and resampling methods.
5. Compare model families:
   - Naive Bayes
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - Linear SVM
   - Neural Network
6. Evaluate models using F1 score and confusion matrices.

## Key Finding

The results suggested that the dataset has a practical performance ceiling under the attempted feature engineering and resampling strategies. The best models reached approximately 75% total accuracy, while the minority-class F1 score remained around the mid-0.4 range.

## How To Run

Open the notebook from this folder so relative paths resolve correctly:

```bash
cd /Users/qiankangwang/Desktop/personal-portfolio/demos/graduate-statistics/statistical-learning/original-repo
jupyter notebook Project.ipynb
```

Install dependencies as needed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn torch
```
