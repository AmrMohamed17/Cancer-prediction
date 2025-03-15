# Breast Cancer Classification

This project implements machine learning models to classify breast cancer tumors as malignant or benign based on features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

## Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset, which includes features computed from a digitized image of a fine needle aspirate of a breast mass. Features describe characteristics of the cell nuclei present in the image.

## Project Structure

```
├── data.csv                # The breast cancer dataset
├── Cancer.ipynb  # Main script for data processing and model training
├── README.md               # This file
└── requirements.txt        # Dependencies
```

## Features

The original dataset contains 32 features, but after correlation analysis, highly correlated features (|correlation| > 0.75 with diagnosis) were removed to reduce dimensionality and avoid multicollinearity.

## Preprocessing

The following preprocessing steps were applied:
1. Removal of unnecessary columns ('id' and 'Unnamed: 32')
2. Label encoding of the 'diagnosis' column (M = Malignant, B = Benign)
3. Feature selection based on correlation with target variable
4. Train-test split (80% training, 20% testing)

## Models

Two classification models were trained and evaluated:

### Random Forest Classifier
- Hyperparameters tuned using RandomizedSearchCV with 5-fold cross-validation
- Parameter grid:
  - n_estimators: [50, 100]
  - max_depth: [3, 5, 7]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]

### CatBoost Classifier
- Hyperparameters tuned using RandomizedSearchCV with 5-fold cross-validation
- Parameter grid:
  - iterations: [200, 300]
  - depth: [1, 2, 4]
  - learning_rate: [0.1, 0.2]
  - l2_leaf_reg: [5, 7]

## Performance

Both models achieved high accuracy on the test set, demonstrating effective classification of breast cancer tumors.

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
catboost
```

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python breast_cancer_classification.py`

## Future Work

- Implement additional models for comparison (SVM, Neural Networks)
- Feature engineering to create more predictive features
- Ensemble methods to improve overall performance
- Cross-validation with different metrics (precision, recall, F1-score)
- Deployment as a web application for clinical use
