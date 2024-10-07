# Telecom Customer Churn Prediction

This project aims to predict whether a customer will leave (churn) a telecom company based on various customer and service features. 
The best model achieves an f1-score of **88%** on the test set. The dataset used for this task is sourced from [Kaggle](https://www.kaggle.com/datasets/levietdat/telecom-churn-prediction/data) and contains 38 features related to customer demographics, services, and account information.

## Project Workflow

### 1. Exploratory Data Analysis
The first step is to explore and clean the data to better understand the relationships between features and the target variable (`Customer Status`).

Key EDA steps include:
- **Removing unnecessary columns**: Some features, such as `Churn Reason` (which is populated only after a customer leaves), are removed as they provide no value for prediction.
- **Visualizing feature distributions**
- **Handling missing data**
- **Outlier treatment**.
- **Reducing categorical feature cardinality**
- **Encoding categorical variables**.

Each decision during this process is supported by an explanation.

The raw data is stored in `telecom_train.csv` and `telecom_test.csv`. After preprocessing, the cleaned and transformed datasets are saved as `train_data.csv` and `test_data.csv`.

All preprocessing steps are encapsulated in the function `data_preprocessing`, which can be found in the `data_preprocessing.ipynb` notebook.

### 2. Feature Selection
Before training the model, feature selection is performed to remove irrelevant or redundant features. Several techniques are used for this, including:
- **Correlation analysis** to identify lowly correlated features with target.
- **SelectKBest** to select the top features based on statistical tests.
- **Mutual Information (mutual_info_classif)** to gauge the relationship between features and the target variable.
- **Recursive Feature Elimination (RFE)** using a `RandomForestClassifier` as the estimator.

The final set of features is determined by the intersection of these methods.

### 3. Model Training
For classification, a **Logistic Regression** model is trained using the selected features. Hyperparameter tuning is done via **GridSearchCV** to identify the best performing model.

The best model achieves an f1-score of **88%** on the test set.

All steps for model training and evaluation are documented in the `model_training.ipynb` notebook.
