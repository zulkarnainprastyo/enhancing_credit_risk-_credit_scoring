# Credit Scoring Analytics for Lending Club Loan Data

## 1. Data Preprocessing
- [data_preprocessing.py](./1.%20Data%20Preprocessing/data_preprocessing.py)
- [data_cleaning.py](./1.%20Data%20Preprocessing/data_cleaning.py)
- [outlier_detection.py](./1.%20Data%20Preprocessing/outlier_detection.py)
- [data_transformation.py](./1.%20Data%20Preprocessing/data_transformation.py)

## 2. Feature Selection
- [feature_selection.py](./2.%20Feature%20Selection/feature_selection.py)
- [traditional_credit_data.py](./2.%20Feature%20Selection/traditional_credit_data.py)

## 3. Deep Metric Learning
- [deep_metric_learning.py](./3.%20Deep%20Metric%20Learning/deep_metric_learning.py)
- [deep_learning_models.py](./3.%20Deep%20Metric%20Learning/deep_learning_models.py)

## 4. Scorecards Development
- [scorecards_development.py](./4.%20Scorecards%20Development/scorecards_development.py)
- [acquisition_scorecards.py](./4.%20Scorecards%20Development/acquisition_scorecards.py)

## 5. False Positive Minimization
- [false_positive_minimization.py](./5.%20False%20Positive%20Minimization/false_positive_minimization.py)
- [decision_thresholds.py](./5.%20False%20Positive%20Minimization/decision_thresholds.py)
- [feature_scaling.py](./5.%20False%20Positive%20Minimization/feature_scaling.py)


## Project Architecture Credit Scorecard Analytics

The project's folder structure is presented below:

```bash
├── 1. Data Preprocessing
│   ├── data_preprocessing.py
│   ├── data_cleaning.py
│   ├── outlier_detection.py
│   ├── data_transformation.py
│
├── 2. Feature Selection
│   ├── feature_selection.py
│   ├── traditional_credit_data.py
│
├── 3. Deep Metric Learning
│   ├── deep_metric_learning.py
│   ├── deep_learning_models.py
│
├── 4. Scorecards Development
│   ├── scorecards_development.py
│   ├── acquisition_scorecards.py
│
├── 5. False Positive Minimization
│   ├── false_positive_minimization.py
│   ├── decision_thresholds.py
│   ├── feature_scaling.py
│
└── README.md
```

## Modeling Accuracies:


Modeling | Accuracy
--- | ---
Random Forest with Randomized                      | 82.18
Logistic Regression with Grid                      | 82.03
K Nearest Neighbors with Grid                      | 76.31
Bagging with Base estimator as Random Forest       | 82.98
Bagging with Base estimator as Logistic Regression | 81.93
AdaBoost Classifier                                | 82.12
MultilLayer Perceptron Classifier                  | 82.18

## Framework of this project

1. Introduction and Background
    * Explain the background and goals of the project.

2. Dataset Description
    * Explain the dataset; describe the variables.

3. Scorecards Development
    * Describe each step of the workflow and its result.

4. Implementation Plan
    * Setting a cutoff.
    * Create the credit process workflow.

5. Conclusion and Recommendation
    Make a summary of the following:
    * The conclusions from the scorecards and analysis.
    * Recommendations for the user.
    * Recommendation for the next project.

6. References
    * Search references