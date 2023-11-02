# Credit Scoring Analytics for Lending Club Loan Data

## Background

The objective of LendingClub is to revolutionize the banking industry by connecting borrowers with investors. The platform offers a transparent overview of various investment opportunities that differ from the conventional choices provided by the banking industry. Borrowers are provided with unique means of financing their projects and businesses. Meanwhile, investors are given alternative investment options to explore.

Lending Club, the largest peer-to-peer lending platform, was established in San Francisco, California in 2007. A year later, it registered its offerings with the Securities and Exchange Commission. Since then, it has been the recipient of many accolades and international recognition, such as the World Economic Forum 2012 Technology Pioneer Award, Forbes’s America’s 20 most promising companies in 2011 and 2012, and one of CNBC’s Disruptor 50 in 2013 and 2014.

## Goals

To create a financial diversification recommender that aims to maximize returns while minimizing the risk of investments.

## Motivation for the Project:

Lending Club periodically publishes both quarterly and annual data regarding the loans that it has facilitated. This data is comprised of vast databases that contain various parameters and past outcomes of both successful and unsuccessful loans. Despite the wealth of information present, these databases are so vast that it is impractical for any human investor to effectively utilize them.

Despite the enormous amount of information available, it offers a prime opportunity to construct lucrative models utilizing data science techniques. Those invested in Lending Club are eager to discern which loans are most promising in terms of generating high returns. Through data science techniques, these models would essentially predict and determine which loans are worthy investments, considering the risk of default and potential interest returns. Establishing effective and precise models would enable investors to invest with greater assurance, thereby promoting greater participation in Lending Club.

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