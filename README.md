# StudentPerformance
A small project for building and comparing different ML models for StudentPerformance model from UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/student%2Bperformance


# 1 Exploratory Data Analysis 
Pandas library is used for loading and processing of CSV


# 2 Data Preprocessing
`LabelEncoder` and `OneHotEncoder` is used for encoding categorical data, no scaling is used at the moment

# 3 Model Building
List of models used in comparision.py:

 ### Regression:
    - Linear Regression
    - Rige Regression (with `GridSearchCV` cross validation score)
    - Lasso Regression(with `GridSearchCV` cross validation score)
    - ELasticNet
    - KNeighborsRegressor
    - DecisionTreeRegressor
    - SVR 
 ### Classification
    - LogisticRegression
    - KNeighborClassifier
    - DecisionTree
    
 # 4 Usage:
 Run `main.py` to see comparision result from different model, current result indicate best predictor as RidgeRegression
