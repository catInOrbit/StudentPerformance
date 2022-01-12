import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from plot_helper import draw_linear_comparision, draw_boxplot_comparision

def model_comparison(model_type, k_splits, data_features, target):
    models_plain_reg = [("LiR", LinearRegression())]
    models_regularized_reg = [("Ridge", Ridge()), ("Las", Lasso())]
    models_other_reg = [("ENet", ElasticNet()), ("KNNR", KNeighborsRegressor()), ("CART", DecisionTreeRegressor()), ("SVR", SVR())]

    kfold = KFold(n_splits=k_splits)
    params = {
        'regularized_regression__alpha': np.geomspace(4, 20, 30)
    }

    result_scores = []
    if model_type == "RE_LE":
        for name, model in models_plain_reg:
            result = cross_val_score(model, data_features, target, cv=kfold, scoring="r2")
            result_scores.append(result)
            print("%s: %.3f (%.3f)" % (name, result.mean(), result.std()))

        draw_boxplot_comparision(models_plain_reg, result_scores)
    elif model_type == "RE_OT":
        for name, model in models_other_reg:
            result = cross_val_score(model, data_features, target, cv=kfold, scoring="r2")
            result_scores.append(result)
            print("%s: %.3f (%.3f)" % (name, result.mean(), result.std()))

        draw_boxplot_comparision(models_plain_reg, result_scores)
    elif model_type == "RE_Reg":
        for name, model in models_regularized_reg:
            estimator = Pipeline([("regularized_regression", model)])
            grid = GridSearchCV(estimator, params, cv=kfold)
            grid_fit = grid.fit(data_features, target)
            print(name, "GridCV best score: ", grid.best_score_, "GridCV best params: ", grid.best_params_)
            y_predict = grid.predict(data_features)
            score = r2_score(target, y_predict)
            result_scores.append(score)
            print("r2 score: ", score)
