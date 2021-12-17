import os

from modelComparison import model_comparison
from preprocessing import data_preprocessing
import pandas as pd
import tensorflow as tf
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

data = pd.read_csv("student-por.csv", delimiter=";")

print(data.head())
print(data.info())

data_encoded = data_preprocessing(data)


feature_cols = [x for x in data_encoded.columns if x != "G3"]
msno.heatmap(data_encoded[feature_cols])
plt.show()

X_data = data_encoded[feature_cols]
y_data = data_encoded["G3"]
chi_test = SelectKBest(score_func=chi2, k=6)

fit = chi_test.fit(X_data, y_data)
cols = fit.get_support(indices=True)
features_df_new = X_data.iloc[:,cols]
print(features_df_new)

model_comparison("RE_LE", 10, X_data, y_data)
model_comparison("RE_Reg", 10, X_data, y_data)
model_comparison("RE_OT", 10, X_data, y_data)

