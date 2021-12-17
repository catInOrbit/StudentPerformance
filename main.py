import os

from modelComparison import model_comparison
from preprocessing import data_preprocessing
import pandas as pd
import tensorflow as tf
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("student-por.csv", delimiter=";")

sns.heatmap(data.corr())
plt.show()

print(data.head())
print(data.info())

data_encoded = data_preprocessing(data)


feature_cols = [x for x in data_encoded.columns if x != "G3"]

X_data = data_encoded[feature_cols]
sns.heatmap(X_data.corr())
plt.show()

y_data = data_encoded["G3"]

model_comparison("RE_LE", 10, X_data, y_data)
model_comparison("RE_Reg", 10, X_data, y_data)
model_comparison("RE_OT", 10, X_data, y_data)

