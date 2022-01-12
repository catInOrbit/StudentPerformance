from modelComparison import model_comparison
from preprocessing import data_preprocessing, normalizing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("student-por.csv", delimiter=";")

data.G3.hist()
plt.show()

print(data.G3.unique())
print(data.G3.value_counts())

print(data.head())
print(data.info())

data = normalizing(data)

data_encoded = data_preprocessing(data)
feature_cols = [x for x in data_encoded.columns if x != "G3"]
# sns.heatmap(data_encoded[feature_cols])
plt.show()

X_data = data_encoded[feature_cols]
y_data = data_encoded["G3"]

model_comparison("RE_LE", 10, X_data, y_data)
model_comparison("RE_Reg", 10, X_data, y_data)
model_comparison("RE_OT", 10, X_data, y_data)