import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.stats import boxcox, normaltest
from sklearn.preprocessing import MinMaxScaler
def data_preprocessing(dataframe):
    encoding_cols = dataframe.dtypes[dataframe.dtypes == object]  # Masking
    encoding_cols = encoding_cols.index.tolist()

    labelEnc, oneHotEncoder = LabelEncoder(), OneHotEncoder()
    num_ohc_cols = (dataframe[encoding_cols].apply(lambda x: x.nunique()).sort_values(ascending=False))
    data_encoded = dataframe.__deepcopy__()

    for col in num_ohc_cols.index:
        dat = labelEnc.fit_transform(data_encoded[col]).astype(int)
        data_encoded = data_encoded.drop(col, axis=1)
        new_encoded_data = oneHotEncoder.fit_transform(dat.reshape(-1, 1))

        num_of_cols = new_encoded_data.shape[1]
        col_names = ['_'.join([col, str(x)]) for x in range(num_of_cols)]

        new_df = pd.DataFrame(new_encoded_data.toarray(), index=data_encoded.index, columns=col_names)
        data_encoded = pd.concat([data_encoded, new_df], axis=1)


    print("Encoded Dataframe: ", data_encoded)
    print(data_encoded.columns)
    print(data_encoded.info())
    return data_encoded

def normalizing(dataframe):
    print(normaltest(dataframe.G3.values))
    scaler = MinMaxScaler()
    dataframe.iloc[:, -3:] = scaler.fit_transform(dataframe.iloc[:, -3:])
    dataframe.loc[dataframe.G3 == 0, 'G3'] = 0.01

    log_G3 = np.log(dataframe.G3)
    log_G3 = log_G3.apply(np.abs, axis=1)
    print(normaltest(log_G3))
    sqrt_G3 = np.sqrt(dataframe.G3)
    print(normaltest(sqrt_G3))
    boxcox_G3 = boxcox(dataframe.G3)
    print(normaltest(boxcox_G3[0]))
    dataframe.iloc[:, -1:] = boxcox_G3[0]

    return dataframe


