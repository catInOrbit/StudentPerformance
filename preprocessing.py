import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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

# def EDA(original_df, processed_df):
#     print("Head: ", original_df.head())
#     print("Dataframe info: ", original_df.head())
#     print(processed_df.info())



