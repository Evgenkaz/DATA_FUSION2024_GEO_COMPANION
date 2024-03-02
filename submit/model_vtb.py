import pandas as pd
import numpy as np
from typing import List
import joblib


def count_transform(df: pd.DataFrame, hexses_data: List[np.ndarray]):
    data_features = []
    for customer_id, data in df.groupby("customer_id"):
        customer_day = data.datetime_id.tolist()
        location = data.h3_09.tolist()

        loc_feature = []
        loc_text = []
        for loc in hexses_data:
            if loc in location:
                loc_count = data[data.h3_09 == loc]['count'].sum()
                loc_feature.append(loc_count)
                loc_text.append(loc)
            else:
                loc_feature.append(-1)
                loc_text.append(loc)

        data_features.append([customer_id] + loc_feature)
    return pd.DataFrame(data_features, columns=['customer_id'] + loc_text)


def dataset_agg(df: pd.DataFrame):
    agg_df = df.groupby('customer_id').agg({'h3_09': lambda x: x.nunique(),
                                            #  'lat': lambda x: x.mean(),
                                            #  'lng': lambda x: x.mean(),
                                            'mcc_code': lambda x: x.nunique(),
                                            'datetime_id': lambda x: x.nunique(),
                                            'count': lambda x: x.mean(),
                                            'sum': lambda x: x.mean(),
                                            'avg': lambda x: x.mean(),
                                            'min': lambda x: x.mean(),
                                            'max': lambda x: x.mean(),
                                            'std': lambda x: x.mean(),
                                            'count_distinct': lambda x: x.mean()})
    return agg_df

def load_models(test: pd. DataFrame, location: str):
    print(test)
    predictions = []
    for i in range(3):
        model = joblib.load(f'{location}/model_{i}.pkl')
        predictions.append(model.predict_proba(test)[:, 1])
    return np.mean(predictions, axis=0)