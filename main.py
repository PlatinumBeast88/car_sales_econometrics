import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.metrics import accuracy_score
import xgboost
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot

train_data = pd.read_excel("car_sales_train.xlsx")
train_data_copy = train_data
rofl = train_data_copy
rofl = rofl.loc[99:156]
rofl = rofl['manufact']


def cleaning(train_data):
    train_data_copy = train_data
    numerical = list(train_data_copy.select_dtypes(include=['int64', 'float64']).columns.values)
    numerical.remove('sales')
    for col in numerical:
        if train_data_copy[col].isnull().values.any():
            train_data_copy[col].fillna(train_data_copy[col].median(), inplace=True)
    return train_data_copy


train_data = cleaning(train_data)


def encoder(text_to_enc):
    enc = LabelEncoder()
    text_to_enc['manufact'] = enc.fit_transform(text_to_enc['manufact'])
    text_to_enc['model'] = enc.fit_transform(text_to_enc['model'])
    return text_to_enc


encoder(train_data)
tr_data = train_data.loc[1:98]
train_data.dropna(subset=['sales'])
test_data = train_data.loc[99:156]

tr_data_y = tr_data['sales']
tr_data_x = tr_data.drop(columns=['sales'])
y_val = test_data['sales']
test_data_x = test_data.drop(columns=['sales'])
# print(tr_data_y,tr_data_x,test_data_x)

# model = LinearRegression().fit(tr_data_x, tr_data_y)
# importance = model.coef_
# for i, v in enumerate(importance):
#     print('Feature: %0d, Score: %.5f' % (i, v))
# pyplot.bar([x for x in range(len(importance))], importance)
# pyplot.show()
model = xgboost.XGBRegressor()
model.fit(tr_data_x, tr_data_y)

predictions = model.predict(test_data_x)

zhizha = pd.DataFrame({"Cars": rofl.values,
                   "sales": predictions
                   })
# x_val = zhizha.drop(columns=['Cars'])


# print(model.score(test_data_x, x_val))
# simple_model_reverse = xgboost.XGBRegressor()
# simple_model_reverse.fit(tr_data_x[['manufact',	'model', 'resale',	'type', 'price', 'engine_s', 'horsepow', 'wheelbas',
#                                     'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']], tr_data_y)
# print(simple_model_reverse.coef_)

# print(predictions)
zhizha.to_excel("predictions.xlsx", index=False)
