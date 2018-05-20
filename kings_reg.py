import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, GridSearchCV, cross_val_score)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso)
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, PolynomialFeatures)
import matplotlib.pyplot as plt
import seaborn as sns



# df = pd.read_csv('data/home_price.csv.csv')

# label = df['price']
# inp = df[['bedrooms', 'condition', 'sqft_lot15', 'sqft_living15', 'lat',
#           'long', 'yr_built', 'floors', 'waterfront', 'view', 'bathrooms', 'zipcode']]

# x_train, y_train, x_test, y_test = train_test_split(inp, label, test_size=0.20)

# lreg = LinearRegression()

# lreg.fit(x_train, x_test)

# pred = lreg.predict(y_train)
# print(pred)
# lreg.score(y_train, y_test) * 100
