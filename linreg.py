import numpy as np
import pandas as pd
# import model
from sklearn.linear_model import LinearRegression
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
# import module to calculate model perfomance metrics
from sklearn import metrics


data_path = "data\pulsar_stars.csv" # or load the dataset directly from the link

data = pd.read_csv(data_path, index_col=0)

feature_names = ['Standard', 'Excess', 'Skewness']

X = data[feature_names]

y = data.Sales

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

linreg = LinearRegression()

linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

