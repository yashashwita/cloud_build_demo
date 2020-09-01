# import libraries
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime

# import data
df = pd.read_csv(r"data/Iris.csv")

# dummify
df = pd.get_dummies(df, drop_first = True)

# train test split
X_train, X_test, y_train, y_test = train_test_split(df[df.columns[df.columns != "PetalWidthCm"]], df["PetalWidthCm"], test_size = 0.2)

# cross validataion
model = RidgeCV(alphas=np.logspace(-6, 6, 13), cv = 3)
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# appending results back
X_test["Predicted"] = y_pred.round(1)
X_test["Actual"] = y_test
#rmse addition
# rmse = sqrt(mean_squared_error(y_test, y_pred))
# X_test["rmse"] = rmse

# save output
X_test.to_csv("data/irisOutput_{}.csv".format(datetime.now().strftime("%Y%m%d")), index = False)
