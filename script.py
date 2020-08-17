# import libraries
import pandas as pd
import os
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
from flask import Flask

app = Flask(__name__)

@app.route('/api',methods=['POST'])
def predict():

	# import data
	df = pd.read_csv(r"Iris.csv")
	
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
	X_test["Predicted"] = y_pred
	
	#rmse addition
	# rmse = sqrt(mean_squared_error(y_test, y_pred))
	# X_test["rmse"] = rmse
	
	# save output
	resp = make_response(X_test.to_csv("iris_output.csv", index = False))
	resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
	resp.headers["Content-Type"] = "text/csv"
	return resp
	
if __name__ == '__main__':
    app.run(port=int(os.environ.get('PORT',8080)),host='0.0.0.0', debug=True)