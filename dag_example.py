# import libraries
from datetime import datetime,timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

import pandas as pd
import numpy as np
import os
import csv
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


#defining functions

def preprocess():
	df = pd.read_csv(r"/home/airflow/gcs/dags/data/Iris_raw.csv")
	df.to_csv(r"/home/airflow/gcs/dags/data/Iris_preprocessed.csv", index = False)

def feature_engineering():
	df = pd.read_csv(r"/home/airflow/gcs/dags/data/Iris_preprocessed.csv")

	#Feature creation
	df["Sepal_Area"] = df["SepalLengthCm"] * df["SepalWidthCm"]
	
	df.to_csv(r"/home/airflow/gcs/dags/data/Iris.csv", index = False)

def iris_model():
	# import data
	df = pd.read_csv(r"/home/airflow/gcs/dags/data/Iris.csv")
	
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
	print('*_*_*_*_*_*_*__*_*_*')
	with open('/home/airflow/gcs/dags/data/iris_output_' + str(datetime.now().strftime("%d%m%Y%H%M%S")) + '.csv', 'w',newline='') as fp:
	    writer = csv.DictWriter(fp, fieldnames=X_test.columns)
	    writer.writeheader()
	    for row in X_test.to_dict('records'):
	        writer.writerow(row)
	print(X_test)
	

# Define some arguments for our DAG
default_args = {
    'owner': 'yashashwita',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Instantiate our DAG
dag = DAG(
    'dag_example',
    default_args=default_args,
    description='A DAG Example',
    schedule_interval=timedelta(days=1),
)

with dag:
    iris_data_preprocessing = PythonOperator(
        task_id='iris_data_preprocessing',
        python_callable=preprocess
    )

    iris_feature_engineering = PythonOperator(
        task_id='iris_feature_engineering',
        python_callable=feature_engineering
    )

    iris_model = PythonOperator(
        task_id='iris_model',
        python_callable=iris_model
    )

    iris_data_preprocessing >> iris_feature_engineering >> iris_model
