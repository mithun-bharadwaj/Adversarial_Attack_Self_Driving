"""Import necessary libraries"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

##################################################################################################

"""Read data"""
column_names = ['Width', 'Color', 'Steering_angle_difference','Other_lane_intersection']
raw_dataset = pd.read_csv('data/new_data_1.csv', sep = ',', skipinitialspace=True, dtype = float)
dataset = raw_dataset.copy()
dataset.isna().sum()

#################################################################################################

"""Implement SVM and evaluate"""

labels = [1 if x>0 else 0 for x in dataset['Other_lane_intersection']]

X = dataset.drop('Other_lane_intersection', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.20)

svclassifier = SVC(kernel='poly', degree=4)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
print(classification_report(y_test, y_pred))
