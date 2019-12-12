"""Import necessary libraries"""

from mpl_toolkits.mplot3d import Axes3D  
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

###############################################################################################################################

"""Read data"""
column_names = ['Width', 'Color', 'Steering_angle_difference','Other_lane_intersection']
raw_dataset = pd.read_csv('data/new_data_1.csv', sep = ',', skipinitialspace=True, dtype = float)
dataset = raw_dataset.copy()
dataset.isna().sum()

#################################################################################################################################
""""Plot graphs"""

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_trisurf(dataset['Color'], dataset['Width'], dataset['Steering_angle_difference'], linewidth=0, antialiased=False)
ax.set_title('Surface plot')
ax.set_xlabel('Color')
ax.set_ylabel('Width')
ax.set_zlabel('Steering angle difference')
plt.show()

data = dataset.pivot(index='Color', columns='Width', values='Steering_angle_difference')
sns.heatmap(data)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(dataset['Color'], dataset['Width'], dataset['Other_lane_intersection'], linewidth=0, antialiased=False)
ax.set_title('Surface plot')
ax.set_xlabel('Color')
ax.set_ylabel('Width')
ax.set_zlabel('Infraction')
plt.show()

data = dataset.pivot(index='Color', columns='Width', values='Other_lane_intersection')
sns.heatmap(data)
plt.show()

########################################################################################################################



