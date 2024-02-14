#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 19:29:33 2024

@author: elnazafzali
"""

import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py

# loading the Dataset
patient_data = pd.read_csv('diabetes.csv')

# Display the first 5 rows of the dataset
display(patient_data.head(5))

# Statistics summary
summary_stats = patient_data.describe().T
print(summary_stats)


    
# Make a deep copy of the dataset
patient_data_copy = patient_data.copy(deep=True)



# Replace 0 values with NaN in specific columns
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
patient_data_copy [columns_to_replace] = patient_data_copy [columns_to_replace].replace(0, np.NaN)

# Show the count of NaN values
nan_counts = patient_data_copy.isnull().sum()
print(nan_counts)
 

# Display distribution for each column in the DataFrame
patient_data.hist(figsize=(30, 30))

# Add titles and labels
#plt.suptitle('Data distribution', x=0.5, y=0.92, fontsize=20)
#plt.xlabel('Values', ha='center', fontsize=16)
#plt.ylabel('Frequency', va='center', fontsize=16)

# Show the plots
plt.show()

# fill missing values with median

patient_data_copy ['Glucose'].fillna(patient_data_copy ['Glucose'].median(), inplace = True)
patient_data_copy['BloodPressure'].fillna(patient_data_copy['BloodPressure'].median(), inplace = True)
patient_data_copy['SkinThickness'].fillna(patient_data_copy['SkinThickness'].median(), inplace = True)
patient_data_copy['Insulin'].fillna(patient_data_copy['Insulin'].median(), inplace = True)
patient_data_copy['BMI'].fillna(patient_data_copy['BMI'].median(), inplace = True)

# Display distribution for each column after filling missing values
patient_data.hist(figsize=(20, 20))
plt.xlabel('Values', ha='center', fontsize=16)
plt.ylabel('Frequency', va='center', fontsize=16)
plt.show()




g = sns.pairplot(patient_data_copy , hue = "Outcome", palette = "husl")



def plot_outliers_box(patient_data_copy, feature, title = None, boxpoints ='suspectedoutliers', colors = None, filename=None):
    
    if feature not in patient_data_copy.columns:
        raise ValueError(f"Feature '{feature}' not found in the DataFrame.")

    if title is None:
        title = f"{feature} Outliers"

    if colors is None:
        colors = {
            'all_points': 'rgb(7,40,89)',
            'only_whiskers': 'rgb(9,56,125)',
            'suspected_outliers': 'rgb(8,81,156)',
            'whiskers_and_outliers': 'rgb(107,174,214)'
        }

    traces = [
        go.Box(
            y = patient_data_copy[feature],
            name = "All Points",
            jitter = 0.3,
            pointpos = -1.8,
            boxpoints = 'all',
            marker = dict(color = colors['all_points']),
            line = dict(color = colors['all_points'])
        ),
        go.Box(
            y = patient_data_copy[feature],
            name = "Only Whiskers",
            boxpoints = False,
            marker = dict(color = colors['only_whiskers']),
            line = dict(color = colors['only_whiskers'])
        ),
        go.Box(
            y = patient_data_copy[feature],
            name = "Suspected Outliers",
            boxpoints = boxpoints,
            marker = dict(
                color = colors['suspected_outliers'],
                outliercolor = 'rgba(219, 64, 82, 0.6)',
                line = dict(outliercolor = 'rgba(219, 64, 82, 0.6)', outlierwidth = 2)
            ),
            line = dict(color = colors['suspected_outliers'])
        ),
        go.Box(
            y = patient_data_copy[feature],
            name = "Whiskers and Outliers",
            boxpoints='outliers',
            marker = dict(color = colors['whiskers_and_outliers']),
            line = dict(color = colors['whiskers_and_outliers'])
        )
    ]

    layout = go.Layout(title = title)
    fig = go.Figure(data = traces, layout = layout)
    if filename is not None:
        py.plot(fig, filename=filename)
    else:
        py.iplot(fig)
    


plot_outliers_box(patient_data_copy, 'Glucose', 'Glucose outliers', filename='Glucose_outliers.html')
plot_outliers_box(patient_data_copy, 'Pregnancies', 'Pregnancies outliers', filename='Pregnancies_outliers.html')
plot_outliers_box(patient_data_copy, 'Age', 'Age outliers', filename='Age_outliers.html')
plot_outliers_box(patient_data_copy, 'BloodPressure', 'BloodPressure outliers', filename='BloodPressure_outliers.html')
plot_outliers_box(patient_data_copy, 'SkinThickness', 'SkinThickness outliers', filename='SkinThickness_outliers.html')
plot_outliers_box(patient_data_copy, 'Insulin', 'Insulin outliers', filename='Insulin_outliers.html')
plot_outliers_box(patient_data_copy, 'BMI', 'BMI outliers', filename='BMI_outliers.html')
plot_outliers_box(patient_data_copy, 'DiabetesPedigreeFunction', 'DiabetesPedigreeFunction outliers', filename='DiabetesPedigreeFunction_outliers.html')



# using the IQR method to identify outliers
Q1 = patient_data_copy.quantile(0.25)
Q3 = patient_data_copy.quantile(0.75)
IQR = Q3 - Q1

# Define a function to identify outliers
def detect_outliers(series):
    return (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))

outliers = patient_data_copy.apply(detect_outliers)

# Calculate median
medians = patient_data_copy.median()


# Reset the index of the outliers Series and medians Series
outliers.reset_index(drop=True, inplace=True)
medians.reset_index(drop=True, inplace=True)

# Replace outliers with median
for col in patient_data_copy.columns:
    patient_data_copy[col] = patient_data_copy[col].mask(outliers[col], medians[col])
    
plot_outliers_box(patient_data_copy, 'Glucose', 'Glucose outliers', filename='Glucose_outliers_after.html')
plot_outliers_box(patient_data_copy, 'Pregnancies', 'Pregnancies outliers', filename='Pregnancies_outliers_after.html')
plot_outliers_box(patient_data_copy, 'Age', 'Age outliers', filename='Age_outliers_after.html')
plot_outliers_box(patient_data_copy, 'BloodPressure', 'BloodPressure outliers', filename='BloodPressure_outliers_after.html')
plot_outliers_box(patient_data_copy, 'SkinThickness', 'SkinThickness outliers', filename='SkinThickness_outliers_after.html')
plot_outliers_box(patient_data_copy, 'Insulin', 'Insulin outliers', filename='Insulin_outliers_after.html')
plot_outliers_box(patient_data_copy, 'BMI', 'BMI outliers', filename='BMI_outliers_after.html')
plot_outliers_box(patient_data_copy, 'DiabetesPedigreeFunction', 'DiabetesPedigreeFunction outliers', filename='DiabetesPedigreeFunction_outliers_after.html')    