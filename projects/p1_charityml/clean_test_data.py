#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 22:55:24 2018

@author: alex
"""

import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer

data = pd.read_csv("test_census.csv")
data = data.drop(['Unnamed: 0'], axis=1)
features_raw = data

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Scale numerical columns
features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)

# Impute missing value
imp = Imputer(missing_values=np.nan, strategy='mean')
features_log_minmax_transform[numerical] = imp.fit_transform(features_log_minmax_transform[numerical])

features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# one hot encoding of categorical features
features_final = pd.get_dummies(features_log_minmax_transform)

# write to csv
features_final.to_csv('cleaned_testing.csv', index=False)