#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 22:12:03 2020

@author: sayalisonawane
"""
from sklearn import preprocessing

def data_preparation(data):
    label_encoder = preprocessing.LabelEncoder()
    # State is string and we want discreet integer values
    data['Intl_Plan'] = label_encoder.fit_transform(data['Intl_Plan'])
    data['Vmail_Plan'] = label_encoder.fit_transform(data['Vmail_Plan'])
    data['Area_Code'] = label_encoder.fit_transform(data['Area_Code'])
    data['Churn'] = label_encoder.fit_transform(data['Churn'])
    #Dropping State Column
    data = data.drop(columns=['State'])
    data = data.drop(columns=['Phone'])
    return data
