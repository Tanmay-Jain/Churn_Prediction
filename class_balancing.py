#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 22:04:18 2020

@author: sayalisonawane
"""
import pandas as pd
from sklearn.utils import resample

# Class balancing 
def balance_class_upsampling(feat_store,target_col):
    """
    This function upsamples minority class and makes 50%-50% class distribution
    """
    # 1. Seperate Class 0 and Class 1 data  
    # define major and minor class 
    class_0 = feat_store[feat_store[target_col]==0]
    class_1 = feat_store[feat_store[target_col]==1]
    
    # 2. Discover majority and minority class
    if (len(class_0)>len(class_1)):
               df_majority = class_0.copy() # copy dataframe in new dataframe 
               df_minority = class_1.copy() # copy dataframe in new dataframe 
    else:
               df_majority = class_1.copy()
               df_minority = class_0.copy()
    
    # 3. Use majority class size to upsample minority class and return upsampled data. 
    # majority class size
    major_class_size = len(df_majority)
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample without replacement
                                 n_samples=major_class_size,     # to match minority class
                                 random_state=123) # reproducible results
    
    # 4. Combine upsampled_minority and majority class in one dataframe. 
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_minority_upsampled, df_majority])
    return df_upsampled