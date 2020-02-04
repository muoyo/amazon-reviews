"""
##### DATA VISUALIZATIONS #####

This module contains our functions for linear regression.

"""


import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def get_train_test_split(df, test_size=.25):

    X = df.loc[:, ['helpful_votes', 'total_votes', 'neg', 'neu', 'pos', 'compound']]
    y = df.loc[:, 'star_rating']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    smote = SMOTE()
    
    X_train, y_train = smote.fit_sample(X_train, y_train) 
    
    return X_train, X_test, y_train, y_test