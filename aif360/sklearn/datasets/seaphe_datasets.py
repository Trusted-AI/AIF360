# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a class for the SEAPHE datasets.
http://www.seaphe.org/databases.php
"""
import os
import pandas as pd
import numpy as np
import tempfile
import requests
import zipfile
from utils import standardize_dataset
from sklearn.model_selection import train_test_split


def load_lawschool_data(target, subset="all", usecols=None, dropcols=None,
                        numeric_only=False, dropna=True):
    """Downloads SEAPHE lawschool data from the SEAPHE webpage.
    For more information, refer to http://www.seaphe.org/databases.php

    :param target: the name of the target variable, either pass_bar or zfygpa
    :type target: str
    :return: pandas.DataFrame with columns
    """
    if target not in ['pass_bar', 'zfygpa']:
        raise ValueError("Only pass_bar and zfygpa are supported targets.")
        
    if subset not in {'train', 'test', 'all'}:
        raise ValueError("subset must be either 'train', 'test', or 'all'; "
                         "cannot be {}".format(subset))
        

    with tempfile.TemporaryDirectory() as temp_dir:
        response = requests.get("http://www.seaphe.org/databases/LSAC/LSAC_SAS.zip")
        temp_file_name = os.path.join(temp_dir, "LSAC_SAS.zip")
        with open(temp_file_name, "wb") as temp_file:
            temp_file.write(response.content)
        with zipfile.ZipFile(temp_file_name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        data = pd.read_sas(os.path.join(temp_dir, "lsac.sas7bdat"))

        # Your data preprocessing steps here...
    data = data[['lsat', 'ugpa', 'race', 'gender', target]]
    

    data = (standardize_dataset(data, prot_attr='race', target='zfygpa',
                               usecols=usecols, dropcols=dropcols,
                               numeric_only=numeric_only, dropna=dropna))
    
    All_features = pd.DataFrame(data.X)
    All_labels = pd.DataFrame(data.y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        All_features, 
        All_labels, 
        test_size= 0.33,
        random_state=123
    )
    
    if subset == "train":
        return X_train, y_train
    elif subset == "test":
        return X_test, y_test
    else:
        return All_features, All_labels
