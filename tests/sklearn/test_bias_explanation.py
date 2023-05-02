import sys
sys.path.append('../..')
from aif360.sklearn.datasets import fetch_adult, fetch_compas, fetch_german
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from aif360.sklearn.explainers import fairxplain_statistical_parity, fairxplain_equalized_odds, fairxplain_predictive_parity
from aif360.sklearn.metrics import intersection, selection_rate
import numpy as np


prot_attr_config_dic = {
    "adult": {
        0 : ['race'],
        1 : ['sex'], 
        2 : ['race', 'sex']
    },
    "compas": {
        0 : ['race'],
        1 : ['sex'],
        2 : ['race', 'sex']
    },
    "german": {
        0 : ['sex'],
        1 : ['age'],
        2 : ['age', 'sex']
    }
}

def get_dataset(dataset, prot_attr):
    if(dataset == "adult"):
        X, y, sample_weight = fetch_adult(numeric_only=True)
    elif(dataset == "compas"):
        X, y = fetch_compas(numeric_only=True, binary_race=True)
    elif(dataset == "german"):
        X, y = fetch_german(numeric_only=True)
        # make age binary if age is used as protected attribute
        if('age' in prot_attr):
            X['age'] = X['age'] > 25
    else:
        raise ValueError("Unknown dataset")
    return X, y


def get_statistical_parity_difference_from_aif360(X, y, prot_attr, clf):
    positive_prediction_probabilities = np.array(
        intersection(selection_rate, y, clf.predict(X), prot_attr=prot_attr)) # considering intersectional protected attributes
    value_statistical_parity = positive_prediction_probabilities.max() - positive_prediction_probabilities.min()
    return value_statistical_parity

def get_equalized_odds_difference_from_aif360(X, y, prot_attr, clf):
    value_equalized_odds = []
    for each_y in y.unique():
        positive_prediction_probabilities = np.array(
            intersection(selection_rate, y[y == each_y], clf.predict(X[y == each_y]), prot_attr=prot_attr)) # considering intersectional protected attributes
        value_statistical_parity = positive_prediction_probabilities.max() - positive_prediction_probabilities.min()
        value_equalized_odds.append(value_statistical_parity)

    # taking the max
    value_equalized_odds = np.max(value_equalized_odds)
    return value_equalized_odds



def _test(dataset):
    for prot_attr_config in [0, 1, 2]:
        prot_attr = prot_attr_config_dic[dataset][prot_attr_config]
        X, y = get_dataset(dataset, prot_attr)

        # check if protected attributes are in the dataset
        for attr in prot_attr:
            assert attr in X.columns

        # learn a model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = LogisticRegression().fit(X_train, y_train)


        result, bias = fairxplain_statistical_parity(
            clf, 
            X_train, 
            prot_attr=prot_attr, 
            maxorder=1, 
            spline_intervals=3, 
            verbose=True, 
            seed=None, 
            return_bias=True 
        )

        # check if the bias value is correct 
        assert bias == get_statistical_parity_difference_from_aif360(X_train, y_train, prot_attr, clf)

        # check if result contains attributes that are in the dataset. Only works when maxorder=1
        for attribute in list(result.index):
            if(attribute == "Residual FIFs"):
                continue
            assert attribute in X.columns

        results, bias_values = fairxplain_equalized_odds(
            clf, 
            X_train, 
            y_train, # For equalized odds, we need to pass the true labels as well
            prot_attr=prot_attr, 
            maxorder=1,
            spline_intervals=3,
            seed=None,
            verbose=False,
            return_bias=True)
        
        # check if the bias value is correct
        index_max_unfairness = np.argmax(bias_values) # we choose the class label with the highest bias for equalized odds
        assert bias_values[index_max_unfairness] == get_equalized_odds_difference_from_aif360(X_train, y_train, prot_attr, clf)



        # check if result contains attributes that are in the dataset. Only works when maxorder=1
        for attribute in list(results[index_max_unfairness].index):
            if(attribute == "Residual FIFs"):
                continue
            assert attribute in X.columns

        results, bias_values = fairxplain_predictive_parity(
            clf, 
            X_train, 
            y_train, # For predictive parity, we need to pass the true labels as well
            prot_attr=prot_attr, 
            maxorder=1, 
            spline_intervals=3,
            seed=None,
            verbose=False,
            return_bias=True)
        
        # check if result contains attributes that are in the dataset. Only works when maxorder=1
        index_max_unfairness = np.argmax(bias_values)
        for attribute in list(results[index_max_unfairness].index):
            if(attribute == "Residual FIFs"):
                continue
            assert attribute in X.columns

        

def test_all():
    for dataset in ["compas", "german", "adult"]:
        _test(dataset)
        
        



