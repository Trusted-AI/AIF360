import sys
sys.path.append('../..')
from aif360.sklearn.datasets import fetch_adult, fetch_compas, fetch_german
from sklearn.linear_model import LogisticRegression
from aif360.sklearn.explainers import fairxplain_statistical_parity, fairxplain_equalized_odds, fairxplain_predictive_parity
from aif360.sklearn.metrics import intersection, selection_rate
import numpy as np

import pytest



class FairnessInstance:
    def __init__(self, dataset, prot_attr_config, classifier):
        self.dataset = dataset
        self.classifier = classifier
        self.prot_attr_config = prot_attr_config
        self.prot_attr = self.get_protected_attribute()
        self.X, self.y = self.get_X_y()

        # learn a model
        self.clf = LogisticRegression().fit(self.X, self.y)

        # compute fairness metrics
        self._get_statistical_parity_difference_from_aif360()
        self._get_equalized_odds_difference_from_aif360()


        
    def get_protected_attribute(self):
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
        return prot_attr_config_dic[self.dataset][self.prot_attr_config]
    
    def get_X_y(self):
        if(self.dataset == "adult"):
            X, y, sample_weight = fetch_adult(numeric_only=True)
        elif(self.dataset == "compas"):
            X, y = fetch_compas(numeric_only=True, binary_race=True)
        elif(self.dataset == "german"):
            X, y = fetch_german(numeric_only=True)
            # make age binary if age is used as protected attribute
            if('age' in self.prot_attr):
                X['age'] = X['age'] > 25
        else:
            raise ValueError("Unknown dataset")
        return X, y
    

    # helper function to compute statistical parity difference
    def _get_statistical_parity_difference_from_aif360(self):
        positive_prediction_probabilities = np.array(
            intersection(selection_rate, self.y, self.clf.predict(self.X), prot_attr=self.prot_attr)) # considering intersectional protected attributes
        self.value_statistical_parity = positive_prediction_probabilities.max() - positive_prediction_probabilities.min()

    # helper function to compute predictive parity difference
    def _get_equalized_odds_difference_from_aif360(self):
        self.value_equalized_odds = []
        for each_y in self.y.unique():
            positive_prediction_probabilities = np.array(
                intersection(selection_rate, self.y[self.y == each_y], self.clf.predict(self.X[self.y == each_y]), prot_attr=self.prot_attr)) # considering intersectional protected attributes
            value_statistical_parity = positive_prediction_probabilities.max() - positive_prediction_probabilities.min()
            self.value_equalized_odds.append(value_statistical_parity)

        # taking the max
        self.value_equalized_odds = np.max(self.value_equalized_odds)
        
    
class FairXplainerInstance:

    def __init__(self, X, y, prot_attr, clf):
        self.X = X
        self.y = y
        self.prot_attr = prot_attr
        self.clf = clf

        # compute explanations
        self._explain_statistical_parity()
        self._explain_equalized_odds()
        self._explain_predictive_parity()

    # individual explanations
    def _explain_statistical_parity(self):
        self.result_statistical_parity, self.bias_statistical_parity = fairxplain_statistical_parity(
            self.clf, 
            self.X, 
            prot_attr=self.prot_attr, 
            maxorder=1, 
            spline_intervals=3, 
            verbose=False, 
            seed=None, 
            return_bias=True 
        )

    def _explain_equalized_odds(self):
        self.result_equalized_odds, self.bias_equalized_odds = fairxplain_equalized_odds(
            self.clf, 
            self.X, 
            self.y, 
            prot_attr=self.prot_attr, 
            maxorder=1, 
            spline_intervals=3, 
            verbose=False, 
            seed=None, 
            return_bias=True 
        )

    def _explain_predictive_parity(self):
        self.result_predictive_parity, self.bias_predictive_parity = fairxplain_predictive_parity(
            self.clf, 
            self.X, 
            self.y, 
            prot_attr=self.prot_attr, 
            maxorder=1, 
            spline_intervals=3, 
            verbose=False, 
            seed=None, 
            return_bias=True
        )


@pytest.fixture(params=[
    ["compas", 0, LogisticRegression()],
    ["compas", 1, LogisticRegression()],
    ["compas", 2, LogisticRegression()],
    ["adult", 0, LogisticRegression()],
    ["adult", 1, LogisticRegression()],
    ["adult", 2, LogisticRegression()],
    ["german", 0, LogisticRegression()],
    ["german", 1, LogisticRegression()],
    ["german", 2, LogisticRegression()],
])
def fair_instance(request):
    return FairnessInstance(request.param[0], request.param[1], request.param[2])

@pytest.fixture
def fairxplainer_instance(fair_instance):
    return FairXplainerInstance(fair_instance.X, fair_instance.y, fair_instance.prot_attr, fair_instance.clf)

def test_attribute_contains_special_characters(fair_instance):
    # check if protected attributes contain special characters: &
    for attribute in fair_instance.X.columns:
        assert '&' not in attribute

def test_protected_attribute(fair_instance):
    # check if protected attributes are in the dataset
    assert all(prot_attr in fair_instance.X.columns for prot_attr in fair_instance.prot_attr)

def test_model_is_trained(fair_instance):
    # check if model is trained
    assert fair_instance.clf is not None
    assert fair_instance.clf.predict(fair_instance.X).shape[0] == fair_instance.X.shape[0]

def test_bias_correct_statistical_parity(fair_instance, fairxplainer_instance):
    # check if bias is correct
    assert fairxplainer_instance.bias_statistical_parity == fair_instance.value_statistical_parity

def test_bias_correct_equalized_odds(fair_instance, fairxplainer_instance):
    # check if bias is correct
    assert np.max(fairxplainer_instance.bias_equalized_odds) == fair_instance.value_equalized_odds

def test_coherent_explanation(fairxplainer_instance):
    # check if explanations are coherent.
    # For now we check whether the attribute matches original dataset columns

    # statistical parity
    for attribute in fairxplainer_instance.result_statistical_parity.index:
        if(attribute == "Residual FIFs"):
                continue
        assert attribute in fairxplainer_instance.X.columns

    # equalized odds
    for result in fairxplainer_instance.result_equalized_odds:
        for attribute in result.index:
            if(attribute == "Residual FIFs"):
                    continue
            assert attribute in fairxplainer_instance.X.columns

    # predictive parity
    for result in fairxplainer_instance.result_predictive_parity:
        for attribute in result.index:
            if(attribute == "Residual FIFs"):
                    continue
            assert attribute in fairxplainer_instance.X.columns