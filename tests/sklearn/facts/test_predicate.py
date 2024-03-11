import numpy as np
import pandas as pd
import string

from aif360.sklearn.detectors.facts.predicate import Predicate, featureChangePred, recIsValid, drop_two_above
from aif360.sklearn.detectors.facts.parameters import ParameterProxy


def test_equal_different_order() -> None:
    n_samples = 1000
    feats = ["".join(np.random.choice(list(string.ascii_letters), l).tolist()) for l in np.random.randint(1, 20, 100)]
    vals = np.random.randint(1, 1000, size=100)

    feat_vals = list(zip(feats, vals))

    samples = [np.random.permutation(feat_vals) for _ in range(n_samples)]
    sample_feats = [[f for f, v in sample.tolist()] for sample in samples]
    sample_vals = [[v for f, v in sample.tolist()] for sample in samples]

    ps = [Predicate(fs, vs) for fs, vs in zip(sample_feats, sample_vals)]

    if ps == []:
        return
    assert all(p == ps[0] for p in ps)

def test_featureChangePred() -> None:
    ifclause = Predicate.from_dict({"a": 1, "b": 2, "c": 3})
    thenclause = Predicate.from_dict({"a": 1, "b": 4, "c": 7})
    diff_penal = lambda x, y: abs(x - y)
    params = ParameterProxy(featureChanges={
        "a": lambda x, y: abs(x - y),
        "b": lambda x, y: 200 * abs(x - y),
        "c": lambda x, y: 50 * abs(x - y)
    })

    assert featureChangePred(ifclause, thenclause, params=params) == 600

def test_recIsValid_nodrop() -> None:
    n_feats = 100
    max_str_len = 20
    max_val = 1000
    feats = ["".join(np.random.choice(list(string.ascii_letters), l).tolist()) for l in np.random.randint(1, max_str_len, n_feats)]
    
    vals = np.random.randint(1, max_val, size=n_feats)
    mock_dataset = pd.DataFrame([vals.tolist()] * 100, columns=feats)

    n_samples = 1000
    for _ in range(n_samples):
        n_feats_subset = np.random.randint(2, n_feats)
        ifclause_feats = np.random.choice(feats, n_feats_subset, replace=False)
        ifclause_vals = np.random.randint(1, max_val, size=n_feats_subset)
        ifclause = Predicate(features=ifclause_feats.tolist(), values=ifclause_vals.tolist())

        if np.random.rand() > 0.5:
            # create valid counterfactual
            thenclause_feats = ifclause_feats
            thenclause_vals = ifclause_vals.copy()
            rand_idxs = np.random.choice(list(range(n_feats_subset)), np.random.randint(1, n_feats_subset), replace=False)
            feat_change = rand_idxs[0]
            feats_rand = rand_idxs[1:]
            thenclause_vals[feat_change] = ifclause_vals[feat_change] + 1
            thenclause_vals[feats_rand] = np.random.randint(1, max_val, size=len(feats_rand))

            thenclause = Predicate(features=thenclause_feats.tolist(), values=thenclause_vals.tolist())

            assert recIsValid(ifclause, thenclause, mock_dataset, drop_infeasible=False)
        else:
            # create invalid counterfactuals
            thenclause_feats_1 = feats[:n_feats_subset + 1]
            thenclause_vals_1 = np.random.randint(1, max_val, size=len(thenclause_feats_1))
            thenclause_1 = Predicate(features=thenclause_feats_1, values=thenclause_vals_1.tolist())

            thenclause_2 = Predicate(features=ifclause_feats.tolist(), values=ifclause_vals.tolist())

def test_recIsValid_withdrop() -> None:
    n_samples = 1000
    n_feats = 7
    max_val = 1000
    feats = ["parents", "age", "ages", "education-num", "PREDICTOR RAT AGE AT LATEST ARREST", "age_cat", "sex"]
    feats_nosmaller = ["parents", "age", "ages", "education-num", "PREDICTOR RAT AGE AT LATEST ARREST", "age_cat"]
    vals = np.random.randint(1, max_val, size=n_feats)

    mock_dataset = pd.DataFrame([vals.tolist()] * 100, columns=feats)

    for _ in range(n_samples):
        n_feats_subset = np.random.randint(1, n_feats)
        ifclause_feats = np.random.choice(feats, n_feats_subset, replace=False)
        ifclause_vals = np.random.randint(1, max_val, size=n_feats_subset)
        dic = {f: v for f, v in zip(ifclause_feats, ifclause_vals)}
        if "age" in ifclause_feats:
            rand_int = np.random.randint(1, max_val)
            dic["age"] = pd.Interval(rand_int, rand_int + np.random.randint(1, max_val))
        ifclause = Predicate.from_dict(dic)

        for f in feats_nosmaller:
            if not f in ifclause_feats:
                continue
            dic = {f: v for f, v in zip(ifclause.features, ifclause.values)}
            
            dic[f] = ifclause.to_dict()[f] + 1
            thenclause = Predicate.from_dict(dic)
            assert recIsValid(ifclause, thenclause, mock_dataset, drop_infeasible=True)
            dic[f] = ifclause.to_dict()[f] - 1
            thenclause = Predicate.from_dict(dic)
            assert not recIsValid(ifclause, thenclause, mock_dataset, drop_infeasible=True)
        
        if "sex" in ifclause_feats:
            dic = {f: v for f, v in zip(ifclause.features, ifclause.values)}
            dic["sex"] = ifclause.to_dict()["sex"] + 1
            assert not recIsValid(ifclause, Predicate.from_dict(dic), mock_dataset, drop_infeasible=True)
        
        dic = {f: v for f, v in zip(ifclause.features, ifclause.values)}
        rand_feat = np.random.choice(ifclause.features)
        dic[rand_feat] = "Unknown"
        thenclause = Predicate.from_dict(dic)
        assert not recIsValid(ifclause, thenclause, mock_dataset, drop_infeasible=True)

def test_drop_two_above() -> None:
    n_samples = 1000
    max_val = 1000
    feats = ['education-num', 'age', 'PREDICTOR RAT AGE AT LATEST ARREST']

    rand_int = np.random.randint(1, max_val)
    ages_age = [pd.Interval(rand_int * i, rand_int * (i + 1)) for i in range(100)]
    ages_rat = np.cumsum(np.random.randint(1, max_val))
    ages_full = [age.left for age in ages_age] + ages_rat.tolist()

    for _ in range(n_samples):
        rand_int = np.random.randint(1, max_val)
        vals = [np.random.randint(1, max_val), np.random.choice(ages_age), np.random.choice(ages_rat)]

        ifclause = Predicate(features=feats, values=vals)

        for offset in range(-3, 3):
            dic = {f: v for f, v in zip(ifclause.features, ifclause.values)}
            dic["education-num"] += offset
            thenclause = Predicate.from_dict(dic)
            assert drop_two_above(ifclause, thenclause, ages_full)
        for offset in range(3, 10):
            dic = {f: v for f, v in zip(ifclause.features, ifclause.values)}
            dic["education-num"] += offset
            thenclause = Predicate.from_dict(dic)
            assert not drop_two_above(ifclause, thenclause, ages_full)
        
        for offset in range(-3, 3):
            dic = {f: v for f, v in zip(ifclause.features, ifclause.values)}
            new_age = ages_age.index(dic["age"]) + offset
            if new_age >= 0 and new_age < len(ages_age):
                dic["age"] = ages_age[ages_age.index(dic["age"]) + offset]
            else:
                continue
            thenclause = Predicate.from_dict(dic)
            assert drop_two_above(ifclause, thenclause, ages_full)
        for offset in range(3, 10):
            dic = {f: v for f, v in zip(ifclause.features, ifclause.values)}
            new_age = ages_age.index(dic["age"]) + offset
            if new_age >= 0 and new_age < len(ages_age):
                dic["age"] = ages_age[ages_age.index(dic["age"]) + offset]
            else:
                continue
            thenclause = Predicate.from_dict(dic)
            assert not drop_two_above(ifclause, thenclause, ages_full)

def test_unequal_different_type() -> None:
    p = Predicate(["sex", "age", "number_of_noses"], ["Male", pd.Interval(20, 25), 2])
    random_stuff = [3, "Hello", pd.Interval(10, 30), 3.14]

    for r in random_stuff:
        assert p != r
    

