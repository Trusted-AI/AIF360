from contextlib import nullcontext
from functools import partial
from inFairness import fairalgo, distances
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from aif360.sklearn.inprocessing import SenSeI, SenSR
from aif360.sklearn.inprocessing.infairness import Dataset


def test_sensei_classification():
    """Tests whether SenSeI output matches original implementation."""
    X, y = make_classification(n_features=10)
    X = X.astype('float32')
    ds = Dataset(X, y)

    dx = distances.SVDSensitiveSubspaceDistance()
    dx.fit(X, 2)
    dy = distances.SquaredEuclideanDistance()
    dy.fit(2)

    torch.random.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 2))
    aif360_sensei = SenSeI(
            mlp, criterion=nn.CrossEntropyLoss, distance_x=dx, distance_y=dy,
            rho=1., eps=0.1, auditor_nsteps=10, auditor_lr=0.01, max_epochs=5,
            optimizer=optim.Adam, lr=1e-3, predict_nonlinearity=None, verbose=0)
    y_pred = aif360_sensei.fit(X, y).predict_proba(X)
    assert aif360_sensei.regression_ == False

    torch.random.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 2))
    orig_sensei = fairalgo.SenSeI(mlp, dx, dy, F.cross_entropy, 1., 0.1, 10, 0.01)
    orig_sensei.train()
    opt = optim.Adam(orig_sensei.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad()
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=len(ds))))
        loss = orig_sensei(*batch).loss
        loss.backward()
        opt.step()
    orig_sensei.eval()
    y_pred_orig = orig_sensei(torch.as_tensor(X), torch.as_tensor(y)).y_pred.detach().numpy()

    assert np.allclose(y_pred, y_pred_orig)

def test_sensr_regression():
    """Tests whether SenSR output matches original implementation."""
    X, y = make_regression(n_features=10)
    X, y = X.astype('float32'), y.astype('float32').reshape(-1, 1)
    ds = Dataset(X, y)

    dx = distances.MahalanobisDistances()
    dx.fit(torch.eye(10))

    torch.random.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 1))
    aif360_sensei = SenSR(
            mlp, criterion=nn.MSELoss, distance_x=dx, eps=0.1, lr_lamb=1.,
            lr_param=1., auditor_nsteps=10, auditor_lr=0.01, max_epochs=5,
            optimizer=optim.Adam, lr=1e-3, predict_nonlinearity=None, verbose=0)
    y_pred = aif360_sensei.fit(X, y).predict(X)
    assert aif360_sensei.regression_ == True

    torch.random.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 1))
    orig_sensei = fairalgo.SenSR(mlp, dx, F.mse_loss, 0.1, 1., 1., 10, 0.01)
    orig_sensei.train()
    opt = optim.Adam(orig_sensei.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad()
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=len(ds))))
        loss = orig_sensei(*batch).loss
        loss.backward()
        opt.step()
    orig_sensei.eval()
    y_pred_orig = orig_sensei(torch.as_tensor(X), torch.as_tensor(y)).y_pred.detach().numpy()

    assert np.allclose(y_pred, y_pred_orig)

@pytest.mark.parametrize(
    "y, criterion, raises", [
    # input unchanged
    ([1, 0, 1, 0], nn.CrossEntropyLoss, None),
    # input unchanged (wrong dtype)
    ([1., 0, 1, 0], nn.CrossEntropyLoss, RuntimeError),
    # input unchanged (must be 2D)
    ([[1.], [0], [1], [0]], nn.BCEWithLogitsLoss, None),
    # binarized to [[0], [1], [0], [1]]
    ([-1, 1, -1, 1], nn.BCEWithLogitsLoss, None),
    # binarized to [[0], [1], [0], [1]] (wrong shape)
    ([-1, 1, -1, 1], nn.CrossEntropyLoss, RuntimeError),
    # binarized to [[0], [1], [0], [1]]
    ([0, 2, 0, 2], nn.BCEWithLogitsLoss, None),
    # binarized to [[0], [1], [0], [1]] (wrong shape)
    ([0, 2, 0, 2], nn.CrossEntropyLoss, RuntimeError),
    # binarize to one-hot
    ([0, 1, 0, 2], nn.CrossEntropyLoss, None),
    # input unchanged -- detected regression (wrong shape)
    ([0.1, 1, 0, 2], nn.CrossEntropyLoss, RuntimeError),
    # input unchanged -- detected regression
    ([[0.1], [1], [0], [2]], nn.MSELoss, None),
    # binarized to [[0], [1], [0], [1]]
    (['a', 'b', 'a', 'b'], nn.BCEWithLogitsLoss, None),
    # input unchanged -- binary
    ([[0., 1], [1, 0], [0, 1], [1, 0]], nn.CrossEntropyLoss, None),
    # input unchanged -- multilabel
    ([[0., 1], [1, 0], [0, 1], [1, 0]], nn.BCEWithLogitsLoss, None),
    # multiclass-multioutput (not supported)
    ([[0, 1], [1, 2], [0, 1], [1, 0]], nn.CrossEntropyLoss, ValueError),
    # input unchanged -- detected multioutput regression
    ([[0.1, 1], [1, 0], [0, 1], [1, 0]], nn.MSELoss, None),
])
def test_target_encoding(y, criterion, raises):
    """Tests the automatic type casting for classification problems."""
    X = np.random.random((4, 2)).astype('float32')
    y = np.array(y)
    if criterion == nn.MSELoss:
        y = np.array(y, dtype='float32')
    classes = np.unique(y).tolist()
    if criterion == nn.BCEWithLogitsLoss or criterion == nn.MSELoss:
        ndim = 1 if y.ndim < 2 else y.shape[1]
    else:
        ndim = len(classes)
    dx = distances.SquaredEuclideanDistance()
    dx.fit(2)
    dy = distances.SquaredEuclideanDistance()
    dy.fit(ndim)
    mlp = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, ndim))
    sensei = SenSeI(mlp, criterion=criterion, distance_x=dx, distance_y=dy,
            rho=1., eps=0.1, auditor_nsteps=1, auditor_lr=0.01, max_epochs=1,
            optimizer=optim.Adam, lr=1e-3, verbose=0)
    with pytest.raises(raises) if raises is not None else nullcontext():
        sensei.fit(X, y)
        assert sensei.regression_ == (criterion == nn.MSELoss)
        if not sensei.regression_:
            assert sensei.classes_.tolist() == classes

dy = distances.SquaredEuclideanDistance()
dy.fit(2)
@pytest.mark.parametrize(
    "estimator_cls", [
    partial(SenSeI, distance_y=dy, rho=1., eps=0.1, auditor_nsteps=10, auditor_lr=0.01),
    partial(SenSR, eps=0.1, lr_lamb=1., lr_param=1., auditor_nsteps=10, auditor_lr=0.01),
])
def test_grid_pipe(estimator_cls):
    """Tests if SenSeI/SenSR works in a Pipeline and GridSearchCV."""
    X, y = make_classification(n_features=10)
    X = X.astype('float32')

    dx = distances.SquaredEuclideanDistance()
    dx.fit(10)

    mlp = nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 2))
    estimator = estimator_cls(mlp, criterion=nn.CrossEntropyLoss, distance_x=dx,
                              optimizer=optim.Adam, lr=1e-3, max_epochs=5, verbose=0)

    pipe = Pipeline([('scaler', StandardScaler()), ('estimator', estimator)])
    params = {'estimator__auditor_nsteps': [0, 10, 25]}
    grid = GridSearchCV(pipe, params, scoring='accuracy')
    grid.fit(X, y)
    assert not pd.DataFrame(grid.cv_results_).isna().any(axis=None)
