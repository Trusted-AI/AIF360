import sys
sys.path.append('../..')
from aif360.sklearn.datasets import fetch_adult, fetch_compas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
# from aif360.metrics import statistical_parity_difference
import pandas as pd


# X, y, sample_weight = fetch_adult(numeric_only=True)
X, y = fetch_compas(numeric_only=True, binary_race=True)


X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)

y = pd.Series(y.factorize(sort=True)[0], index=y.index)


scaler = MinMaxScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
print(X)



(X_train, X_test,
 y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=1234567)

clf = LogisticRegression(solver='liblinear').fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

# print(statistical_parity_difference(y_test, y_pred, prot_attr='sex'))

from aif360.sklearn.explainers.bias_explainer import explain_statistical_parity
explain_statistical_parity(clf, X_train, sensitive_features=['race'], maxorder=2, draw_plot=True, verbose=True)


