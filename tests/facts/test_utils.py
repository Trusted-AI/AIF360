import os
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

from aif360.algorithms.postprocessing.facts.utils import (
    save_state,
    save_rules_by_if,
    save_model,
    save_object,
    save_test_data_used,
    load_state,
    load_model,
    load_object,
    load_rules_by_if,
    load_test_data_used,
    rules_to_latex,
    table_to_latex,
)
from aif360.algorithms.postprocessing.facts.predicate import Predicate

class MockModel:
    def predict(self, X: ArrayLike) -> ArrayLike:
        return np.ones(X.shape[0])
    
def test_state() -> None:
    mock_rules = {"a": 1, "b": 2}
    mock_X = pd.DataFrame([[1, 2], [3, 4]], columns=["e", "f"])
    mock_model = MockModel()

    save_state("temp", mock_rules, mock_X, mock_model)
    r, X, m = load_state("temp")
    assert  r == mock_rules
    assert (X == mock_X).all().all()
    os.remove("temp")

def test_rules_by_if() -> None:
    mock_rules = {"a": 1, "b": 2}

    save_rules_by_if("temp", mock_rules)
    r = load_rules_by_if("temp")
    assert  r == mock_rules
    os.remove("temp")

def test_model() -> None:
    mock_model = MockModel()

    save_model("temp", mock_model)
    m = load_model("temp")
    assert m.predict(pd.DataFrame([[1, 2], [3, 4]])).shape[0] == 2
    os.remove("temp")

def test_object() -> None:
    obj = 2
    save_object("temp", obj)
    o = load_object("temp")
    assert o == obj

    obj = [1, 2, 3, 4, 5]
    save_object("temp", obj)
    o = load_object("temp")
    assert o == obj

    obj = {1, 2, 3, 4, 5}
    save_object("temp", obj)
    o = load_object("temp")
    assert o == obj

    obj = {"one": 1, "two": 2}
    save_object("temp", obj)
    o = load_object("temp")
    assert o == obj

    os.remove("temp")

def test_test_data_used() -> None:
    mock_X = pd.DataFrame([[1, 2], [3, 4]], columns=["e", "f"])

    save_test_data_used("temp", mock_X)
    X = load_test_data_used("temp")
    assert (X == mock_X).all().all()
    os.remove("temp")

def test_rules_to_latex() -> None:
    rules = {
        Predicate.from_dict({"a": 13}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15}), 0.35),
            (Predicate.from_dict({"a": 17}), 0.7),
        ]),
        "Female": (0.25, [])},
        Predicate.from_dict({"a": 13, "b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.5),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.99),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.8),
        ])},
    }
    bias = {
        Predicate.from_dict({"a": 13}): ("onestr", "twostr", 3.14),
        Predicate.from_dict({"a": 13, "b": 45}): ("astr", "anotherstr", 2.72),
    }

    ret = rules_to_latex(rules, bias, subgroup_names=["1", "2"])
    expected = r"""
\begin{figure}[h]
\centering
\begin{minipage}{1\linewidth}
\begin{lstlisting}[style = base,escapechar=+]
+\textbf{Subgroup 1}+
If a = 13:
    Protected Subgroup = `Male', !20.00%! covered
        Make @a = 15@ with effectiveness &35.00%&
        Make @a = 17@ with effectiveness &70.00%&
    Protected Subgroup = `Female', !25.00%! covered
        """ "\t\t" r"""@No recourses for this subgroup.@
    _Bias against `onestr' due to twostr. Unfairness score = 3.14._
\end{lstlisting}
\begin{lstlisting}[style = base,escapechar=+]
+\textbf{Subgroup 2}+
If a = 13, b = 45:
    Protected Subgroup = `Male', !20.00%! covered
        Make @a = 15, b = 40@ with effectiveness &50.00%&
        Make @a = 17, b = 38@ with effectiveness &99.00%&
    Protected Subgroup = `Female', !25.00%! covered
        Make @a = 15, b = 40@ with effectiveness &45.00%&
        Make @a = 17, b = 38@ with effectiveness &80.00%&
    _Bias against `astr' due to anotherstr. Unfairness score = 2.72._
\end{lstlisting}

\caption{}
\label{}
\end{minipage}
\end{figure}
"""
    assert ret == expected

def test_table_to_latex() -> None:
    comb_df = pd.DataFrame(
        [
            [1, 2, 5, 6, 9, 10],
            [3, 4, 7, 8, 11, 12]
        ],
        index=[Predicate.from_dict({"a": 1}), Predicate.from_dict({"b": 13})]
    )
    comb_df.columns = pd.MultiIndex.from_tuples([
        ("col1", "rank"), ("col1", "score"), ("col1", "bias against"),
        ("col2", "rank"), ("col2", "score"), ("col2", "bias against")
    ])
    subgroups = [Predicate.from_dict({"a": 1}), Predicate.from_dict({"b": 13})]
    metric_names = [("col1", "colname1"), ("col1", "colname1")]

    ret = table_to_latex(comb_df, subgroups, metric_names)
    expected = r"""
\begin{table}[ht]
\caption{}
  \label{}
  \centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
\multicolumn{1}{r}{} & \multicolumn{3}{c}{\textbf{Subgroup 1}} & \multicolumn{3}{c}{\textbf{Subgroup 2}}  \\ \cmidrule(r){2-7}

\multicolumn{1}{c}{} & \multicolumn{1}{c}{rank} & \multicolumn{1}{c}{bias against} & \multicolumn{1}{c}{unfairness score} & \multicolumn{1}{c}{rank} & \multicolumn{1}{c}{bias against} & \multicolumn{1}{c}{unfairness score} \\ \midrule
colname1 & \textbf{\textcolor{red}{1}} & 5 & 2 & 3 & 7 & 4 \\
colname1 & \textbf{\textcolor{red}{1}} & 5 & 2 & 3 & 7 & 4 \\


\bottomrule
\end{tabular}%
}
\end{table}
"""
    print(ret)
    assert ret == expected