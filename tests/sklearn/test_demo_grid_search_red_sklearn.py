from ..notebook_runner import notebook_run
import os


def test_demo_grid_search_classification():
    nb, errors = notebook_run(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..','examples', 'sklearn',
            'demo_grid_search_reduction_classification_sklearn.ipynb'))

    if len(errors) > 0:
        for err in errors:
            for tbi in err['traceback']:
                print(tbi)
        raise AssertionError("errors in notebook testcases")

def test_demo_grid_search_regression():
    nb, errors = notebook_run(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..','examples', 'sklearn',
            'demo_grid_search_reduction_regression_sklearn.ipynb'))

    if len(errors) > 0:
        for err in errors:
            for tbi in err['traceback']:
                print(tbi)
        raise AssertionError("errors in notebook testcases")
