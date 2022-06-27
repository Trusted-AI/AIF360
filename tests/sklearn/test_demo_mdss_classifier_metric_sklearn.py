from ..notebook_runner import notebook_run
import os

def test_demo_mdss_classifier_metric_sklearn():
    nb, errors = notebook_run(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..','examples', 'sklearn',
            'demo_mdss_classifier_metric_sklearn.ipynb'))

    if len(errors) > 0:
        for err in errors:
            for tbi in err['traceback']:
                print(tbi)
        raise AssertionError("errors in notebook testcases")
