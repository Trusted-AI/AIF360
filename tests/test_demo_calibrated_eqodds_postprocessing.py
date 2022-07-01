# Adapted from:
# http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html

from .notebook_runner import notebook_run
import os

def test_calibrated_eqodds_postprocessing():
    nb, errors = notebook_run(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'examples', 'demo_calibrated_eqodds_postprocessing.ipynb'))

    if len(errors) > 0:
        for err in errors:
            for tbi in err['traceback']:
                print(tbi)
        raise AssertionError("errors in notebook testcases")
