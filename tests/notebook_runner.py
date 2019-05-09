# Adapted from:
# http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html

import os
import subprocess
import tempfile

import nbformat

def notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    dirname, __ = os.path.split(path)
    os.chdir(dirname)

    kername = "python3"

    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=600",
                "--ExecutePreprocessor.allow_errors=True",
                "--ExecutePreprocessor.kernel_name={}".format(kername),
                "--output", fout.name, path]

        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
                     for output in cell["outputs"]
                     if output.output_type == "error"]

    return nb, errors
