import subprocess
import os
import shutil
import tempfile
import atexit
import logging


def execute(cmd, env=None, pipe_output=True, wait=True):
    env = env or {}
    env['PATH'] = env.get('PATH') or os.environ.get('PATH')

    kwargs = {
        'shell': True,
        'env': env
    }

    if not pipe_output:
        return subprocess.check_output(cmd, **kwargs)
    process = subprocess.Popen(cmd, bufsize=-1, **kwargs)
    if wait:
        result = process.wait()
        return result
    return process


def new_tmp_dir():
    return new_tmp_file_or_dir(dir=True)


def chmod_r(path, mode):
    """ Recursive chmod """
    os.chmod(path, mode)

    def try_chmod(filepath, mode):
        try:
            os.chmod(filepath, mode)
        except Exception:
            # potentially a symlink where we cannot chmod the target
            if not os.path.islink(filepath):
                raise

    for root, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            try_chmod(os.path.join(root, dirname), mode)
        for filename in filenames:
            try_chmod(os.path.join(root, filename), mode)


def rm_rf(path):
    """
    Recursively removes a file or directory
    """
    if not path or not os.path.exists(path):
        return
    # Make sure all files are writeable and dirs executable to remove
    chmod_r(path, 0o777)
    if os.path.isfile(path):
        os.remove(path)
    else:
        shutil.rmtree(path)


def new_tmp_file_or_dir(dir=False):
    tmp_path = tempfile.mkstemp()[1]
    if dir:
        os.remove(tmp_path)
        os.mkdir(tmp_path)

    def delete_tmp_file():
        rm_rf(tmp_path)

    atexit.register(delete_tmp_file)
    return tmp_path


def get_logger(name, log_level=logging.INFO, filename=None):
    logger = logging.getLogger(name)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=filename)
    logger.setLevel(log_level)
    return logger