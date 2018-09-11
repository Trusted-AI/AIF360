#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
training logistic regression

SYNOPSIS::

    SCRIPT [options]

Description
===========

The last column indicates binary class.

Options
=======

-i <INPUT>, --in <INPUT>
    specify <INPUT> file name
-o <OUTPUT>, --out <OUTPUT>
    specify <OUTPUT> file name
-C <REG>, --reg <REG>
    regularization parameter (default 1.0)
-e <eta>, --eta <eta>
    fairness penalty parameter (default 1.0)
-l <LTYPE>, --ltype <LTYPE>
    likehood fitting type (default 4)
-t <NTRY>, --try <NTRY>
    the number of trials with random restart. if 0, all coefficients are
    initialized by zeros, and a model is trained only once. (default 0)
-n <ITYPE>, --itype <ITYPE>
    method to initialize coefficients. 0: by zero, 1: at random following
    normal distribution, 2: learned by standard LR, 3: separately learned by
    standard LR (default 3)
-q, --quiet
    set logging level to ERROR, no messages unless errors
--rseed <RSEED>
    random number seed. if None, use /dev/urandom (default None)
--version
    show version

Attributes
==========
N_NS : int
    the number of non sensitive features
"""

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2012/08/26"
__version__ = "3.0.0"
__copyright__ = "Copyright (c) 2011 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"
__docformat__ = "restructuredtext en"

#==============================================================================
# Imports
#==============================================================================


import sys
import argparse
import os
import platform
import commands
import logging
import datetime
import pickle
import numpy as np

# private modeules ------------------------------------------------------------
import site
site.addsitedir('.')

from fadm import __version__ as fadm_version
from sklearn import __version__ as sklearn_version
from fadm.util import fill_missing_with_mean
from fadm.lr.pr import *

#==============================================================================
# Public symbols
#==============================================================================

__all__ = []

#==============================================================================
# Constants
#==============================================================================

N_NS = 1

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

#==============================================================================
# Functions
#==============================================================================

def train(X, y, ns, opt):
    """ train model

    Parameters
    ----------
    X : ary, shape=(n_samples, n_features)
        features
    y : ary, shape=(n_samples)
        classes
    ns : int
        the number of sensitive features
    opt : object
        options

    Returns
    -------
    clr : classifier object
        trained classifier
    """
    if opt.ltype == 4:
        clr = LRwPRType4(eta=opt.eta, C=opt.C)
        clr.fit(X, y, ns, itype=opt.itype)
    else:
        sys.exit("Illegal likelihood fitting type")

    return clr

#==============================================================================
# Main routine
#==============================================================================

def main(opt):
    """ Main routine that exits with status code 0
    """

    ### pre process

    # read data
    D = np.loadtxt(opt.infile)

    # split data and process missing values
    y = np.array(D[:, -1])
    X = fill_missing_with_mean(D[:, :-1])
    del D

    ### main process

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    opt.start_time = start_time.isoformat()
    logger.info("start time = " + start_time.isoformat())

    # init constants
    ns = 1

    # train
    if opt.ntry <= 0:
        # train only once with zero coefficients
        clr = train(X, y, ns, opt)
        opt.final_loss = clr.f_loss_
        logger.info('final_loss = ' + str(opt.final_loss))
    else:
        # train multiple times with random restarts
        clr = None
        best_loss = np.inf
        best_trial = 0
        for trial in range(opt.ntry):
            logger.info("Trial No. " + str(trial + 1))
            tmp_clr = train(X, y, ns, opt)
            logger.info("loss = " + str(tmp_clr.f_loss_))
            if tmp_clr.f_loss_ < best_loss:
                clr = tmp_clr
                best_loss = clr.f_loss_
                best_trial = trial + 1
        opt.final_loss = best_loss
        logger.info('final_loss = ' + str(opt.final_loss))
        opt.best_trial = best_trial
        logger.info('best_trial = ' + str(opt.best_trial))

    # set end and elapsed time
    end_time = datetime.datetime.now()
    end_utime = os.times()[0]
    logger.info("end time = " + end_time.isoformat())
    opt.end_time = end_time.isoformat()
    logger.info("elapsed_time = " + str((end_time - start_time)))
    opt.elapsed_time = str((end_time - start_time))
    logger.info("elapsed_utime = " + str((end_utime - start_utime)))
    opt.elapsed_utime = str((end_utime - start_utime))

    ### output

    # add info
    opt.nos_samples = X.shape[0]
    logger.info('nos_samples = ' + str(opt.nos_samples))
    opt.nos_features = X.shape[1]
    logger.info('nos_features = ' + str(X.shape[1]))
    opt.classifier = clr.__class__.__name__
    logger.info('classifier = ' + opt.classifier)
    opt.fadm_version = fadm_version
    logger.info('fadm_version = ' + opt.fadm_version)
    opt.sklearn_version = sklearn_version
    logger.info('sklearn_version = ' + opt.sklearn_version)
#    opt.training_score = clr.score(X, y)
#    logger.info('training_score = ' + str(opt.training_score))

    # write file
    pickle.dump(clr, opt.outfile)
    info = {}
    for key, key_val in vars(opt).items():
        info[key] = str(key_val)
    pickle.dump(info, opt.outfile)

    ### post process

    # close file
    if opt.infile is not sys.stdin:
        opt.infile.close()

    if opt.outfile is not sys.stdout:
        opt.outfile.close()

    sys.exit(0)


### Preliminary processes before executing a main routine
if __name__ == '__main__':
    ### set script name
    script_name = sys.argv[0].split('/')[-1]

    ### init logging system
    logger = logging.getLogger(script_name)
    logging.basicConfig(level=logging.INFO,
                        format='[%(name)s: %(levelname)s'
                               ' @ %(asctime)s] %(message)s')

    ### command-line option parsing

    ap = argparse.ArgumentParser(
        description='pydoc is useful for learning the details.')

    # common options
    ap.add_argument('--version', action='version',
                    version='%(prog)s ' + __version__)

    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(verbose=True)
    apg.add_argument('--verbose', action='store_true')
    apg.add_argument('-q', '--quiet', action='store_false', dest='verbose')

    ap.add_argument("--rseed", type=int, default=None)

    # basic file i/o
    ap.add_argument('-i', '--in', dest='infile',
                    default=None, type=argparse.FileType('r'))
    ap.add_argument('infilep', nargs='?', metavar='INFILE',
                    default=sys.stdin, type=argparse.FileType('r'))
    ap.add_argument('-o', '--out', dest='outfile',
                    default=None, type=argparse.FileType('wb'))
    ap.add_argument('outfilep', nargs='?', metavar='OUTFILE',
                    default=sys.stdout, type=argparse.FileType('wb'))

    # script specific options
    ap.add_argument('-C', '--reg', dest='C', type=float, default=1.0)
    ap.set_defaults(ns=False)
    ap.add_argument('-e', '--eta', type=float, default=1.0)
    ap.add_argument('-l', '--ltype', type=int, default=4)
    ap.add_argument('-n', '--itype', type=int, default=3)
    ap.set_defaults(ns=False)
    ap.add_argument('--ns', dest='ns', action='store_true')
    ap.add_argument('-t', '--try', dest='ntry', type=int, default=0)

    # parsing
    opt = ap.parse_args()

    # post-processing for command-line options
    # disable logging messages by changing logging level
    if not opt.verbose:
        logger.setLevel(logging.ERROR)

    # set random seed
    np.random.seed(opt.rseed)

    # basic file i/o
    if opt.infile is None:
        opt.infile = opt.infilep
    del vars(opt)['infilep']
    logger.info("input_file = " + opt.infile.name)
    if opt.outfile is None:
        opt.outfile = opt.outfilep
    del vars(opt)['outfilep']
    logger.info("output_file = " + opt.outfile.name)

    ### set meta-data of script and machine
    opt.script_name = script_name
    opt.script_version = __version__
    opt.python_version = platform.python_version()
    opt.sys_uname = platform.uname()
    if platform.system() == 'Darwin':
        opt.sys_info =\
        commands.getoutput('system_profiler'
                           ' -detailLevel mini SPHardwareDataType')\
        .split('\n')[4:-1]
    elif platform.system() == 'FreeBSD':
        opt.sys_info = commands.getoutput('sysctl hw').split('\n')
    elif platform.system() == 'Linux':
        opt.sys_info = commands.getoutput('cat /proc/cpuinfo').split('\n')

    ### suppress warnings in numerical computation
    np.seterr(all='ignore')

    ### call main routine
    main(opt)
