#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predict classes for logistic regression model

SYNOPSIS::

    SCRIPT [options]

Description
===========

Columns of Outputs:

1. true sample class number
2. predicted class number
3. sensitive feature
4. class 0 probability
5. class 1 probability

Delimiters of columns are a single space.

Options
=======

-i <INPUT>, --in <INPUT>
    specify <INPUT> file name
-o <OUTPUT>, --out <OUTPUT>
    specify <OUTPUT> file name
-m <MODEL>, --model <MODEL>
    trained classifier (default "classification.model")
--ns
    ignore sensitive features
--hideinfo
    suppress output meta information
-q, --quiet
    set logging level to ERROR, no messages unless errors
--rseed <RSEED>
    random number seed. if None, use /dev/urandom (default None)

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
__copyright__ = "Copyright (c) 2012 Toshihiro Kamishima all rights reserved."
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
from fadm.util import fill_missing_with_mean

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

#==============================================================================
# Main routine
#==============================================================================

def main(opt):
    """ Main routine that exits with status code 0
    """

    ### pre process

    # load model file
    clr = pickle.load(opt.model)
    clr_info = pickle.load(opt.model)

    # read data
    D = np.loadtxt(opt.infile)

    # split data and process missing values
    y = np.array(D[:, -1])
    if opt.ns:
        X = fill_missing_with_mean(D[:, :-(1 + N_NS)])
    else:
        X = fill_missing_with_mean(D[:, :-1])
    S = np.atleast_2d(D[:, -(1 + N_NS):-1])

    ### main process

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    opt.start_time = start_time.isoformat()
    logger.info("start time = " + start_time.isoformat())

    # prediction and write results
    p = clr.predict_proba(X)

    # output prediction
    n = 0
    m = 0
    for i in range(p.shape[0]):
        c = np.argmax(p[i, :])
        opt.outfile.write("%d %d " % (y[i], c))
        opt.outfile.write(" ".join(S[i, :].astype(str)) + " ")
        opt.outfile.write(str(p[i, 0]) + " " + str(p[i, 1]) + "\n")
        n += 1
        m += 1 if c == y[i] else 0

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

    # add meta info
    opt.nos_samples = n
    logger.info('nos_samples = ' + str(opt.nos_samples))
    opt.nos_correct_samples = m
    logger.info('nos_correct_samples = ' + str(opt.nos_correct_samples))
    opt.accuracy = m / float(n)
    logger.info('accuracy = ' + str(opt.accuracy))
    opt.negative_mean_prob = np.mean(p[:, 0])
    logger.info('negative_mean_prob = ' + str(opt.negative_mean_prob))
    opt.positive_mean_prob = np.mean(p[:, 1])
    logger.info('positive_mean_prob = ' + str(opt.positive_mean_prob))

    # output meta information
    if opt.info:
        for key in clr_info.keys():
            opt.outfile.write("#classifier_%s=%s\n" %
                              (key, str(clr_info[key])))

        for key, key_val in vars(opt).items():
            opt.outfile.write("#%s=%s\n" % (key, str(key_val)))

    ### post process

    # close file
    if opt.infile != sys.stdin:
        opt.infile.close()

    if opt.outfile != sys.stdout:
        opt.outfile.close()

    if opt.model != sys.stdout:
        opt.model.close()

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
                    default=None, type=argparse.FileType('w'))
    ap.add_argument('outfilep', nargs='?', metavar='OUTFILE',
                    default=sys.stdout, type=argparse.FileType('w'))

    # script specific options
    ap.add_argument('-m', '--model', type=argparse.FileType('rb'),
                    required=True)
    ap.set_defaults(ns=False)
    ap.add_argument("--ns", dest="ns", action="store_true")
    ap.set_defaults(info=True)
    ap.add_argument('--hideinfo', dest='info', action='store_false')

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
