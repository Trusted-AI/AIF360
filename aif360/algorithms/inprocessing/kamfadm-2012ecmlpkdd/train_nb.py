#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training naive Bayes

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
-b <BETA>, --beta <BETA>
    coefficient of a prior for feature probabilities (default 1.0)
-f <NFV>, --nfv <NFV>
    a list of the numbers of feature domains, which are separated by ':'. if a
    single number is specified, all sizes of domeins are set to the number. By
    default, all domains are set to 2.
--ns
    ignore sensitive features
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
from fadm.nb import CompositeNaiveBayes
from fadm.util import decode_nfv

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

    # read data
    D = np.loadtxt(opt.infile)

    # split data and process missing values
    y = D[:, -1]
    if opt.ns:
        X = D[:, :-(1 + N_NS)]
        nf = X.shape[1]
        nfv = decode_nfv(opt.nfv, nf)
    else:
        X = D[:, :-1]
        nf = X.shape[1]
        nfv = np.r_[decode_nfv(opt.nfv, nf - N_NS),
                    2 * np.ones(N_NS)].astype(int)

    ### main process

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    opt.start_time = start_time.isoformat()
    logger.info("start time = " + start_time.isoformat())

    # main process
    clr = CompositeNaiveBayes(2, nf, nfv, beta=opt.beta)
    clr.fit(X, y)

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
#    opt.training_score = clr.score(X, y)
#    logger.info('training_score = ' + str(opt.training_score))

    # write file
    pickle.dump(clr, opt.outfile)
    info = {}
    for key, key_val in vars(opt).iteritems():
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
                    default=None, type=argparse.FileType('w'))
    ap.add_argument('outfilep', nargs='?', metavar='OUTFILE',
                    default=sys.stdout, type=argparse.FileType('w'))

    # script specific options
    ap.add_argument('-b', '--beta', type=float, default=1.0)
    ap.add_argument('-f', '--nfv', type=str, default="2")
    ap.set_defaults(ns=False)
    ap.add_argument("--ns", dest="ns", action="store_true")

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
