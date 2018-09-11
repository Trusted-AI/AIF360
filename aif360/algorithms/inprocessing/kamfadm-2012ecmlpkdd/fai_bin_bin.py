#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute various types of fairness-aware indexes.

SYNOPSIS::

    SCRIPT [options] [<INPUT> [<OUTPUT>]]

Description
===========

Input
-----

Both a class and a sensitive attribute are assumed to be binary. As default,
the first, the second, and the third columns indicate a correct class, an
estimated class, and a sensitive attribute.

Output
------

1. Counts of contingency table
2. Marginal counts of contingency table elements
3. Accuracy
4. Mutual Information with natural log
5. Contingency table with sensitive attribute:
6. Mutual Information between correct classes and sensitive attributes.
   (natural log)
7. Mutual Information between estimated classes and sensitive attributes.
   (natural log)
8. KL-divergence between correct and estimated conditional distributions given
   sensitive attributes.
9. Hellinger distance between correct and estimated distributions jointed with
   distributions given sensitive attributes
10. Caldars-Verwer score.

Options
=======

-i <INPUT>, --in <INPUT>
    Specify <INPUT> file name. if this option is not specified and non-optional
    argument is specified, the first argument is used as input file name
    (default sys.stdin)
-o <OUTPUT>, --out <OUTPUT>
    Specify <OUTPUT> file name (default sys.stdout)
-c <CORRECT>, --correct <CORRECT>
    The column number for the data of the correct class, starting from 1
    (default 1)
-e <ESTIMATED>, --estimated <ESTIMATED>
    The column number for the data of the estimated class, starting from 1
    (default 2)
-s <SENSITIVE>, --sensitive <SENSITIVE>
    The column number for the data of the sensitive attribute class, starting
    from 1 (default 3)
-d <DL>, --delimiter <DL>
    Column delimiter string. 't' specifies TAB character.(default " ")
-g <IGNORE>, --ignore <IGNORE>
    Ignore line if the line start with char included in this string
    (default "#")
-r, --raw
    simply separated with the specified delimiter
-n, --negate
    As default, a value 1 indicates a positive class. If specified, this
    meaning is negated.
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2012/08/26"
__version__ = "2.0.0"
__copyright__ = "Copyright (c) 2012 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"
__docformat__ = "restructuredtext en"

#==============================================================================
# Imports
#==============================================================================

import sys
import argparse
import numpy as np

# private modeules -------------------------------------------------------------

import site
site.addsitedir('.')

from fadm.eval import BinClassBinSensitiveStats

#==============================================================================
# Public symbols
#==============================================================================

__all__ = []

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

#==============================================================================
# Functions
#==============================================================================

def read_01_file(opt):
    """ read data from file
    
    Parameters
    ----------
    opt : option
        parsed options

    Returns
    -------
    n : array, shape=(2, 2, 2), dtype=int
        array: (s,1,1)=TP, (s,1,0)=FN, (s,0,1)=FP, (s,0,0)=TN, where s= 0 or 1
    """

    # init stats
    n = np.zeros((2, 2, 2))

    # read from file
    line_no = 0
    for line in opt.infile.readlines():
        line = line.rstrip('\r\n')
        line_no += 1

        # skip empty line
        if line == "":
            continue

        # test top char if this line is comment
        t = line[0]
        if opt.ignore.find(t) < 0:
            f = line.split(opt.dl)

            try:
                s = int(f[opt.sensitive])
                c = int(f[opt.correct])
                e = int(f[opt.estimated])

                n[s, c, e] += 1 # count up
            except IndexError:
                sys.exit("Parse error in line %d" % line_no)

    return n

#==============================================================================
# Main routine
#==============================================================================

def main(opt):
    """ Main routine that exits with status code 0
    """

    ### main process

    # read file
    ct = read_01_file(opt)
    fai = BinClassBinSensitiveStats(ct)

    # check negation
    if opt.negate:
        fai.negate()

    # output results
    if opt.format:
        print(fai.str_all(), file=opt.outfile)
    else:
        print(opt.dl.join(map(str, fai.all())), file=opt.outfile)

    ### post process

    # close file
    if opt.infile is not sys.stdin:
        opt.infile.close()

    if opt.outfile is not sys.stdout:
        opt.outfile.close()

    sys.exit(0)

### Check if this is call as command script
if __name__ == '__main__':
    ### set script name
    script_name = sys.argv[0].split('/')[-1]

    ### command-line option parsing
    ap = argparse.ArgumentParser(
        description='pydoc is useful for learning the details.')

    # common options
    ap.add_argument('--version', action='version',
                    version='%(prog)s ' + __version__)

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
    ap.add_argument("-c", "--correct", type=int, default=1)
    ap.add_argument("-e", "--estimated", type=int, default=2)
    ap.add_argument("-s", "--sensitive", type=int, default=3)
    ap.add_argument("-d", "--dlimiter", type=str, dest="dl", default=" ")
    ap.add_argument("-g", "--ignore", type=str, default="#")
    ap.set_defaults(format=True)
    ap.add_argument("-r", "--raw", dest="format", action="store_false")
    ap.set_defaults(negate=False)
    ap.add_argument("-n", "--negate", dest="negate", action="store_true")

    # parsing
    opt = ap.parse_args()

    ### post-processing for command-line options
    # basic file i/o
    if opt.infile is None:
        opt.infile = opt.infilep
    del vars(opt)['infilep']
    if opt.outfile is None:
        opt.outfile = opt.outfilep
    del vars(opt)['outfilep']

    # set meta-data of script and machine
    opt.script_name = script_name
    opt.script_version = __version__

    # the specified delimiter is TAB?
    if opt.dl == 't' or opt.dl == 'T':
        opt.dl = '\t'

    # check columns
    if opt.correct <= 0 or opt.estimated <= 0 or opt.sensitive <= 0\
       or opt.correct == opt.estimated\
       or opt.correct == opt.sensitive\
    or opt.estimated == opt.sensitive:
        sys.exit("Incorrect specification of data columns")
    opt.estimated -= 1
    opt.correct -= 1
    opt.sensitive -= 1

    ### call main routine
    main(opt)
