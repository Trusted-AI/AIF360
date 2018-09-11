#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add quadratic terms to tab/comma/space separated data

SYNOPSIS::

    SCRIPT [options] [<INPUT> [<OUTPUT>]]

Description
-----------
Quadratic terms of specified variables are genereted, and these are inserted
just before the last <LAST> columns.

Options
-------
-i <INPUT>, --in <INPUT>
    specify <INPUT> file name
-o <OUTPUT>, --out <OUTPUT>
    specify <OUTPUT> file name
-d <DL>, --delimiter <DL>
    column delimiter string. 't' specifies tab character.(default " ")
-g <IGNORE>, --ignore <IGNORE>
    ignore line if the line start with char included in this string
    (default "#")
-m <MODE>, --mode <MODE>
    <MODE>=squared: only self squared terms are generated,
    <MODE>=cross: cross terms of two variables are generated, and
    <MODE>=both: both squared and cross terms are generated. (default=both)
-f <FIRST>, --first <FIRST>
    for the first <FIRST> columns, quadratic terms are not generated.
    (default 0)
-l <LAST>, --last <LAST>
    for the last <LAST> columns, quadratic terms are not generated.
    (default 0)
--version
    show version
"""

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2012/04/11"
__version__ = "1.1.0"
__copyright__ = "Copyright (c) 2012 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"
__docformat__ = "restructuredtext en"

#==============================================================================
# Imports
#==============================================================================

import sys
import argparse
import itertools

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

def gen_quad_terms(f, mode):
    """
    generate quadratic terms

    Parameters
    ----------
    f : list, type=str
        list of columns to generated quadratic terms
    mode : str, ['squared', 'cross', 'both']
        types of quadratic terms to generate


    Returns
    -------
    quad : list, type=str
        generated
    """

    g = [float(x) for x in f]
    q = []

    # add squared terms
    if mode == 'squared' or mode == 'both':
        for x in g:
            q.append(x ** 2)

    # add squared terms
    if mode == 'cross' or mode == 'both':
        for x in itertools.combinations(g, 2):
            q.append(x[0] * x[1])

    return [ str(int(x)) if x % 1.0 == 0 else str(x) for x in q ]

#==============================================================================
# Main routine
#==============================================================================

def main(opt):
    """ Main routine that exits with status code 0
    """

    ### main process
    # read from file
    line_no = 0
    try:
        for line in opt.infile.readlines():
            line = line.rstrip('\r\n')
            line_no += 1

            # skip empty line and comment line
            if line == "" or opt.ignore.find(line[0]) >= 0:
                continue

            # split into columns
            f = line.split(opt.dl)
            l = len(f)

            # process
            q = gen_quad_terms(f[opt.first:l - opt.last], opt.mode)

            # output
            opt.outfile.write(opt.dl.join(f[:l - opt.last]) + opt.dl +
                              opt.dl.join(q) + opt.dl +
                              opt.dl.join(f[l - opt.last:]) + '\n')

    except IndexError:
        sys.exit("Parse error in line %d" % line_no)
    except IOError:
        sys.exit("File error in line %d" % line_no)

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

    # options for a data format
    ap.add_argument('-d', '--delimiter', dest='dl', default=' ')
    ap.add_argument('-g', '--ignore', default='#')

    # script specific options
    ap.add_argument('-m', '--mode', default='both',
                    choices=['squared', 'cross', 'both'])
    ap.add_argument('-f', '--first', type=int, default=0)
    ap.add_argument('-l', '--last', type=int, default=0)

    ap.add_argument('-c', '--choices', type=str,
                    default='a', choices=['a', 'b', 'c'])

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

    ### call main routine
    main(opt)
