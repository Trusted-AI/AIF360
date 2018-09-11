#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Discretize *adult* data

SYNOPSIS::

    SCRIPT [options]

Description
===========

Discretize *adult* data as the procedure written in [DMKD2010]_.

- Integer attributes are divided into 4 bins each of which contains
  equal numbers of samples.
- Nominal attribute values whose counts are less than 50 are merged
  into *Pool* value.

Options
=======

-i <INPUT>, --in <INPUT>
    specify <INPUT> file name
-o <OUTPUT>, --out <OUTPUT>
    specify <OUTPUT> file name
-b <BIN>, --bin <BIN>
    number of bins that numerical attributes are discretized. (default 4)
-m <MINF>, --minfreq <MINF>
    the attribute values whose counts are less or equal than <MINF> are
    meerged int *Pool* attribute value. (default 50)

:Variables:
    `script_name` : str
        name of this script

.. [DMKD2010] T.Calders and S.Verwer "Three naive Bayes approaches for
    discrimination-free classification" Data Mining and Knowledge Discovery,
    vol.21 (2010)
"""

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2011/02/07"
__version__ = "2.0.0"
__copyright__ = "Copyright (c) 2011 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License http://www.opensource.org/licenses/mit-license.php"
__docformat__ = "restructuredtext en"

#==============================================================================
# Imports
#==============================================================================

import sys
import os
import optparse
import numpy as np
from scipy.io.arff import loadarff

#==============================================================================
# Public symbols
#==============================================================================

__all__ = []

#==============================================================================
#{ Constants
#==============================================================================

#==============================================================================
#{ Module variables
#==============================================================================

script_name = os.path.basename(sys.argv[0])

#==============================================================================
#{ Classes
#==============================================================================

#==============================================================================
#{ Functions 
#==============================================================================

def discretize_numerical_attribute(attr, data, meta, name, n_bin):
    """ Discretize numerical attribute into the specified
    number of bins.
    
    :Parameters:
        attr : dict, shape=(2)
            updated attribute information
        data : arylike, shape=(n_samples, n_features)
            data
        meta : scipy.io.arff.Metadata
            metadata of features
        name : str
            feature name
        n_bin : int
            numbers of bins
    """

    # find thresholds
    vlist = []
    for d in data:
        if d[name] != '?':
            vlist.append(d[name])
    vlist.sort()

    tlist = []
    for t in xrange(n_bin - 1):
        i = int(len(vlist) * (t + 1) / n_bin)
        tlist.append(vlist[i])
    tlist.append(vlist[-1])

    # update attribute info
    attr[name] = ('nominal', map(lambda x: 'v' + str(x), np.arange(n_bin)))

    # discretize numerical attributes
    for d in data:
        if d[name] != '?':
            for v in xrange(n_bin):
                if d[name] <= tlist[v]:
                    break
            d[name] = 'v' + str(v)

def merge_low_freq_vals(attr, data, meta, name, min_freq):
    """ Merge low frequency attribute values into a *Pool* value. 
    
    :Parameters:
        attr : dict, shape=(2)
            updated attribute information
        data : arylike, shape=(n_samples, n_features)
            data
        meta : scipy.io.arff.Metadata
            metadata of features
        name : str
            feature name
        min_freq : int
            minimum frequency
    """

    # countup data
    count = {}
    count['?'] = 0
    for k in attr[name][1]:
        count[k] = 0

    for d in data:
        count[d[name]] += 1

    # generate conversion table
    conv = {}
    conv['?'] = '?'
    del count['?']

    for k in attr[name][1]:
        if count[k] <= 0:
            pass
        elif count[k] <= min_freq:
            conv[k] = 'Pool'
        else:
            conv[k] = k

    # convert data
    for d in data:
        d[name] = conv[d[name]]

    # generate new attr_list
    new_attr = conv.values()
    new_attr.remove('?')
    for i in xrange(new_attr.count('Pool') - 1):
        new_attr.remove('Pool')

    # update attribute info
    attr[name] = ['nominal', new_attr]

def savearff(out, attr, data, meta):
    """ write arff file
    """

    # write header
    out.write("@relation adultd\n")
    for name in meta.names():
        if name == 'income':
            out.write("@attribute income {'>50K','<=50K'}\n")
        else:
            out.write("@attribute " + name +
                      " {" + ",".join(attr[name][1]) + "}\n")

    # write body
    out.write("@data\n")
    for d in data:
        d[-1] = "'" + d[-1] + "'"
        out.write(",".join(d) + "\n")

#==============================================================================
#{ Main routine
#==============================================================================

def main(opt, arg):
    """ Main routine that exits with status code 0
    """

    # open input file
    if opt.input == None:
        if len(arg) > 0:
            infile = open(arg.pop(0), "r")
        else:
            infile = sys.stdin
    else:
        infile = open(opt.input, "r")

    # open output file
    if opt.output == None:
        if len(arg) > 0:
            outfile = open(arg.pop(0), "w")
        else:
            outfile = sys.stdout
    else:
        outfile = open(opt.output, "w")

    # read arff file
    data, meta = loadarff(infile)
    data = data.astype([(name, np.dtype(object)) for name in meta.names()])

    # process each attribute
    attr = {}
    for name in meta:
        if meta[name][0] == 'numeric':
            discretize_numerical_attribute(attr, data, meta, name, opt.bin)
        else:
            attr[name] = [meta[name][0], list(meta[name][1])]

    # merge low frequent values
    for name in meta:
        merge_low_freq_vals(attr, data, meta, name, opt.minf)

    # write file
    savearff(outfile, attr, data, meta)

    # close file
    if infile != sys.stdin:
        infile.close()

    if outfile != sys.stdout:
        outfile.close()

    sys.exit(0)

#==============================================================================
# Check if this is call as command script
#==============================================================================
if __name__ == '__main__':

    # command-lien option parsing
    parser = optparse.OptionParser(usage="Usage: %prog [options] args...",
                                   description="For details, use pydoc or epydoc.",
                                   version="%prog " + __version__)

#  ----------------------------------------------------------------------------
    # additional command line args

    parser.add_option("-i", "--in", dest="input")
    parser.add_option("-o", "--out", dest="output")
    parser.add_option("-b", "--bin", dest="bin", type="int")
    parser.set_defaults(bin=4)
    parser.add_option("-m", "--minfreq", dest="minf", type="int")
    parser.set_defaults(minf=50)

#  ----------------------------------------------------------------------------

    (opt, arg) = parser.parse_args()

    # call main routine
    main(opt, arg)
