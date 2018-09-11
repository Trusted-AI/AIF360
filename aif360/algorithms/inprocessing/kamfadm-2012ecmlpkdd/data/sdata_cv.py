#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Generate Artificial Data Set for Discrimination/Fairness-aware learning

SYNOPSIS::

    SCRIPT [options]

Description
===========

.. math::

    \Pr[C, L, P, A_1, \cdots, A_f] =
        \Pr[L] \Pr[A] \Pr[C | L, A] \Pr[A_1 | L, S] \cdots \Pr[A_f | L, S]

Features :math:`A_i` are non-sensitive and a feature :math:`S` is sensitive.
A Class :math:`L` is unobserved and non-discriminating. A Class :math:`C` is
observed and discriminating. [DMKD2010]_

Output Format
=============

* the first <FEATURE> columns: non-sensitive features
* the (<FEATURE> + 1)-th column: sensitive features
* the (<FEATURE> + 2)-th column: latent unbiased class
* the (<FEATURE> + 3)-th column: observed biased class

All features and classes are binary.

Options
=======

-o <OUTPUT>, --out <OUTPUT>
    specify <OUTPUT> file name
-f <FEATURE>, --feature <FEATURE>
    the number of features (default 20)
-n <SAMPLE>, --nossample <SAMPLE>
    the number of samples (default 20000)
-l <LBOUND>, --lbound <LBOUND>
    bound for *|P[A|L=1] - P[A|L=0]|* (default 0.8).
-s <SBOUND>, --sbound <SBOUND> 
    bound for *|P[A|S=1] - P[A|S=0]|* (default 0.8).
--arff
    By default, data are written in space-separated text. If this option is
    specified, write in arff format.
-q, --quiet
    set logging level to ERROR, no messages unless errors
--rseed <RSEED>
    random number seed. if None, use /dev/urandom (default None)

:Variables:
    `PC_BASE` : array-like, shape(2, 2), dtype=float
        PC_BASE[L, S] = Pr[C = 1 | L, S]
    `PL_BASE` : float
        a base probability of Pr[L = 1]
    `PS_BASE` : float
        a base probability of Pr[S = 1]

:Variables:
    `script_name` : str
        name of this script
    `info` : {str = any}
        meta information about learning 

.. [DMKD2010] T.Calders and S.Verwer "Three naive Bayes approaches for
    discrimination-free classification" Data Mining and Knowledge Discovery,
    vol.21 (2010)
"""

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2011/02/07"
__version__ = "1.0.2"
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

#==============================================================================
# Public symbols
#==============================================================================

__all__ = []

#==============================================================================
#{ Constants
#==============================================================================

PC_BASE = np.array([[0.1, 0.2], [0.8, 0.9]])
PL_BASE = 0.5
PS_BASE = 0.5

#==============================================================================
#{ Module variables
#==============================================================================

script_name = os.path.basename(sys.argv[0])
info = {}

#==============================================================================
#{ Classes
#==============================================================================

class AData(object):
    """ Artificial Data Generator for Fairness-aware Learning

    :IVariables:
        `f` : int
            the number of features
        `lb` : float
            bound for | Pr[A|L=1] - Pr[A|L=0] |
        `sb` : float
            bound for | Pr[A|S=1] - Pr[A|S=0] |
        `n` : int
            the number of samples
        `pa` : list[ array-like, shape(2, 2), dtype=float ], len=self.f
            list of distributions of A=1 given L and S
        `pc` : array-like, shape(2, 2), dtype=float
            distribution of C=1 given L and S
        `pl` : float
            probability of L=1
        `ps` : float
            probability of S=1
        `s` : array-like, shape(self.n), dryps=int
            instances of S (sensitive feature)
        `l` : array-like, shape(self.n), dtype=int
            instances of L (hidden true class)
        `a` : array-like, shape(self.n, self.f), dtype=int
            instances of A (non-sensitive features)
        `c` : array-like, shape(self.n), dtype=int
            instances of C (observed class)
    """

    def __init__(self, f, lb, sb):
        """ Constructor
        
        :Parameters:
            `f` : int
                the number of features
            `lb` : float
                bound for | Pr[A|L=1] - Pr[A|L=0] |
            `sb` : float
                bound for | Pr[A|S=1] - Pr[A|S=0] |
        """

        self.f = f
        self.lb = lb
        self.sb = sb

        self.pc = PC_BASE
        self.pl = PL_BASE
        self.ps = PS_BASE
        self.pa = []
        for i in xrange(self.f):
            pa = np.empty((2, 2))
            pa[1, 1] = np.random.random()
            pa[1, 0] = np.random.random()
            while np.abs(pa[1, 1] - pa[1, 0]) > self.sb:
                pa[1, 0] = np.random.random()
            pa[0, 1] = np.random.random()
            while np.abs(pa[1, 1] - pa[0, 1]) > self.lb:
                pa[0, 1] = np.random.random()
            pa[0, 0] = np.random.random()
            while (np.abs(pa[1, 0] - pa[0, 0]) > self.lb) \
                or (np.abs(pa[0, 1] - pa[0, 0]) > self.sb) \
                or ((pa[1, 0] - pa[1, 1]) * (pa[0, 0] - pa[0, 1]) < 0) \
                or ((pa[0, 1] - pa[1, 1]) * (pa[0, 0] - pa[1, 0]) < 0):
                pa[0, 0] = np.random.random()
            self.pa.append(pa)

        self.n = 0
        self.s = None
        self.l = None
        self.a = None
        self.c = None

    def generate(self, n):
        """  generate L and S
        
        :Parameters:
            `n` : int
                the number of samples
        """

        self.n = n

        self.s = np.random.binomial(1, self.ps, self.n)
        self.l = np.random.binomial(1, self.pl, self.n)
        self.c = np.random.binomial(1, self.pc[self.l, self.s])

        self.a = np.empty((self.n, self.f))
        for i in xrange(self.f):
            self.a[:, i] = np.random.binomial(1, self.pa[i][self.l, self.s])

    def write_arff(self, f):
        """ write data in ARFF format
        
        :Parameters:
            `f` : file object
                file hundle to write
        """

        f.write("% pl=" + str(self.pl) + "\n")
        f.write("% ps=" + str(self.ps) + "\n")
        f.write("%% pc=[%s, %s]\n" % (str(self.pc[0, :]), str(self.pc[1, :])))
        for i in xrange(self.f):
            f.write("%% pa[%d]=[%s, %s]\n" % \
                    (i, str(self.pa[i][0, :]), str(self.pa[i][1, :])))
        f.write("% nos_sample=" + str(self.n) + "\n")
        f.write("% nos_feature=" + str(self.f) + "\n")
        f.write("% l_bound=" + str(self.lb) + "\n")
        f.write("% s_bound=" + str(self.sb) + "\n")

        # write header
        f.write("\n@relation fldata\n\n")
        for i in xrange(self.f):
            f.write("@attribute a%02d { 0, 1 }\n" % (i + 1))
        f.write("@attribute s   { 0, 1 }\n")
        f.write("@attribute l   { 0, 1 }\n")
        f.write("@attribute c   { 0, 1 }\n")
        f.write("\n@data\n")

        # write body
        for i in xrange(self.n):
            for j in xrange(self.f):
                f.write(str(int(self.a[i, j])) + ", ")
            f.write("%d, %d, %d\n" % (self.s[i], self.l[i], self.c[i]))

    def write_txt(self, f):
        """ write data in space-sparated text
        
        :Parameters:
            `f` : file object
                file hundle to write
        """

        # write body
        for i in xrange(self.n):
            for j in xrange(self.f):
                f.write(str(int(self.a[i, j])) + " ")
            f.write("%d %d %d\n" % (self.s[i], self.l[i], self.c[i]))

        # write parameters
        f.write("# pl=" + str(self.pl) + "\n")
        f.write("# ps=" + str(self.ps) + "\n")
        f.write("# pc=[%s, %s]\n" % (str(self.pc[0, :]), str(self.pc[1, :])))
        for i in xrange(self.f):
            f.write("# pa[%d]=[%s, %s]\n" % \
                    (i, str(self.pa[i][0, :]), str(self.pa[i][1, :])))
        f.write("# nos_sample=" + str(self.n) + "\n")
        f.write("# nos_feature=" + str(self.f) + "\n")
        f.write("# l_bound=" + str(self.lb) + "\n")
        f.write("# s_bound=" + str(self.sb) + "\n")

#==============================================================================
#{ Functions 
#==============================================================================

#==============================================================================
#{ Main routine
#==============================================================================
def main(opt, arg):
    """ Main routine that exits with status code 0
    """

    global info

    # set metadata of script and machine
    info['script'] = script_name
    info['script_version'] = __version__
    info['random_seed'] = opt.rseed
    info['verbose_mode'] = opt.verbose

# Open Files ------------------------------------------------------------------
    # open output file
    if opt.output == None:
        if len(arg) > 0:
            info['output_file'] = arg[0]
            outfile = open(arg.pop(0), "w")
        else:
            info['output_file'] = "<stdout>"
            outfile = sys.stdout
    else:
        info['output_file'] = str(opt.output)
        outfile = open(opt.output, "w")

# Process data ----------------------------------------------------------------

    d = AData(opt.feature, opt.lbound, opt.sbound)
    d.generate(opt.sample)

# Output ----------------------------------------------------------------------
    if opt.arff:
        for key in info.keys():
            outfile.write("%% %s=%s\n" % (key, str(info[key])))
        d.write_arff(outfile)
    else:
        d.write_txt(outfile)
        for key in info.keys():
            outfile.write("# %s=%s\n" % (key, str(info[key])))

# End Process -----------------------------------------------------------------
    if outfile != sys.stdout:
        outfile.close()

    sys.exit(0)

#==============================================================================
# Check if this is call as command script
#==============================================================================
if __name__ == '__main__':

    # command-lien option parsing
    parser = optparse.OptionParser(usage="Usage: %prog [options] args...",
                                   description="use pydoc or epydoc.",
                                   version="%prog " + __version__)
    parser.add_option("--verbose", action="store_true", dest="verbose")
    parser.add_option("-q", "--quiet", action="store_false", dest="verbose")
    parser.set_defaults(verbose=True)

# additional command line args ------------------------------------------------

    parser.add_option("-o", "--out", dest="output")
    parser.add_option("--rseed", dest="rseed", type="int")
    parser.add_option("-f", "--feature", dest="feature", type="int")
    parser.set_defaults(feature=20)
    parser.add_option("-n", "--nossample", dest="sample", type="int")
    parser.set_defaults(sample=20000)
    parser.add_option("-l", "--lbound", dest="lbound", type="float")
    parser.set_defaults(lbound=0.8)
    parser.add_option("-s", "--sbound", dest="sbound", type="float")
    parser.set_defaults(sbound=0.8)
    parser.add_option("--arff", dest="arff", action="store_true")
    parser.set_defaults(arff=False)

#  ----------------------------------------------------------------------------

    (opt, arg) = parser.parse_args()

    # set random seed
    np.random.seed(opt.rseed)

    # call main routine
    main(opt, arg)
