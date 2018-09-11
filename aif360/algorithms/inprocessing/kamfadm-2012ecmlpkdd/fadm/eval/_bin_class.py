#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
import from 50b745c1d18d5c4b01d9d00e406b5fdaab3515ea @ KamLearn

Compute various statistics between estimated and correct classes in binary
cases
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

#==============================================================================
# Module metadata variables
#==============================================================================

#==============================================================================
# Imports
#==============================================================================

import logging
import numpy as np

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['BinClassStats']

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

class BinClassStats(object):
    """ Compute various statistics of 2class sample data

    Parameters
    ----------
    tp : float
        The number of True-Positives = n[1, 1]
    fn : float
        The number of False-Negatives = n[1, 0]
    fp : float
        The number of False-Positives = n[0, 1]
    tn : float
        The number of True-Negatives = n[0, 0]

    Attributes
    ----------
    n : array-like, shape=(2, 2), dtype=float
        Contingency table of the correct and estimated samples. Rows and
        columns correspond to the correct and the estimated samples.
    c : array-like, shape(2, 0), dtype=float
        Marginal counts of the correct(=true) samples
    e : array-like, shape(2, 0), dtype=float
        Marginal counts of the estimated samples
    t : float
        The number of total samples
    """

    def __init__(self, tp, fn, fp, tn):

        self.n = np.empty((2, 2))
        self.n[1, 1] = float(tp)
        self.n[1, 0] = float(fn)
        self.n[0, 1] = float(fp)
        self.n[0, 0] = float(tn)

        self.c = np.sum(self.n, axis=1)
        self.e = np.sum(self.n, axis=0)
        self.t = np.sum(self.n)

        if self.t <= 0.0 or np.any(self.n < 0.0) \
            or np.any(np.isinf(self.n)) or np.any(np.isnan(self.n)):
            raise ValueError("Illegal values are specified")

    def negate(self):
        """ negate the meanings of positives and negatives
        """

        self.n[1, 1], self.n[0, 0] = self.n[0, 0], self.n[1, 1]
        self.n[1, 0], self.n[0, 1] = self.n[0, 1], self.n[1, 0]

        self.c = np.sum(self.n, axis=1)
        self.e = np.sum(self.n, axis=0)
        self.t = np.sum(self.n)

    def ct(self):
        """ Counts of contingency table elements

        Returns
        -------
        tp : float
            n[1, 1], the number of true positive samples
        fn : float
            n[1, 0], the number of false negative samples
        fp : float
            n[0, 1], the number of false positive samples
        tn : float
            n[0, 0], the number of true negative samples
        """

        return self.n[1, 1], self.n[1, 0], self.n[0, 1], self.n[0, 0]

    def str_ct(self, header=True):
        """ Strings for ct()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        tp, fn, fp, tn = self.ct()

        pr = []
        if header:
            pr.append("### Contingency Table ###")
        pr.append("[ TP(1,1), FN(1,0) ] = [ %6.15g, %6.15g ]" % (tp, fn))
        pr.append("[ FP(0,1), TN(0,0) ] = [ %6.15g, %6.15g ]" % (fp, tn))

        return pr

    def mct(self):
        """ Marginal counts of contingency table elements

        Returns
        -------
        cp : float
            sum of correct positives
        cn : float
            sum of correct negatives
        ep : float
            sum of estimated positives
        en : float
            sum of estimated negatives
        tc : float
            total count
        """

        return self.c[1], self.c[0], self.e[1], self.e[0], self.t

    def str_mct(self, header=True):
        """ Strings for mct()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        cp, cn, ep, en, t = self.mct()

        pr = []
        if header:
            pr.append("### Marginal/Total Counts ###")
        pr.append("True [ P, N ] = [ %6.15g, %6.15g ]" % (cp, cn))
        pr.append("Est  [ P, N ] = [ %6.15g, %6.15g ]" % (ep, en))
        pr.append("Total       = %.15g" % (t))

        return pr

    def acc(self):
        """ Accuracy

        Returns
        -------
        acc : float
            accuracy
        sd : float
            s.d. of accuracy
        """

        acc = (self.n[1, 1] + self.n[0, 0]) / self.t
        sd = np.sqrt(acc * (1.0 - acc) / self.t)

        return acc, sd

    def str_acc(self, header=True):
        """ Strings for acc()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        acc, sd = self.acc()

        pr = []
        if header:
            pr.append("### Accuracy ###")
        pr.append("Acc / S.D. = [ %.15g, %.15g ]" % (acc, sd))

        return pr

    def jaccard(self):
        """ Jaccard / Dice coefficients

        Returns
        -------
        jaccard : float
            Jaccard coefficient
        njaccard : float
            Negated Jaccard coefficient
        dice : float
            Dice coefficient
        ndice : float
            Negated Dice coefficient
        """

        jaccard = self.n[1, 1] / (self.t - self.n[0, 0])
        njaccard = self.n[0, 0] / (self.t - self.n[1, 1])
        dice = 2.0 * self.n[1, 1] / (self.c[1] + self.e[1])
        ndice = 2.0 * self.n[0, 0] / (self.c[0] + self.e[0])

        return jaccard, njaccard, dice, ndice

    def str_jaccard(self, header=True):
        """ Strings for jaccard()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        jaccard, njaccard, dice, ndice = self.jaccard()

        pr = []
        if header:
            pr.append("### Jaccard / Dice Coefficients ###")
        pr.append("Jaccard [ P, N ] = [ %.15g, %.15g ]" % (jaccard, njaccard))
        pr.append("Dice    [ P, N ] = [ %.15g, %.15g ]" % (dice, ndice))

        return pr

    def kldiv(self):
        """ KL divergence

        Returns
        -------
        kldivc : float
            D( Correct || Estimated ) with natural log.
            KL divergence from correct to estimated.
        kldive : float
            D( Estimated || Correct ) with natural log.
            KL divergence from estimated to correct.
        kldivc2 : float
            D( Correct || Estimated ) with log2.
            KL divergence from correct to estimated.
        kldive2 : float
            D( Estimated || Correct ) with log2.
            KL divergence from estimated to correct.
        """

        i = lambda n, m: 0.0 if n == 0.0 else \
            np.inf if m == 0.0 else n * np.log(n / m)

        kldivc = (i(self.c[0], self.e[0]) + i(self.c[1], self.e[1])) \
            / self.t
        kldive = (i(self.e[0], self.c[0]) + i(self.e[1], self.c[1])) \
            / self.t

        i2 = lambda n, m: 0.0 if n == 0.0 else \
            np.inf if m == 0.0 else n * np.log2(n / m)

        kldivc2 = (i2(self.c[0], self.e[0]) + i2(self.c[1], self.e[1])) \
            / self.t
        kldive2 = (i2(self.e[0], self.c[0]) + i2(self.e[1], self.c[1])) \
            / self.t

        return kldivc, kldive, kldivc2, kldive2

    def str_kldiv(self, header=True):
        """ Strings for kldiv()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        kldivc, kldive, kldivc2, kldive2 = self.kldiv()

        pr = []
        if header:
            pr.append("### KL Divergence ###")
        pr.append("[ D(C||E), D(E||C) ] with ln   = [ %.15g, %.15g ]"
                  % (kldivc, kldive))
        pr.append("[ D(C||E), D(E||C) ] with log2 = [ %.15g, %.15g ]"
                  % (kldivc2, kldive2))

        return pr

    def mi(self):
        """ Mutual Information with natural log

        Returns
        -------
        mi : float
            I(C; E) = H(C) + H(E).- H(C, E) mutual information
        nmic : float
            I(C; E) / H(C). MI normalized by H(C)
        nmie : float
            I(C; E) / H(E). MI normalized by H(E)
        amean : float
            Arithmetic mean of two normalized mutual informations.
        gmean : float
            Geometric mean of two normalized mutual informations.
        """

        # joint entropy of the pmf function n / sum(n)
        en = lambda n: np.sum([0.0 if i == 0.0
                               else (-i / self.t) * np.log(i / self.t)
                               for i in np.ravel(n)])

        hc = en(self.c)
        he = en(self.e)
        hj = en(self.n)

        mi = np.max((0.0, hc + he - hj))
        nmic = 1.0 if hc == 0.0 else mi / hc
        nmie = 1.0 if he == 0.0 else mi / he

        return mi, nmic, nmie, (nmic + nmie) / 2.0, np.sqrt(nmic * nmie)

    def str_mi(self, header=True):
        """ Strings for mi()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        mi, nmic, nmie, amean, gmean = self.mi()

        pr = []
        if header:
            pr.append("### Mutual Information (natual log) ###")
        pr.append("I(C;E) = %.15g" % (mi))
        pr.append("[ I(C;E)/H(C), I(C;E)/H(E) ] = [ %.15g, %.15g ]" % \
                  (nmic, nmie))
        pr.append("Arithmetic Mean = %.15g" % (amean))
        pr.append("Geometric Mean = %.15g" % (gmean))

        return pr

    def mi2(self):
        """ Mutual Information with log2

        Returns
        -------
        mi : float
            I(C; E) = H(C) + H(E).- H(C, E) mutual information
        nmic : float
            I(C; E) / H(C). MI normalized by H(C)
        nmie : float
            I(C; E) / H(E). MI normalized by H(E)
        amean : float
            Arithmetic mean of two normalized mutual informations.
        gmean : float
            Geometric mean of two normalized mutual informations.
        """

        # joint entropy of the pmf function n / sum(n)
        en = lambda n: np.sum([0.0 if i == 0.0
                               else (-i / self.t) * np.log2(i / self.t)
                               for i in np.ravel(n)])

        hc = en(self.c)
        he = en(self.e)
        hj = en(self.n)

        mi = np.max((0.0, hc + he - hj))
        nmic = 1.0 if hc == 0.0 else mi / hc
        nmie = 1.0 if he == 0.0 else mi / he

        return mi, nmic, nmie, (nmic + nmie) / 2.0, np.sqrt(nmic * nmie)

    def str_mi2(self, header=True):
        """ Strings for mi2()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        mi, nmic, nmie, amean, gmean = self.mi2()

        pr = []
        if header:
            pr.append("### Mutual Information (log2) ###")
        pr.append("I(C;E) = %.15g" % (mi))
        pr.append("[ I(C;E)/H(C), I(C;E)/H(E) ] = [ %.15g, %.15g ]" % \
                  (nmic, nmie))
        pr.append("Arithmetic Mean = %.15g" % (amean))
        pr.append("Geometric Mean = %.15g" % (gmean))

        return pr

    def prf(self, alpha=0.5):
        """ Precision, recall, and F-measure

        Parameters
        ----------
        alpha : float, default=0.5
            weight of precision in calculation of F-measures

        Returns
        p : float
            Precision for a positive class
        r : float
            Recall for a positive class
        f : float
            F-measure for a positive class
        """

        p = self.n[1, 1] / (self.n[1, 1] + self.n[0, 1])
        r = self.n[1, 1] / (self.n[1, 1] + self.n[1, 0])
        f = 1.0 / (alpha * (1.0 / p) + (1.0 - alpha) * (1.0 / r))

        return p, r, f

    def str_prf(self, alpha=0.5, header=True):
        """ Strings for prf()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        p, r, f = self.prf()

        pr = []
        if header:
            pr.append("### Precision, Recall, and F-measure ###")
        pr.append("Precision = %.15g" % (p))
        pr.append("Recall = %.15g" % (r))
        pr.append("F-measure = %.15g" % (f))

        return pr

    def all(self):
        """ all above statistics

        Returns
        -------
        stats : float
            list of all statistics
        """

        stats = []

        stats += self.ct()
        stats += self.mct()
        stats += self.acc()
        stats += self.jaccard()
        stats += self.kldiv()
        stats += self.mi()
        stats += self.mi2()
        stats += self.prf()

        return tuple(stats)

    def str_all(self, header=True):
        """ Strings for all()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        ret_str = ""

        ret_str += "\n".join(self.str_ct(header)) + "\n\n"
        ret_str += "\n".join(self.str_mct(header)) + "\n\n"
        ret_str += "\n".join(self.str_acc(header)) + "\n\n"
        ret_str += "\n".join(self.str_jaccard(header)) + "\n\n"
        ret_str += "\n".join(self.str_kldiv(header)) + "\n\n"
        ret_str += "\n".join(self.str_mi(header)) + "\n\n"
        ret_str += "\n".join(self.str_mi2(header)) + "\n\n"
        ret_str += "\n".join(self.str_prf(header))

        return ret_str

#==============================================================================
# Functions
#==============================================================================

#==============================================================================
# Module initialization
#==============================================================================

# init logging system ---------------------------------------------------------

logger = logging.getLogger('fadm')
if not logger.handlers:
    logger.addHandler(logging.NullHandler)

#==============================================================================
# Test routine
#==============================================================================

def _test():
    """ test function for this module
    """

    # perform doctest
    import sys
    import doctest

    doctest.testmod()

    sys.exit(0)

# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
