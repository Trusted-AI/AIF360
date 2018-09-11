#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute various types of fairness-aware indexes.
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

# private modules -------------------------------------------------------------

import site
site.addsitedir('.')

from ._bin_class import BinClassStats

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['BinClassBinSensitiveStats']

#==============================================================================
#{ Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

class BinClassBinSensitiveStats(BinClassStats):
    """ Calculate fairness-aware indexes where.

    Parameters
    ----------
    m : array-like, shape=(2, 2, 2), dtype=float
        Contingency table with sensitive attribute.

    Attributes
    ----------
    m : array-like, shape=(2, 2, 2), dtype=float
        Contingency table.
        The 1st argument is a value of a sensitive attribute.
        The 2nd argument is a correct/true class.
        The 3rd argument is a estimated class.
    s : array-like, shape(2,2), type=float
        Marginal by the sensitive attribute.
        Equivalent to "np.sum(s, axis=0).

    Raises
    ------
    ValueError
        parameters are out of ranges
    """

    def __init__(self, m):
        if np.any(m < 0.0) or np.any(np.isinf(m)) or np.any(np.isnan(m)):
            raise ValueError("Illegal values are specified")

        self.m = m
        self.s = np.sum(self.m, axis=0)

        super(BinClassBinSensitiveStats, self)\
            .__init__(self.s[1, 1], self.s[1, 0], self.s[0, 1], self.s[0, 0])

    def negate(self):
        """ Negate the meanings of positive and negative classes.
        """

        super(BinClassBinSensitiveStats, self).negate()

        for s in [0, 1]:
            self.m[s, 1, 1], self.m[s, 0, 0] = self.m[s, 0, 0], self.m[s, 1, 1]
            self.m[s, 1, 0], self.m[s, 0, 1] = self.m[s, 0, 1], self.m[s, 1, 0]
        self.s = np.sum(self.m, axis=0)

    def sct(self):
        """ Contingency table with sensitive attribute.

        Returns
        -------
        m(0)...m(7) : dtype=float
            Contingency table with sensitive attribute.
        """

        return self.m[0, 1, 1], self.m[0, 1, 0], \
               self.m[0, 0, 1], self.m[0, 0, 0], \
               self.m[1, 1, 1], self.m[1, 1, 0], \
               self.m[1, 0, 1], self.m[1, 0, 0]

    def str_sct(self, header=True):
        """ Strings for sct()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        m = self.sct()

        pr = []
        if header:
            pr.append("### Contingency Table with Sensitive Attribute ###")
        pr.append("S=0 [ TP(1,1), FN(1,0) ] = [ %6.15g, %6.15g ]" % \
                  (m[0], m[1]))
        pr.append("    [ FP(0,1), TN(0,0) ] = [ %6.15g, %6.15g ]" % \
                  (m[2], m[3]))
        pr.append("S=1 [ TP(1,1), FN(1,0) ] = [ %6.15g, %6.15g ]" % \
                  (m[4], m[5]))
        pr.append("    [ FP(0,1), TN(0,0) ] = [ %6.15g, %6.15g ]" % \
                  (m[6], m[7]))

        return pr

    def kldiv(self):
        """ KL divergence

        Returns
        -------
        kldivc : float
            D( Correct || Estimated ) with natural log.
            KL divergence from correct distribution to estimated distribution
        kldive : float
            D( Estimated || Correct ) with natural log.
            KL divergence from estimated distribution to correct distribution
        kldivc2 : float
            D( Correct || Estimated ) with log2.
            KL divergence from correct distribution to estimated distribution
        kldive2 : float
            D( Estimated || Correct ) with log2.
            KL divergence from estimated distribution to correct distribution
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

    def mics(self):
        """ Mutual Information between correct classes and sensitive
        attributes. (natural log)

        Returns
        -------
        mi : float
            I(C; S) = H(C) + H(S) - H(C, S), mutual information
        nmic : float
            I(C; S) / H(C), normalized by H(C).
        nmis : float
            I(C; S) / H(S), normalized by H(S).
        amean : float
            Arithmetic mean of two types of normalized mutual information
        gmean : float
            Geometric means of two types of normalized mutual information
        """

        # joint entropy of the pmf function n / sum(n)
        en = lambda n: np.sum([0.0 if i == 0.0
                               else (-i / self.t) * np.log(i / self.t)
                               for i in np.ravel(n)])

        j = np.sum(self.m, axis=2)
        hj = en(j)
        hc = en(self.c)
        hs = en(np.sum(j, axis=1))

        mi = np.max((0.0, hc + hs - hj))
        nmic = 1.0 if hc == 0.0 else mi / hc
        nmis = 1.0 if hs == 0.0 else mi / hs

        return mi, nmic, nmis, (nmic + nmis) / 2.0, np.sqrt(nmic * nmis)

    def str_mics(self, header=True):
        """ Strings for mics()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        mi, nmic, nmis, amean, gmean = self.mics()

        pr = []
        if header:
            pr.append("### Mutual Information (Correct, Sensitive) ###")
        pr.append("I(C;S) = %.15g" % (mi))
        pr.append("[ I(C;S)/H(C), I(C;ES)/H(S) ] = [ %.15g, %.15g ]" % \
                  (nmic, nmis))
        pr.append("Arithmetic Mean = %.15g" % (amean))
        pr.append("Geometric Mean = %.15g" % (gmean))

        return pr

    def mies(self):
        """ Mutual Information between estimated classes and sensitive
        attributes. (natural log)

        Returns
        -------
        mi : float
            I(C; S) = H(C) + H(S) - H(C, S). mutual information
        nmie : float
            I(C; S) / H(C), normalized by H(C)
        nmis : float
            I(C; S) / H(S), normalized by H(S)
        amean : float
            Arithmetic mean of two types of normalized mutual information
        gmean : float
            Geometric means of two types of normalized mutual information
        """

        # joint entropy of the pmf function n / sum(n)
        en = lambda n: np.sum([0.0 if i == 0.0
                               else (-i / self.t) * np.log(i / self.t)
                               for i in np.ravel(n)])

        j = np.sum(self.m, axis=1)
        hj = en(j)
        he = en(self.e)
        hs = en(np.sum(j, axis=1))

        mi = np.max((0.0, he + hs - hj))
        nmie = 1.0 if he == 0.0 else mi / he
        nmis = 1.0 if hs == 0.0 else mi / hs

        return mi, nmie, nmis, (nmie + nmis) / 2.0, np.sqrt(nmie * nmis)

    def str_mies(self, header=True):
        """ Strings for mies()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        mi, nmie, nmis, amean, gmean = self.mies()

        pr = []
        if header:
            pr.append("### Mutual Information (Estimated, Sensitive) ###")
        pr.append("I(C;S) = %.15g" % (mi))
        pr.append("[ I(C;S)/H(C), I(C;ES)/H(S) ] = [ %.15g, %.15g ]" % \
                  (nmie, nmis))
        pr.append("Arithmetic Mean = %.15g" % (amean))
        pr.append("Geometric Mean = %.15g" % (gmean))

        return pr

    def klgivens(self):
        """ KL-divergence between correct and estimated conditional
        distributions given sensitive attributes (natural log)

        Returns
        -------
        kldivc : float
            D(C|S || E|S). KL-divergence from correct to estimated.
        kldive : float
            D(E|S || C|S). KL-divergence from estimated to correct.
        """

        c = np.sum(self.m, axis=2)
        e = np.sum(self.m, axis=1)

        # KL-divergence from correct to estimated
        if np.any(e <= 0.0):
            kldivc = np.inf
        else:
            kldivc = 0.0
            for i, j in zip(np.ravel(c), np.ravel(e)):
                if i > 0:
                    kldivc += i * (np.log(i) - np.log(j))

        # KL-divergence from estimated to correct
        if np.any(c <= 0.0):
            kldive = np.inf
        else:
            kldive = 0.0
            for i, j in zip(np.ravel(e), np.ravel(c)):
                if i > 0:
                    kldive += i * (np.log(i) - np.log(j))

        return kldivc / self.t, kldive / self.t

    def str_klgivens(self, header=True):
        """ Strings for klgivens()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        kldivc, kldive = self.klgivens()

        pr = []
        if header:
            pr.append("### KL-Divergence Given Sensitive ###")
        pr.append("D(C|S || E|S) = %.15g" % (kldivc))
        pr.append("D(E|S || C|S) = %.15g" % (kldive))

        return pr

    def hdjoints(self):
        """ Hellinger distance between correct and estimated distributions
        jointed with distributions given sensitive attributes
        (natural log)

        Returns
        -------
        hdjoints : float
            d_H(P(C,S), P(E,S)). Hellinger distance between correct and
            estimated
        nhdjoints : float
            d_H(P(C,S), P(E,S)) / Sqrt(2). normalized to [0,1]
        """

        c = np.sum(self.m, axis=2)
        e = np.sum(self.m, axis=1)

        # Bhattacharyya coefficient
        bc = 0.0
        for i, j in zip(np.ravel(c), np.ravel(e)):
            bc += np.sqrt(i * j)

        # compute Hellinger disatance
        hd = 1.0 - bc / self.t

        return np.sqrt(2.0 * hd), np.sqrt(hd)

    def str_hdjoints(self, header=True):
        """ Strings for hdjoints()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        hdjoints, nhdjoints = self.hdjoints()

        pr = []
        if header:
            pr.append("### Hellinger distance jointed with S ###")
        pr.append("d_H(P(C,S), P(E,S)) = %.15g" % (hdjoints))
        pr.append("Normalized = %.15g" % (nhdjoints))

        return pr

    def cvs(self):
        """ Caldars-Verwer score.

        Returns
        -------
        cvsc : float
            Pr[C=1 | S=1] - Pr[C=1 | S=0], CV score on correct class
        cvse : float
            Pr[E=1 | S=1] - Pr[E=1 | S=0], CV score on estimated class
        """

        c = np.sum(self.m, axis=2)
        e = np.sum(self.m, axis=1)

        cvsc = c[1, 1] / (c[1, 0] + c[1, 1]) - c[0, 1] / (c[0, 0] + c[0, 1])
        cvse = e[1, 1] / (e[1, 0] + e[1, 1]) - e[0, 1] / (e[0, 0] + e[0, 1])

        return cvsc, cvse

    def str_cvs(self, header=True):
        """ Strings for cvs()

        Parameters
        ----------
        header : boolean, default=True
            include header info

        Returns
        -------
        pr : list, type=str
            list of message strings
        """

        cvsc, cvse = self.cvs()

        pr = []
        if header:
            pr.append("### Caldars-Verwer score ###")
        pr.append("Pr[C=1 | S=1] - Pr[C=1 | S=0] = %.15g" % (cvsc))
        pr.append("Pr[E=1 | S=1] - Pr[E=1 | S=0] = %.15g" % (cvse))

        return pr

    def all(self):
        """ all above statistics

        Returns
        -------
        stats : float, shape=(n_stats)
            list of all statistics
        """

        stats = super(BinClassBinSensitiveStats, self).all()

        stats += self.sct()
        stats += self.mics()
        stats += self.mies()
        stats += self.klgivens()
        stats += self.hdjoints()
        stats += self.cvs()

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
        ret_str += "\n".join(self.str_mi(header)) + "\n\n"

#        ret_str = super(FairnessAwareIndexBinClassBinSensitive, self).\
#            str_all(header)
        ret_str += "\n".join(self.str_sct(header)) + "\n\n"
        ret_str += "\n".join(self.str_mics(header)) + "\n\n"
        ret_str += "\n".join(self.str_mies(header)) + "\n\n"
        ret_str += "\n".join(self.str_klgivens(header)) + "\n\n"
        ret_str += "\n".join(self.str_hdjoints(header)) + "\n\n"
        ret_str += "\n".join(self.str_cvs(header))

        return ret_str

#==============================================================================
# Functions
#==============================================================================

#==============================================================================
# Module initialization
#==============================================================================

# init logging system

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

# Check if this is call as command script

if __name__ == '__main__':
    _test()
