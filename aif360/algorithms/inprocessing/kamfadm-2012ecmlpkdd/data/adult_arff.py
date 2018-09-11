#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert *adult.data* or *adult.test* to ARFF format

adult data set (a.k.a. census income data set)
http://archive.ics.uci.edu/ml/datasets/Adult

SYNOPSIS::

    SCRIPT [options]

Options
=======

-i <INPUT>, --in <INPUT>
    specify <INPUT> file name
-o <OUTPUT>, --out <OUTPUT>
    specify <OUTPUT> file name

:Variables:
    `script_name` : str
        name of this script
"""

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2011/02/04"
__version__ = "1.0.0"
__copyright__ = "Copyright (c) 2011 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License http://www.opensource.org/licenses/mit-license.php"
__docformat__ = "restructuredtext en"

#==============================================================================
# Imports
#==============================================================================

import sys
import os
import optparse

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

def write_header(outfile):
    """ Write Header of ARFF file
    
    :Parameters:
        `outfile` : file
            file descriptor to write
    """

    outfile.write("""% Adult Data Set / Census Income Data Set
% http://archive.ics.uci.edu/ml/datasets/Adult
% http://archive.ics.uci.edu/ml/datasets/Census+Income

@relation adult

@attribute age integer
@attribute workclass {Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked}
@attribute fnlwgt integer
@attribute education {Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool}
@attribute education-num integer
@attribute marital-status {Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse}
@attribute occupation {Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces}
@attribute relationship {Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried}
@attribute race {White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black}
@attribute sex {Female, Male}
@attribute capital-gain integer
@attribute capital-loss integer
@attribute hours-per-week integer
@attribute native-country {United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands}
@attribute income {>50K, <=50K}

@data
""")

    return 0

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

    # write attributes
    write_header(outfile)

    # read from file
    for line in infile.readlines():
        line = line.rstrip('\r\n')

        # skip empty line and comment line
        if (line == "") or (line[0] == "|"):
            continue

        # remove tail period
        line = line.rstrip('.')

        # write output
        outfile.write(line + "\n")

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

#  ----------------------------------------------------------------------------

    (opt, arg) = parser.parse_args()

    # call main routine
    main(opt, arg)

