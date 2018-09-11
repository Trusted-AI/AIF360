#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Move the specified features to the last position

SYNOPSIS::

    SCRIPT [options]

Options
=======

-i <INPUT>, --in <INPUT>
    specify <INPUT> file name
-o <OUTPUT>, --out <OUTPUT>
    specify <OUTPUT> file name
-f <FEATURE>, --feature <FEATURE>
    the feature number to move, starting from 0 (default 0)
-d <DL>, --delimiter <DL>
    column delimiter string (default " ")
-g <IGNORE>, --ignore <IGNORE>
    ignore line if the line start with char included in this string
    (default "#")
-n, --negate
    negate a class
-r, --reverse
    negate the selected sensitive feature

:Variables:
    `script_name` : str
        name of this script
"""

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2011/05/07"
__version__ = "1.2.1"
__copyright__ = "Copyright (c) 2011 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"
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

#==============================================================================
#{ Main routine
#==============================================================================
def main(opt, arg):
    """ Main routine that exits with status code 0
    """

# Open Files ------------------------------------------------------------------

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

# Process ---------------------------------------------------------------------

    # read from file
    line_no = 0
    for line in infile.readlines():
        line = line.rstrip('\r\n')
        line_no += 1

        try:
            # skip empty line and comment line
            if line == "" or opt.ignore.find(line[0]) >= 0:
                outfile.write(line + "\n")
                continue

            # split into columns
            f = line.split(opt.dl)

            # re-order features
            c = f[-1]
            if opt.negate:
                c = '0' if c == '1' else '1'
            s = f[opt.feature]
            if opt.reverse:
                s = '0' if s == '1' else '1'
            out = list(f[0:opt.feature]) + list(f[opt.feature + 1:-1]) + \
                list(s) + list(c)
            outfile.write(opt.dl.join(out) + "\n")
        except IndexError:
            sys.exit("Parse error in line %d" % (line_no))
        except IOError:
            break

# Output ----------------------------------------------------------------------

# End Process -----------------------------------------------------------------

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
                                   description="use pydoc or epydoc.",
                                   version="%prog " + __version__)

# additional command line arguments -------------------------------------------

    parser.add_option("-i", "--in", dest="input")
    parser.add_option("-o", "--out", dest="output")
    parser.add_option("-f", "--feature", dest="feature", type="int")
    parser.set_defaults(feature=0)
    parser.add_option("-d", "--delimiter", dest="dl")
    parser.set_defaults(dl=" ")
    parser.add_option("-g", "--ignore", dest="ignore")
    parser.set_defaults(ignore="#")
    parser.add_option("-n", "--negate", dest="negate", action="store_true")
    parser.set_defaults(negate=False)
    parser.add_option("-r", "--reverse", dest="reverse", action="store_true")
    parser.set_defaults(reverse=False)

#  ----------------------------------------------------------------------------

    (opt, arg) = parser.parse_args()

    # call main routine
    main(opt, arg)
