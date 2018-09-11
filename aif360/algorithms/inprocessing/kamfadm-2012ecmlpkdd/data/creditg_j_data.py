#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert credit-g.data => creditg_p.data
"""

import sys

for line in sys.stdin.readlines():
    line = line.rstrip('\r\n')

    f = line.split(" ")
    sys.stdout.write(" ".join(f[0:16]) + " ")
    sys.stdout.write(" ".join(f[17:20]) + " ")
    if f[16] == "0":
        sys.stdout.write("0 ")
    else:
        sys.stdout.write("1 ")
    sys.stdout.write(str(1 - int(f[20])) + "\n")

