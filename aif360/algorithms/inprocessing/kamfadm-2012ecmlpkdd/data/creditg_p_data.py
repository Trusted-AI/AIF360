#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert credit-g.data => creditg_p.data
"""

import sys

for line in sys.stdin.readlines():
    line = line.rstrip('\r\n')

    f = line.split(" ")
    sys.stdout.write(" ".join(f[0:8]) + " ")
    sys.stdout.write(" ".join(f[9:20]) + " ")
    if f[8] == "1":
        sys.stdout.write("0 ")
    else:
        sys.stdout.write("1 ")
    sys.stdout.write(str(1 - int(f[20])) + "\n")

