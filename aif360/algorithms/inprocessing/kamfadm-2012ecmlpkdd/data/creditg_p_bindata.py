#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert credit-g.data => creditg_p.data
"""

import sys

def write_data_with_binary(a_list, d_list):

    # for each attribute
    for a, d in zip(a_list, d_list):

        # numeric attribute?
        if a <= 2:
            sys.stdout.write(d)
        else:
            v = ['0'] * a
            v[int(d)] = '1'
            sys.stdout.write(" ".join(v))
        sys.stdout.write(" ")

nfv = [4, 0, 5, 11, 0, 5, 5, 0, 5, 3, 0, 4, 0, 3, 3, 0, 4, 0, 2, 2]


for line in sys.stdin.readlines():
    line = line.rstrip('\r\n')

    f = line.split(" ")
    write_data_with_binary(nfv[0:8], f[0:8])
    write_data_with_binary(nfv[9:20], f[9:20])
    if f[8] == "1":
        sys.stdout.write("0 ")
    else:
        sys.stdout.write("1 ")
    sys.stdout.write(str(1 - int(f[20])) + "\n")
