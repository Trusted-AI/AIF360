# Copyright (c) 2017 Niels Bantilan
# This software is the same as the original software licensed
# under the MIT License.
#
# https://github.com/cosmicBboy/themis-ml/blob/master/LICENSE
#
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2023 Fujitsu Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for doing checks."""


def check_binary(x):
    """
    Binary check.

    Parameters
    ----------
    x : numpy.ndarray
        Target

    Returns
    -------
    x : numpy.ndarray
        ValueError if not binary.
    """
    if not is_binary(x):
        raise ValueError("%s must be a binary variable" % x)
    return x


def is_binary(x):
    """
    Check if numpy multidimensional array consists of {0,1}

    Parameters
    ----------
    x : numpy.ndarray
        Target

    Returns
    -------
    result : boolean
        Check result
    """
    return set(x.ravel()).issubset({0, 1})
