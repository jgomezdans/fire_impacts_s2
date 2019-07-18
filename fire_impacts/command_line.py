#!/usr/bin/env python
"""A command line tool to run the fcc/fire impacts code from the command line.
Main idea is to produce some output."""

# fire_impacts_s2 A fcc model calculator for S2 & L8
# Copyright (c) 2019 J Gomez-Dans. All rights reserved.
#
# This file is part of fire_impacts.
#
# fire_impacts is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# fire_impacts is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with fire_impacts.  If not, see <http://www.gnu.org/licenses/>.


__author__ = "J Gomez-Dans"
__email__ = "j.gomez-dans@ucl.ac.uk"

import logging
import argparse
import os 
import sys 

import datetime as dt

from pathlib import Path

from .fcc_calculation import FireImpacts, Observations

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(module)s." +
                    "%(funcName)s - " +
                    "- %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def get_args():
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description="""fcc processor for Sentinel 2 & Landsat 8
                    By J Gomez-Dans""")
    # Add arguments
    parser.add_argument("pre", help="Pre-fire image", action="store", type=str)
    parser.add_argument("post", help="Post-fire image", action="store",
                        type=str)
    parser.add_argument("-o", "--output", help="Output folder", type=str,
                        default=".")
    parser.add_argument("-q", "--quantise", help="Quantise output to Byte",
    type=bool, default=False)
    parser.add_argument("--a0",
                        help="User-supplied value of a0 for entire scene",
                        type=float, default=None)
    parser.add_argument("--a1",
                        help="User-supplied value for a1 for entire scene",
                        type=float, default=None)
    args = parser.parse_args()
    print(args)
    return args

def main():
    t0 = dt.datetime.now()
    argz = get_args()
    if not Path(argz.pre).exists():
        raise IOError(f"Pre-fire image ({argz.pre:s}) not present!")
    if not Path(argz.post).exists():
        raise IOError(f"Post-fire image ({argz.post:s}) not present!")


    f=FireImpacts(Observations(argz.pre, argz.post), output_dir=argz.output,
                 quantise=argz.quantise, user_a0=argz.a0, user_a1=argz.a1)
    f.launch_processor()
    t1 = dt.datetime.now()
