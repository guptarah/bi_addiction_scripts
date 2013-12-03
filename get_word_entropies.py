#! /usr/bin/python

import sys
import math
from os import system
import numpy

count_file=sys.argv[1]

counts=numpy.genfromtxt(count_file,dtype=<type 'float'>)
cumulative_counts=numpy.sum(counts, axis=0)
temp_division_matrix=numpy.tile(cumulative_counts,(counts.shape[0],1))


