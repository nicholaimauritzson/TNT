# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Quick plotting script for analog aquadaq.
# Argument needs to be hdf5 file (pandas)
#
# By Nicholai Mauritzson 
# ------------------------------------------------------------------------------

import sys
sys.path.insert(0, "/home/gheed/Documents/projects/TNT/code_repo","r") #Import my own libraries
from pyTagAnalysis import load_data
import nicholai_math as nm
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_hdf(path=sys.argv[0])

bins=8*1024
figure(0)
plt.hist()
ax.set_yscale("log")


