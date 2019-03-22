import sys
sys.path.insert(0, "/home/gheed/Documents/projects/TNT/code_repo") #Import my own libraries
sys.path.insert(1, "/home/gheed/Documents/projects/TNT/code_repo/analog") #Import my own libraries
from pyTagAnalysis import load_data
import nicholai_math as nm
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Analysis script for analog data from aquadaq
# By Nicholai Mauritzson 
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Load and save data as HDF5 file.
df = load_data("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1811_cooked.root")
df.to_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1811_DF.hdf5", "a")
# ------------------------------------------------------------------------------