#http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow
# ----------------------------------------------------------
# ██╗   ██╗████████╗██╗██╗     ██╗████████╗██╗   ██╗
# ██║   ██║╚══██╔══╝██║██║     ██║╚══██╔══╝╚██╗ ██╔╝
# ██║   ██║   ██║   ██║██║     ██║   ██║    ╚████╔╝ 
# ██║   ██║   ██║   ██║██║     ██║   ██║     ╚██╔╝  
# ╚██████╔╝   ██║   ██║███████╗██║   ██║      ██║   
#  ╚═════╝    ╚═╝   ╚═╝╚══════╝╚═╝   ╚═╝      ╚═╝   
#           A library of utility methods.
#       
#   Author: Nicholai Mauritzson 2019-...
#           nicholai.mauritzson@nuclear.lu.se
# ----------------------------------------------------------

import pandas as pd
import numpy as np
import sys
from tqdm import tqdm #Library for progress bars
sys.path.insert(0, "/home/gheed/Documents/projects/TNT/code_repo") #Import my own libraries
from pyTagAnalysis import load_data

def printFormatting(title, descriptions, values, errors=None, unit=('Units missing!')):
    """
    Method which prints information to console in a nice way.
    - 'title'..........String containing desired title of the print-out.
    - 'descritpions'...List of strings containing the descriptions of each line to be printed. description=('variable1, varible2, ...).
    - 'values'.........List of variables for each description. value=(val1, val2, ...).
    - 'errors'.........List of errors for each variable (optional). errors=(err1, err2, ...).
    - 'units'..........List of strings containing the unit of each variable. units=(unit1, unit2, ...).
    """
    numEnt = len(descriptions)
    str_len = []
    dots = []

    for i in range(numEnt):
        str_len.append(len(descriptions[i]))

    for i in range(numEnt):
        dots.append(str_len[i]*'.')
    max_dots = len(max(dots, key=len))

    print_dots=[]
    for i in range(numEnt):
        print_dots.append((max_dots-str_len[i]+5)*'.')
        
    print()#Create vertical empty space in terminal
    print('______________________________________________________________________') 
    print('<<<<< %s >>>>>'% title) #Print title
    if errors is not None:
        for i in range(numEnt):
            print('%s%s%.4f (+/-%.4f %s)'%(descriptions[i], print_dots[i], values[i], errors[i], units[i]))
            
    print('______________________________________________________________________')
    print()#Create vertical empty space in terminal
    return 0

def convertCookedData(load_path, save_path):
    """
    Method takes "cooked" root file as input ('load_path'), converts it to pandas data frame and saves it to 'save_path' as hdf5 file.
    
    - 'load_path'....List of paths to each data file (strings). Takes these, converts them and saves them all as one file.
    - 'save_path'....String of path to where converted data shall be saved.
    """
    if type(load_path) == tuple:
        frames = []
        for i in tqdm(range(len(load_path))):
            df_temp = load_data(load_path[i])
            frames.append(df_temp)
        df = pd.concat(frames)
    elif type(load_path) == str:
        df = load_data(load_path)
    else:
        print('ERROR: load_path needs to be string or list of strings')
        return 0

    df.to_hdf(save_path, key="w")

def randomGauss(mean, sigma, numEnt)
    """
    Methods takes mean and sigma as well as number of entries and returns an array of normally distributed values around 'mean'
    
    """
    return np.random.normal(mean, sigma, numEnt)