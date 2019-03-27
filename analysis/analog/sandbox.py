import sys
sys.path.insert(0, "/home/gheed/Documents/projects/TNT/code_repo") #Import my own libraries
from pyTagAnalysis import load_data
import nicholai_math as nm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import cos, pi, radians
from scipy import signal
from tqdm import tqdm
import pyTagAnalysis as pta
from scipy.optimize import curve_fit

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Analysis script for analog data from aquadaq
# By Nicholai Mauritzson 
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Load and save data as HDF5 file.
# ------------------------------------------------------------------------------

# df1 = load_data("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1838_cooked.root")
# df2 = load_data("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1816_cooked.root")
# df3 = load_data("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1817_cooked.root")
# frames = [df1, df2, df3]
# df4 = pd.concat(frames)
# df1.to_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1838_DF.hdf5", key="w")

# ------------------------------------------------------------------------------

# def ratioPlot(data1, data2):
#     return data1/data2


# df = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1825_DF.hdf5")
# df2 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1815_DF.hdf5")
# plt.hist(df.qdc_det0, bins=1024*8)
# plt.show()
# w1=np.empty(len(df.qdc_det0))
# w1.fill(1/56520)
# w2=np.empty(len(df2.qdc_det0))
# w2.fill(1/246300)

# n= plt.hist(df.qdc_det0, bins=1000, weights=w1, label="PuBe", histtype="step")
# n2= plt.hist(df2.qdc_det0, bins=1000, weights=w2, label="Cs-137", histtype="step")
# plt.legend()
# plt.show()
# bins=8192
# plt.hist()
# ax.set_yscale("log")
# ratio = ratioPlot(n2[0],n[0])

# def TDCcalibration(data)




# ::::::::::::::::::::::::::::::::::::::::::::::
# TDC calibration
# TDCcal_data = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1819_DF.hdf5")
# n = plt.hist(TDCcal_data.tdc_det0_yap0, bins=500)

# const, mean, sigma = nm.gaussFit(n[1], n[0], 114, 122)
# print(mean)
# x_values=np.linspace(580,600)
# y_values = nm.gaussFunc(x_values, const, mean, sigma)

# plt.plot(x_values, y_values)
# plt.show()
# plt.plot(y_val[1][:-1],y_val[0])

# X_VAL = [235.39, 182.11, 52.94]
# Y_VAL = [66, 51 ,15]

# plt.plot(X_VAL, Y_VAL)
# plt.show()
#::::::::::::::::::::::::::::::::::::::::::::::::


#:::::::::::::::::::::::::::::::::::::::::::
# Compton edge fitting
# data = pta.load_data('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1825_cooked.root')
data = pd.read_hdf('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1825_DF.hdf5')
col = "qdc_det0"
min = 430
max = 550
def compton_fit(data, col, min, max, Ef, fit_lim=None):
    """
    Method is designed to fit and return the edge positions and maximum electron recoil energy of a Compton distribution.
    1) input pandas df 'data' 'col' 'min' 'max' 'Ef' 'fit_lim'
    2) Create histogram, saves x,y data
    3) 
    """
    #data is pandas df
    #col is the name of the column containing the data
    #Ef is the orignal foton energy
    #fit_lim contains. Default is None. Otherwise fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]. These are boundary constraints for the Gaussian fit functions. Put to inf or -inf to igore. 

    N = np.histogram(data[col], range=[min, max], bins=(max-min)) #Create histogram and save x,y data.
    x = N[1][:-1]   #Set bin positions as x values
    y = N[0]        #Set bin heights as y values
    meanTEMP = sum(x * y) / sum(y) #Calcaulate the mean value of distibution. Used as first quess for Gaussian fit.
    sigmaTEMP = np.sqrt(sum(y * (x - meanTEMP)**2) / sum(y)) #Calculate stdev of distribution. Used as first geuss for Gaussian fit.

    # const_min = 11651*0.98
    # mean_min  = (452)
    # sigma_min = -np.inf
    # const_max = 11651*1.02
    # mean_max  = np.inf#(451+10)
    # sigma_max = np.inf
    # fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]

    if fit_lim != None: #Check if boundary limits are enforced for Gaussian fit
        popt, pcov = curve_fit(nm.gaussFunc, x, y, p0 = [np.max(y), meanTEMP, sigmaTEMP], bounds=fit_lim)
    else:
        popt, pcov = curve_fit(nm.gaussFunc, x, y, p0 = [np.max(y), meanTEMP, sigmaTEMP])
    const, mean, sigma = popt[0], popt[1], popt[2] #saving result of fit as individual variables.
    #Print to console: optimal fitting parameters
    print('--- Optimal fitting parameters (scipy.curve_fit(gaussianFunc, const, mean, sigma)) ---')
    print('const=%.4f (max height of gaussian)' % const)
    print('mean=%.4f (mean position of gaussian)' % mean)
    print('sigma=%.4f (standard deviation of gaussian)' % sigma)
    print('---------------------------------------------------------------------------------------')
    print() #Create vertical space on terminal
    E_recoil_max = nm.comptonMax(eF) #Calculate maximum electron recoil energy
    
    for i in range(min,max): #Loop for finding 89% of maximum
        if nm.gaussFunc(i, const, mean, sigma)<=0.89*const:
            p=nm.gaussFunc(i, const, mean, sigma) #Saving compton edge value
            print('-> Compton edge found at ADC value: %.4f' % p) #Printing compton edge value (ADC) to console
            break
    print('-> Maximum electron recoil energy: %.4f' % E_recoil_max)

    #Test plotting
    plt.hist(data.qdc_det0,range=[200,800], bins=600)
    plt.plot(x,nm.gaussFunc(x, const, mean, sigma))
    plt.show()

return p, E_recoil_max


# Fuction which fits the compton spectrum as above but will show plot for itterative testing to determine best fit.
# Function compton fit should:
# - plot the data and fit so that user can make changes
# - return ADC value of compton edge
# - print to console final Gaussian fit parameters for diagnostic.

#PSD TESTING

# LG = df.qdc_det0
# SG = df.qdc_det0
# plt.hexbin(LG, (LG-SG)/LG)
# ply.show()