import sys
sys.path.insert(0, "/home/gheed/Documents/projects/TNT/code_repo") #Import my own libraries
from pyTagAnalysis import load_data
import nicholai_math as nm
import nicholai_utility as nu
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

# df1 = load_data("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1841_cooked.root")
# df2 = load_data("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1816_cooked.root")
# df3 = load_data("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1817_cooked.root")
# frames = [df1, df2, df3]
# df4 = pd.concat(frames)
# df1.to_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1841_DF.hdf5", key="w")
# nu.convertCookedData(("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1847_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1848_cooked.root"), "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1847_DF.hdf5")
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



BG_data = pd.read_hdf('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1841_DF.hdf5')
# Co60_data = pd.read_hdf('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1831_DF.hdf5')
Na22_data = pd.read_hdf('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1838_DF.hdf5')
# Cs137_data = pd.read_hdf('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1825_DF.hdf5')

BG_time = 14375
# Co60_time = 510.29 
# Cs137_time = 3698.68
Na22_time = 603.77

BG_w    = np.empty(len(BG_data.qdc_det0)) #Make array same length as data set
# Co60_w  = np.empty(len(Co60_data.qdc_det0)) #Make array same length as data set
Na22_w  = np.empty(len(Na22_data.qdc_det0)) #Make array same length as data set
# Cs137_w = np.empty(len(Cs137_data.qdc_det0)) #Make array same length as data set

BG_w.fill(1/BG_time)
# Co60_w.fill(1/Co60_time)
Na22_w.fill(1/Na22_time)
# Cs137_w.fill(1/Cs137_time)

# BG    = np.histogram(BG_data.qdc_det0,      bins=1024, weights=BG_w)
# Co60  = np.histogram(Co60_data.qdc_det0,    bins=1024, weights=Co60_w)
# Na22  = np.histogram(Na22_data.qdc_det0,    bins=1024, weights=Na22_w)
# Cs137 = np.histogram(Cs137_data.qdc_det0,   bins=1024, weights=Cs137_w)

# Co60_new = Co60[0]-BG[0]
# Na22_new = Na22[0]-BG[0]
# Cs137_new = Cs137[0]-BG[0]

# plt.plot(Co60[1][:-1], Co60[0], label='Co60 - BG')
# plt.plot(Co60[1][:-1], Co60_new, label='Co60 raw')

# plt.plot(Na22[1][:-1], Na22[0], label='Na22 - BG')
# plt.plot(Na22[1][:-1], Na22_new, label='Na22 raw')

# plt.plot(Cs137[1][:-1], Cs137[0], label='Cs137 - BG')
# plt.plot(Cs137[1][:-1], Cs137_new, label='Cs137 raw')

# plt.legend()
# plt.show()



#_____Na-22 (0.511 keV)Boundary conditions______
# const_min = -np.inf
# mean_min  = -np.inf #...
# sigma_min = -np.inf
# const_max = np.inf
# mean_max  = 444#np.inf
# sigma_max = np.inf
# fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]
# min = 400#441
# max = 545
# Ef = .511

# #_____Cs-137 (1.3325 MeV)Boundary conditions______
# const_min = 11651*0.98
# mean_min  = (452)
# sigma_min = -np.inf
# const_max = 11651*1.02
# mean_max  = np.inf#(451+10)
# sigma_max = np.inf
# fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]

# #_____Cs-137 (0.6615 MeV)Boundary condisions______
# const_min = -np.inf
# mean_min  = -np.inf#454#-np.inf
# sigma_min = -np.inf
# const_max = 3.15#np.inf
# mean_max  = np.inf
# sigma_max = np.inf
# fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]
# min = 460
# max = 550#535
# Ef = .6615

##_____Na-22 (1.275 keV)Boundary condisions______
const_min = -np.inf
mean_min  = -np.inf
sigma_min = -np.inf
const_max = np.inf
mean_max  = np.inf
sigma_max = np.inf
fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]
min = 953
max = 1400
Ef = 1.275

col = "qdc_det0"
nm.comptonEdgeFit(Na22_data, col, min, max, Ef, Na22_w, BG_w, BG_data, fit_lim)


#ERROR PROPAGATION TEST
# print(nm.errorPropAdd((.25,25,3,100,0.010)))
# print(nm.errorPropMulti(100, (1,2,3), (0.1,1.1,0.001)))
# print(nm.errorPropPower(100, 25, 0.001, 3))
# print(nm.errorPropExp(100, 0.1))
# print(nm.errorPropGauss(12000, 400, 12500, 100, 350, 52, 100, 20))



# from uncertainties import ufloat
# from uncertainties.umath import * #Get all methods for library

# x=491.720 #My error: 5.5350
# #MY Y values and error: 2.8118 +/- 0.0367 (Na22 @ 89%)
# uconst =ufloat(3.1593, 0.0250)
# umean = ufloat(442.1512, 2.8206)
# usigma = ufloat(102.1599, 4.5066)

# const = 3.1593
# const_err = .025
# mean = 442.1512
# mean_err = 2.8206
# sigma = 102.1599
# sigma_err = 4.5066
# Y = nm.gaussFunc(x, const, mean, sigma)
# print(nm.errorPropGauss(Y, x, const, const_err, mean, mean_err, sigma, sigma_err))