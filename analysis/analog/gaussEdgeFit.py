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
from scipy.stats import chisquare
from tqdm import tqdm
import pyTagAnalysis as pta
from scipy.optimize import curve_fit

# BG_data = pd.read_hdf('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1841_DF.hdf5')
# Co60_data = pd.read_hdf('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1851_DF.hdf5')
# Na22_data = pd.read_hdf('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1862_DF.hdf5')
# Cs137_data = pd.read_hdf('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1850_DF.hdf5')
# PuBe_data = pd.read_hdf('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1876_DF.hdf5')

# BG_time = 14375
# Co60_time = 510.29 
# Na22_time = 603.77
# Cs137_time = 3698.68
# PuBe_time = 11251.2

#...Calculating weights
# BG_w    = np.empty(len(BG_data.qdc_det0)) #Make array same length as data set
# Co60_w  = np.empty(len(Co60_data.qdc_det0)) #Make array same length as data set
# Na22_w  = np.empty(len(Na22_data.qdc_det0)) #Make array same length as data set
# Cs137_w = np.empty(len(Cs137_data.qdc_det0)) #Make array same length as data set
# PuBe_w = np.empty(len(PuBe_data.qdc_det0)) #Make array same length as data set

# BG_w.fill(1/BG_time)
# Co60_w.fill(1/Co60_time)
# Na22_w.fill(1/Na22_time)
# Cs137_w.fill(1/Cs137_time)
# PuBe_w.fill(1/PuBe_time)

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

#_____Na-22 (0.511 keV)Boundary conditions______Data1862
# const_min = -np.inf
# mean_min  = 630#-np.inf #...
# sigma_min = -np.inf
# const_max = np.inf
# mean_max  = np.inf
# sigma_max = np.inf
# fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]
# min = 620#441
# max = 800
# Ef = .511

##_____Na-22 (1.275 keV)Boundary condisions______
# const_min = -np.inf
# mean_min  = -np.inf
# sigma_min = -np.inf
# const_max = np.inf
# mean_max  = np.inf
# sigma_max = np.inf
# fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]
# min = 953
# max = 1400
# Ef = 1.275

##_____Na-22 (1.275 keV)Boundary condisions______Data1862
# const_min = -np.inf
# mean_min  = -np.inf
# sigma_min = -np.inf
# const_max = np.inf
# mean_max  = np.inf
# sigma_max = np.inf
# fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]
# min = 953
# max = 1445
# Ef = 1.275

# #_____Co-60 (1.3325 MeV)Boundary conditions______Data1851 collimated
# const_min = -np.inf#11651*0.98
# mean_min  = -np.inf#(452)
# sigma_min = -np.inf
# const_max = np.inf#11651*1.02
# mean_max  = np.inf#(451+10)
# sigma_max = np.inf
# fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]
# min = 821#850
# max = 1137
# Ef = 1.3325

# #_____Co-60 (1.3325 MeV)Boundary conditions______Data1831
# const_min = -np.inf#11651*0.98
# mean_min  = -np.inf#(452)
# sigma_min = -np.inf
# const_max = np.inf#11651*1.02
# mean_max  = np.inf#(451+10)
# sigma_max = np.inf
# fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]
# min = 850
# max = 1200
# Ef = 1.3325

# #_____Cs-137 (0.6615 MeV)Boundary condisions______ (Data1850, collimated)
# const_min = -np.inf
# mean_min  = 580#-np.inf#454#-np.inf
# sigma_min = -np.inf
# const_max = np.inf#3.15
# mean_max  = np.inf
# sigma_max = np.inf
# fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]
# min = 570
# max = 667#550
# Ef = .6615

# #_____PuBe (2.230 MeV) Boundary condisions______ Data1876
# const_min = -np.inf
# mean_min  = 960#-np.inf
# sigma_min = -np.inf
# const_max = np.inf
# mean_max  = np.inf
# sigma_max = np.inf
# fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]
# min = 920
# max = 1272
# Ef = 2.230

# #_____PuBe (4.439 MeV) Boundary condisions______
# const_min = -np.inf
# mean_min  = -np.inf
# sigma_min = -np.inf
# const_max = 3000#np.inf
# mean_max  = np.inf
# sigma_max = np.inf
# fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]]
# min = 1272
# max = 2500
# Ef = 4.439

# col = "qdc_det0"
# x_CE, y_CE, variables, errors = nm.comptonEdgeFit(Na22_data, col, min, max, Ef, fit_lim)
