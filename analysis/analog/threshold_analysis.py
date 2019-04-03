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

# df1 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1894_25mV.hdf5") 
# min = 150
# max = 300
# df2 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1895_50_7mV.hdf5") 
# min = 450
# max = 650
# df3 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1896_75_7mV.hdf5") 
# min = 583
# max = 800
# df4 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1897_100_9mV.hdf5") 
# min = 751
# max = 1000
# df5 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1898_151mV.hdf5") 
# min = 774
# max = 980
# df6 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1899_200_4mV.hdf5") 
# min = 1015
# max = 1250
# df7 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1900_249_7mV.hdf5") 
# min = 1222
# max = 1480
df8 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1901_299_4mV.hdf5") 
min = 1400
max = 1700

# w1=np.empty(len(df1.qdc_det0))
# w1.fill(1/209.72)
# w2=np.empty(len(df2.qdc_det0))
# w2.fill(1/280.25)
# w3=np.empty(len(df3.qdc_det0))
# w3.fill(1/349.37)
# w4=np.empty(len(df4.qdc_det0))
# w4.fill(1/419.07)
# w5=np.empty(len(df5.qdc_det0))
# w5.fill(1/588.19)
# w6=np.empty(len(df6.qdc_det0))
# w6.fill(1/594.17)
# w7=np.empty(len(df7.qdc_det0))
# w7.fill(1/257.5)
w8=np.empty(len(df8.qdc_det0))
w8.fill(1/249.4)

# N1 = np.histogram(df1.qdc_det0, range=(min, max), bins=(max-min), weights=w1) #Data1894
# N2 = np.histogram(df2.qdc_det0, range=(min, max), bins=(max-min), weights=w2)#Data1895
# N3 = np.histogram(df3.qdc_det0, range=(min, max), bins=(max-min), weights=w3)#Data1896
# N4 = np.histogram(df4.qdc_det0, range=(min, max), bins=(max-min), weights=w4)#Data1897
# N5 = np.histogram(df5.qdc_det0, range=(min, max), bins=(max-min), weights=w5)#Data1898
# N6 = np.histogram(df6.qdc_det0, range=(min, max), bins=(max-min), weights=w6)#Data1899
# N7 = np.histogram(df7.qdc_det0, range=(min, max), bins=(max-min), weights=w7)#Data1900
N8 = np.histogram(df8.qdc_det0, range=(min, max), bins=(max-min), weights=w8)#Data1901
X=N8[1][:-1]
Y=N8[0]
X_long=np.arange(min, max, 0.01)
popt, pcov = curve_fit(nm.gaussFunc, X, Y, p0=(np.max(Y), np.mean(X), 100))

p=[0, 0]

# for i in tqdm(range(len(X_long))):
#     val = nm.gaussFunc(X_long[i], popt[0],popt[1],popt[2])
#     if val >= 0.5*popt[0]:
#         p = (X_long[i], val) #save x and y values at 50%
#         break
print('Threshold found at ADC channel: %.4f'%popt[1])

plt.plot(X, nm.gaussFunc(X, popt[0],popt[1],popt[2]), label='A=%.4f, mu=%.4f, std=%.4f'%(popt[0],popt[1],popt[2]))
plt.hist(df8.qdc_det0, range=(0,4000), bins=4000, weights=w8, histtype='step')
plt.scatter(p[0], p[1], marker='o', color='black')
# nu.printFormatting('Fit errors', descriptions=('Constant','Mean','Sigma'), values=(popt[0], popt[1], popt[2]), errors=None, unit=('','',''))
# # plt.title('collimated sources')
# plt.xlabel('QDC values [arb. units]')
# plt.ylabel('counts [s$^{-1}$]')
# plt.grid(which='both')
plt.legend()
# plt.yscale("log")
plt.show()
