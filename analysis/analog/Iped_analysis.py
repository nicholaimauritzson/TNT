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

# Make root cooked root files into pandas DF
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1970_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1970_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1971_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1971_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1972_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1972_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1973_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1973_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1974_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1974_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1975_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1975_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1976_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1976_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1977_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1977_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1978_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1978_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1979_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1979_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1980_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1980_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1981_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1981_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1982_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1982_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1983_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1983_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1984_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1984_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1985_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1985_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1986_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1986_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1987_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1987_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1988_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1988_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1989_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1989_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1991_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1991_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1993_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1993_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1994_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1994_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1995_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1995_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1996_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1996_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1997_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1997_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1998_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1998_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1999_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1999_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data2000_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data2000_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data2001_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data2001_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data2002_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data2002_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data2003_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data2003_DF.hdf5")
# nu.convertCookedData("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data2004_cooked.root", "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data2004_DF.hdf5")


time60 = 10.93
time70 = 10.21
time80 = 9.66
time90 = 9.26
time100 = 13.91
time110 = 13.01
time120 = 10.52
time130 = 10.98
time140 = 10.38
time150 = 13.38
time160 = 13.45
time170 = 20.56
time180 = 20.92
time190 = 10.95
time200 = 13.48
time210 = 11.85
time220 = 23.51
time230 = 8.33
time240 = 11.17
time250 = 11.02

Iped60 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1982_DF.hdf5")
Iped70 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1981_DF.hdf5")
Iped80 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1980_DF.hdf5")
Iped90 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1979_DF.hdf5")
Iped100 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1978_DF.hdf5")
Iped110 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1977_DF.hdf5")
Iped120 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1976_DF.hdf5")
Iped130 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1975_DF.hdf5")
Iped140 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1974_DF.hdf5")
Iped150 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1973_DF.hdf5")
Iped160 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1972_DF.hdf5")
Iped170 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1971_DF.hdf5")
Iped180 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1970_DF.hdf5")
Iped190 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1983_DF.hdf5")
Iped200 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1984_DF.hdf5")
Iped210 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1985_DF.hdf5")
Iped220 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1986_DF.hdf5")
Iped230 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1987_DF.hdf5")
Iped240 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1988_DF.hdf5")
Iped250 = pd.read_hdf("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1989_DF.hdf5")

# w=np.empty(len(df.qdc_det0)).fill(1/time60) #Make array same length as data set
# w.fill(1/56520) #Fill array with 1/live_time
min = 500
max = 1850
numBins=200

plt.figure(0)
plt.title('QDC spectrums vs I$_{ped}$ settings (signal generator)')
plt.xlabel('QDC [arb. units]')
plt.ylabel('counts s$^{-1}')

w = np.empty(len(Iped60.qdc_det0))
w.fill(1/time60)
plt.hist(Iped60.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=60, mean=%.1f'%np.mean(Iped60.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped70.qdc_det0))
w.fill(1/time70)
plt.hist(Iped70.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=70, mean=%.1f'%np.mean(Iped70.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped80.qdc_det0))
w.fill(1/time80)
plt.hist(Iped80.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=80, mean=%.1f'%np.mean(Iped80.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped90.qdc_det0))
w.fill(1/time90)
plt.hist(Iped90.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=90, mean=%.1f'%np.mean(Iped90.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped100.qdc_det0))
w.fill(1/time100)
plt.hist(Iped100.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=100, mean=%.1f'%np.mean(Iped100.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped110.qdc_det0))
w.fill(1/time110)
plt.hist(Iped110.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=110, mean=%.1f'%np.mean(Iped110.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped120.qdc_det0))
w.fill(1/time120)
plt.hist(Iped120.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=120, mean=%.1f'%np.mean(Iped120.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped130.qdc_det0))
w.fill(1/time130)
plt.hist(Iped130.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=130, mean=%.1f'%np.mean(Iped130.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped140.qdc_det0))
w.fill(1/time140)
plt.hist(Iped140.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=140, mean=%.1f'%np.mean(Iped140.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped150.qdc_det0))
w.fill(1/time150)
plt.hist(Iped150.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=150, mean=%.1f'%np.mean(Iped150.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped160.qdc_det0))
w.fill(1/time160)
plt.hist(Iped160.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=160, mean=%.1f'%np.mean(Iped160.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped170.qdc_det0))
w.fill(1/time170)
plt.hist(Iped170.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=170, mean=%.1f'%np.mean(Iped170.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped180.qdc_det0))
w.fill(1/time180)
plt.hist(Iped180.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=180, mean=%.1f'%np.mean(Iped180.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped190.qdc_det0))
w.fill(1/time190)
plt.hist(Iped190.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=190, mean=%.1f'%np.mean(Iped190.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped200.qdc_det0))
w.fill(1/time200)
plt.hist(Iped200.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=200, mean=%.1f'%np.mean(Iped200.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped210.qdc_det0))
w.fill(1/time210)
plt.hist(Iped210.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=210, mean=%.1f'%np.mean(Iped210.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped220.qdc_det0))
w.fill(1/time220)
plt.hist(Iped220.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=220, mean=%.1f'%np.mean(Iped220.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped230.qdc_det0))
w.fill(1/time230)
plt.hist(Iped230.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=230, mean=%.1f'%np.mean(Iped230.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped240.qdc_det0))
w.fill(1/time240)
plt.hist(Iped240.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=240, mean=%.1f'%np.mean(Iped240.qdc_det0)) #Apply weighting factor

w = np.empty(len(Iped250.qdc_det0))
w.fill(1/time250)
plt.hist(Iped250.qdc_det0, range=(min, max), bins=numBins, weights=w, label='I$_{ped}$=250, mean=%.1f'%np.mean(Iped250.qdc_det0)) #Apply weighting factor

plt.legend()




means=[
np.mean(Iped60.qdc_det0),
np.mean(Iped70.qdc_det0),
np.mean(Iped80.qdc_det0),
np.mean(Iped90.qdc_det0),
np.mean(Iped100.qdc_det0),
np.mean(Iped110.qdc_det0),
np.mean(Iped120.qdc_det0),
np.mean(Iped130.qdc_det0),
np.mean(Iped140.qdc_det0),
np.mean(Iped150.qdc_det0),
np.mean(Iped160.qdc_det0),
np.mean(Iped170.qdc_det0),
np.mean(Iped180.qdc_det0),
np.mean(Iped190.qdc_det0),
np.mean(Iped200.qdc_det0),
np.mean(Iped210.qdc_det0),
np.mean(Iped220.qdc_det0),
np.mean(Iped230.qdc_det0),
np.mean(Iped240.qdc_det0),
np.mean(Iped250.qdc_det0),
]
Iped_values = [60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250]

plt.figure(1)
plt.title('Iped settings vs mean value')
plt.xlabel('I$_{ped}$ settings [arb. units]')
plt.ylabel('Mean value [QDC units]')
plt.plot(Iped_values, means, linewidth=2, label='Data')
plt.legend()

plt.show()


#Do the same for Iped data with Cs source!!!!