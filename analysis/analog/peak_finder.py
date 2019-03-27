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

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# peak finder test
f = open("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NaI/Na-22_spectrum_NaI_detector.txt", "r")
def compton_scatter(E1, angle): #energi given in MeV, angle is given in degrees
    deg = pi/180 #conversion from degrees to radians
    angle = angle*deg #Convert angle input to radians
    me = 510.9989461 # keV/c^2

    return E1/(1+(E1/(me))*(1-cos(radians(angle))))


test_data = np.loadtxt("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NaI/Na-22_spectrum_NaI_detector.txt")

prominance = max(test_data)/40

peaks = signal.find_peaks(test_data, distance=10, width=5, prominence=300)

channelNum = np.arange(1,2049,1)
x_values=1.463602*channelNum-14.432985
plt.plot(x_values, test_data)

for i in peaks[0]:
    plt.axvline(x=i, color='k', linestyle='--')

angels = np.arange(0, 181, 1)
energies = []
for i in range(181): #Calculate compton scattering energy values
    energies.append(compton_scatter(510.9989461, angels[i]))


# plt.plot(compton_scatter(510.9989461, angles))
plt.axvline(x=max(energies), color='r', linewidth=2, linestyle='--')

plt.show()
