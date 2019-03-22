import sys
sys.path.insert(0, "/home/gheed/Documents/projects/TNT/code_repo") #Import my own libraries
import pyTagAnalysis as pyTag
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ------------------------------------------------------------------------------
# Quick plotting script for analog aquadaq.
# Argument needs to be hdf5 file (pandas)
#
# By Nicholai Mauritzson 
# ------------------------------------------------------------------------------

path="/media/gheed/Seagate_Expansion_Drive1/Data/TNT/NE213_cup/Data1811_DF.hdf5"
df = pd.read_hdf(path) #Load data
pyTag.calculate_ps(df) #Add pulse shape colum to df


# plt.figure(0)
# plt.hist(df.qdc_det0, 256, linewidth=2, histtype="step",label='LG, entries={}'.format(np.count_nonzero(df.qdc_det0)))
# plt.legend()
# plt.title("Long gate QDC neutron detector")
# plt.ylabel("counts")
# plt.xlabel("QDC [arb. unit]")
# plt.grid(zorder=0)
# plt.yscale("log")

# plt.figure(1)
# plt.hist(df.qdc_sg_det0, 256, linewidth=2, histtype="step",label='SG, entries={}'.format(np.count_nonzero(df.qdc_sg_det0)))
# plt.legend()
# plt.title("Short gate QDC neutron detector")
# plt.ylabel("counts")
# plt.xlabel("QDC [arb. unit]")
# plt.grid(zorder=0)
# plt.yscale("log")

# plt.figure(2)
# plt.hist(df.qdc_yap0, 256, histtype="step", linewidth=2, label='YAP_0, entries={}'.format(np.count_nonzero(df.qdc_yap0)))
# plt.hist(df.qdc_yap1, 256, histtype="step", linewidth=2, label='YAP_1, entries={}'.format(np.count_nonzero(df.qdc_yap1)))
# plt.hist(df.qdc_yap2, 256, histtype="step", linewidth=2, label='YAP_2, entries={}'.format(np.count_nonzero(df.qdc_yap2)))
# plt.hist(df.qdc_yap3, 256, histtype="step", linewidth=2, label='YAP_3, entries={}'.format(np.count_nonzero(df.qdc_yap3)))
# plt.legend()
# plt.title("QDC YAPs")
# plt.ylabel("counts")
# plt.xlabel("QDC [arb. unit]")
# plt.grid(zorder=0)
# plt.yscale("log")

# plt.figure(3)
# plt.hist(df.tdc_det0_yap0, 3000, histtype="step", linewidth=2, label='YAP_0, entries={}'.format(np.count_nonzero(df.tdc_det0_yap0)))
# plt.hist(df.tdc_det0_yap1, 3000, histtype="step", linewidth=2, label='YAP_1, entries={}'.format(np.count_nonzero(df.tdc_det0_yap1)))
# plt.hist(df.tdc_det0_yap2, 3000, histtype="step", linewidth=2, label='YAP_2, entries={}'.format(np.count_nonzero(df.tdc_det0_yap2)))
# plt.hist(df.tdc_det0_yap3, 3000, histtype="step", linewidth=2, label='YAP_3, entries={}'.format(np.count_nonzero(df.tdc_det0_yap3)))
# plt.legend()
# plt.title("Time of Flight")
# plt.xlabel("TOF [arb. unit]")
# plt.ylabel("counts")
# plt.grid(zorder=0)
# plt.yscale("log")

new_df=df.query('0 <= ps_det0 <= 1')
plt.figure(4)
plt.hist2d(new_df.qdc_det0, new_df.ps_det0, bins=256, norm=LogNorm())
plt.colorbar()
plt.xlim(0,4200)
plt.title("Pulse Shape")
plt.xlabel("QDC LG [arb. unit]")
plt.ylabel("PS")

plt.show()

