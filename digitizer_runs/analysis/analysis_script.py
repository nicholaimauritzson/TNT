import sys
sys.path.insert(0, "/home/gheed/Documents/projects/TNT/code_repo") #Import my own libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import WD as wd

df = pd.read_csv("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/digitizer/first_tests/wave0_4.txt")
h_df = []
space = 19005
idx = 0
for evt in range(round(len(df)/space)):
    h_df.append(df[idx:idx+space].max().item())
    idx += space

plt.hist(h_df, bins=100)
plt.show()
    
