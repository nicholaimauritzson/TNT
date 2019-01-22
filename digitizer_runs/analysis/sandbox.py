import sys
sys.path.insert(0, "/home/gheed/Documents/projects/TNT/code_repo") #Import my own libraries
sys.path.insert(0, "/home/gheed/Documents/projects/TNT/code_repo/Rasmus_TOF") #Import my own libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import WD as wd
import pickle
import tof as tof
import dask.dataframe as dd
import DigiMethods as dm
import re
from tqdm import tqdm


#IMPORTING PARQUET FILES

# df = pd.read_parquet('/media/gheed/Seagate_Expansion_Drive1/Data/rasmus_data/test_ch3/part.0.parquet')
# df.head()

df = dd.read_parquet('/media/gheed/Seagate_Expansion_Drive1/Data/rasmus_data/test_ch3/part.0.parquet')
# tof.dask_chewer('/media/gheed/Seagate_Expansion_Drive1/Data/rasmus_data/test_ch3.txt', '/media/gheed/Seagate_Expansion_Drive1/Data/rasmus_data/', 10, 99999)






# textfile = open("wave0_1.txt") #open txt file
# df = pd.DataFrame({"evtno": [], "ts": [], "sample": []}) #Create pandas DataFrame
# pattern = [re.compile("Record"), re.compile("Event"), re.compile("Trigger"), re.compile("DC")]

# length = np.nan
# evtno = np.nan
# ts = np.nan
# sample_idx = np.nan

# sample_counter = 1
# evtno_list = []
# ts_list = []
# sample_list = []

# with textfile as f: #Go through text file line by line
#     for idx, line in enumerate(f, 1):
#         if pattern[0].search(line): #RECORD LENGHT"
#             tmp = line.split()
#             length = int(tmp[-1]) #save number of samples for event

#         if pattern[1].search(line): #EVENT NUMBER"
#             tmp = line.split() 
#             print("Event Number: "+tmp[-1])
#             evtno = int(tmp[-1]) #save event number value

#         if pattern[2].search(line): #TIME STAMP"
#             tmp = line.split()
#             ts = int(tmp[-1]) #save time stamp value
        
#         if pattern[3].search(line): #FIRST SAMPLE index"
#             sample_idx = int(idx + 1) #save index of first sample point
        
#         if ~np.isnan(length) and ~np.isnan(evtno) and ~np.isnan(ts) and ~np.isnan(sample_idx) and idx >= sample_idx and sample_counter <= length-1:
#             evtno_list.append(evtno)                    #Save event number to list
#             ts_list.append(ts)                          #Save time stamp to list
#             sample_TMP = line.rstrip('\n')
#             sample_list.append(int(sample_TMP)) #Save sample to list
#             sample_counter += 1
            
#             if sample_counter > length-1: # Reset all variable for next event
#                 sample_counter = 1
#                 sample_idx = np.nan
#                 length = np.nan
#                 evtno = np.nan
#                 ts = np.nan

# df['evtno'] = evtno_list 
# df['ts'] = ts_list 
# df['sample'] = sample_list 
# pickle.dump(df, open("df1.pkl", "wb")) #save ph data to file