import sys
sys.path.insert(0, "/home/gheed/Documents/projects/TNT/code_repo") #Import my own libraries
import WD as wd
import PMCA as pmca
import pandas as pd
import pickle
import glob
import os

#List of data files for import/conversion:

# pmca.importData('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/digitizer/digitizer_mca_comparison/MCA_180s.mca')
# pmca.importData('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/digitizer/digitizer_mca_comparison/MCA_300s.mca')

# wd.load_data_simple('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/digitizer/digitizer_mca_comparison/wave0_1.txt', 19005)



file_list = os.listdir("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/digitizer/digitizer_mca_comparison/wave0_1/") #glob.glob(os.path.join(os.getcwd(), "/media/gheed/Seagate_Expansion_Drive1/Data/TNT/digitizer/digitizer_mca_comparison/wave0_1", "*.txt"))
evt = 0
ph = pd.DataFrame({'ph': []}) #ph = pd.DataFrame(columns='ph')
for file_path in file_list:
    with open("/media/gheed/Seagate_Expansion_Drive1/Data/TNT/digitizer/digitizer_mca_comparison/wave0_2/"+file_path) as f_input:
        print('Event number:', evt)
        evt += 1
        ph = ph.append({'ph':max(map(float, f_input))}, ignore_index=True)


pickle.dump(ph, open("ph.pkl", "wb")) #save ph data to file