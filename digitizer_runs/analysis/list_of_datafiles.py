import sys
sys.path.insert(0, "/home/gheed/Documents/projects/TNT/code_repo") #Import my own libraries

import PMCA as pmca

#List of data files for conversion:

pmca.importData('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/digitizer/digitizer_mca_comparision/mca180.mca')
pmca.importData('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/digitizer/digitizer_mca_comparision/mca300.mca')