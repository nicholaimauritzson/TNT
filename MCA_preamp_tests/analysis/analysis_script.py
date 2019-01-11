import sys
sys.path.insert(0, "/home/gheed/Documents/projects/TNT/code_repo") #Import my own libraries
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import PMCA as pmca
import nicholai_math as nm


norm = 500
# run13 = pmca.e3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run13.pkl', 'rb')),norm)
# run14 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run14.pkl', 'rb')),norm)
# run15 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run15.pkl', 'rb')),norm)
# run16 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run16.pkl', 'rb')),norm)
# run17 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run17.pkl', 'rb')),norm)
# run18 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run18.pkl', 'rb')),norm)
# run19 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run19.pkl', 'rb')),norm)
# run20 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run20.pkl', 'rb')),norm)
# run21 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run21.pkl', 'rb')),norm)
# run22 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run22.pkl', 'rb')),norm)
# run23 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run23.pkl', 'rb')),norm)
# run24 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run24.pkl', 'rb')),norm)
# run25 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run25.pkl', 'rb')),norm)
# run26 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run26.pkl', 'rb')),norm)
# run27 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run27.pkl', 'rb')),norm)
# run28 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run28.pkl', 'rb')),norm)
# run29 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run29.pkl', 'rb')),norm)
# run30 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run30.pkl', 'rb')),norm)
# run31 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run31.pkl', 'rb')),norm)
# run32 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run32.pkl', 'rb')),norm)
# run33 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run33.pkl', 'rb')),norm)
# run34 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run34.pkl', 'rb')),norm)
# run35 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run35.pkl', 'rb')),norm)
# run36 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run36.pkl', 'rb')),norm)
# run37 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run37.pkl', 'rb')),norm)
# run38 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run38.pkl', 'rb')),norm)
# run39 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run39.pkl', 'rb')),norm)

## - 12us shaping as function of voltage
# run40 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run40.pkl', 'rb')),norm)
# run41 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run41.pkl', 'rb')),norm)
# run42 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run42.pkl', 'rb')),norm)
# run43 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run43.pkl', 'rb')),norm)
# run44 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run44.pkl', 'rb')),norm)
# run45 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run45.pkl', 'rb')),norm)
# run46 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run46.pkl', 'rb')),norm)
# run47 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run47.pkl', 'rb')),norm)
# run48 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run48.pkl', 'rb')),norm)
# run49 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run49.pkl', 'rb')),norm)
# run50 = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/MCA_preamp_tests/Run50.pkl', 'rb')),norm)

## - Background measurements
BG = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/background/background.pkl', 'rb')),norm)
BG_B = pmca.he3Shift(pickle.load(open('/media/gheed/Seagate_Expansion_Drive1/Data/TNT/background/background_B.pkl', 'rb')),norm)



start = 450 
stop = 550

# const, mean, sigma = nm.gaussFit(run13, 'ADC', 'rate', start, stop)
# run13_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run14, 'ADC', 'rate', start, stop)
# run14_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run15, 'ADC', 'rate', start, stop)
# run15_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run16, 'ADC', 'rate', start, stop)
# run16_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run17, 'ADC', 'rate', start, stop)
# run17_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run18, 'ADC', 'rate', start, stop)
# run18_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run19, 'ADC', 'rate', start, stop)
# run19_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run20, 'ADC', 'rate', start, stop)
# run20_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run21, 'ADC', 'rate', start, stop)
# run21_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run22, 'ADC', 'rate', start, stop)
# run22_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run23, 'ADC', 'rate', start, stop)
# run23_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run24, 'ADC', 'rate', start, stop)
# run24_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run25, 'ADC', 'rate', start, stop)
# run25_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run26, 'ADC', 'rate', start, stop)
# run26_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run27, 'ADC', 'rate', start, stop)
# run27_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run28, 'ADC', 'rate', start, stop)
# run28_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run29, 'ADC', 'rate', start, stop)
# run29_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run30, 'ADC', 'rate', start, stop)
# run30_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run31, 'ADC', 'rate', start, stop)
# run31_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run32, 'ADC', 'rate', start, stop)
# run32_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run33, 'ADC', 'rate', start, stop)
# run33_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run34, 'ADC', 'rate', start, stop)
# run34_res = nm.resCalc(mean,sigma)*100

# const, mean, sigma = nm.gaussFit(run35, 'ADC', 'rate', start, stop)
# run35_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run36, 'ADC', 'rate', start, stop)
# run36_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run37, 'ADC', 'rate', start, stop)
# run37_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run38, 'ADC', 'rate', start, stop)
# run38_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run39, 'ADC', 'rate', start, stop)
# run39_res = nm.resCalc(mean,sigma)*100


# const, mean, sigma = nm.gaussFit(run40, 'ADC', 'rate', 420 , 570)
# run40_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run41, 'ADC', 'rate', 491, 663)
# run41_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run42, 'ADC', 'rate', 636, 777)
# run42_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run43, 'ADC', 'rate', 766, 932)
# run43_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run44, 'ADC', 'rate', 972, 1149)
# run44_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run45, 'ADC', 'rate', 1263, 1470)
# run45_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run46, 'ADC', 'rate', 1677, 1883)
# run46_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run47, 'ADC', 'rate', 2200, 2513)
# run47_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run48, 'ADC', 'rate', 2975, 3356)
# run48_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run49, 'ADC', 'rate', 4074, 4593)
# run49_res = nm.resCalc(mean,sigma)*100
# const, mean, sigma = nm.gaussFit(run50, 'ADC', 'rate', 5694, 6480)
# run50_res = nm.resCalc(mean,sigma)*100

# resolution = [run40_res, run41_res, run42_res, run43_res, run44_res, run45_res, run46_res, run47_res, run48_res, run49_res, run50_res]
# voltage = [800,850,900,950,1000,1050,1100,1150,1200,1250,1300]

# plt.plot(run13['ADC'], run13['rate'], linewidth=2, drawstyle='steps', label='run13, R={:.2f}%'.format(run13_res))
# plt.plot(run14['ADC'], run14['rate'], linewidth=2, drawstyle='steps', label='run14, R={:.2f}%'.format(run14_res))
# plt.plot(run15['ADC'], run15['rate'], linewidth=2, drawstyle='steps', label='run15, R={:.2f}%'.format(run15_res))
# plt.plot(run16['ADC'], run16['rate'], linewidth=2, drawstyle='steps', label='run16, R={:.2f}%'.format(run16_res))
# plt.plot(run17['ADC'], run17['rate'], linewidth=2, drawstyle='steps', label='run17, R={:.2f}%'.format(run17_res))
# plt.plot(run18['ADC'], run18['rate'], linewidth=2, drawstyle='steps', label='run18, R={:.2f}%'.format(run18_res))
# plt.plot(run19['ADC'], run19['rate'], linewidth=2, drawstyle='steps', label='run19, R={:.2f}%'.format(run19_res))
# plt.plot(run20['ADC'], run20['rate'], linewidth=2, drawstyle='steps', label='run20, R={:.2f}%'.format(run20_res))
# plt.plot(run21['ADC'], run21['rate'], linewidth=2, drawstyle='steps', label='run21, R={:.2f}%'.format(run21_res))
# plt.plot(run22['ADC'], run22['rate'], linewidth=2, drawstyle='steps', label='run22, R={:.2f}%'.format(run22_res))
# plt.plot(run23['ADC'], run23['rate'], '--',linewidth=2, drawstyle='steps', label='run23, R={:.2f}%'.format(run23_res))
# plt.plot(run24['ADC'], run24['rate'], '--',linewidth=2, drawstyle='steps', label='run24, R={:.2f}%'.format(run24_res))
# plt.plot(run25['ADC'], run25['rate'], '--',linewidth=2, drawstyle='steps', label='run25, R={:.2f}%'.format(run25_res))
# plt.plot(run26['ADC'], run26['rate'], '-',linewidth=2, drawstyle='steps', label='1$\mu$s, R={:.2f}%'.format(run26_res))
# plt.plot(run27['ADC'], run27['rate'], '-',linewidth=2, drawstyle='steps', label='2$\mu$s, R={:.2f}%'.format(run27_res))
# plt.plot(run28['ADC'], run28['rate'], '-',linewidth=2, drawstyle='steps', label='3$\mu$s, R={:.2f}%'.format(run28_res))
# plt.plot(run29['ADC'], run29['rate'], '-',linewidth=2, drawstyle='steps', label='6$\mu$s, R={:.2f}%'.format(run29_res))
# plt.plot(run30['ADC'], run30['rate'], '-',linewidth=2, drawstyle='steps', label='1$\mu$s, R={:.2f}%'.format(run30_res))
# plt.plot(run31['ADC'], run31['rate'], '-',linewidth=2, drawstyle='steps', label='2$\mu$s, R={:.2f}%'.format(run31_res))
# plt.plot(run32['ADC'], run32['rate'], '-',linewidth=2, drawstyle='steps', label='3$\mu$s, R={:.2f}%'.format(run32_res))
# plt.plot(run33['ADC'], run33['rate'], '-',linewidth=2, drawstyle='steps', label='5$\mu$s, R={:.2f}%'.format(run33_res))
# plt.plot(run34['ADC'], run34['rate'], '-',linewidth=2, drawstyle='steps', label='10$\mu$s, R={:.2f}%'.format(run34_res))

# plt.plot(run35['ADC'], run35['rate'], '-',linewidth=2, drawstyle='steps', label='1$\mu$s, R={:.2f}%'.format(run35_res))
# plt.plot(run36['ADC'], run36['rate'], '-',linewidth=2, drawstyle='steps', label='2$\mu$s, R={:.2f}%'.format(run36_res))
# plt.plot(run37['ADC'], run37['rate'], '-',linewidth=2, drawstyle='steps', label='3$\mu$s, R={:.2f}%'.format(run37_res))
# plt.plot(run38['ADC'], run38['rate'], '-',linewidth=2, drawstyle='steps', label='6$\mu$s, R={:.2f}%'.format(run38_res))
# plt.plot(run39['ADC'], run39['rate'], '-',linewidth=2, drawstyle='steps', label='12$\mu$s, R={:.2f}%'.format(run39_res))

# plt.plot(run40['ADC'], run40['counts'], '-',linewidth=2, drawstyle='steps', label='+800VDC, R={:.2f}%'.format(run40_res))
# plt.plot(run41['ADC'], run41['counts'], '-',linewidth=2, drawstyle='steps', label='+850VDC, R={:.2f}%'.format(run41_res))
# plt.plot(run42['ADC'], run42['counts'], '-',linewidth=2, drawstyle='steps', label='+900VDC, R={:.2f}%'.format(run42_res))
# plt.plot(run43['ADC'], run43['counts'], '-',linewidth=2, drawstyle='steps', label='+950VDC, R={:.2f}%'.format(run43_res))
# plt.plot(run44['ADC'], run44['counts'], '-',linewidth=2, drawstyle='steps', label='+1000VDC, R={:.2f}%'.format(run44_res))
# plt.plot(run45['ADC'], run45['counts'], '-',linewidth=2, drawstyle='steps', label='+1050VDC, R={:.2f}%'.format(run45_res))
# plt.plot(run46['ADC'], run46['counts'], '-',linewidth=2, drawstyle='steps', label='+1100VDC, R={:.2f}%'.format(run46_res))
# plt.plot(run47['ADC'], run47['counts'], '-',linewidth=2, drawstyle='steps', label='+1150VDC, R={:.2f}%'.format(run47_res))
# plt.plot(run48['ADC'], run48['counts'], '-',linewidth=2, drawstyle='steps', label='+1200VDC, R={:.2f}%'.format(run48_res))
# plt.plot(run49['ADC'], run49['counts'], '-',linewidth=2, drawstyle='steps', label='+1250VDC, R={:.2f}%'.format(run49_res))
# plt.plot(run50['ADC'], run50['counts'], '-',linewidth=2, drawstyle='steps', label='+1300VDC, R={:.2f}%'.format(run50_res))
plt.plot(BG['ADC'], BG['counts'], '-',linewidth=2, drawstyle='steps', label='Background')
plt.plot(BG_B['ADC'], BG_B['counts'], '-',linewidth=2, drawstyle='steps', label='Background /w Boron sleeve')
plt.plot(BG['ADC'], pmca.ratio(BG_B, BG, 'counts'),linewidth=2, label='ratio')

# plt.ylim((0.000001, 0.007))
# plt.xlim((0, 13000))
# plt.yscale('log')
plt.legend()#loc=1,ncol=1)
plt.xlabel('ADC [a.u.]')
plt.ylabel('Counts')
plt.title('Background measurements')#'Ortec 673 Shaper/Amp')
plt.grid()
plt.show()

# plt.figure()
# plt.plot(voltage, resolution, linewidth=2)
# plt.xlabel('Voltage [+VDC]')
# plt.ylabel('Resolution [%]')
# plt.title('12$\mu$s shaping time')#'Ortec 673 Shaper/Amp')
# plt.grid()
# plt.show()

