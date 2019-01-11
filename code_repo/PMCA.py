import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nicholai_math as nm

# ------------------------------------------------------------------------------
# Library of methods for manipulation of data files from MCA8000D (Pocket MCA)
# By Nicholai Mauritzson
# ------------------------------------------------------------------------------

def importData(path):
    """
        -------- PMCA Method (pocket MCA) for manipulation of data. ---------
        1) Takes a file path to .mca file as input. This is data from the pocketMCA.
        2) Reads in the data and finds the live time, ADC channels and number of counts for each.
        3) Normalizes the number of counts with the live time to get rate in 1/s for each ADC channel.
        4) Saves a .pkl (pickle) containg the 'df' with four columns: 'ADC', 'counts', 'rate' and 'live_time'
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2018-12-12
    """
    print('~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~')
    print('-> RUNNING: PMCA_import()')
    print('-> PATH: {}'.format(path))

    f = open(path, encoding='latin-1') #Opens file in 'path' and makes sure formatting is readable.
    dataIdx = [0,0] #Counter for number of data entries of file. Contains [start+1, stop-1] index of data set.
    dataTemp = [] #Temporary holder for data, will be written to pandas data frame at the end of method.
    with f as fp:  
        for cnt, line in enumerate(fp):

            if '<<END>>' in line:
                dataIdx[1] = cnt - 1 #Save index of start of data set.

            if dataIdx[0] != 0: #Check that data is being read from f.
                if dataIdx[1] == 0: #Save data only until end of data has been found.
                    dataTemp.append(float(line.strip('\n')))

            if '<<DATA>>' in line:
                dataIdx[0] = cnt+1 #Saves index of end of data set.

            if "LIVE_TIME" in line: #Finds the line with live time.
                live_time = line.split(' - ') #Splits string containing the live time of measurement
                del live_time[0] 
                live_time[-1] = live_time[-1].strip() #Removed 'new line' command (/n) at the end of string in string.
    
    print('-> NUMBER OF DATA ENTRIES: {:.4f}'.format(sum(dataTemp)))
    print('-> LIVE_TIME [s]: {}'.format((live_time[0])))
    print('-> RATE [s^-1]: {:.4f}'.format(sum(dataTemp)/float(live_time[0])))
    print('~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~')

    #Saving data to pandas data frame.
    df = pd.DataFrame(columns=['ADC', 'counts', 'rate', 'LT']) #Create pandas data frame.
    df['ADC'] = range(dataIdx[1]-dataIdx[0]+1) #Saving channel numbers ('ADC').
    df['counts'] = dataTemp #Saving data ('counts').
    df['rate'] = df['counts'].apply(lambda x: x/float(live_time[0]))
    # df['rate'] = df.loc['counts']/) #Calulates rate for each ADC value based on the 'counts' column.
    df.at[0,'LT'] = float(live_time[0]) #Save the live time as first entry in its own column.
    pickle.dump(df, open(path.strip('.mca')+'.pkl', 'wb'))

def customNorm(df, normConst=1): #WORK IN PROGRESS
    """
    -------- PMCA Method (pocket MCA) for manipulation of data. ---------
    1) Takes pandas data frame 'df' as input. Format given by 'pocketMCA_import()' method. 
    2) Normalizes the counts in each ADC channel with the live time of the run.
    3) 'normConst' is the normalization constant which will be used to normalize each value in the 'counts' column.
    4) Returns modified data frame with one new column: 'customNorm' in units 1/s.
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2018-12-11
    """
    live_time = df['live_time'][0]
    df['customNorm'] = 0
    df['rate'] = df['counts']/normConst
    
    # df.at()

    return df

def he3Shift(df, fullAbsPos):

    """
        1) Takes pandas data frame 'df' as input.
        2) Finds full absorption peak, fits it with a gaussian function 'gaussFit()'.
        3) Shifts the 'ADC' values so that the full absorptions peak of the spectrum ends up centered on the 'fullAbsPos' value.
        4) Returns shifted He-3 spectrum as df.
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2018-12-11
    """
    print('~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~')
    print('-> RUNNING: PMCA_he3Shift()')

    size = len(df)
    print('-> NUMBER OF DATA ENTRIES: {}'.format(size))
    maxValuePos = df.loc[220:size]['counts'].idxmax() #The the ADC value of the maximum counts peak (full absorption)
    const, mean, sigma = nm.gaussFit(df, 'ADC', 'counts', maxValuePos-100, maxValuePos+100) #Fit the entire spectrum
    coeff = fullAbsPos/mean
    df['ADC'] = df['ADC'].apply(lambda x: x*coeff)
    
    print('~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~')
    return df

def ratio(df1, df2, col):
    """
        1) Takes two pandas data frame 'df1' and 'df2' as inputs.
        2) Calculated the ratio between each data value in column 'col' as df1(i)/df2(i).
           NOTE: if df1(i) = 0 then ratio(i) = 0
           NOTE: if df2(i) = 0 then ratio(i) = 0
        3) Returns list 'ratio'
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2019-01-11
    """
    ratio = []
    for i in range(len(df1)):
        if df1[str(col)][i] == 0 or df2[str(col)][i] == 0: #Treat nominator and denominator = 0 instances.
            ratio.append(0)
        else:
            ratio.append(df1[str(col)][i] / df2[str(col)][i])
    return ratio