import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------------------------------------
#  __  __       _   _     
# |  \/  | __ _| |_| |__  
# | |\/| |/ _` | __| '_ \ 
# | |  | | (_| | |_| | | |
# |_|  |_|\__,_|\__|_| |_|                      
# ----------------------------------------------------------
# Library of methods for regular mathematical operations. 
# Some methods are based on pandas DataFrames.
# 
# By Nicholai Mauritzson
# ----------------------------------------------------------

def simpleIntegrate(df, col=None, start=0, stop=None): #WORK IN PROGRESS
    """
        'col' is the column name containing the data to be integrated.
        Option: if no index value is given to 'stop'. Integration will go until 
                last index value of column.
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2018-12-12
    """
    if stop==None:
        stop = len(df[col])

    return df.loc[start:stop][col].sum()

def resCalc(mean, sigma):
    """
    Takes a mean 'mean' and standard diviation 'sigma' of a distribution as input.
    Returns the resolution (decimal form) at 'mean' value.
    NOTE: method assumes a Gaussian distribution.
    """
    return (sigma*2*np.sqrt(2*np.log(2)))/mean

def gaussFit(df, col1=None, col2=None, start=0, stop=None):
    """
        1) Takes panda data frame 'df', 'col1' and 'col2' which are the names of the x and y values respectively.
           'start' and 'stop' define the range in with values in terms of 'col1' for the fit.
        2) Fits the selected intervall with a Gaussian function 'gaussFunc()'.
        3) Returns the constant ('const'), mean value ('mean') and the standard diviation ('sigma') of the fit.
        4) Optional: If no values for start and stop are given then the default is to try and fit the entire spectrum.
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2018-12-12
    """
    if stop == None:
        stop = len(df)
    if col1 == None or col2 == None:
        raise ValueError('Argument(s) missing! No imput for col1 and/or col2 was given.')

    x = np.array(df[col1][(df[col1]>=start) & (df[col1]<=stop)]) #Get x-values in range to fit
    y = np.array(df[col2][(df[col1]>=start) & (df[col1]<=stop)]) #Get y-values in range to fit


    meanTEMP = sum(x * y) / sum(y)
    sigmaTEMP = np.sqrt(sum(y * (x - meanTEMP)**2) / sum(y))

    popt, pcov = curve_fit(gaussFunc, x, y, p0 = [max(y), meanTEMP, sigmaTEMP])
    const = popt[0]
    mean = popt[1]
    sigma = popt[2]

    return const, mean, sigma

def gaussFunc(x, a, x0, sigma):
    """
        A generic Gaussian function.
        - 'x' is the varable data (list format).
        - 'a' is the constant.
        - 'x0' is the mean value of distribution position.
        - 'sigma' is the standard diviation of the distribution.
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2018-12-11
    """

    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def ratio(df1, df2, col):
    """
        1) Takes two pandas data frames 'df1' and 'df2' as inputs and name of column 'col' to use.
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