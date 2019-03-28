#http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow
# ----------------------------------------------------------
#          ███╗   ███╗ █████╗ ████████╗██╗  ██╗
#          ████╗ ████║██╔══██╗╚══██╔══╝██║  ██║
#          ██╔████╔██║███████║   ██║   ███████║
#          ██║╚██╔╝██║██╔══██║   ██║   ██╔══██║
#          ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║
#          ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
#  A library of methods for regular mathematical operations.
#       Some methods are based on pandas DataFrames
#       
#   Author: Nicholai Mauritzson 2018-2019
#           nicholai.mauritzson@nuclear.lu.se
# ----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

def gaussFitpd(df, col1=None, col2=None, start=0, stop=None):
    """
        1) Takes panda data frame 'df', 'col1' and 'col2' which are the names of the x and y values respectively.
           'start' and 'stop' define the range of the fit using 'col1'.
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

    popt, pcov = curve_fit(gaussFunc, x, y, p0 = [max(y), np.mean(y), sigmaTEMP])
    const = popt[0]
    mean = popt[1]
    sigma = popt[2]

    return const, mean, sigma

def gaussFit(x_input=None, y_input=None, start=0, stop=None):
    """
        1) Takes 'x_input' and 'y_input' values as lists, 'start' and 'stop' define the range of the fit in terms of 'x'.
        2) Fits the selected intervall with a Gaussian function 'gaussFunc()'.
        3) Returns the constant ('const'), mean value ('mean') and the standard diviation ('sigma') of the fit.
        4) Optional: If no values for start and stop are given then the default is to try and fit the entire spectrum.
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2019-03-27
    """
    if stop == None:
        stop = len(x_input)
    # if x_input == None or y_input == None:
        # raise ValueError('Argument(s) missing! No imput for col1 and/or col2 was given.')

    y = []
    x = []
    for i in range(len(x_input)):
        if x_input[i] >= start and x_input[i] <= stop:
            x.append(x_input[i]) #Get x-values in range to fit
            y.append(y_input[i]) #Get y-values in range to fit
    # x = [x for x in x_value if x <= stop and x >= start]
    # x = np.array(x>=start & x<=stop)]) 
    # y = np.array(df[col2][(df[col1]>=start) & (df[col1]<=stop)]) #Get y-values in range to fit

    # print(x)
    # print(y)
    place_holder = 0
    for i in range(len(x)):
        place_holder += x[i] * y[i]
    meanTEMP = place_holder/sum(y)
    place_holder = 0

    # meanTEMP = sum((lambda x,y:x*y,x,y) / sum(y))
    for i in range(len(x)):
        place_holder =+ (y[i] * (x[i] - meanTEMP)*(x[i] - meanTEMP))
    
    sigmaTEMP = np.sqrt(place_holder / sum(y))

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


def comptonMax(Ef):
    """
    Takes photon energy input and returns the maximum electron recoil energy
    -------------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2019-03-26
    """
    return 2*Ef**2/(0.5109989+2*Ef)


def comptonEdgeFit(data, col, min, max, Ef, fit_lim=None):
    """
    Method is designed to fit and return the edge positions and maximum electron recoil energy of a Compton distribution.
    ---------------------------------------------------------------------
    Inputs: 
    - 'data' : A pandas data frame
    - 'col' : The name of the column containg data to be fitted
    - 'min' : The minimum x-value (ADC) for fit
    - 'max' : The maximum x-value (ADC) for fit
    - 'Ef' : The photon energy in MeV
    - 'fit_lim' : Boundary parameters for scipy.optimize.curve_fit method (Guassian):
                  Format: fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]] 

    1) Creates a histogram, saves x,y data.
    2) Fits the data and prints to console optimal fit parameters.
    3a) Finds the compton edge @ 89% of compton edge maximum (Knox method).
    3b) Finds the compton edge @ 50% of compton edge maximum (Flynn method).
    4) Calculates the maximum electron recoil energy based on 'Ef'.
    5) Plots histogram of original data, Gaussian fit and markers for 89% and 50% of maximum.

    Return:
    Method returns array containing the 89% of maximum [ADC], 50% of maximum [ADC] and the calculated maximum electron recoil energy (MeV)
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2019-03-28
    """

    #1)____________MAKE HISTOGRAM AND EXTRACT X,Y VALUES___________
    N = np.histogram(data[col], range=[min, max], bins=(max-min)) #Create histogram and save x,y data.
    x = N[1][:-1]   #Set bin positions as x values
    y = N[0]        #Set bin heights as y values
    meanTEMP = np.max(y)#sum(x * y) / sum(y)  #Calcaulate the mean value of distibution. Used as first quess for Gaussian fit.
    sigmaTEMP = np.sqrt(sum(y * (x - meanTEMP)**2) / sum(y)) #Calculate stdev of distribution. Used as first geuss for Gaussian fit.

    #2)____________FITS HISTOGRAM DATA WITH GAUSSIAN_______________
    if fit_lim != None: #Check if boundary limits are enforced for Gaussian fit
        popt, pcov = curve_fit(gaussFunc, x, y, p0 = [np.max(y), meanTEMP, sigmaTEMP], bounds=fit_lim)
    else: #If not...
        popt, pcov = curve_fit(gaussFunc, x, y, p0 = [np.max(y), meanTEMP, sigmaTEMP])
    const, mean, sigma = popt[0], popt[1], popt[2] #saving result of fit as individual variables.
    fitError = np.sqrt(np.diag(pcov)) #Calculate and store the errors of each variable from Gaussian fit errors = [const_err, mean_err, sigma_err]

    #Print to console: optimal fitting parameters
    print('______________________________________________________________________')
    print('>>>> Optimal fitting parameters (Gaussian) <<<')
    print('Method: scipy.curve_fit(gaussianFunc, const, mean, sigma))')
    print('-> Maximum height.......const = %.4f (+/- %.4f)' % (const, fitError[0]))
    print('-> Mean position........mean = %.4f (+/- %.4f)' % (mean, fitError[1]))
    print('-> Standard deviation...sigma = %.4f (+/- %.4f)' % (sigma, fitError[2]))
    print('______________________________________________________________________')
    print() #Create vertical space on terminal

    #3a)___________FINDING COMPTON EDGE @ 89% OF MAX_______________Knox Method   
    for i in tqdm(np.arange(min, max, 0.0001), desc='Finding CE @ 89% of max'): #Loop for finding 89% of maximum with 4 decimal points
        if gaussFunc(i, const, mean, sigma)<=0.89*const:
            p = i #Saving compton edge value
            break
    else:
        p = np.nan
        print('FAILURE TO LOCATE CE @ 89%%...')
    
    #3b)___________FINDING COMPTON EDGE @ 50% OF MAX_______________Flynn Method
    for i in tqdm(np.arange(min, max, 0.0001), desc='Finding CE @ 50% of max'): #Loop for finding 50% of maximum with 4 decimal points
        if gaussFunc(i, const, mean, sigma)<=0.5*const:
            p2 = i #Saving compton edge value
            break
    else:
        p2 = np.nan
        print('FAILURE TO LOCATE CE @ 50%...')

    #4)____________MAXIMUM ELECTRON RECOIL ENERGY__________________
    E_recoil_max = comptonMax(Ef) #Calculate maximum electron recoil energy
    print()
    print('______________________________________________________________________')
    print('>>>> comptonEdgeFit() returned <<<<')
    print('-> 89%% Compton edge found at ADC value: %.4f' % p) #Printing compton edge value (ADC) to console
    print('-> 50%% Compton edge found at ADC value: %.4f' % p2) #Printing compton edge value (ADC) to console
    print('-> Maximum electron recoil energy: %.4f MeV' % E_recoil_max)
    print('______________________________________________________________________')

    #5)____________PLOTTING________________________________________ 
    x_long = np.arange(min, max, 0.0001) #Increse plotting points for Gaussian plot by 10000
    plt.hist(data.qdc_det0, range=(min-200, max+200), bins=(max-min+400), label='data') #Plot histogram of input data
    plt.plot(x_long, gaussFunc(x_long, const, mean, sigma), color='r', linewidth=3, label='Gaussian fit') #Plot Gaussian fit
    plt.plot(p, gaussFunc(p, const, mean, sigma), color='black', marker='o', markersize=10, label='Compton edge (89%)') #Mark 89% of maximum point
    plt.plot(p2, gaussFunc(p2, const, mean, sigma), color='green', marker='o', markersize=10, label='Compton edge (50%)') #Mark 50% of maximum point
    plt.legend()
    plt.show()

    return p, p2, E_recoil_max