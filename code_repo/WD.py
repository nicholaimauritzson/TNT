
#http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow
# -------------------------------------------------------------------
#       A collections of methods for importing and manipulating 
#                 outputs from CAEN WaveDump program.
#
#   Author: Rasmus Kjaer Hoier, Nicholai Mauritzson 2018-2019
#                               nicholai.mauritzson@nuclear.lu.se
# -------------------------------------------------------------------

import numpy as np
import pandas as pd
import csv
import sys
from itertools import islice
import time
# import matplotlib.pyplot as plt
from math import sqrt
from math import atan
# import dask.dataframe as dd
import dask as dd

def dask_chewer(filename, outpath):
    A = dd.read_csv(filename, header=None, names=['a','b','c','index','e','timestamp','g', 'samples'])
    A=A[['index','timestamp','samples']]
    A.compute()
    A.to_hdf(outpath, 'a')


def load_data(filename, threshold, frac=0.3, skip_badevents=True, chunksize=2**16, outpath='data/chunk', use_dask=False, daskoutpath='data/daskframe.h5'):
    """
        load_data()\nArguments and default inputs: 
        \nfilename: path to datafile, 
        \nthreshold: absolute value, unit ADC count, range 0 to 1023, 
        \nskip_badevents=True: Wether to skip events where baseline was noisy or threshold was not surpassed, 
        \nchunksize=2**16: the size of the chunks. for 8gB RAM 2**16-2**17 seems to be the limit, 
        \noutpath='data/chunk': path to outputfile location, 
        \nuse_dask=False, 
        \ndaskoutpath='data/daskframe.h5'):
        """
    t0 = time.time()
    if use_dask:
        print("processig file with dask")
        dask_chewer(filename, daskoutpath)
        print("opening dask generated dataframe")
        Chunks=pd.read_hdf(daskoutpath, chunksize=chunksize)
        t1 = time.time()
        print("dask processing time: ", t1-t0, ' seconds')
    else:
        print("Scanning the file to get number of chunks:")
        nChunks = int(round(0.5 + sum(1 for row in open(filename, 'r'))/chunksize))
        t1 = time.time()
        print("Scan time: ", t1-t0, ' seconds')
        print("Will generate ", nChunks)
        Chunks = pd.read_csv(filename, header=None, usecols=[5,7], names=['timestamp', 'samples'], chunksize=chunksize)
    count=0
    tdummy1=t1
    nTimesReset = 0
    
    for df in Chunks:
        df= df.reset_index()
        tdummy2=tdummy1
        print("Chunk number", count + 1, "/", nChunks)
        df['samples'] = df.samples.str.split().apply(lambda x: np.array(x, dtype=np.int16))

        #dummy variable used to compare consecutive timestamps. reset to zero when processing a new chunk of the df
        timestampDummy = 0
        #Arrays for the data we will put into the df chunks columns.
        samples = np.array([None]*df.shape[0])
        timestamp = np.array([0]*df.shape[0], dtype=np.int64)
        amplitude = np.array([0]*df.shape[0], dtype=np.int16)
        peak_index = np.array([0]*df.shape[0], dtype=np.int16)
        valid_event = np.array([True]*df.shape[0], dtype=np.int16)
        ref_point_rise = np.array([0]*df.shape[0], dtype=np.int32)
        ref_point_fall = np.array([0]*df.shape[0], dtype=np.int32)

        for i in range(0, df.shape[0]):
            #u = chunksize*count + i
            k = round(100*i/df.shape[0])
            sys.stdout.write("\rGenerating dataframe %d%%" % k)
            sys.stdout.flush()

            Baseline = int(round(np.average(df.samples[i][0:20])))
            peak_index[i] = np.argmin(df.samples[i])

            #Accept only only events above threshold and for which the first 20 samples can give a good baseline.
            if (skip_badevents==True) and (abs(df.samples[i][peak_index[i]] - Baseline) < threshold or (max(df.samples[i][0:20])-min(df.samples[i][0:20])) > 3):
                valid_event[i] = False
                continue
            else:
                #subtract baseline, get cfd refpoint and get pulse amplitude.
                samples[i] = df['samples'][i] - Baseline
                ref_point_rise[i], ref_point_fall[i] = cfd(samples=samples[i], frac=frac, peak_index=peak_index[i])
                amplitude[i] = samples[i][peak_index[i]]
                #Correct the timestamp resetting done by Wavedump at t=2**32
                if ((df['timestamp'][i] + nTimesReset*2147483647) < timestampDummy):
                    nTimesReset += 1
                timestamp[i] = df['timestamp'][i] + nTimesReset*2147483647
                timestampDummy = timestamp[i]


# def load_data_simple(path, numEnt, sign='positive'):
#     """ 
#         A quick and dirty pulse height generator for wavedump files. 
#                     !! VERY MEMORY INTENSIVE !!

#         1) Takes 'path' to WaveDump txt file (string) and number of entries 'numEnt' (int), were
#             numEnt is the number of samples point for each event. The polarity of the signal is signified by
#             the third entry 'sign' as 'positive' (default) or 'negative'.
#         2) Finds the largest value for each event (pulse height) and stores these in an array.
#         3) Saves the array in 'path' as 'ph.pkl'.
#         NOTE Assumes a file format with NO headers between events. 
#         ---------------------------------------------------------------------
#         Nicholai Mauritzson
#         Edit: 2019-01-17
#     """

#     print('Loading raw data...')
#     df = pd.read_csv(path)
#     print('Load complete...')
#     ph = [] #Pre allocation of memory for pulse height array.
#     idx = 0
#     if sign=='positive':
#         for evt in range(round(len(df)/numEnt)): #Loop through each event and 
#             print('Event number:', evt)
#             ph.append(df[idx:idx+numEnt].max().item())
#             idx += numEnt
            
#     elif sign=='negative':
#         for evt in range(round(len(df)/numEnt)): #Loop through each event and 
#             print('Event number:', evt)
#             ph.append(abs(df[idx:idx+numEnt].min().item()))
#             idx += numEnt

#     pickle.dump(ph, open("ph.pkl", "wb")) #Save pulse height array as pickle file.


# def load_data(filename, threshold, frac=0.3, nlines=0, startline=0, nTimesReset=0, no_skip=False, chunksize=2**18, outpath='data/chunk'):
#     """
#         Method needs general documenting.

#         Inputs: What are the inputs and what do they do/represent? Explain each one and any relevant expections which exists.

#         ---------------------------------------------------------------------
#         Nicholai Mauritzson (nicholai.mauritzson@nuclear.lu.se)
#         Edit: 2019-01-16
#         [original code by Rasmus Kjaer Hoier 2018-2019]
#     """
#     t0 = time.time() 
#     print("Scanning the file to get number of chunks:")
#     nChunks = int(round(0.5 + sum(1 for row in open(filename, "r"))/chunksize))
#     t1 = time.time()
#     print("Scan time: ",t1-t0, "seconds")
#     print("Will generate ",nChunks," chunks")
#     Chunks = pd.read_csv(filename, header=None, usecols=[5,7], names=["timestamp", "samples"], chunksize=chunksize)
    
#     count = 0 # Counter for numbering chunks for loop add +1 for itterating over 
#     tdummy1 = t1
    
#     # Looping through all chunks and saving/calculating relevant data
#     for df in Chunks:
#         tdummy2 = tdummy1
#         print("Chunk number", count + 1, "/", nChunks)
#         df["samples"] = df.samples.str.split().apply(lambda x: np.array(x, dtype=np.int16))

#         # -------------------------------------------------------
#         # This section of code seems to preallocated memory for the 
#         # various lists which will go into the final DataFrame.
        
#         # QUESTION: why is this run over and over in the loop? To clear the previous chunk's data?
#         samples = np.array([None]*df.shape[0])
#         timestamp = np.array([0]*df.shape[0], dtype=np.int64)
#         amplitude = np.array([0]*df.shape[0], dtype=np.int16)
#         peak_index = np.array([0]*df.shape[0], dtype=np.int16)
#         valid_event = np.array([True]*df.shape[0], dtype=np.int16)
#         ref_point_rise = np.array([0]*df.shape[0], dtype=np.int32)
#         ref_point_fall = np.array([0]*df.shape[0], dtype=np.int32)
#         nTimesReset = 0
#         # -------------------------------------------------------

#         # -------------------------------------------------------
#         # QUESTION: What are "u"and "k"?
#         for i in range(0, df.shape[0]):
#             u = chunksize*count + i
#             k = round(100*i/df.shape[0])
#             sys.stdout.write("\rGenerating dataframe %d%%" % k)
#             sys.stdout.flush()

#             Baseline = int(round(np.average(df.samples[u][0:20])))
#             peak_index[i] = np.argmin(df.samples[u])

#             #Check that only events above threshold are accepted and that first 20 samples can give a good baseline.
#             if abs(df.samples[u][peak_index[i]] - Baseline) < threshold or (max(df.samples[u][0:20])-min(df.samples[u][0:20])) > 3:
#                 valid_event[i] = False
#                 continue
#             else:
#                 samples[i] = df["samples"][u] - Baseline
#                 ref_point_rise[i], ref_point_fall[i] = cfd(samples=samples[i], frac=frac, peak_index=peak_index[i])
#                 amplitude[i] = samples[i][peak_index[i]]
#                 timestamp[i] = df["timestamp"][u]
#                 if i > 0:
#                     if timestamp[i] < timestamp[i-1]-nTimesReset*2147483647: #QUESTION: What is this number? Is it OK for hardcode?
#                         nTimesReset += 1
#                     timestamp[i] += nTimesReset*2147483647
        
#         # -------------------------------------------------------
#         # Section of code for saving all collected and calculated information in
#         # relevant DataFrame columns.
#         df["timestamp"] = timestamp
#         df["samples"] = samples
#         df["valid_event"] = valid_event
#         df["amplitude"] = amplitude
#         df["peak_index"] = peak_index
#         df["ref_point_rise"] = ref_point_rise
#         df["ref_point_fall"] = ref_point_fall
#         df = df.query("valid_event == True").reset_index()
        
#         # -------------------------------------------------------
#         # Section for saving DataFrame
#         df.to_hdf(outpath+".h5", key="key%s"%count)
#         df = df.drop("samples", axis = 1)
#         df.to_hdf(outpath+"cooked.h5", key="key%s"%count)
#         # -------------------------------------------------------
#         tdummy1=time.time()
#         print("chunk", count, "processed in", tdummy1-tdummy2, "seconds"  ) 
#         count += 1  
#         # END OF CHUNK LOOP
#         # -------------------------------------------------------


# def basic_framer(filename, threshold, frac=0.3, nlines=0, startline=0, nTimesReset=0, no_skip=False):
#     #Get number of lines
#     if nlines == 0:
#         nlines=sum(1 for line in (open(filename)))
#     nevents = int(nlines/8)
#     samples = [None]*nevents
#     timestamp = np.array([0]*nevents, dtype=np.int64)
#     refpoint_rise = np.array([0]*nevents, dtype=np.int32)
#     refpoint_fall = np.array([0]*nevents, dtype=np.int32)
#     peak_index = np.array([0]*nevents, dtype=np.int16)
#     height = np.array([0]*nevents, dtype=np.int16)
#     acq_window = 0
#     try:
#         with open(filename, newline='\n') as f:
#             reader = csv.reader(f)
#             line_index = startline
#             event_index = 0
#             idx=0
#             #Scan forwards through the file until you are at the provided start time
#             if line_index > 0:
#                 for row in reader:
#                     if idx == startline-1:
#                         break
#                     else:
#                         idx+=1
#             #go to the startline
#             for row in reader:
#                 line_index +=1
#                 #only go through lines belonging to the current block
#                 if line_index >= startline+nlines:
#                     break
#                 if line_index%8 == 6:
#                     k = 100*(line_index-startline)/nlines+1
#                     sys.stdout.write("\rGenerating dataframe %d%%" % k)
#                     sys.stdout.flush()
#                     dummytimestamp = int(row[0].split()[3])
#                     if event_index > 0:
#                         if dummytimestamp < timestamp[event_index-1]-nTimesReset*2147483647:
#                             nTimesReset += 1
#                             #print('\ntime reset!\n')
#                     timestamp[event_index]= (dummytimestamp+nTimesReset*2147483647)
#                 if line_index%8 == 0:#every eigth row is the data
#                     dummy = row[0].split()
#                     dummy=[int(i) for i in dummy]
#                     samples[event_index] = np.array(dummy ,dtype=np.int16)
#                     #B is the number of samples used to calculate baseline
#                     #We don't care about events that have large peaks or noise in this interval
#                     B = 20
#                     baseline = int(sum(samples[event_index][0:B])/B)
#                     samples[event_index] -= baseline
#                     #check the polarity and check if the pulse crosses threshold and if it is properly contained
#                     peak_index[event_index] = np.argmax(np.absolute(samples[event_index]))
#                     if (np.absolute(samples[event_index][peak_index[event_index]]) < threshold and no_skip==False):
#                         continue
#                     else:
#                         if samples[event_index][peak_index[event_index]] < 0:
#                             samples[event_index] *= -1
#                         #get pulse height and pulse edge bins
#                         height[event_index] = samples[event_index][peak_index[event_index]]
#                         refpoint_rise[event_index], refpoint_fall[event_index] = cfd(samples[event_index], frac, peak_index[event_index])
#                         #throw away events marked problematic by cfd alg. and events without room for tail.
#                         if ((refpoint_rise[event_index]<0 and no_skip==False) or  (refpoint_fall[event_index]<0 and no_skip==False)):
#                             continue
#                         event_index += 1
#         #throw away empty rows.
#         samples = samples[0:event_index]
#         timestamp = timestamp[0:event_index]
#         height = height[0:event_index]
#         peak_index = peak_index[0:event_index]
#         refpoint_rise = refpoint_rise[0:event_index]
#         refpoint_fall = refpoint_fall[0:event_index]
#     except IOError:
#         return None
#     return pd.DataFrame({'timestamp': timestamp,
#                          'samples' : samples,
#                          'height' : height,
#                          'peak_index':peak_index,
#                          'refpoint_rise' : refpoint_rise,
#                          'refpoint_fall' : refpoint_fall}), nTimesReset

# def get_gates(frame, lg=500, sg=55, offset=10):
#     longgate=np.array([0]*len(frame), dtype=np.int16)
#     shortgate=np.array([0]*len(frame), dtype=np.int16)
#     pulsetail=np.array([0]*len(frame), dtype=np.int16)
#     theta=np.array([0]*len(frame), dtype=np.int16)
#     #species=np.array([-1]*len(frame), dtype=np.int8)
#     for i in range(0, len(frame)):
#         k = round(100*i/len(frame))
#         sys.stdout.write("\rCalculating gates %d%%" % k)
#         sys.stdout.flush()

#         #start = int(round(frame.refpoint_rise[i]/1000))-offset
#         start = frame.peak_index[i]-offset
#         longgate[i] = np.trapz(frame.samples[i][start:start+lg])
#         shortgate[i] = np.trapz(frame.samples[i][start:start+sg])
#         theta[i] = atan(2*frame.height[i]/frame.refpoint_fall[i]-frame.refpoint_rise[i])

#         #send weird events to quarantine bins
#         if shortgate[i]>longgate[i]:
#             #workaround. need to deal with reflections properly!
#             longgate[i]=20000
#             shortgate[i]=20000
#         if longgate[i]<=0 or shortgate[i]<=0:
#             longgate[i]=20000
#             shortgate[i]=20000

#         #tail
#         pulsetail[i] = np.trapz(frame.samples[i][int(frame.refpoint_fall[i]/1000):int(frame.refpoint_fall[i]/1000)+lg])

#     frame['ps'] = (longgate-shortgate)/longgate
#     frame['longgate']=longgate
#     frame['shortgate']=shortgate
#     frame['pulsetail']=pulsetail
#     frame['theta']=theta
#     return 0

# def get_species(df, X=[0, 1190,2737, 20000], Y=[0, 0.105, 0.148, 0.235]):
#     species=np.array([-1]*len(df), dtype=np.int8)
#     #loop through pulses
#     for n in range(0, len(df)):
#         k = round(100*n/len(df))
#         sys.stdout.write("\rGetting species %d%%" % k)
#         sys.stdout.flush()
#         #If we are to the left of the exclusion zone
#         if df.longgate[n]<X[1]:
#             #inside exclusion box=>indistinguishable
#             if df.ps[n]<Y[1]:
#                 species[n]=-1
#                 #above exclusion box=>neutron
#             else:
#                 species[n]=1
#         #If we are to the right of the exclusion zone
#         #then loop through coordinates
#         elif df.longgate[n]>=X[1]:
#             for i in range(1,len(X)):
#                 #find the interval the pulse belongs to
#                 if df.longgate[n]<X[i]:
#                     if X[i]>=X[1]:
#                         #are we below(gamma) or above(neutron) of the discrimination line
#                         if df.ps[n]<Y[i-1]+(df.longgate[n]-X[i-1])*(Y[i]-Y[i-1])/(X[i]-X[i-1]):
#                             species[n] = 0
#                         else:
#                             species[n] = 1
#                         break
#     df['species'] = species

# def cfd(samples, frac, peak_index):
#     peak = samples[peak_index]
#     print('frac*peak = %d0'%(peak*frac))
#     rise_index = 0
#     fall_index = 0
#     #find the cfd rise point
#     for i in range(0, peak_index):
#         if samples[i] < frac * peak:
#             rise_index = i
#             break
#         else:
#             rise_index = 0
#         #find the cfd fall point
#         for i in range(peak_index, len(samples)):
#             if samples[i] > frac*peak:
#                 fall_index = i
#                 break
#             else:
#                 fall_index = 0
#         slope_rise = (samples[rise_index] - samples[rise_index-1])#divided by 1ns
#         slope_fall = (samples[fall_index] - samples[fall_index-1])#divided by 1ns
#         #slope equal 0 is a sign of error. fx a pulse located
#         #in first few bins and already above threshold in bin 0.
#         #rise
#         if slope_rise == 0:
#             print('\nslope == 0!!!!\nindex=', rise_index,'\n', samples[rise_index-5:rise_index+5])
#             tfine_rise = -1
#         else:
#             tfine_rise = 1000*(rise_index-1) + int(round(1000*(peak*frac-samples[rise_index-1])/slope_rise))
#             #fall
#         if slope_fall == 0:
#             print('\nslope == 0!!!!\nindex=', fall_index,'\n', samples[fall_index-5:fall_index+5])
#             tfine_fall = -1
#         else:
#             tfine_fall = 1000*(fall_index-1) + int(round(1000*(peak*frac-samples[fall_index-1])/slope_fall))
#         return tfine_rise, tfine_fall

# def cfd(samples, frac, peak_index):
#     peak = samples[peak_index]
#     rise_index = 0
#     fall_index = 0
#     #find the cfd rise point
#     for i in range(0, peak_index):
#         if samples[i] < frac * peak:
#             rise_index = i
#             break
#         else:
#             rise_index = 0
#     #find the cfd fall point
#     for i in range(peak_index, len(samples)):
#         if samples[i] > frac*peak:
#             fall_index = i
#             break
#         else:
#             fall_index = 0
#     slope_rise = (samples[rise_index] - samples[rise_index-1])#divided by 1ns
#     slope_fall = (samples[fall_index] - samples[fall_index-1])#divided by 1ns
#     #slope equal 0 is a sign of error. fx a pulse located
#     #in first few bins and already above threshold in bin 0.
#     #rise
#     if slope_rise == 0:
#         print('\nslope == 0!!!!\nindex=', rise_index,'\n', samples[rise_index-5:rise_index+5])
#         tfine_rise = -1
#     else:
#         tfine_rise = 1000*(rise_index-1) + int(round(1000*(peak*frac-samples[rise_index-1])/slope_rise))
#     #fall
#     if slope_fall == 0:
#         print('\nslope == 0!!!!\nindex=', fall_index,'\n', samples[fall_index-5:fall_index+5])
#         tfine_fall = -1
#     else:
#         tfine_fall = 1000*(fall_index-1) + int(round(1000*(peak*frac-samples[fall_index-1])/slope_fall))
#     return tfine_rise, tfine_fall

# def get_frames(filename, threshold, frac=0.3, no_skip=False, outpath='/home/rasmus/Documents/ThesisWork/code/tof/data/'):
#     time0 = time.time()
#     nlines=sum(1 for line in (open(filename)))
#     nlinesBlock = 2**21 # lines per block
#     #Number of full blocks
#     nBlocks = int(nlines/nlinesBlock)
#     #Number of lines in the final block
#     nlinesBlockF = (nlines%nlinesBlock)
#     Blocklines =[nlinesBlock]*(nBlocks+1)
#     Blocklines[-1] = nlinesBlockF
#     #we need nBlocks +1 dataframes
#     #FrameList=[0]*len(Blocklines)
#     nTimesReset = 0
#     for i in range(0, (nBlocks+1)):
#         print('\n -------------------- \n frame', i+1, '/', (nBlocks+1), '\n --------------------')
#         Frame, nTimesReset = basic_framer(filename, threshold, frac, nlines=Blocklines[i], startline=i*nlinesBlock, nTimesReset=nTimesReset, no_skip=no_skip)
#         get_gates(Frame)
#         get_species(Frame)
#         if outpath!='':
#             Frame.to_hdf(outpath+'%s.h5'%i, 'a')
#     time1 = time.time()
#     deltaT=time1-time0

#     #make the cooked frame
#     D=[0]*(nBlocks+1)
#     for i in range(0, (nBlocks+1)):
#         D[i]=pd.read_hdf(outpath+'%d.h5'%i)
#         D[i].drop('samples', axis=1)
#     D=pd.concat(D).reset_index()
#     D.to_hdf(outpath+'_cooked.h5', 'a')

#     print('Runtime: ', deltaT/60, 'minutes')
#     return 0

# def tof_spectrum(ne213, yap, fac=8, tol_left=0, tol_right=120):
    # ymin=0
    # tof_hist = np.histogram([], tol_left+tol_right, range=(tol_left, tol_right))
    # dt=np.array([0]*len(ne213), dtype=np.int32)
    # #tof_hist1 = np.histogram([], tol_left+tol_right, range=(tol_left, tol_right))
    # #tof_hist2 = np.histogram([], tol_left+tol_right, range=(tol_left, tol_right))

    # #for ne in range(0, len(ne213)):
    # counter=0
    # for row in ne213.itertuples():
    #     ne=row[0]
    #     counter += 1
    #     k = 100*counter/len(ne213)
    #     sys.stdout.write("\rGenerating tof spectrum %d%%" % k)
    #     sys.stdout.flush()
    #     for y in range(ymin, len(yap)):
    #         Delta=int(round(((fac*1000*ne213.timestamp[ne]+ne213.refpoint_rise[ne])-(fac*1000*yap.timestamp[y]+yap.refpoint_rise[y]))/1000))
    #         if Delta > tol_right:
    #             ymin = y
    #         if tol_left <= Delta < tol_right:
    #             tof_hist[0][tol_left+int(Delta)] += 1
    #             if dt[ne] == 0:
    #                 dt[ne]=Delta
    #             else:
    #                 print('Multiple matches!!! taking the first one!')
    #         elif Delta < -tol_right:
    #             break
    #     ne213['dt']=dt
    # return tof_hist
