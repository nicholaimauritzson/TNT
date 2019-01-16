import numpy as np
import pandas as pd
import csv
import sys
#from itertools import islice
import time
#import matplotlib.pyplot as plt
# from math import sqrt
# from math import atan

def load_data(filename, threshold, frac=0.3, nlines=0, startline=0, nTimesReset=0, no_skip=False, chunksize=2**18, outpath='data/chunk'):
    t0 = time.time() 
    print("Scanning the file to get number of chunks:")
    nChunks = int(round(0.5 + sum(1 for row in open(filename, "r"))/chunksize))
    t1 = time.time()
    print("Scan time: ",t1-t0, "seconds")
    print("Will generate ",nChunks, "chunks")

    Chunks = pd.read_csv(filename, header=None, usecols=[5,7], names=["timestamp", "samples"], chunksize=chunksize)
    
    count = 0
    tdummy1 = t1
    for df in Chunks:
        tdummy2 = tdummy1
        print("Chunk number", count + 1, "/", nChunks)
        df["samples"] = df.samples.str.split().apply(lambda x: np.array(x, dtype=np.int16))

        samples = np.array([None]*df.shape[0])
        timestamp = np.array([0]*df.shape[0], dtype=np.int64)
        amplitude = np.array([0]*df.shape[0], dtype=np.int16)
        peak_index = np.array([0]*df.shape[0], dtype=np.int16)
        valid_event = np.array([True]*df.shape[0], dtype=np.int16)
        ref_point_rise = np.array([0]*df.shape[0], dtype=np.int32)
        ref_point_fall = np.array([0]*df.shape[0], dtype=np.int32)
        nTimesReset = 0

        for i in range(0, df.shape[0]):
            u = chunksize*count + i
            k = round(100*i/df.shape[0])
            sys.stdout.write("\rGenerating dataframe %d%%" % k)
            sys.stdout.flush()

            Baseline = int(round(np.average(df.samples[u][0:20])))
            peak_index[i] = np.argmin(df.samples[u])

            #Check that only events above threshold are accepted and that first 20 samples can give a good baseline.
            if abs(df.samples[u][peak_index[i]] - Baseline) < threshold or (max(df.samples[u][0:20])-min(df.samples[u][0:20])) > 3:
                valid_event[i] = False
                continue
            else:
                samples[i] = df["samples"][u] - Baseline
                ref_point_rise[i], ref_point_fall[i] = cfd(samples=samples[i], frac=frac, peak_index=peak_index[i])
                amplitude[i] = samples[i][peak_index[i]]
                timestamp[i] = df["timestamp"][u]
                if i > 0:
                    if timestamp[i] < timestamp[i-1]-nTimesReset*2147483647:
                        nTimesReset += 1
                    timestamp[i] += nTimesReset*2147483647
        df["timestamp"] = timestamp
        df["samples"] = samples
        df["valid_event"] = valid_event
        df["amplitude"] = amplitude
        df["peak_index"] = peak_index
        df["ref_point_rise"] = ref_point_rise
        df["ref_point_fall"] = ref_point_fall
        df = df.query("valid_event == True").reset_index()
        df.to_hdf(outpath+".h5", key="key%s"%count)
        df = df.drop("samples", axis = 1)
        df.to_hdf(outpath+"cooked.h5", key="key%s"%count)
        tdummy1=time.time()
        print("chunk", count, "processed in", tdummy1-tdummy2, "seconds"  )
        count += 1
