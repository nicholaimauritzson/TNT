#!/usr/bin/env python3
"""

Accesses AquaDAQ "cooked" data files and plots ToF

"""

import uproot as ur # to access ROOT files
import pandas as pd

import matplotlib
matplotlib.use('Qt5Agg')  # nice, but issue with interactive use e.g. in
                          # Jupyter; see
                          # http://matplotlib.org/faq/usage_faq.html#what-is-a-backend
import matplotlib.pyplot as plt

import numpy as np
import scipy

import logging
import sys
import os

import argparse

#import fithelpers as fh

# from: https://stackoverflow.com/a/19361027
def plot_binned_data(axes, binedges, data, *args, **kwargs):
    #The dataset values are the bin centres
    x = (binedges[1:] + binedges[:-1]) / 2.0
    #The weights are the y-values of the input binned data
    weights = data
    return axes.hist(x, bins=binedges, weights=weights, *args, **kwargs)

def getBinCenters(bins):
    """ calculate center values for given bins """
    return np.array([np.mean([bins[i],bins[i+1]]) for i in range(0, len(bins)-1)])

def get_raw_df_from_file(file_name, branch_list):
    tfile = ur.open(file_name)
    # TODO verify that tree exists and report if not
    ttree = tfile.get("cooked_data")
    # now load all the above channels from the TTree
    data = ttree.arrays(branch_list)
    return pd.DataFrame(data)

def load_data(file_name):
    """
    loads and configures DataFrame from AquaDAQ root file ("cooked" format)
    """
    log = logging.getLogger('tof_analysis')  # set up logging
    # TODO extend to allow several detectors (list of lists and loop/flatten where needed)
    # TODO make the channel assignment configurable (e.g. from config file)
    column_config = {}
    # TODO use *one* dict with e.g. "qdc_yap:qdc_ch47" instead of the lists below
    # map AquaDAQ-internal names to more descriptive names used later for the columns
    column_config['qdc_det'] = ['qdc_ch0'] # n detector QDC long gate
    column_config['qdc_sg_det'] = ['qdc_ch32'] # n detector QDC short gate
    column_config['tdc_det0_yap'] = ['tdc_ch0', 'tdc_ch1', 'tdc_ch2', 'tdc_ch3'] # n detector/YAP TDC
    column_config['tdc_st_det'] = ['tdc_ch4'] # n detector self-timing TDC
    column_config['qdc_yap'] = ['qdc_ch44', 'qdc_ch45', 'qdc_ch46', 'qdc_ch47']
    # load data from file based on above channel list
    df = get_raw_df_from_file(file_name, [v for key, value in column_config.items() for v in value])
    # rename columns to replace ch# reference with content identifier
    # TODO: extend to include several detectors
    for key in column_config:
        df.rename(columns={k.encode(): "{}{}".format(key, i) for i,k in enumerate(column_config[key])}, inplace=True)
    mem_used = df.memory_usage(deep=True).sum()
    log.info("Approximately {} MB of memory used for data loaded from {}".format(round(mem_used/(1024*1024),2),file_name))
    # need to fix certain runs' TDC data (subtract 16384) due to an issue in AquaDAQ with the Caen 775

    log.info("Fixing TDC values (subtracting 16384)")
    # TODO: could this be done more efficiently with df.update(df.query().eval("tdc = ... ")) ?
    mask = df.filter(regex='tdc_det[0-9]_yap[0-9]').lt(16384) # find all TDC values < 16384
    df.update(df.filter(regex='tdc_det[0-9]_yap[0-9]').where(mask,other=df-16384)) # subtract

    return df

def calculate_ps(df):
    """
    calculates pulse-shape figure-of-merit from long and short gate QDCs. Requires prior pedestal subtraction.
    """
    # TODO: clean ps values from strange ones (e.g. nan)
    log = logging.getLogger('tof_analysis')  # set up logging
    log.info("Calculating pulse-shape values for detector 0")
    df['ps_det0'] = (df['qdc_det0']-df['qdc_sg_det0'])/df['qdc_det0']

def find_pedestal(df, col, max_ch = 200, min_amplitude = .4):
    """
    searches for pedestal in given dataframe column
    """
    log = logging.getLogger('tof_analysis')  # set up logging
    values, bins = np.histogram(df.loc[df.eval(f'{col}>0 and {col}<{max_ch}'),col], bins=range(1,max_ch))
    peaks = scipy.signal.find_peaks_cwt(values, [1,4,1])
    log.debug(f"Found pedestal candidates in {col}: {peaks}")
    # check peaks for first one with significant amplitude
    for p in peaks:
        if values[p]/np.max(values) > min_amplitude:
            return bins[p]
    log.warning(f"Could not find any pedestal candidate with significant amplitude in {col}")
    return -1

def find_all_pestals(df):
    """
    searches for pedestal in all dataframe columns whose name starts with with 'qdc'
    and plots the result.
    """
    # TODO: divide plotting and finding pedestals into two separate functions
    pedestals = {}
    for col in df.columns.values[df.columns.str.startswith('qdc')]:
        plt.figure()
        p = find_pedestal(df,col)
        pedestals[col] = p
        df.loc[df.eval(f"{col}>0 and {col}<200"),col].plot.hist(bins=range(1,200))
        plt.axvline(p,0,1,color="red")
        plt.title(col)
    plt.ylim(ymin=0.1)
    plt.yscale('log')
    return pedestals

def get_t0(tdc_calibration, distance, gfpos):
    """ returns T0 calculated from distance, gamma flash position and TDC time calibration (in ns/bin)"""
    speed_of_light = 2.99792458e8 # in [m/s]
    return gfpos + (distance / speed_of_light)/tdc_calibration

def fit_gf_in_hist(bins, values):
    fits = fh.fit_all_gaussians(getBinCenters(bins), values, npoints=4, widths = np.arange(1,5,1), logger=log.name)
    # remove Gaussians that are not narrow enough nor have a significant amplitude
    fits[:] = [f for f in fits if f.sigma < 1 and np.abs(1-(f.A/np.max(values))) < 0.5]
    return fits

def find_all_gf(df, roi, criteria=None):
    """
    plots TDC spectra and determines gamma flash positions
    inputs:
    df = dataframe
    roi = list of two values defining boundaries of region-of-interest
    criteria is an optional string to filter the elements shown using DataFrame.eval() syntax
    """
    log = logging.getLogger('tof_analysis')  # set up logging
    cols = df.columns.values[df.columns.str.startswith('tdc_det')]
    gf = {}
    for c in cols:
        # construct filter criteria for ROI
        if criteria is None:
            criteria = f"{c} > {roi[0]} and {c} < {roi[1]}"
        else:
            criteria = f"{criteria} and {c} > {roi[0]} and {c} < {roi[1]}"
        # create histogram
        vals, bins = np.histogram(df.loc[df.eval(criteria),c], bins=range(roi[0],roi[1]))
        log.info("Finding gamma flash peak in {}:".format(c))
        fits = fit_gf_in_hist(bins, vals)
        if len(fits) == 0:
            log.warning("No gamma flash candidates found in {} data".format(c))
        if len(fits)> 1:
            log.warning("More than one ({}) gamma flash candidate found in {} data".format(len(fits),c))
        for f in fits:
            log.info("Found gf at: {}".format(f.as_string()))
        gf[c] = fits
    return gf


def calculate_tof(df, gf, tdc_calibration, distance):
    """
    calculates time-of-flight from TDC spectrum, its calibration factor (in [ns]), distance, list of gamma flash positions and stores them in columns of the DataFrame
    """
    log = logging.getLogger('tof_analysis')  # set up logging
    cols = df.columns.values[df.columns.str.startswith('tdc_det')]
    for c in cols:
        gfpos = gf[c][0].mu # take first gaussian in list; TODO: make sure it exists first or handle exception!
        T0 = get_t0(tdc_calibration, distance, gfpos)
        tof_c = c.replace('tdc_', 'tof_')
        df[tof_c] = ((-1) * df[c] + T0) * tdc_calibration
        # values below 0 do not make sense; replace with NaN
        df.loc[df[tof_c]<0, tof_c] = np.nan

def plot_gf(df, roi, gf, criteria=None):
    fig, ax = plot_tdc(df,roi,criteria=criteria)
    gf = find_all_gf(df,roi,criteria=criteria)
    cols = df.columns.values[df.columns.str.startswith('tdc_det')]
    for c in cols:
        fits = gf[c]
        for f in fits:
            log.info(f.as_string())
            x = np.arange(f.mu-3*f.sigma, f.mu+3*f.sigma, 0.1)
            plt.plot(x, f.value(x), label=r'Gaussian fit, $\mu={}$, $\sigma={}$'.format(round(f.mu),round(f.sigma)))
    plt.legend()
    return fig, ax

def plot_tof(df, criteria=None):
    log = logging.getLogger('tof_analysis')  # set up logging
    fig, ax = plt.subplots()
    cols = df.columns.values[df.columns.str.startswith('tof_det')]
    for c in cols:
        if criteria is None:
            criteria = "{c} == {c}" # exclude nan values (which are not equal to themselves)
        else:
            criteria = "{criteria} and {c} == {c}"
        vals, bins = np.histogram(df.loc[df.eval(criteria),c], bins=100)
        plot_binned_data(ax, bins, vals, alpha = 0.75, label = "{}".format(c))
    plt.legend()
    plt.xlabel("neutron time-of-flight [ns]")
    plt.ylabel("counts")
    return fig, ax


def plot_col(df, col='tdc_det', roi=None, criteria=None, show_all=False):
    """
    plots all DataFrame columns matching '{col}*' with optional filter criteria applied
    inputs:
    df = dataframe
    col = string that matches start of column name(s) to use
    roi = list of two values defining boundaries of region-of-interest
    criteria is an optional string to filter the elements shown using DataFrame.eval() syntax
    """
    log = logging.getLogger('tof_analysis')  # set up logging
    fig, ax = plt.subplots()
    cols = df.columns.values[df.columns.str.startswith(col)]
    for c in cols:
        # handle (optional) parameters
        if roi is None:
            roi_filter = f"{c}=={c}" # True if not nan
            bins = 150
        else:
            roi_filter = f"{c} > {roi[0]} and {c} < {roi[1]}"
            bins = range(roi[0],roi[1])
        # use filter criteria for ROI and the user-specified parameter
        if criteria is not None:
            vals_mask, bins = np.histogram(df.loc[df.eval(f"{roi_filter} and {criteria}"),c], bins=bins)
            label = f"{c}, {criteria}"
        else:
            vals_mask, bins = np.histogram(df.loc[df.eval(f"{roi_filter}"),c], bins=bins)
            label = f"{c}"
        _n, _bins, patches = plot_binned_data(ax, bins, vals_mask, alpha = 0.75, label=label)
        # determine the color used for the plot:
        color = patches[0].get_facecolor()
        if show_all and criteria is not None:
            vals_all, bins = np.histogram(df.loc[df.eval(f"{roi_filter}"),c], bins=bins)
            vals_nomask, bins = np.histogram(df.loc[df.eval(f"not ({criteria}) and {roi_filter}"),c], bins=bins)
            # plot the full and inverted data sets (reusing the color used in the prev. plot)
            plot_binned_data(ax, bins, vals_all, alpha = 0.35, label = "{}, full".format(c), ls="-",color=color,histtype="step")
            plot_binned_data(ax, bins, vals_nomask, alpha = 0.5, label = "{}, inv. mask".format(c),ls="--",color=color,histtype="step")
    plt.legend()
    #plt.ylim(ymin=0.5)
    #plt.yscale('log')
    return fig, ax

def plot_qdc(df):
    log = logging.getLogger('tof_analysis')  # set up logging
    _f, _a = plt.subplots()
    cols = df.columns.values[df.columns.str.startswith('qdc_yap')]
    for c in cols:
        df[df[c]>0][c].plot.hist(bins = 150, label=c)
    plt.legend()
    plt.yscale('log')

    _f, _a = plt.subplots()
    cols = df.columns.values[df.columns.str.startswith('qdc_det')]
    for c in cols:
        df[df[c]>0][c].plot.hist(bins = 150, label=c)
    plt.legend()
    plt.yscale('log')

    _f, _a = plt.subplots()
    cols = df.columns.values[df.columns.str.startswith('qdc_sg_det')]
    for c in cols:
        df[df[c]>0][c].plot.hist(bins = 150, label=c)
    plt.legend()
    plt.yscale('log')

def plot_psd(df, detector = "det0", criteria = None):
    """
    plots pulse-shape discrimination plots for the DataFrame 'df'
    criteria is an optional string to filter the elements shown using DataFrame.eval() syntax
    """
    log = logging.getLogger('tof_analysis')  # set up logging
    # if no selection criteria was given, we take all elements except over flow bins
    if criteria is None:
        # set filter for cutting away QDC bins > 4100 (overflow bins)
        criteria = f"qdc_{detector} > 0 and qdc_{detector} < 4100"
    else:
        criteria = f"{criteria} and qdc_{detector} > 0 and qdc_{detector} < 4100"
    log.debug(f"Plotting PSD spectra with selection criteria: {criteria}")
    # generate 2D histograms
    # banana plot
    plt.figure()
    df.loc[df.eval(f'ps_{detector}<1 and {criteria}')].plot(f'qdc_{detector}',f'ps_{detector}', kind='hexbin',gridsize=(25,25))
    # TODO: the criteria are not applied to the SG side!!
    H, xedges, yedges = np.histogram2d(df.loc[df.eval(criteria), f'qdc_{detector}'], df.loc[df.eval(criteria),f'qdc_sg_{detector}'], bins=100)
    H = H.T  # Let each row list bins with common y range.
    import matplotlib.colors as colors
    norm = colors.LogNorm()
    _f, _a = plt.subplots()
    heatmap = plt.imshow(H, interpolation='nearest', cmap='inferno',
                         norm=norm, origin='low',
                         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.xlabel(f"QDC {detector}")
    plt.ylabel(f"QDC short-gate {detector}")
    cbar = plt.colorbar(heatmap)     # plot color legend
    # 1D plot with gate ratio
    _f, _a = plt.subplots()
    plt.hist(df.loc[df.eval(criteria),f'qdc_sg_{detector}']/df.loc[df.eval(criteria),f'qdc_{detector}'], bins=200)
    plt.xlabel(f"QDC ratio short/long gates {detector}")
    plt.ylabel("counts")
    # 1D plot with PS distribution
    _f, _a = plt.subplots()
    # filter out unreasonable values of PS (>1) that spoil the binning
    plt.hist(df.loc[df.eval(f'{criteria} and ps_{detector}<1'), f'ps_{detector}'], bins=100)
    plt.xlabel("pulse shape FOM")
    plt.ylabel("counts")
    # plot TDC versus PS
    # filter out unreasonable values of PS (>1) that spoil the binning
    H, xedges, yedges = np.histogram2d(df.loc[df.eval(f'{criteria} and ps_{detector}<1'), f'tdc_{detector}_yap0'], df.loc[df.eval(f'{criteria} and ps_{detector}<1'), f'ps_{detector}'], bins=100)
    H = H.T  # Let each row list bins with common y range.
    import matplotlib.colors as colors
    norm = colors.LogNorm()
    _f, _a = plt.subplots()
    heatmap = plt.imshow(H, interpolation='nearest', cmap='inferno',
                         norm=norm, origin='low',
                         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cbar = plt.colorbar(heatmap)     # plot color legend
    #plt.ylim(ymin=0.001)
    #plt.yscale('log')
    plt.xlabel(f"TDC {detector} [ch]")
    plt.ylabel("pulse shape FOM")


if __name__ == "__main__":
    log = logging.getLogger('tof_analysis')  # set up logging
    formatter = logging.Formatter('%(asctime)s %(name)s(%(levelname)s): %(message)s',"%H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    log.addHandler(handler_stream)
    # using this decorator, we can count the number of error messages
    class callcounted(object):
        """Decorator to determine number of calls for a method"""
        def __init__(self,method):
            self.method=method
            self.counter=0
        def __call__(self,*args,**kwargs):
            self.counter+=1
            return self.method(*args,**kwargs)
    log.error=callcounted(log.error)

    # command line argument parsing
    argv = sys.argv
    progName = os.path.basename(argv.pop(0))
    parser = argparse.ArgumentParser(
        prog=progName,
        description=
        "ToF analysis script" # TODO: make this more descriptive
    )
    parser.add_argument(
        "-l",
        "--log-level",
        default="info",
        help=
        "Sets the verbosity of log messages where LEVEL is either debug, info, warning or error",
        metavar="LEVEL")
    parser.add_argument(
        "-i",
        "--interactive",
        action='store_true',
        help=
        "Drop into an interactive IPython shell instead of showing default plots"
    )
    parser.add_argument(
        "file_name",
        help=
        "'cooked' ROOT file to open"
    )

    # parse the arguments
    args = parser.parse_args(argv)
    # set the logging level
    numeric_level = getattr(logging, "DEBUG",
                            None)  # set default
    if args.log_level:
        # Convert log level to upper case to allow the user to specify --log-level=DEBUG or --log-level=debug
        numeric_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            log.error('Invalid log level: %s' % args.log_level)
            sys.exit(2)
    log.setLevel(numeric_level)

    log.debug("Command line arguments used: %s ", args)
    log.debug("Libraries loaded:")
    log.debug("   - Matplotlib version {}".format(matplotlib.__version__))
    log.debug("   - Pandas version {}".format(pd.__version__))
    log.debug("   - Numpy version {}".format(np.__version__))

    log.info("Loading data from {}".format(args.file_name))
    df = load_data(args.file_name)

    if (args.interactive):
        print(" Interactive IPython shell ")
        print(" ========================= ")
        print(" Quick command usage:")
        print("  - 'who' or 'whos' to see all (locally) defined variables")
        print("  - if the plots are shown only as black area, run '%gui qt'")
        print("  - to make cmd prompt usable while plots are shown use 'plt.ion()' for interactive mode")
        import IPython
        IPython.embed()
    else:
        # analysis parameters
        # TODO: make these configurable through cmdline arguments
        tdc_roi = [1, 4000] # region of interest in TDC units
        distance = 1 # distance from source to detector in [m]
        seconds_per_TDC_channel = 0.265 # to convert from TDC bins to [ns]
        # plot TDC spectra
        log.info("Plotting TDC spectra from {} to {} TDC ADC".format(tdc_roi[0],tdc_roi[1]))
        plot_col(df=df, roi=tdc_roi, col="tdc_det0", show_all=True)

        # finding QDC pedestals and plot spectra
        #log.info("finding QDC pedestals and plot spectra")
        #find_all_pestals(df)

        # plot QDC spectra
        log.info("Plotting QDC values")
        plot_qdc(df)

        # plot PSD spectra
        # TODO: clean ps values from strange ones (e.g. nan)
        #calculate_ps(df)
        #log.info("Plotting det0 PSD spectra for YAP in TDC region-of-interest")
        #plot_psd(df, criteria = f'tdc_det0_yap0 > {tdc_roi[0]} and tdc_det0_yap0 < {tdc_roi[1]}')

        # finding gamma flash
        #log.info("Determining gamma-flash positions and calculating neutron ToF spectrum")
        #gf = find_all_gf(df, tdc_roi, criteria=f"ps_det0<{psd_cut}")
        # calculating and plotting ToF
        #calculate_tof(df, gf, seconds_per_TDC_channel, distance)
        #plot_tof(df, mask=df['ps_det0']>psd_cut)

        plt.show(block=False)  # block to allow "hit enter to close"
        plt.pause(0.001)  # <-------
        input("<Hit Enter To Close>")
        plt.close('all')

    if log.error.counter>0:
        log.warning("There were "+str(log.error.counter)+" error messages reported")
