#!/usr/bin/env

import numpy as np
from matplotlib import pyplot as plt


table_path = '/Users/rfinn/Dropbox/Research/APPSS/SDSSphot/run_sep14/'
latest_run = 'a100.code12.SDSSvalues170914.csv'
infile = table_path+latest_run

sdss = np.recfromcsv(infile)

agc_cross = 'a100.sdsscross.code12.170914.csv'
agc = np.recfromcsv(table_path+agc_cross)


# plot a color-mag diagram
def plotugvsr():
    color = sdss['modelmag_u'] - sdss['modelmag_g']
    plt.figure()
    plt.scatter(sdss['modelmag_r'],color,s=60,c=sdss['petror90_r'])

