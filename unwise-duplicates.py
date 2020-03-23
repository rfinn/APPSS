#!/usr/bin/python

import numpy as np
import os
import sys

from astropy.io import fits, ascii
from astropy.table import Table, join, hstack, Column, MaskedColumn
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.visualization import simple_norm

from astroquery.skyview import SkyView

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from urllib.parse import urlencode
from urllib.request import urlretrieve

import pandas as pd

homedir = os.getenv('HOME')
sys.path.append(homedir+'/github/Virgo/programs/')
from create-super-sample import getlegacy, getlegacyimage



# read in a100-sdss-wise table
cat = fits.getdata(homedir+'/research/tables/a100-sdss-wise.fits')

# identify a100 galaxies with multiple matches

unique, counts = np.unique(cat['AGC'],return_counts=True)
print('number of AGC with multiple unWISE matches = ',sum(counts > 1))
print('')
print('AGC number of sources with multiple unWISE matches = ')
print(unique[counts > 1])

# create postage stamp images, with positions of a100 and unWISE sources marked

