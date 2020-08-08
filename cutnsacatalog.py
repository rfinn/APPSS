#!/usr/bin/env python

'''
GOAL:
- read in NSA and gswlc catalogs
- save version with cz < 20000

INPUT:
- nsa catalog
- gswlc catalog

OUTPUT:
- cut nsa catalog
- cut gswlc catalog

'''

import os
from astropy.table import Table

homedir = os.getenv("HOME")
outdir = homedir+'/research/APPSS/tables/'
cat = homedir+'/research/NSA/nsa_v1_0_1.fits'
nsa = Table.read(cat)
flag = nsa['Z']*3.e5 < 2.e4
nsa = nsa[flag]
nsa.write(outdir+'nsa_v1_0_1_vmax20k.fits',overwrite=True)


cat = outdir+'/gswlc-A2-sdssphot.v2.fits'
gsw = Table.read(cat)
flag = gsw['Z']*3.e5 < 2.e4
gsw = gsw[flag]
gsw.write(outdir+'gswlc-A2-sdssphot.v2.vmax20k.fits',overwrite=True)
