#!/usr/bin/env python

'''
GOAL: create table for paper

column 1 - flag for good photometry photFlag gi2
column 2 - distance in Mpc,
column 3- HI mass log10MHI in M,
column 4 - heliocentric velocity of the HI profile midpoint,
column 5 - internal extinction coefficient in g band γg calculated as explained in section 2.1,
column 6 - internal extinction coefficient in i band γi calculated using the method described in section 2.1,
column 7- (g-i) color obtained using modelMag magnitudes
from SDSS, corrected for both galactic and internal extinction, as explained in section 2.1,
column 8 - absolute magnitude in i band Mi obtained using cModelMag i corrected for both galactic and
internal extinction,
column 9 - stellar mass (using Taylor et al. (2011), as explained in section 3).

'''

from astropy.io import fits
from astropy.table import Table

## READ IN A100-SDSS TABLE

filename = '/Users/rfinn/github/APPSS/tables/a100-sdss.fits'
a100 = fits.getdata(filename)

## CREATE NEW TABLE WITH SELECTED COLUMNS
#newphotflag = 
colnames = ['AGC_number','RA', 'DEC','phot_Flag','dist','logM_HI','vhelio','g_i','absMag_g_corr','absMag_i_corr','logMstar_Taylor']
#colnames = ['RA', 'DEC','photFlag','DIST','logMHI','vhelio','gmi','absMag_g_corr','absMag_i_corr','logMstarTaylor']
newt = Table([a100.AGC,\
              a100.RAdeg_Use,\
              a100.DECdeg_Use, \
              a100.photFlag_gi, \
              # do we want to make this a 1/0 flag? - no, leave it as 0, 1, 2
              a100.Dist, \
              a100.logMH, \
              a100.Vhelio, \
              # missing internal ext coeff, need to update match_catalogs.py to write this out to a100-sdss.fits
              a100.gmi_corr, \
              a100.absMag_g_corr, \
              a100.absMag_i_corr, \
              a100.logMstarTaylor], names=colnames)

## WRITE OUT SELECTED COLUMNS AS FITS FILE
newt.write('paper-table.fits',overwrite=True)#,newt,names=colnames)


## WRITE OUT FIRST 20 ENTRIES AS LATEX FILE
newt[0:20].write('paper-table.tex',overwrite=True)
