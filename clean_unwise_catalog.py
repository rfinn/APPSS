#!/usr/bin/env python

import numpy as np
import numpy.ma as ma
import os
from astropy.io import fits
from astropy.io import ascii
from astropy import constants as c
from astropy import units as u
from astropy.table import Table, join, hstack, Column, MaskedColumn
from astropy.coordinates import SkyCoord
import sys
from astropy.cosmology import WMAP9 as cosmo

homedir = os.getenv('HOME')
sys.path.append(homedir+'/github/appss/')
from join_catalogs import make_new_cats, join_cats

import time
start_time = time.time()

if homedir.find('Users') > -1:
    # running on macbook
    tabledir = homedir+'/github/APPSS/tables/'
else:
    tabledir = homedir+'/research/APPSS/tables/'
    tabledir = homedir+'/github/a100sdss/tables/'    

H0 = 70. # km/s/Mpc


class unwise():
    def __init__(self,catalog):
        ##############################################################
        # READ IN UNWISE CATALOG
        ##############################################################
        self.w = Table(fits.getdata(catalog))
        
    def get_duplicates(self):

        ##############################################################
        # IDENTIFY A100 SOURCES WITH MULTIPLE UNWISE MATCHES
        ##############################################################

        unique, arindex, counts = np.unique(self.w['agc'],return_counts=True, return_index=True)
        print('number of AGC with multiple unWISE matches = ',sum(counts > 1))
        #print('')
        #print('AGC number of sources with multiple unWISE matches = ')
        #print(unique[counts > 1])
        self.doubles = unique[counts > 1]
        print('number of sources with multiple matches = ',len(self.doubles))
        print('\t number of sources with 2 matches = ',np.sum(counts == 2))
        print('\t number of sources with 3 matches = ',np.sum(counts == 3))        
    def combine_duplicates(self):
        ##############################################################
        # COMBINE UNWISE FLUXES/MAGNITUDES FOR OBJECTS WITH MULTIPLE MATCHES
        ##############################################################

        # sum flux: w1, w2, w3, w4
        flux_cols = ['_nanomaggies','_nanomaggies_ivar','_pronpix','_proflux']

        for d in self.doubles:
            dindex = self.w['agc'] == d
            for i in range(4):
                w = i + 1 # wise band
                for f in flux_cols:
                    colname = 'w'+str(w)+f
                    if colname.find('ivar') > -1:
                        #handle inverse variance
                        self.w[colname][dindex[0]] = np.sum(1./self.w[colname][dindex])
                    else:
                        
                        self.w[colname][dindex[0]]= np.sum(self.w[colname][dindex])

                # sum magnitudes - but how??

                # maybe just calc mag from flux in nanomaggies?  except what is ZP

                # mag = 22.5 - 2.5*log10(flux_nanomaggies)
                fluxcol = 'w'+str(w)+'_nanomaggies'
                ivar = 'w'+str(w)+'_nanomaggies_ivar'                
                mag1 = 22.5 - 2.5*np.log10(self.w[fluxcol])
                mag2 = 22.5 - 2.5*np.log10(self.w[fluxcol] + np.sqrt(1./self.w[ivar]))                
                
                self.w['w'+str(w)+'_mag'] = mag1
                self.w['w'+str(w)+'_mag_err'] = mag1-mag2


    def remove_duplicates(self):
        ##############################################################
        # REMOVE DUPLICATES
        ##############################################################
        remove_rows = []
        for d in self.doubles:
            dindex = np.where(self.w['agc'] == d)
            # remove all but first row
            #print(dindex)
            remove_rows = remove_rows + dindex[0][1:].tolist()
        # remove rows from table
        self.remove_rows = remove_rows
        #print('rows to be removed:')
        #print(remove_rows)
        self.w.remove_rows(remove_rows)
    def write_table(self):
        ##############################################################
        # WRITE THE CLEANED TABLE
        ##############################################################
        outfile = tabledir+'/a100.SDSSObjID.191001.match3.unwise-cleaned.fits'
        self.w.write(outfile, format='fits', overwrite=True)
if __name__ == '__main__':
    wisefile = tabledir+'/a100.SDSSObjID.191001.match3.unwise.fits'
    w = unwise(wisefile)
    w.get_duplicates()
    w.combine_duplicates()
    w.remove_duplicates()
    w.write_table()
    
