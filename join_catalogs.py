#!/usr/bin/env python

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import numpy as np
from astropy.table import Table
import pandas as pd
def join_cats(ra1,dec1,ra2,dec2,maxoffset=15.,velocity1=None,velocity2=None,maxveloffset=None):
    '''
    INPUT:
    - enter two sets of coordinates: ra1, dec1 and ra2, dec2
       - these are assumed to be in deg
    - to add velocity offset constraint, include velocity1 and velocity2, as well as max allowable velocity offset
    - maxoffset = max separation to consider a match, in arcsec

    OUTPUT:
    - 4 arrays, with length equal to the number of unique galaxies

      - cat1_index = index of galaxy in catalog 1
      - cat1_flag = boolean array, true if galaxy is in catalog 1
      - cat2_index = index of galaxy in catalog 2
      - cat2_flag = boolean array, true if galaxy is in catalog 2

    
    '''
    # explicitly assign offset to units of arcseconds
    maxoffset = maxoffset*u.arcsec
    c1 = SkyCoord(ra1*u.deg,dec1*u.deg, frame='icrs')
    c2 = SkyCoord(ra2*u.deg,dec2*u.deg, frame='icrs')

    ######################################################
    # match catalog2 to catalog1
    ######################################################
    id2, d2d, d3d = c1.match_to_catalog_sky(c2)
    
    # keep matches that have offsets < maxoffset
    # (match_to_catalog_sky finds closest object,
    # so some are not real matches
    # 
    # if velocity offset is provided, check that too
    if maxveloffset is None:
        matchflag2to1 = d2d < maxoffset
    else:
        velmatch_flag = abs(velocity1 - velocity2[id2]) < maxveloffset
        matchflag2to1 = (d2d < maxoffset) & velmatch_flag

    # check if there are duplicates
    nmatch = sum(matchflag2to1)
    nunique = set(id2[matchflag2to1].tolist())

    # get list of cat2 that weren't matched to cat1

    nomatchids = np.setxor1d(np.arange(len(ra2)),id2[matchflag2to1])


    '''
    ######################################################
    # match catalog1 to catalog2
    ######################################################
    id1, d2d, d3d = c2.match_to_catalog_sky(c1)
    
    # keep matches that have offsets < maxoffset
    # (match_to_catalog_sky finds closest object,
    # so some are not real matches
    # 
    # if velocity offset is provided, check that too
    if maxveloffset is None:
        matchflag1to2 = d2d < maxoffset
    else:
        velmatch_flag = abs(velocity2 - velocity1[id1]) < maxveloffset
        matchflag1to2 = (d2d < maxoffset) & velmatch_flag

    print('number of cat2 objects matched to cat1 = ',sum(matchflag2to1))
    print('number of unique cat2 objects = ',len(nunique))
    print('number of cat1 objects matched to cat2 = ',sum(matchflag1to2))
    print('number of unique cat1 objects = ',len(set(id1[matchflag1to2])))
    print('number of cat2 not matched to cat1 = ',sum(~matchflag1to2))

    # check if
    '''
    ######################################################
    # joined catalog will contain all objects in cat1
    # plus any in cat2 that weren't matched to cat1
    ######################################################
    join_index = np.arange(len(ra1) + len(nomatchids))
    #print(len(join_index))
    cat1_index = np.zeros(len(join_index),'i')
    
    # flag for objects in cat 1 that are in the joined catalog
    # should be all of the objects in cat 1
    cat1_flag = join_index < len(ra1)
    cat1_index[cat1_flag] = np.arange(len(join_index))[cat1_flag]
    cat2_index = np.zeros(len(join_index),'i')
    cat2_index[cat1_flag] = id2
    
    #print(cat2_index[cat1_flag][matchflag2to1],len(cat2_index[cat1_flag][matchflag2to1]))
    #print(id2[matchflag2to1])
    cat2_flag = np.zeros(len(join_index),'bool')
    # set cat2 flag for galaxies that are in cat 1
    cat2_flag[cat1_flag] = matchflag2to1
    # this zeros the index of anything that doesn't have a match to cat1
    # otherwise index would be showing closest match to cat 1,
    # even if offset is > maxoffset
    cat2_index[~cat2_flag]=np.zeros(sum(~cat2_flag),'bool')


    # fill in details for objects in cat2 that weren't matched to cat1
    #print('number in cat2 that are not in cat1 = ',sum(~matchflag1to2))
    c2row = np.arange(len(ra2))
    cat2_index[~cat1_flag] = nomatchids
    cat2_flag[~cat1_flag] = np.ones(sum(~cat1_flag),'bool')
    return cat1_index,cat1_flag, cat2_index, cat2_flag

def make_new_cats(cat1, cat2, RAkey1='RA',DECkey1='DEC',RAkey2='RA',DECkey2='DEC',maxoffset=15,velocity1=None,velocity2=None,maxveloffset=None):
    '''
    GOAL:
    - matches two catalogs by RA and DEC
    - user can specify the maximum allowable offset in arcsec

    INPUT:
    - two catalogs
    - assumes both catalogs have RA and DEC columns
    
    OUTPUT:
    - returns tables with one line for each unique galaxy
    - tables contain contents of original two catalogs, but now line matched.
    - For example, table1 may have a row of zeros for a galaxy that is not in tab1 but it is in tab2.
    '''
    cat1_index,cat1_flag, cat2_index, cat2_flag = join_cats(cat1[RAkey1],cat1[DECkey1],cat2[RAkey2],cat2[DECkey2],maxoffset=maxoffset, velocity1=velocity1, velocity2=velocity2,maxveloffset=maxveloffset)
    #newcat1 = np.zeros(len(cat1_index),dtype=cat1.dtype)
    newcat1 = np.ma.masked_array(np.zeros(len(cat1_index),dtype=cat1.dtype))
    cat1 = Table(cat1)
    newcat1 = Table(newcat1)
    newcat1[cat1_flag] = cat1[cat1_index[cat1_flag]]
    newcat2 = np.ma.masked_array(np.zeros(len(cat1_index),dtype=cat2.dtype))
    cat2 = Table(cat2)
    newcat2 = Table(newcat2)
    try:
        newcat2[cat2_flag] = cat2[cat2_index[cat2_flag]]
        #print('simple transformation worked')
    except ValueError: # go brute force method, kept getting data coersion error - "ValueError: Datatype coercion is not allowed"
        # need to go back and fix this
        # -- one option is to convert masked values to nans
        #    (this is what it does when you step through element by element)
        # -- another option is to figure out how to pass the mask?
        #
        # YAAAAAY
        # fixed this by converting both cat2 and newcat2 to astropy Tables before
        # splicing in rows.   
        print('problem matching - prepare for a code crash')
        return
        newcat2_index = np.arange(len(newcat2))[cat2_flag]
        oldcat2_index = cat2_index[cat2_flag]
        for i,j in zip(newcat2_index,oldcat2_index):
            for z in range(len(newcat2[i])):
                newcat2[i][z] = cat2[j][z]
    return Table(newcat1), cat1_flag, Table(newcat2), cat2_flag
if __name__ == '__main__':
    nsa = fits.getdata('NRGs27_nsa.fits')
    agc = fits.getdata('NRGs27_agc.fits')
    t = join_cats(nsa.RA,nsa.DEC,agc.RA,agc.DEC)
    j = make_new_cats(nsa,agc)
    
    
