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

H0 = 70. # km/s/Mpc
class phot_functions():
    def calc_distance_quantities(self):
        #redshift = self.a100sdss['Vhelio']/c.c.to('km/s').value
        #dL = Column(cosmo.luminosity_distance(redshift),name='distance',unit=u.Mpc)
        self.absMag_g =self.a100sdss['cModelMag_g']+ 5+ - 5*np.log10(self.a100sdss['Dist']*1.e6) - self.a100sdss['extinction_g']
        c1 = MaskedColumn(self.absMag_g,name='absMag_g',unit=u.mag)
        self.absMag_i = self.a100sdss['cModelMag_i']+ 5+ - 5*np.log10(self.a100sdss['Dist']*1.e6) - self.a100sdss['extinction_i']
        c2 = MaskedColumn(self.absMag_i,name='absMag_i',unit=u.mag)
        self.a100sdss.add_columns([c1,c2])

    def internal_extinction(self, gflag = False):

        # calc internal extinction

        # this is exponential fit a/b (not ba)
        ab = self.a100sdss['expAB_r']
        ba = 1./ab
        gamma_g = np.zeros(len(self.a100sdss[self.ref_column]))
        # only apply correction for bright galaxies
        mag_flag_g = self.a100sdss['absMag_g'] <= -17.
        # equation from paper
        gamma_g[mag_flag_g] = -0.35*self.a100sdss['absMag_g'][mag_flag_g] - 5.95
        extinction_g = gamma_g*np.log10(1.*ba) 
        # correct the absolute mag for internal AND galactic extinction
        gmag_corr = self.a100sdss['cModelMag_g'] - extinction_g - self.a100sdss['extinction_g']

        gamma_i = np.zeros(len(self.a100sdss[self.ref_column]))
        mag_flag_i = self.a100sdss['absMag_i'] <= -17.
        # equation from paper
        gamma_i[mag_flag_i] = -0.15*self.a100sdss['absMag_i'][mag_flag_i] - 2.55
        extinction_i = gamma_i*np.log10(1.*ba) 
        # correct the absolute mag for internal AND galactic extinction
        imag_corr = self.a100sdss['cModelMag_i'] - extinction_i - self.a100sdss['extinction_i']
        gmi_corr = self.a100sdss['modelMag_g'] - extinction_g - self.a100sdss['extinction_g'] \
          -(self.a100sdss['modelMag_i'] - extinction_i - self.a100sdss['extinction_i']) 

        # calculate abs mag in g and i, corrected for MW and internal extinction

        # first use distance to get distance modulus
        absMag_i_corr = imag_corr - 5*np.log10(self.a100sdss['Dist']*1.e6) +5
        absMag_g_corr = gmag_corr - 5*np.log10(self.a100sdss['Dist']*1.e6) +5

        # calculate Shao values - these do
        # https://iopscience.iop.org/article/10.1086/511131/fulltext/
        G_Shao = self.a100sdss['absMag_g']  -(1.68*np.log10(ba)) 
        I_Shao = self.a100sdss['absMag_i']  -(1.08*np.log10(ba))
        gmi_Shao = self.a100sdss['modelMag_g'] -self.a100sdss['extinction_g']- 1.68*np.log10(ba) \
           - (self.a100sdss['modelMag_i'] -self.a100sdss['extinction_i']-1.08*np.log10(ba))

        #G_halfShao = absMag_i>-20.5? absMag_g : G_Shao
        #I_halfShao = absMag_i>-20.5? absMag_i : I_Shao
        #Gmi_halfShao = absMag_i>-20.5? (modelMag_g)-(modelMag_i) : gmi_Shao

        #gmi_no_int = self.a100sdss['absMag_g'] - self.a100sdss['absMag_i']
        gmi_no_int =(self.a100sdss['modelMag_g'] - self.a100sdss['extinction_g']) -(self.a100sdss['modelMag_i'] - self.a100sdss['extinction_i']) 
        # append corrected mag and gmi color to table

        c1 = MaskedColumn(gmag_corr, name='gmag_corr',unit = u.mag)
        c2 = MaskedColumn(imag_corr, name='imag_corr',unit = u.mag)
        c3 = MaskedColumn(gmi_corr, name='gmi_corr',unit = u.mag)
        c4 = MaskedColumn(absMag_g_corr,name='absMag_g_corr',unit=u.mag)
        c5 = MaskedColumn(absMag_i_corr,name='absMag_i_corr',unit=u.mag)
        c6 = MaskedColumn(gmi_no_int,name='gmi_no_int',unit=u.mag)
        c7 = MaskedColumn(G_Shao,name='G_Shao',unit=u.mag)
        c8 = MaskedColumn(I_Shao,name='I_Shao',unit=u.mag)
        c9 = MaskedColumn(gmi_Shao,name='gmi_Shao',unit=u.mag)
        # add the extinction coefficients (these will be included in paper table)
        c10 = MaskedColumn(gamma_g,name='gamma_g')
        c11 = MaskedColumn(gamma_i,name='gamma_i')
        
        self.a100sdss.add_columns([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11])
        
    def define_photflag(self):
        # define flag to denote objects with good photometry
        # add photflag column to a100sdss

        # galaxies that have sdss photometry
        sdssflag = self.a100sdss[self.ref_column_objid] > 0

        # galaxies that have good photometry
        photflag_1 = ((self.a100sdss['modelMagErr_g']<0.05) &\
                      (self.a100sdss['modelMagErr_i']<0.05) &\
                      (self.a100sdss['cModelMagErr_g']<0.05) &\
                      (self.a100sdss['cModelMagErr_i']<0.05))
                      
        photflag_2 =  ((self.a100sdss['modelMagErr_g']>0.05) |\
                      (self.a100sdss['modelMagErr_i']>0.05) |\
                      (self.a100sdss['cModelMagErr_g']>0.05) |\
                      (self.a100sdss['cModelMagErr_i']>0.05))
        photflag_gi = np.zeros(len(self.a100sdss[self.ref_column]),'i')
        temp = np.ones(len(self.a100sdss[self.ref_column]),'i')
        photflag_gi[photflag_1 & sdssflag] = temp[photflag_1 & sdssflag]
        photflag_gi[photflag_2 & sdssflag] = 2*temp[photflag_2 & sdssflag]

        c1 = Column(photflag_gi,name='photFlag_gi')
        self.a100sdss.add_column(c1)
    def taylor_mstar(self):
        # calc stellar mass
        logMstarTaylor=1.15+0.70*(self.a100sdss['gmi_corr']) -0.4*(self.a100sdss['absMag_i_corr'])
        ###
        # NSA abs mags are for H0=100, need to correct for our assumed cosmology
        # so need to add 5*np.log10(h), where h = cosmo.H(0)/100.
        #logMstarTaylor = logMstarTaylor - 0.4*(5*np.log10(cosmo.H(0).value/100.))
        # not doing this b/c absMag_i_corr already has distance corrected for H0

        # -0.68 + .7*gmi_cor + (Mi-4.56)/-2.5
        # add taylor_mstar column to a100sdss
        # only set values for galaxies with photflag == 1
        flag = self.a100sdss['photFlag_gi'] == 1
        goodMstar = np.zeros(len(self.a100sdss[self.ref_column]),'f')
        goodMstar[flag] = logMstarTaylor[flag]
                             
        c1 = MaskedColumn(goodMstar,name='logMstarTaylor', mask = ~flag)
        self.a100sdss.add_column(c1)

        
        
class a100(phot_functions):
    def __init__(self, a100_catalog,sdss_catalog):
        '''
        INPUT:
        - a100_catalog = csv version of a100 catalog
        - sdss_catalog = csv version of Martha's sdss catalog that is line-matched to A100

        '''
        # read in a100 catalog
        self.a = ascii.read(a100_catalog,format='csv')
        self.a.rename_column('AGCNr','AGC')
        # read in sdss catalog
        self.s = ascii.read(sdss_catalog,format='csv')
        # join cats
        self.a100sdss = join(self.a,self.s,keys='AGC')

        # now calculate quantities that we use in our paper
        self.ref_column = 'ra'
        self.ref_column_objid = 'objID'
        self.calc_distance_quantities()
        self.internal_extinction()

        self.define_photflag()
        self.taylor_mstar()
        self.write_a100sdss()
        
    def write_a100sdss(self):
        # write full catalog
        self.a100sdss.write(homedir+'/github/APPSS/tables/a100-sdss.fits',format='fits',overwrite=True)

class gswlc(phot_functions):
    def __init__(self, catalog):
        '''
        INPUT:
        - gswlc-A2-sdssphot.fits (GSWLC-A2 file, with SDSS phot that Adriana downloaded attached)

        OUTPUT:
        - gswlc-A2-sdssphot-corrected.fits - input file with columns attached for
          - inclination-corrected colors
          - Taylor stellar mass

        PROCEDURE:
        - downloaded catalog gswlc-A2 (640,659 objects)
          - made a version with the header line preceded by #
          - read into topcat as an ascii file
        - downloaded Adriana's catalog with SDSS photometry and errors
          - https://drive.google.com/drive/folders/1P3ooZ5euqK8DvpsgT_2b5eOgtma6XT0-
          - Table_sdss_...
        - matched tables in topcat according to Adriana's directions
          - Sky match (RA, DEC), 0.5" error
          - output is gswlc-A2-sdssphot.fits
        '''
        # read in gswlc-A2-sdssphot.fit catalog
        # this is the 
        self.a100sdss = Table(fits.getdata(catalog))

        self.ref_column = 'RA_1'
        self.ref_column_objid = 'objID_2'
        self.run_all()
    def run_all(self):
        # calculate column Dist and append to table
        # this is technically not the right distance to use for the GSWLC,
        # we should be using the distance in the A100 catalog, which has a flow model applied.
        dist = self.a100sdss['Z']*3.e5/cosmo.H(0).value
        c1 = Column(dist,name='Dist')
        self.a100sdss.add_column(c1)

        # now move on to calculating phot quantities, corrected for inclination
        self.calc_distance_quantities()
        self.internal_extinction(gflag=True)

        self.define_photflag()
        self.taylor_mstar()
        self.write_a100sdss()
        
    def write_a100sdss(self):
        # write full catalog
        
        self.a100sdss.write(homedir+'/github/APPSS/tables/gswlc-A2-sdssphot-corrected.fits',format='fits',overwrite=True)

    def match2zoo(self):
        zoo = fits.getdata('/Users/rfinn/research/GalaxyZoo/GalaxyZoo1_DR_table2.fits')
        # nothing here yet
        # galaxy zoo SDSS objIDs don't match what we have in a100 - like zero matches
        # not sure why
        
class match2a100sdss():
    def __init__(self, a100sdss=None):
        self.a100sdss = fits.getdata(a100sdss)
 
    def match_nsa(self,nsacat):
        # read in nsa
        self.nsa = fits.getdata(nsacat)
        a100 = self.a100sdss
        print('FULL CATALOGS, BEFORE MATCHING')
        print('total number in A100 = ', len(a100))
        print('total number in NSA = ',len(self.nsa))
        
        # match to agc overlap region        # match catalogs
        velocity1 = a100.Vhelio
        velocity2 = self.nsa.Z*(c.c.to('km/s').value)
        voffset = 300.
        a1002, a100_matchflag, nsa2, nsa_matchflag = make_new_cats(a100, self.nsa,RAkey1='RAdeg_Use',DECkey1='DECdeg_Use', velocity1=velocity1, velocity2=velocity2, maxveloffset = voffset,maxoffset=15.)
        
        # join a100-sdss and nsa into one table
        joined_table = hstack([a1002,nsa2])

        # print match statistics for full catalogs
        print('AFTER MATCHING')
        print('total number in A100 = ',sum(a100_matchflag))
        print('total number in NSA = ',sum(nsa_matchflag))
        print('number of unique galaxies = ',len(a1002))
        print('number of matches between A100 and NSA = ',sum(a100_matchflag & nsa_matchflag))
        print('number in A100 but not in NSA = ',sum(a100_matchflag & ~nsa_matchflag))
        print('number in NSA but not in A100 = ',sum(~a100_matchflag & nsa_matchflag))

        #######################################################################################
        # Now match overlap volumes only
        #######################################################################################
        
        # define overlap region in terms of a100sdss columns and cull a100
        keepa100 = (self.a100sdss.RAdeg_OC > 140.) &\
          (self.a100sdss.RAdeg_OC < 230.) &\
          (self.a100sdss.DECdeg_OC > 0) & (self.a100sdss.DECdeg_OC < 35.) &\
          (self.a100sdss.Vhelio < 15000) 
        # cull a100
        a100 = self.a100sdss[keepa100]

        self.nsa = fits.getdata(nsacat)
        # define overlap in terms of nsa quantities
        keepnsa = (self.nsa.Z < 0.05) & (self.nsa.RA > 140.) & (self.nsa.RA < 230.) \
          & (self.nsa.DEC > 0) & (self.nsa.DEC < 35.)
        # keep overlap region
        nsa = self.nsa[keepnsa]
        print('')
        print('OVERLAP VOLUME, BEFORE MATCHING')
        print('total number in A100 = ', len(a100))
        print('total number in NSA = ',len(nsa))

        # match catalogs
        velocity1 = a100.Vhelio
        velocity2 = nsa.Z*(c.c.to('km/s').value)
        voffset = 300.
        a1002, a100_matchflag, nsa2, nsa_matchflag = make_new_cats(a100, nsa,RAkey1='RAdeg_Use',DECkey1='DECdeg_Use', velocity1=velocity1, velocity2=velocity2, maxveloffset = voffset,maxoffset=15.)
        
        # join a100-sdss and nsa into one table
        joined_table = hstack([a1002,nsa2])

        # print match statistics
        print('')
        print('OVERLAP VOLUME, AFTER MATCHING')
        print('total number in A100 = ',sum(a100_matchflag))
        print('total number in NSA = ',sum(nsa_matchflag))
        print('number of unique galaxies = ',len(a1002))
        print('number of matches between A100 and NSA = ',sum(a100_matchflag & nsa_matchflag))
        print('number in A100 but not in NSA = ',sum(a100_matchflag & ~nsa_matchflag))
        print('number in NSA but not in A100 = ',sum(~a100_matchflag & nsa_matchflag))
        
        # add columns that track if galaxy is in agc and in nsa
        c1 = Column(a100_matchflag,name='a100Flag',dtype='i')
        c2 = Column(nsa_matchflag,name='nsaFlag',dtype='i')
        joined_table.add_columns([c1,c2])
        
        # write out joined a100-sdss-nsa catalog
        joined_table.write(homedir+'/github/APPSS/tables/a100-nsa.fits',format='fits',overwrite=True)


    def match_gswlc(self,gswcat):
        a100 = self.a100sdss
        #gsw = ascii.read(gswcat)
        self.gsw = fits.getdata(gswcat)
        # match to agc     
        velocity1 = a100.Vhelio
        velocity2 = self.gsw['Z']*(c.c.to('km/s').value)
        voffset = 300.
        self.aindex,self.aflag, self.gindex, self.gflag = join_cats(a100['RAdeg_Use'],a100['DECdeg_Use'],self.gsw['RA_1'], self.gsw['DEC_1'],maxoffset=15.,maxveloffset=voffset,  velocity1=velocity1, velocity2=velocity2)
        a1002, a100_matchflag, gsw2, gsw_matchflag = make_new_cats(a100, self.gsw,RAkey1='RAdeg_Use',DECkey1='DECdeg_Use',RAkey2='RA_1',DECkey2='DEC_1', velocity1=velocity1, velocity2=velocity2, maxveloffset = voffset,maxoffset=15.)
    
        # print match statistics
        print('FULL CATALOGS, AFTER MATCHING')
        print('total number in A100 = ',sum(a100_matchflag))
        print('total number in GSWLC-A2 = ',sum(gsw_matchflag))
        print('number of unique galaxies = ',len(a1002))
        print('number of matches between A100 and GSWLC-A2 = ',sum(a100_matchflag & gsw_matchflag))
        print('number in A100 but not in GSWLC-A2 = ',sum(a100_matchflag & ~gsw_matchflag))
        print('number in GSWLC but not in A100 = ',sum(~a100_matchflag & gsw_matchflag))

        # reset catalogs to match overlap region only
        #a100 = self.a100sdss
        # read in gsw
        #self.gsw = fits.getdata(gswcat)

        # define overlap region
        ramin = 140.
        ramax = 230.
        decmin = 0.
        decmax = 35.
        zmax = 0.05
        vmax = 15000

        # keep overlap region
        keepa100 = (self.a100sdss.RAdeg_OC > ramin) &\
          (self.a100sdss.RAdeg_OC < ramax) &\
          (self.a100sdss.DECdeg_OC > decmin) &\
          (self.a100sdss.DECdeg_OC < decmax) &\
          (self.a100sdss.Vhelio < vmax) 
        # cull a100
        a100 = self.a100sdss[keepa100]

        # define overlap in terms of gsw quantities
        keep2 = (self.gsw['Z'] < zmax) &\
           (self.gsw['RA_1'] > ramin) &\
           (self.gsw['RA_1'] < ramax) & \
           (self.gsw['DEC_1'] > decmin) & \
           (self.gsw['DEC_1'] < 35.) 
        gsw = self.gsw[keep2]
        # join cats
        # match to agc overlap region     
        velocity1 = a100.Vhelio
        velocity2 = gsw['Z']*(c.c.to('km/s').value)
        voffset = 300.

        self.aindex,self.aflag, self.gindex, self.gflag = join_cats(a100['RAdeg_Use'],a100['DECdeg_Use'],gsw['RA_1'], gsw['DEC_1'],maxoffset=15.,maxveloffset=voffset,  velocity1=velocity1, velocity2=velocity2)
        a1002, a100_matchflag, gsw2, gsw_matchflag = make_new_cats(a100, gsw,RAkey1='RAdeg_Use',DECkey1='DECdeg_Use', RAkey2='RA_1',DECkey2='DEC_1', velocity1=velocity1, velocity2=velocity2, maxveloffset = voffset,maxoffset=15.)
        
        # join a100-sdss and nsa into one table
        joined_table = hstack([a1002,gsw2])
        #self.a1002 = a100
        #self.gsw2 = gsw2 
        # print match statistics
        print('')
        print('OVERLAP VOLUME, AFTER MATCHING')
        print('total number in A100 = ',sum(a100_matchflag))
        print('total number in GSWLC-A2 = ',sum(gsw_matchflag))
        print('number of unique galaxies = ',len(a1002))
        print('number of matches between A100 and GSWLC-A2 = ',sum(a100_matchflag & gsw_matchflag))
        print('number in A100 but not in GSWLC-A2 = ',sum(a100_matchflag & ~gsw_matchflag))
        print('number in GSWLC but not in A100 = ',sum(~a100_matchflag & gsw_matchflag))
        # add columns that track if galaxy is in agc and in nsa
        c1 = Column(a100_matchflag,name='a100Flag',dtype='i')
        c2 = Column(gsw_matchflag,name='gswFlag',dtype='i')
        joined_table.add_columns([c1,c2])
        # append photometry columns for GSWLC data

        # color with inclination correction

        
        # write out joined a100-sdss-gswlc catalog
        joined_table.write(homedir+'/github/APPSS/tables/a100-gswlcA2.fits',format='fits',overwrite=True)
    def match_s4g(self,s4gcat):
        a100 = self.a100sdss
        s4g = ascii.read(s4gcat)

        # create new RA and DEC columns WITHOUT units attached
        ra = s4g['ra']/u.deg
        dec = s4g['dec']/u.deg
        c1 = Column(ra,name='RA')
        c2 = Column(dec,name='DEC')
        s4g.add_columns([c1,c2])
        # join cats
        velocity1 = a100.Vhelio
        velocity2 = s4g['vrad'].to('km/s')/u.km*u.s
        voffset = 300.
        a1002, a100_matchflag, s4g2, s4g_matchflag = make_new_cats(a100, s4g,RAkey1='RAdeg_Use',DECkey1='DECdeg_Use', velocity1=velocity1, velocity2=velocity2, maxveloffset = voffset,maxoffset=15.)
        
        print('FULL CATALOGS, AFTER MATCHING')
        print('total number in A100 = ',sum(a100_matchflag))
        print('total number in S4G = ',sum(s4g_matchflag))
        print('number of unique galaxies = ',len(a1002))
        print('number of matches between A100 and S4G = ',sum(a100_matchflag & s4g_matchflag))
        print('number in A100 but not in S4G = ',sum(a100_matchflag & ~s4g_matchflag))
        print('number in S4G but not in A100 = ',sum(~a100_matchflag & s4g_matchflag))

        # define overlap region
        ramin = 140
        ramax = 230
        decmin = 0
        decmax = 35
        # not sure what RA range is based on paper draft
        ramin2 = 0
        ramax2 = 30
        decmin2 = 0
        decmax2 = 20
        # third overlap region
        ramin3 = 330
        ramax3 = 360
        decmin3 = 0
        decmax3 = 20
        
        zmax = 0.01
        vmax = zmax*c.c.to('km/s').value
        
        # keep overlap region
        keepa100 = (((self.a100sdss.RAdeg_OC > ramin) & (self.a100sdss.RAdeg_OC < ramax) &\
          (self.a100sdss.DECdeg_OC > decmin) & (self.a100sdss.DECdeg_OC < decmax)) |\
          ((self.a100sdss.RAdeg_OC > ramin2) & (self.a100sdss.RAdeg_OC < ramax2) &\
          (self.a100sdss.DECdeg_OC > decmin2) & (self.a100sdss.DECdeg_OC < decmax2)) &\
          ((self.a100sdss.RAdeg_Use > ramin3) & (self.a100sdss.RAdeg_Use < ramax3) &\
          (self.a100sdss.DECdeg_Use > decmin3) & (self.a100sdss.DECdeg_Use < decmax3)) &\
          (self.a100sdss.Vhelio < vmax) )
        # cull a100
        a100 = self.a100sdss[keepa100]

        # read in s4g
        s4g = ascii.read(s4gcat)
        self.s4g = s4g
        # keep overlap region
        keep2 = (((s4g['ra'] > ramin) & (s4g['ra'] < ramax) &\
          (s4g['dec'] > decmin) & (s4g['dec'] < decmax)) |\
          ((s4g['ra'] > ramin2) & (s4g['ra'] < ramax2) &\
          (s4g['dec'] > decmin2) & (s4g['dec'] < decmax2)) &\
          ((s4g['ra'] > ramin3) & (s4g['ra'] < ramax3) &\
          (s4g['dec'] > decmin3) & (s4g['dec'] < decmax3)) &\
          (s4g['vrad'] < vmax))
        s4g = s4g[keep2]
        # create new RA and DEC columns WITHOUT units attached
        ra = s4g['ra']/u.deg
        dec = s4g['dec']/u.deg
        c1 = Column(ra,name='RA')
        c2 = Column(dec,name='DEC')
        s4g.add_columns([c1,c2])
        # join cats
        velocity1 = a100.Vhelio
        velocity2 = s4g['vrad'].to('km/s')/u.km*u.s
        voffset = 300.
        a1002, a100_matchflag, s4g2, s4g_matchflag = make_new_cats(a100, s4g,RAkey1='RAdeg_Use',DECkey1='DECdeg_Use', velocity1=velocity1, velocity2=velocity2, maxveloffset = voffset,maxoffset=15.)
        
       # join a100-sdss and nsa into one table
        joined_table = hstack([a1002,s4g2])

        print('')
        print('OVERLAP VOLUME, AFTER MATCHING')
        print('total number in A100 = ',sum(a100_matchflag))
        print('total number in S4G = ',sum(s4g_matchflag))
        print('number of unique galaxies = ',len(a1002))
        print('number of matches between A100 and S4G = ',sum(a100_matchflag & s4g_matchflag))
        print('number in A100 but not in S4G = ',sum(a100_matchflag & ~s4g_matchflag))
        print('number in S4G but not in A100 = ',sum(~a100_matchflag & s4g_matchflag))
        # add columns that track if galaxy is in agc and in nsa
        c1 = Column(a100_matchflag,name='a100Flag',dtype='i')
        c2 = Column(s4g_matchflag,name='s4gFlag',dtype='i')
        joined_table.add_columns([c1,c2])

        # write joined a100sdss - s4g catalog
        

        joined_table.write(homedir+'/github/APPSS/tables/a100-s4g.fits',format='fits',overwrite=True)


    def plot_a100_skycoverage(self):
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="mollweide")
        ax.scatter(self.a100sdss['RAdeg_Use'], self.a100sdss['DECdeg_Use'])
        pass

if __name__ == '__main__':
    make_a100sdss = True
    if make_a100sdss:
        a100_file = homedir+'/github/APPSS/tables/a100.HIparms.191001.csv'
        # read in sdss phot, line-matched catalogs
        sdss_file = homedir+'/github/APPSS/tables/a100.SDSSparms.191001.csv'
        a = a100(a100_file,sdss_file)

    make_gswlc_sdss = True
    if make_gswlc_sdss:
        gswlc_file = homedir+'/github/APPSS/tables/gswlc-A2-sdssphot.fitsvar.fits'
        # read in sdss phot, line-matched catalogs
        g = gswlc(gswlc_file)


    # next part - match a100 to other catalogs
    match2a100Flag = True
    if match2a100Flag:
        a100sdsscat = homedir+'/github/APPSS/tables/a100-sdss.fits'
        a = match2a100sdss(a100sdss=a100sdsscat)

        print('################################')
        print('\nMATCHING TO NSA \n')
        print('################################')
        # match to NSA
        #nsacat = homedir+'/research/NSA/nsa_v0_1_2.fits'
        nsacat = homedir+'/research/NSA/nsa_v1_0_1.fits'
        a.match_nsa(nsacat)
        print('################################')
        print('\nMATCHING TO GSWLC-A2 \n')
        print('################################')
        # match to GSWLC-A2

        # calculate internal extinction, colors, and taylor stellar mass
        
        #gsw = homedir+'/github/APPSS/tables/gswlc-A2-withheader.dat'
        gsw = homedir+'/github/APPSS/tables/gswlc-A2-sdssphot-corrected.fits'
        a.match_gswlc(gsw)
        print('################################')
        print('\nMATCHING TO S4G \n')
        print('################################')
        # match to S4G
        s4gcat = homedir+'/github/APPSS/tables/spitzer.s4gcat_5173.tbl'
        a.match_s4g(s4gcat)

print("--- %s seconds ---" % (time.time() - start_time))
