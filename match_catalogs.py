#!/usr/bin/env python
'''
GOALS:
- match A100 catalog to:
  - sdss phot cat
  - unwise matches

- match a100-sdss-wise cat to:
  - nsa
  - gswlc
  - s4g
- this calculates match statistics for full surveys
- re-do match for overlap regions
  - print statistics for this match too
- produces catalogs for OVERLAP REGION ONLY

- also matches full a100 to other catalogs to produce 
  one catalog with everything
  - this cuts the nsa and gswlc by vr < 15500 before matching to a100 to save memory

USAGE:

- to create catalogs for overlap regions:

  python ~/github/APPSS/match_catalogs.py --matchoverlap

- to create catalog for full a100 table:

  python ~/github/APPSS/match_catalogs.py --matchoverlap

OUTPUTS:

- a100-sdss.fits (FULL A100)
  - a100 matched to sdss phot
- a100-sdss-wise.fits (FULL A100)
  - a100+sdss phot matched to unwise
- a100-nsa.fits (OVERLAP REGION)
  - a100 matched to NSA in OVERLAP REGION
- a100-gswlcA2.fits (OVERLAP REGION)
  - a100 matched to GSWLCA-2 in OVERLAP REGION
- a100-s4g.fits (OVERLAP REGION)
  - a100 matched to S4G in OVERLAP REGION

- full-a100-sdss-wise-nsa-gswlcA2.fits (FULL A100)
 - a100+sdss+wise matched to nsa
 - then matched to gswlc
 - one line for a100 entry


'''


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

H0 = 70. # km/s/Mpc

# magnitude of Sun in WISE filter
# from Jarrett+2013 https://iopscience.iop.org/article/10.1088/0004-6256/145/1/6
W1_sun = 3.24
W2_sun = 3.27
W3_sun = 3.23
W4_sun = 3.25
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
        ba = self.a100sdss['expAB_r']
        #ba = 1./ab
        gamma_g = np.zeros(len(self.a100sdss[self.ref_column]))
        # only apply correction for bright galaxies
        mag_flag_g = self.a100sdss['absMag_g'] <= -17.
        # equation from paper
        gamma_g[mag_flag_g] = -0.35*self.a100sdss['absMag_g'][mag_flag_g] - 5.95
        extinction_g = -1*gamma_g*np.log10(1.*ba) 
        # correct the absolute mag for internal AND galactic extinction
        gmag_corr = self.a100sdss['cModelMag_g'] - extinction_g - self.a100sdss['extinction_g']

        gamma_i = np.zeros(len(self.a100sdss[self.ref_column]))
        mag_flag_i = self.a100sdss['absMag_i'] <= -17.
        # equation from paper
        gamma_i[mag_flag_i] = -0.15*self.a100sdss['absMag_i'][mag_flag_i] - 2.55
        extinction_i = -1.*gamma_i*np.log10(1.*ba) 
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
        G_Shao = self.a100sdss['absMag_g']  +(1.68*np.log10(ba)) 
        I_Shao = self.a100sdss['absMag_i']  +(1.08*np.log10(ba))
        gmi_Shao = self.a100sdss['modelMag_g'] -self.a100sdss['extinction_g']+ 1.68*np.log10(ba) \
           - (self.a100sdss['modelMag_i'] -self.a100sdss['extinction_i']+1.08*np.log10(ba))

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
        goodMstar = np.empty(len(self.a100sdss[self.ref_column]),'f')
        goodMstar[flag] = logMstarTaylor[flag]
                             
        c1 = MaskedColumn(goodMstar,name='logMstarTaylor', mask = ~flag)
        self.a100sdss.add_column(c1)

class wise_functions():
    def calc_wise_mstar(self):
        '''
        from jarrett+2013, eqn 7; stellar M/L from W1-W2
        WISE3.4μm: log(M/L)(M/L)=−0.75 + 3.42((W1−W2)−2.5log(ζ2/ζ1)),(7)
        '''
        # calculate absolute magnitude using distance modulus formula
        Mabs_W1 =self.a100sdsswise['w1_mag']+ 5+ - 5*np.log10(self.a100sdsswise['Dist']*1.e6)
        # if Dustin's magnitudes are AB
        #Mabs_W1 =self.a100sdsswise['w1_mag'] - 2.699 + 5+ - 5*np.log10(self.a100sdsswise['Dist']*1.e6)
        # Jarrett
        self.logL_W1_sun = (-0.4*(Mabs_W1 - W1_sun))
        # from Culver+2014 https://iopscience.iop.org/article/10.1088/0004-637X/782/2/90
        log_ML = -1.96*(self.a100sdsswise['w1_mag']-self.a100sdsswise['w2_mag']) -  0.03
        # for star-forming (lower mass-to-light systems)
        # log10(M/LW1) = -1.93*(W1-W2) - 0.04 
        log_ML = -1.93*(self.a100sdsswise['w1_mag']-self.a100sdsswise['w2_mag']) -  0.04
        # for low-z resolved sources 
        # log10(M/LW1) = -1.93*(W1-W2) - 0.17
        log_ML = -2.54*(self.a100sdsswise['w1_mag']-self.a100sdsswise['w2_mag']) -  0.04
        self.logMstarWise = self.logL_W1_sun + log_ML
        
        # mcgaugh stellar mass
        # M*/LW1 = 0.45 Msun/Lsun

        self.logMstarMcGaugh = np.log10(.45) +self.logL_W1_sun
        
        c1 = MaskedColumn(self.logL_W1_sun,name='logLW1')
        c2 = MaskedColumn(self.logMstarWise,name='logMstarCluver')
        c3 = MaskedColumn(self.logMstarMcGaugh,name='logMstarMcGaugh')
        self.a100sdsswise.add_columns([c1,c2,c3])


    def calc_sfr12(self):
        '''
        from jarrett+2013. eqn 1 and 2
        WISEW3: SFRIR(±0.28) (Myr−1)=4.91(±0.39)×10−10νL12(L),(1)
        WISEW4: SFRIR(±0.04) (Myr−1)=7.50(±0.07)×10−10νL22(L).
        '''
        # cluver+2017
        Mabs_W3 =self.a100sdsswise['w3_mag']+ 5+ - 5*np.log10(self.a100sdsswise['Dist']*1.e6)
        # if Dustin's magnitudes are AB
        #Mabs_W3 =self.a100sdsswise['w3_mag']-5.174+ 5+ - 5*np.log10(self.a100sdsswise['Dist']*1.e6)
        # Jarrett
        self.logL_W3_sun = (-0.4*(Mabs_W3 - W3_sun))
        # Cluver+2018
        #log SFR (M yr−1 ) = (0.889 ± 0.018) log L12µm(L) − (7.76 ± 0.15),
        self.logSFR_W3 = 0.889*self.logL_W3_sun - 7.76
        c1 = MaskedColumn(self.logL_W3_sun,name='logL12')
        c2 = MaskedColumn(self.logSFR_W3,name='logSFR12')
        self.a100sdsswise.add_columns([c1,c2])
        pass
    def calc_sfr22(self):
        # cluver+2017
        Mabs_W4 =self.a100sdsswise['w4_mag']+ 5+ - 5*np.log10(self.a100sdsswise['Dist']*1.e6)
        # if  Dustin's magnitudes are AB
        #Mabs_W4 =self.a100sdsswise['w4_mag'] - 6.620+ 5+ - 5*np.log10(self.a100sdsswise['Dist']*1.e6)
        # Jarrett
        self.logL_W4_sun = (-0.4*(Mabs_W4 - W4_sun))
        # Cluver+2018
        #log SFR (M yr−1 ) = (0.889 ± 0.018) log L12µm(L) − (7.76 ± 0.15),
        self.logSFR_W4 = 0.915*self.logL_W4_sun - 8.01
        c1 = MaskedColumn(self.logL_W4_sun,name='logL22')
        c2 = MaskedColumn(self.logSFR_W4,name='logSFR22')
        self.a100sdsswise.add_columns([c1,c2])
        
    def calc_sfr22_ke(self):
        distance = self.a100sdsswise['Dist']#*self.a100Flag #+ self.a100sdsswisensa['ZDIST']*3.e5/70.
        wavelength_22 = 22*u.micron
        freq_22 = c.c/wavelength_22

        # need to convert W4 flux from vega magnitude to Jansky

        # AB to Vega conversion is about 6 mag for W4
        w4_ab_mag = self.a100sdsswise['w4_mag']+6.620
        
        # flux zp in AB mag is 3630 Jy
        fluxzp_22_jy = 3631.*u.Jy
        
        self.Fnu22 = fluxzp_22_jy*10**(-1*w4_ab_mag/2.5)
        # caculate nuFnu22

        self.nuFnu22 = self.Fnu22*freq_22
        # then calculate nu L_nu, using distance
        
        self.nuLnu22_ZDIST = self.nuFnu22 * 4 * np.pi * (distance*u.Mpc)**2        
        self.logSFR_IR_KE = np.log10(self.nuLnu22_ZDIST.cgs.value)-42.69

        
        c1 = MaskedColumn(self.logSFR_IR_KE,name='logSFR22_KE')
        #c2 = MaskedColumn(self.logSFR_IR_KE,name='logSFR22_KE_err')        
        self.a100sdsswise.add_columns([c1])



        
class a100(phot_functions,wise_functions):
    def __init__(self, a100_catalog,sdss_catalog,sdss_catalog2=None):
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
        print('number of rows in a100 table AFTER joining HI and SDSS params = ',len(self.a100sdss))
        
        # add column ipcode from a100.SDSSparms.191001.csv
        # other columns that should be included
        a1002 = ascii.read(sdss_catalog2,format='csv')
        a1002 = Table([a1002['AGC'],a1002['icode'],a1002['pcode'],a1002['ipcode'],a1002['photoID'],a1002['spectID'],a1002['sdss_z']])
        # match these columns to the a100sdss table
        self.a100sdss = join(self.a100sdss,a1002,keys='AGC')
        print('number of rows in a100 table AFTER adding other sdss cols like ipcode = ',len(self.a100sdss))
        
        # now calculate quantities that we use in our paper
        self.ref_column = 'ra'
        self.ref_column_objid = 'objID'
        self.calc_distance_quantities()
        self.internal_extinction()

        self.define_photflag()
        self.taylor_mstar()

        # create a new column to indicate of galaxy is outside the sdss footprint
        # photFlag_gi = 0 - no sdss phot
        #               1 - good sdss phot (errors < 0.05)
        #               2 - bad sdss phot (one of errors > 0.05)
        # we are now breaking zero into zero
        #               0 - outside sdss footprint
        #               3 - within sdss footprint but no sdss photometry
        # CHECK THIS WITH MARY!!!
        photCode = self.a100sdss['photFlag_gi'].copy()
        flag = (self.a100sdss['photFlag_gi']==0) & (self.a100sdss['ipcode'] != 0)
        photCode[flag] = 3*np.ones(len(self.a100sdss),'i')[flag]
        c = Column(photCode,name='sdssPhotFlag')
        self.a100sdss.add_column(c)
        
        # match unWISE photometry to table
        self.get_wise()

        self.a100sdsswise = join(self.a100sdss,self.wise,keys='AGC',join_type='left')
        self.calc_wise_mstar()
        self.calc_sfr12()
        self.calc_sfr22()
        self.calc_sfr22_ke()
        #self.calc_sfrnuv_ke()        

        # write out table
        self.write_joined_table()
    def get_wise(self):
        # read in unWISE photometry from Dustin
        wisefile = tabledir+'/a100.SDSSObjID.191001.match3.unwise.fits'
        # removed the duplicates in the unWISE catalog
        wisefile = tabledir+'/a100.SDSSObjID.191001.match3.unwise-cleaned.fits'        
        self.wise = fits.getdata(wisefile)
        self.wise = Table(self.wise)
        self.wise.rename_column('objid','unwise_objid')
        self.wise.rename_column('finn_objid','objID')
        self.wise.rename_column('agc','AGC')
        pass

    def write_joined_table(self):
        # write full catalog
        #self.a100sdss.write(tabledir+'/a100-sdss.fits',format='fits',overwrite=True)
        # adding wise to a100+sdss tables so that we can compare WISE
        # quantities (Mstar, SFR) with values from other tables
        self.a100sdss.write(tabledir+'/a100-sdss.fits',format='fits',overwrite=True)
        self.a100sdsswise.write(tabledir+'/a100-sdss-wise.fits',format='fits',overwrite=True)
        
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

        - redid this with sdss phot file that Adriana matches by sdss objid
        - saved ut as gswlc-A2-sdssphot.v2.fits
        '''
        # read in gswlc-A2-sdssphot.fit catalog
        # this is the
        print(catalog)
        self.a100sdss = Table(fits.getdata(catalog))
        try:
            t = self.a100sdss['RA_1']
            self.ref_column = 'RA_1'
        except KeyError:
            #t = self.a100sdss['RA_1']
            self.ref_column = 'ra_1'
        try:
            t = self.a100sdss['objID_2']
            self.ref_column_objid = 'objID_2'
        except:
            self.ref_column_objid = 'ObjID_2'
        self.run_all()
    def run_all(self):
        # calculate column Dist and append to table
        # this is technically not the right distance to use for the GSWLC,
        # we should be using the distance in the A100 catalog, which has a flow model applied.
        try:
            dist = self.a100sdss['Z']*3.e5/cosmo.H(0).value
        except KeyError:
            dist = self.a100sdss['z']*3.e5/cosmo.H(0).value
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
        
        self.a100sdss.write(tabledir+'/gswlc-A2-sdssphot-corrected.fits',format='fits',overwrite=True)

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
        
        # try nsa catalog to keep local galaxies only
        # use vmax of a100 plus a margin of error (500 km/s)

        flag = self.nsa['Z']*3e5 < 20000
        self.nsa = self.nsa[flag]

        
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
        self.a100sdsswisensa = joined_table
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
        self.a100Flag = a100_matchflag
        self.nsaFlag = nsa_matchflag
        self.a100sdsswisensa.add_columns([c1,c2])
        self.calc_sfrnuv_ke()        
        # write out joined a100-sdss-nsa catalog
        self.a100sdsswisensa.write(tabledir+'/a100-nsa.fits',format='fits',overwrite=True)

    def calc_sfrnuv_ke(self):
        # distance is a100 Distance when available
        # otherwise use SDSS NSA ZDIST
        distance = self.a100sdsswisensa['Dist']#*self.a100Flag + self.a100sdsswisensa['ZDIST']*3.e5/70.*(~self.a100Flag)
        # NUV is 230 nm, according to Kennicutt & Evans
        wavelength_NUV = 230.e-9*u.m
        freq_NUV = c.c/wavelength_NUV
        
        # convert NSA NUV abs mag to nuLnu_NUV
        #flux_10pc = 10.**((22.5-self.s.ABSMAG[:,1])/2.5)
        # assume ABSMAG is in AB mag, with ZP = 3631 Jy
        # *** need to correct ABSMAG to H0=70  ****
        # NSA magnitudes are already corrected for galactic extinction
        flux_10pc = 3631.*10**(-1.*(self.a100sdsswisensa['SERSIC_ABSMAG'][:,1])/2.5)*u.Jy
        dist = 10.*u.pc
        self.nuLnu_NUV = flux_10pc*4*np.pi*dist**2*freq_NUV
        # calculate using A100 distances
        nuv_mag = np.zeros(len(self.a100sdsswisensa),'f')
        fnu_nuv = np.zeros(len(self.a100sdsswisensa),'f')*u.Jy
        flagNUV = self.a100sdsswisensa['SERSIC_NMGY'][:,1] > 0.

        nuv_mag[flagNUV] = 22.5 - np.log10(self.a100sdsswisensa['SERSIC_NMGY'][:,1][flagNUV])
        fnu_nuv[flagNUV] = 3631*10**(-1*nuv_mag[flagNUV]/2.5)*u.Jy
        self.nuLnu_NUV = fnu_nuv*4*np.pi*(distance*u.Mpc)**2*freq_NUV


        
        # GET IR VALUES
        wavelength_22 = 22*u.micron
        freq_22 = c.c/wavelength_22
        # need to convert W4 flux from vega magnitude to Jansky
        # AB to Vega conversion is about 6 mag for W4
        w4_ab_mag = self.a100sdsswisensa['w4_mag']+6.620

        flag22 = self.a100sdsswisensa['w4_mag'] > 0
        # flux zp in AB mag is 3630 Jy
        fluxzp_22_jy = 3631.
        self.Fnu22 = np.zeros(len(flag22),'f')*u.Jy
        self.Fnu22[flag22] = fluxzp_22_jy*10**(-1*w4_ab_mag[flag22]/2.5)*u.Jy
        # caculate nuFnu22
        self.nuFnu22 = self.Fnu22*freq_22
        # then calculate nu L_nu, using distance
        self.nuLnu22_ZDIST = self.nuFnu22 * 4 * np.pi * (distance*u.Mpc)**2        
        
        # correct NUV luminosity by IR flux
        myunit = self.nuLnu_NUV.unit
        self.nuLnu_NUV_cor = np.zeros(len(self.nuLnu_NUV))*myunit
        #self.nuLnu_NUV_cor = self.nuLnu_NUV

        # don't need these two lines b/c I set nuLnu22 = 0 if w4_mag = 0
        #self.nuLnu_NUV_cor[flag] = self.nuLnu_NUV[flag] + 2.26*self.nuLnu22_ZDIST[flag]
        #self.nuLnu_NUV_cor[~flag] = self.nuLnu_NUV[~flag]
        flag = flag22 & flagNUV
        self.nuLnu_NUV_cor[flag] = self.nuLnu_NUV[flag] + 2.26*self.nuLnu22_ZDIST[flag]
        # need relation for calculating SFR from UV only
        #
        # eqn 12
        # log SFR(Msun/yr) = log Lx - log Cx
        # NUV - log Cx = 43.17
        # 24um - logCx = 42.69
        # Halpha - log Cx = 41.27
        
        #self.logSFR_NUV_KE = np.log10(self.nuLnu_NUV.value)+np.log10(9.52141e13) - 43.17
        #self.logSFR_NUVIR_KE = np.log10(self.nuLnu_NUV_cor.value)+np.log10(9.52141e13) - 43.17
        self.logSFR_NUV_KE = -99*np.ones(len(self.nuLnu_NUV))
        self.logSFR_NUVIR_KE = -99*np.ones(len(self.nuLnu_NUV))        
        
        self.logSFR_NUV_KE[flagNUV] = np.log10(self.nuLnu_NUV.cgs.value[flagNUV]) - 43.17
        self.logSFR_NUVIR_KE[flag] = np.log10(self.nuLnu_NUV_cor.cgs.value[flag]) - 43.17

        # write columns out to table
        c0 = MaskedColumn(self.nuLnu_NUV,name='nuLnu_NUV')        
        c1 = MaskedColumn(self.logSFR_NUV_KE,name='logSFR_NUV_KE')
        c2 = MaskedColumn(self.logSFR_NUVIR_KE,name='logSFR_NUVIR_KE')        
        self.a100sdsswisensa.add_columns([c0,c1,c2])
        self.a100sdsswisensa.write(tabledir+'/a100-sdss-wise-nsa.fits',overwrite=True)

    def match_gswlc(self,gswcat):
        a100 = self.a100sdss
        print(gswcat)
        #gsw = ascii.read(gswcat)
        self.gsw = fits.getdata(gswcat)
        
        # try gsw catalog to keep local galaxies only
        # use vmax of a100 plus a margin of error (500 km/s)

        flag = self.gsw['Z']*3e5 < 20000
        self.gsw = self.gsw[flag]

        
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
        joined_table.write(tabledir+'/a100-gswlcA2.fits',format='fits',overwrite=True)
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
        ramin = 138
        ramax = 232
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
        vmax = .01*3e5
        print('vmax for S4G comparison = ',vmax)
        
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
        

        joined_table.write(tabledir+'/a100-s4g.fits',format='fits',overwrite=True)


    def plot_a100_skycoverage(self):
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="mollweide")
        ax.scatter(self.a100sdss['RAdeg_Use'], self.a100sdss['DECdeg_Use'])
        pass

class matchfulla100():
    def __init__(self, a100sdss=None):
        self.a100sdss = fits.getdata(a100sdss)
 
    def match_nsa(self,nsacat):
        # read in nsa
        self.nsa = fits.getdata(nsacat)
        a100 = self.a100sdss
        # try nsa catalog to keep local galaxies only
        # use vmax of a100 plus a margin of error (500 km/s)

        flag = self.nsa['Z']*3e5 < 20000
        self.nsa = self.nsa[flag]
        
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


        self.a100sdsswisensa = joined_table

        # add columns that track if galaxy is in agc and in nsa
        c1 = Column(a100_matchflag,name='a100Flag',dtype='i')
        c2 = Column(nsa_matchflag,name='nsaFlag',dtype='i')
        self.a100Flag = a100_matchflag
        self.nsaFlag = nsa_matchflag
        self.a100sdsswisensa.add_columns([c1,c2])
        self.calc_sfrnuv_ke()

        # keep only a100 galaxies

    
        
        # write out joined a100-sdss-nsa catalog
        # keep a100 galaxies only
        # THIS IS OVERLAP REGION ONLY
        self.a100sdsswisensa[a100_matchflag].write(tabledir+'/full-a100-sdss-wise-nsa.fits',format='fits',overwrite=True)

    def calc_sfrnuv_ke(self):
        distance = self.a100sdsswisensa['Dist']#*self.a100Flag + self.a100sdsswisensa['ZDIST']*3.e5/70.*(~self.a100Flag)
        # NUV is 230 nm, according to Kennicutt & Evans
        wavelength_NUV = 230.e-9*u.m
        freq_NUV = c.c/wavelength_NUV
        
        # convert NSA NUV abs mag to nuLnu_NUV
        #flux_10pc = 10.**((22.5-self.s.ABSMAG[:,1])/2.5)
        # assume ABSMAG is in AB mag, with ZP = 3631 Jy
        # NSA magnitudes are already corrected for galactic extinction
        #flux_10pc = 3631.*10**(-1.*(self.a100sdsswisensa['SERSIC_ABSMAG'][:,1])/2.5)*u.Jy
        flagNUV = self.a100sdsswisensa['SERSIC_NMGY'][:,1] > 0.1
        # adjust Hubble constant from 100 to 70
        flux_10pc = 3631.*10**(-1.*(self.a100sdsswisensa['SERSIC_ABSMAG'][:,1])/2.5)*u.Jy 
        dist = 10.*u.pc
        #self.nuLnu_NUV = flux_10pc*4*np.pi*dist**2*freq_NUV
        # calculate using A100 distances
        nuv_mag = 22.5 - np.log10(self.a100sdsswisensa['SERSIC_NMGY'][:,1])
        fnu_nuv = 3631*10**(-1*nuv_mag/2.5)*u.Jy

        self.nuLnu_NUV = fnu_nuv*4*np.pi*(distance*u.Mpc)**2*freq_NUV

        
        # GET IR VALUES
        wavelength_22 = 22*u.micron
        freq_22 = c.c/wavelength_22
        # need to convert W4 flux from vega magnitude to Jansky
        # AB to Vega conversion is about 6 mag for W4
        w4_ab_mag = self.a100sdsswisensa['w4_mag']+6.620
        
        # flux zp in AB mag is 3630 Jy
        fluxzp_22_jy = 3631.*u.Jy
        
        self.Fnu22 = fluxzp_22_jy*10**(-1*w4_ab_mag/2.5)
        # caculate nuFnu22
        self.nuFnu22 = self.Fnu22*freq_22
        # then calculate nu L_nu, using distance
        self.nuLnu22_ZDIST = self.nuFnu22 * 4 * np.pi * (distance*u.Mpc)**2        
        
        # correct NUV luminosity by IR flux
        myunit = self.nuLnu_NUV.unit
        self.nuLnu_NUV_cor = -99*np.ones(len(self.nuLnu_NUV))*myunit
        flag22 = self.a100sdsswisensa['w4_mag'] > 0.
        #self.nuLnu_NUV_cor = self.nuLnu_NUV
        flag = flag22 & flagNUV        
        ### adjust nuv luminosity for galaxies with a 22 um detection
        self.nuLnu_NUV_cor[flag] = self.nuLnu_NUV[flag] + 2.26*self.nuLnu22_ZDIST[flag]
        #self.nuLnu_NUV_cor[~flag] = self.nuLnu_NUV[~flag]

        # need relation for calculating SFR from UV only
        #
        # eqn 12
        # log SFR(Msun/yr) = log Lx - log Cx
        # NUV - log Cx = 43.17
        # 24um - logCx = 42.69
        # Halpha - log Cx = 41.27

        # not sure what the extra term is here
        #self.logSFR_NUV_KE = np.log10(self.nuLnu_NUV.value)+np.log10(9.52141e13) - 43.17
        #self.logSFR_NUVIR_KE = np.log10(self.nuLnu_NUV_cor.value)+np.log10(9.52141e13) - 43.17

        self.logSFR_NUV_KE = -99*np.ones(len(self.nuLnu_NUV))
        self.logSFR_NUVIR_KE = -99*np.ones(len(self.nuLnu_NUV))        

        self.logSFR_NUV_KE[flagNUV] = np.log10(self.nuLnu_NUV.cgs.value[flagNUV]) - 43.17
        self.logSFR_NUVIR_KE[flag] = np.log10(self.nuLnu_NUV_cor.cgs.value[flag]) - 43.17
        
        #self.logSFR_NUV_KE = np.log10(self.nuLnu_NUV.cgs.value) - 43.17
        #self.logSFR_NUVIR_KE = np.log10(self.nuLnu_NUV_cor.cgs.value) - 43.17

        # write columns out to table
        c0 = MaskedColumn(self.nuLnu_NUV,name='nuLnu_NUV')        
        c1 = MaskedColumn(self.logSFR_NUV_KE,name='logSFR_NUV_KE')
        c2 = MaskedColumn(self.logSFR_NUVIR_KE,name='logSFR_NUVIR_KE')
        c3 = MaskedColumn(flagNUV,name='flagNUV')
        c4 = MaskedColumn(flag22,name='flag22')                        
        self.a100sdsswisensa.add_columns([c0,c1,c2,c3,c4])
    def match_gswlc(self,gswcat):
        self.a100sdsswisensa = fits.getdata(tabledir+'/full-a100-sdss-wise-nsa.fits')
        a100 = Table(self.a100sdsswisensa)
        
        # add column with NSA ra if it exists, otherwise RAdeg_Use
        nsaFlag = a100['nsaFlag'] == 1
        bestra = a100['RA']*nsaFlag + a100['RAdeg_Use']*~nsaFlag
        bestdec = a100['DEC']*nsaFlag + a100['DECdeg_Use']*~nsaFlag        
        c1 = Column(bestra,'bestRA')
        c2 = Column(bestdec,'bestDEC')
        a100.add_columns([c1,c2])
        print(gswcat)
        #gsw = ascii.read(gswcat)
        self.gsw = Table(fits.getdata(gswcat))
        # try nsa catalog to keep local galaxies only
        # use vmax of a100 plus a margin of error (500 km/s)

        flag = self.gsw['Z']*3e5 < 20000
        self.gsw = self.gsw[flag]
        
        # match to agc     
        velocity1 = a100['Vhelio']
        velocity2 = self.gsw['Z']*(c.c.to('km/s').value)
        voffset = 300.
        self.aindex,self.aflag, self.gindex, self.gflag = join_cats(a100['bestRA'],a100['bestDEC'],self.gsw['RA_1'], self.gsw['DEC_1'],maxoffset=15.,maxveloffset=voffset,  velocity1=velocity1, velocity2=velocity2)
        
        a1002, a100_matchflag, gsw2, gsw_matchflag = make_new_cats(a100, self.gsw,RAkey1='RAdeg_Use',DECkey1='DECdeg_Use',RAkey2='RA_1',DECkey2='DEC_1', velocity1=velocity1, velocity2=velocity2, maxveloffset = voffset,maxoffset=15.)
    
        # print match statistics
        print('FULL CATALOGS, AFTER MATCHING')
        print('total number in A100+NSA = ',sum(a100_matchflag))
        print('total number in GSWLC-A2 = ',sum(gsw_matchflag))
        print('number of unique galaxies = ',len(a1002))
        print('number of matches between A100 and GSWLC-A2 = ',sum(a100_matchflag & gsw_matchflag))
        print('number in A100 but not in GSWLC-A2 = ',sum(a100_matchflag & ~gsw_matchflag))
        print('number in GSWLC but not in A100 = ',sum(~a100_matchflag & gsw_matchflag))
        # write out temp files
        #print('writing joined tables in two pieces ')
        a1002[a100_matchflag].write(tabledir+'/a100-sdss-wise-nsa-gswlcA2-left.fits',format='fits',overwrite=True)
        gsw2[a100_matchflag].write(tabledir+'/a100-sdss-wise-nsa-gswlcA2-right.fits',format='fits',overwrite=True)
        print('Trying to join a1002 and gsw2')
        # join a100-sdss and nsa into one table
        joined_table = hstack([a1002[a100_matchflag],gsw2[a100_matchflag]])
        
        # add columns that track if galaxy is in agc and in nsa
        #c1 = Column(a100_matchflag,name='a100Flag',dtype='i')
        c2 = Column(gsw_matchflag[a100_matchflag],name='gswFlag',dtype='i')
        joined_table.add_columns([c2])
        
        # write out joined a100-sdss-gswlc catalog
        joined_table.write(tabledir+'/full-a100-sdss-wise-nsa-gswlcA2.fits',format='fits',overwrite=True)
        

def make_a100sdss():
    a100_file = tabledir+'/a100.HIparms.191001.csv'
    # read in sdss phot, line-matched catalogs
    sdss_file2 = tabledir+'/a100.SDSSparms.191001.csv'
    sdss_file = tabledir+'/a100.code12.SDSSvalues200409.csv'
    ## UPDATING TO READ IN COLUMN IPCODE FROM SDSSPARMS SO THAT WE CAN
    ## UPDATE THE SDSS FLAG TO INDICATE GALAXIES THAT ARE NOT IN THE SDSS FOOTPRINT
    a = a100(a100_file,sdss_file,sdss_catalog2=sdss_file2)


if __name__ == '__main__':
    import argparse
    ###########################
    ##### SET UP ARGPARSE
    ###########################

    parser = argparse.ArgumentParser(description ='Match catalogs for A100-SDSS paper')
    parser.add_argument('--matchoverlap', dest = 'overlap', default = False, action='store_true',help = 'set this to match catalogs in overlap region and write out catalogs for paper analysis.')
    parser.add_argument('--matchfull', dest = 'full', default = False, action='store_true',help = 'set this to match catalogs to the full a100 catalog.  this will produce full-a100-sdss-wise-nsa-gswlcA2.fits, the full a100+sdss catalog with AALLL the other columns that you could ever want.')    

    args = parser.parse_args()

    if args.overlap:
        # this also creates WISE catalog
        make_a100sdss()
        #gswlc_file = tabledir+'/gswlc-A2-sdssphot.fits'

        ### UDPATING JULY 6, 2020 TO USE CATALOG WITH
        ### SDSS PHOT MATCHED BY OBJID INSTEAD OF BY RA, DEC
        #gswlc_file = tabledir+'/gswlc-A2-sdssphot.v2.fits'
        # using version with vmax < 20,000
        gswlc_file = tabledir+'/gswlc-A2-sdssphot.v2.vmax20k.fits'        
        # read in sdss phot, line-matched catalogs
        g = gswlc(gswlc_file)


        # next part - match a100 to other catalogs
        a100sdsscat = tabledir+'/a100-sdss-wise.fits'
        a = match2a100sdss(a100sdss=a100sdsscat)

        print('################################')
        print('\nMATCHING TO NSA \n')
        print('################################')
        # match to NSA
        #nsacat = homedir+'/research/NSA/nsa_v0_1_2.fits'
        nsacat = homedir+'/research/NSA/nsa_v1_0_1.fits'
        # using vmax < 20k version
        nsacat = tabledir+'/nsa_v1_0_1_vmax20k.fits'        
        
        a.match_nsa(nsacat)
        print('################################')
        print('\nMATCHING TO GSWLC-A2 \n')
        print('################################')
        # match to GSWLC-A2

        # calculate internal extinction, colors, and taylor stellar mass
        
        #gsw = tabledir+'/gswlc-A2-withheader.dat'
        gsw = tabledir+'/gswlc-A2-sdssphot-corrected.fits'
        a.match_gswlc(gsw)
        print('################################')
        print('\nMATCHING TO S4G \n')
        print('################################')
        # match to S4G
        s4gcat = tabledir+'/spitzer.s4gcat_5173.tbl'
        a.match_s4g(s4gcat)

    if args.full: 
        # this matches NSA to the full a100 and saves the resulting table
        # as opposed to matching to the restricted overlap region only
        # I think I am doing this so I can match to LCS or Virgo
        a100sdsscat = tabledir+'/a100-sdss-wise.fits'
        afull = matchfulla100(a100sdsscat)
        print('matching full a100 to full NSA')
        nsacat = homedir+'/research/NSA/nsa_v1_0_1.fits'
        # using vmax < 20k version
        nsacat = tabledir+'/nsa_v1_0_1_vmax20k.fits'        
        afull.match_nsa(nsacat)
        print('matching full a100 to full GSWLC')        
        gsw = tabledir+'/gswlc-A2-sdssphot-corrected.fits'
        afull.match_gswlc(gsw)

print("--- %s seconds ---" % (time.time() - start_time))
