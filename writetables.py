#!/usr/bin/env python
'''
GOAL:
* write out tables for A100-SDSS paper (20 rows)
* write out fits tables for full A100 sample

OUTPUT:
* table1.tex
  - table 1 from Durbala+2020

* table2.tex 
  - table 2 from Durbala+2020

* fits versions of table1 and table2
  - Durbala2020.table1.DATE.fits
  - Durbala2020.table2.DATE.fits

USAGE:

python writetables.py

UPDATES:
* 2020-08-06
  - making empty array elements nan instead of, e.g. -99

* 2021-08-27
'''


import numpy as np
import os
import shutil
from astropy.io import fits, ascii
from astropy.table import Table
from datetime import datetime
from astropy.cosmology import WMAP9 as cosmo


homedir = os.getenv('HOME')
tablepath = homedir+'/research/APPSS/tables/'
tablepath = homedir+'/github/a100sdss/output-tables/'
latextablepath = homedir+'/research/APPSS/latex-tables/'


class latextable():
    def __init__(self):
        # read in data table
        self.tab = fits.getdata(tablepath+'a100-sdss-wise.fits')
        # the next table has the NSA data appended to the end columns
        # it contains the full NSA, but the first rows should be line-matched to the a100-sdss-wise

        ### THIS IS THE WRONG TABLE
        ### SHOULD BE USING a100-sdss-wise-nsa-gswlcA2.fits
        #self.tab2 = fits.getdata(tablepath+'full-a100-sdss-wise-nsa-gswlcA2.fits')
        self.tab2 = fits.getdata(tablepath+'a100-sdss-wise-nsa-gswlcA2.fits')        

        
        self.tab2 = self.tab2[0:len(self.tab)] # trim table to keep only A100 rows
        print('length of tab2 = ',len(self.tab2),' should be 31502')
        self.agcdict2=dict((a,b) for a,b in zip(self.tab2['AGC'],np.arange(len(self.tab2['AGC']))))
        pass
    def calculate_errors(self):
        ###################################
        ### TABLE 1
        ###################################        
        # expABr
        self.expAB_r_err = self.tab['expABErr_r']
        # set any error that is less than 0.01 to 0.01

        ###################################
        ### SET MIN ERROR TO 0.01
        ###################################        
        
        flag = self.expAB_r_err < 0.01
        self.expAB_r_err[flag] = 0.01*np.ones(sum(flag))
        # repeat for model mag
        self.tab['cModelMagErr_i'][self.tab['cModelMagErr_i'] < 0.01] = 0.01*np.ones(sum(self.tab['cModelMagErr_i'] < 0.01))
        self.tab['cModelMagErr_g'][self.tab['cModelMagErr_g'] < 0.01] = 0.01*np.ones(sum(self.tab['cModelMagErr_g'] < 0.01))
        # and error in NSA ABS MAG
        minerr = 0.01
        minivar = 1./minerr**2
        self.tab2['SERSIC_AMIVAR'][:,1][self.tab2['SERSIC_AMIVAR'][:,1] == 0] = minivar*np.ones(sum(self.tab2['SERSIC_AMIVAR'][:,1] == 0))

        
        # need to update the error for this
        self.mw_ext_i_err = 0.2*self.tab['extinction_i']
        self.mw_ext_g_err = 0.2*self.tab['extinction_g']

        # cmodel_i
        
        ###################################
        ### TABLE 2
        ###################################
        self.absMag_i_err = np.sqrt(self.tab['cModelMagErr_i']**2 + self.mw_ext_i_err**2 + (5/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2)
        
        # gamma_g
        self.gammag_err = 0.35*np.sqrt(self.tab['cModelMagErr_g']**2 + self.mw_ext_g_err**2 + (5/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2)
        zero_err_flag = self.tab['absMag_g_corr'] >= -17
        self.gammag_err[zero_err_flag] = np.zeros(np.sum(zero_err_flag),'d')
        # gamma_i
        self.gammai_err = 0.15*np.sqrt(self.tab['cModelMagErr_i']**2 + self.mw_ext_i_err**2 + (5/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2)
        zero_err_flag = self.tab['absMag_i_corr'] >= -17
        self.gammai_err[zero_err_flag] = np.zeros(np.sum(zero_err_flag),'d')

        # internal extinction

        #self.internal_ext_i_err = np.sqrt(np.log10(self.tab['expAB_r'])**2*self.gammai_err**2 + (self.tab['gamma_i']/(self.tab['expAB_r']*np.log(10)))**2*self.expAB_r_err**2)
        #self.internal_ext_g_err = np.sqrt(np.log10(self.tab['expAB_r'])**2*self.gammag_err**2 + (self.tab['gamma_g']/(self.tab['expAB_r']*np.log(10)))**2*self.expAB_r_err**2)        
        ## instead using a constant value of 0.3 for the error in gamma_g and gamma_i
        self.internal_ext_i_err = np.sqrt(np.log10(self.tab['expAB_r'])**2*(0.3)**2 + (self.tab['gamma_i']/(self.tab['expAB_r']*np.log(10)))**2*self.expAB_r_err**2)
        self.internal_ext_g_err = np.sqrt(np.log10(self.tab['expAB_r'])**2*(0.3)**2 + (self.tab['gamma_g']/(self.tab['expAB_r']*np.log(10)))**2*self.expAB_r_err**2)        

        
        # absolute Mi and Mg, corrected for internal extinction
        self.absMag_i_corr_err = np.sqrt(self.tab['cModelMagErr_i']**2 + self.mw_ext_i_err**2 + self.internal_ext_i_err**2 + (5/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2)

        self.absMag_g_corr_err = np.sqrt(self.tab['cModelMagErr_g']**2 + self.mw_ext_g_err**2 + self.internal_ext_g_err**2 + (5/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2)        
        
        # (g-i) corr
        #self.gmi_corr_err = np.sqrt(self.tab['cModelMagErr_g']**2 + self.tab['cModelMagErr_i']**2 + self.mw_ext_i_err**2 + self.mw_ext_g_err**2 + self.internal_ext_g_err**2 + self.internal_ext_i_err**2)
        # use 0.02 for uncertainty of galactic extinction
        self.gmi_corr_err = np.sqrt(self.tab['cModelMagErr_g']**2 + self.tab['cModelMagErr_i']**2 + .02**2 + .02**2 + self.internal_ext_g_err**2 + self.internal_ext_i_err**2)
        ###################################
        ### STELLAR MASS
        ###################################
        
        # logMstarTaylor
        # reporting for galaxies with good sdss phot
        self.logMstarTaylor_err = np.zeros(len(self.gmi_corr_err),'f')*np.nan

        flag = self.tab['sdssPhotFlag'] == 1
        self.logMstarTaylor_err[flag] = np.sqrt(0.49*self.gmi_corr_err[flag]**2 + 0.16*self.absMag_i_corr_err[flag]**2)
        
        # McGaugh
        # reporting for all values
        self.absMag_W1_err = np.sqrt(self.tab['w1_mag_err']**2 +  (5/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2)
        self.logMstarMcGaugh_err = 0.4*self.absMag_W1_err        
        #flag = np.ones(len(self.tab))
        #self.logMstarMcGaugh_err = np.empty(len(self.tab),'f')
        #self.logMstarMcGaugh_err[flag] = 0.4*self.absMag_W1_err[flag]
        # SFR NUV_corr

        ###################################
        ### SFR
        ###################################

        # replace inf values in w4_mag_err with 0.2
        # (these had error = 0 in catalog from Dustin)
        flag = self.tab['w4_mag_err'] == np.inf
        self.tab['w4_mag_err'][flag] = 0.2*np.ones(sum(flag),'f')
        
        # SFR 22
        self.logSFR22_KE_err = np.array(np.sqrt(0.16*self.tab['w4_mag_err']**2 + (2/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2),'d')

        self.logSFR_NUV_KE_err = np.array(np.sqrt(0.16*self.tab2['SERSIC_ABSMAG'][:,1]**2 + (2/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2),'d')

        # hold on to your hat for this one...

        #A = 1/22*np.power(10,-1*(self.tab['w4_mag']+6.62)/2.5) + 2.26/0.23*np.power(10,-1*self.tab2['SERSIC_ABSMAG'][:,1]/2.5)

        ###  NEED TO FIX THIS SO THAT I'M ONLY CALCULATING ERRORS
        ###  FOR GALAXIES THAT HAVE VALID UV AND IR MEASUREMENTS
        ###  OTHERWISE WE GET INF IN ERRORS
        A_w4 =  1/22*np.power(10,-1*(self.tab['w4_mag']+6.62)/2.5)
        A_nuv = 2.26/0.23*np.power(10,-1*self.tab2['SERSIC_ABSMAG'][:,1]/2.5)
        A = A_w4 + A_nuv

        sigsq_d = (2/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2
        sigsq_nuv = A_nuv**2*1./self.tab2['SERSIC_AMIVAR'][:,1]
        sigsq_w4 = A_w4**2*self.tab['w4_mag_err']**2
        #self.logSFR_NUVIR_KE_err = np.zeros(len(self.tab),'d')
        self.logSFR_NUVIR_KE_err = np.array(np.sqrt(sigsq_d + (1./A/2.5)**2*(sigsq_nuv + sigsq_w4)),'d')
        self.sigsq_nuv = sigsq_nuv
        self.A_w4 = A_w4
        self.A_nuv = A_nuv
    def clean_arrays(self):
        '''
        remove bogus values from SFR estimates and other arrays with null values

        '''
        self.sfr22flag = (self.tab['w4_mag'] > 0) & (~np.isnan(self.tab['w4_mag']))
        self.sfr22 = np.zeros(len(self.tab),'f')*np.nan
        self.sfr22_err = np.zeros(len(self.tab),'f')*np.nan
        self.sfr22[self.sfr22flag] = self.tab['logSFR22_KE'][self.sfr22flag]
        self.sfr22_err[self.sfr22flag] = self.logSFR22_KE_err[self.sfr22flag]

        self.sfrnuvflag = self.tab2['SERSIC_ABSMAG'][:,1] < 0.
        self.sfrnuv = np.zeros(len(self.tab),'f')*np.nan
        self.sfrnuv_err = np.zeros(len(self.tab),'f')*np.nan

        self.sfrnuv[self.sfrnuvflag] = self.tab2['logSFR_NUV_KE'][self.sfrnuvflag]
        self.sfrnuv_err[self.sfrnuvflag] = self.logSFR_NUV_KE_err[self.sfrnuvflag]

        self.sfrnuvir = np.zeros(len(self.tab),'f')*np.nan
        self.sfrnuvir_err = np.zeros(len(self.tab),'f')*np.nan                
        flag = self.sfrnuvflag & self.sfr22flag
        # the above flag didn't work.  some -99 values still came through
        flag = self.tab2['flagNUV'] & self.tab2['flag22']
        self.sfrnuvir[flag] = self.tab2['logSFR_NUVIR_KE'][flag]
        self.sfrnuvir_err[flag] = self.logSFR_NUVIR_KE_err[flag]
        inf_flag = np.isinf(self.sfrnuvir_err)
        print('galaxies with UV+IR sfr err = inf')
        printindices = np.arange(len(inf_flag))[inf_flag]

        # setting GSWLC entries to -99 if no match
        gswflag = (self.tab2['logMstar'] > 0) 
        self.gsw_sfr = np.zeros(len(self.tab2),'f')*np.nan
        self.gsw_mstar = np.zeros(len(self.tab2),'f')*np.nan
        
        self.gsw_sfr[gswflag] = self.tab2['logSFR'][gswflag]
        self.gsw_mstar[gswflag] = self.tab2['logMstar'][gswflag]
        
        self.gsw_sfr_err = np.zeros(len(self.tab2),'f')*np.nan
        self.gsw_mstar_err = np.zeros(len(self.tab2),'f')*np.nan
        
        self.gsw_sfr_err[gswflag] = self.tab2['logSFR_err'][gswflag]
        self.gsw_mstar_err[gswflag] = self.tab2['logMstar_err'][gswflag]   
        
        
        #
        # error in sfr22_err is infinity for some
        #
        # traces to some entries with w4_mag_err = inf
        # even though these galaxies have reasonable (and not necessarily faint)
        # magnitudes
        #
        for i in printindices:
            print('{0:02d} {1:.2f} {2:.2f} {3:.2f} {4:.2f} {5:.2e} {6:.2e} {7:.2f} {8:.2f} {9:.2f} {10:.2f} {11:.2e} {12:.2e} {13:.2e} {14:.2f}'.format(i,self.sfr22[i],self.sfr22_err[i],self.tab['w4_mag_err'][i],self.tab['Dist'][i],self.tab['sigdist'][i],self.sfrnuv[i],self.sfrnuv_err[i],self.sfrnuvir[i],self.sfrnuvir_err[i],self.tab2['SERSIC_ABSMAG'][:,1][i],1./np.sqrt(self.tab2['SERSIC_AMIVAR'][:,1][i]),self.A_w4[i],self.A_nuv[i],self.tab['w4_mag'][i]))

        # 
        # fix stellar mass
        #

        ## mask sdss phot values if sdssPhotFlag != 1
        ## arrays to mask for paper tab 2: gamma_g, gamma_i, absMag_i_corr, absMag_e_err,
        ## gmi_corr, gmi_corr_err
        maskflag = self.tab['sdssPhotFlag'] != 1
        replacement = np.zeros(sum(maskflag))*np.nan
        self.tab['gamma_g'][maskflag] = replacement
        self.tab['gamma_i'][maskflag] = replacement
        self.tab['absMag_i_corr'][maskflag] = replacement
        self.absMag_i_corr_err[maskflag] = replacement
        self.tab['gmi_corr'][maskflag] = replacement
        self.gmi_corr_err[maskflag] = replacement

        ## for galaxies with no sdss id
        ## set id to nan (rather than 999999)

        maskflag = self.tab['objID_1'] == 999999
        replacement = np.zeros(sum(maskflag))*np.nan
        self.tab['objID_1'][maskflag] = replacement

    def get_HIdef(self):
        ''' calculate HIdef and write out table '''

        # use scaling relation from
        ## https://www.aanda.org/articles/aa/full_html/2013/12/aa21326-13/T1.html
        ## M. Argudo-Fernández1, 2013, A&A, 560, A9
        ## Answers vary between 1.23 and 1.58 depending on sample
        ##
        ## 1.58 for isolated galaxies
        ##

        radius_scalefactor = 1.4
        self.radius = self.tab['petroR90_r']*radius_scalefactor
        
        '''
        use relationship from Boselli+Gavazzi 2009
        https://www.aanda.org/articles/aa/full_html/2009/46/aa12658-09/aa12658-09.html

        HI def = log MHI_ref - log MHI_obs
        h^2 MHI_ref = c + d log(hdiam)^2

        Type 	c 	d 	Ref.
        E-S0a 	6.88 	0.89 	HG84
        Sa-Sab 	7.75 	0.59 	S96
        Sb 	7.82 	0.62 	S96
        Sbc 	7.84 	0.61 	S96
        Sc 	7.16 	0.87 	S96
        Scd-Im-BCD 	7.45 	0.70 	G10 

        email from Martha:

        First, Carmen Toribio looked at a subset of isolated galaxies in
        a40 to derive scaling relations
        https://ui.adsabs.harvard.edu/abs/2011ApJ...732...93T
        https://ui.adsabs.harvard.edu/abs/2015ApJ...802...72T/     (erratum to
        above)
        
        More recently, Mike led an effort to establish relations for the AMIGA
        sample (of isolated galaxies):
        https://ui.adsabs.harvard.edu/abs/2018A%26A...609A..17J
        
        They use isophotal radii, and refer to a discussion of deriving D25 from
        SDSS in the
        earlier paper by Argudo-Fernandez 2013:
        https://ui.adsabs.harvard.edu/abs/2013A%26A...560A...9A
        
        I think there are some others too, but these are what I am most familiar
        with.
        
        Martha
        
        '''
        # use distance from A100 catalog
        distance = self.tab['Dist']

        # use HI mass from A100 catalog
        self.logMHI = self.tab['logMH']
        
        # calculate HI deficiency using Toribio et al 2011 results
        # their relation is
        # log(M_HI/Msun) = 8.72 + 1.25 log(D_25,r/kpc)
        # and
        # log D_25 = log D_25(obs) + beta log(b/a), where beta = 0.35 in r-band
        # NOTE: SDSS isophotal radii are given in pixels!!!!
        # convert from arcsec to kpc with self.AngDistance (which is in units of kpc/arcsec)
        # multiply by 2 to convert from radius to diameter
        # multiply by sdss pixel scale (0.39) b/c isophotal radii are given in pixels

        # use the radius measurements that I collated for John's group catalog
        # these should approximate D25
        # only calculate for galaxies with radiusflag = True

        # returns DA in Mpc/radians
        #DA=cosmo.angular_diameter_distance(self.tab['Vhelio']/3.e5)

        # try using distance in A100
        vr = self.tab['Dist']*cosmo.H0
        # returns DA in Mpc/radians
        DA = cosmo.angular_diameter_distance(vr/3.e5)
        D25obskpc=2.*self.radius/206264*DA.to('kpc')

        
        # apply correction from toribio et al 2011 
        self.logD25kpc=np.log10(D25obskpc.value) + 0.35*np.log10(self.tab['expAB_r'])

        # use toribio et al relation to predict the expected HI mass,
        # including factor of 2 correction
        logHImassExpected = 8.72 + 1.25*(self.logD25kpc-np.log10(2.))
        
        # use jones+2018, A&A, 609, A17 (AMIGA sample
        # relation to predict the expected HI mass
        # using Maximum Liklihood Estimator for Detections
        logHImassExpected_jones = 7.32 + 0.86*(2*self.logD25kpc)
        # relation for ALL galaxies is similar
        # intrinsic scatter is 0.21
        #
        # fits by morphological type
        # <3 	1.04 ± 0.21 	6.44 ± 0.59 	0.27 ± 0.08 	0.67
        # 3–5 	0.93 ± 0.06 	7.14 ± 0.18 	0.16 ± 0.02 	0.74
        # >5 	0.81 ± 0.09 	7.53 ± 0.24 	0.17 ± 0.03 	0.73

        # calculate deficiency as log expected - log observed
        self.HIdef = np.empty(len(self.tab),'f')
        self.HIdef_jones = np.empty(len(self.tab),'f')

        # I don't have Ttype for the full A100 catalog, so skipping this measure
        # of HI def
        self.HIdef_jones_bytype = np.empty(len(self.tab),'f')        

        # report values if the galaxy has a valid value for petroR90_r
        # about 4 galaxies do not
        flag = self.tab['petroR90_r'] > -1.
        self.HIdef[flag] = logHImassExpected[flag] - self.logMHI[flag]
        self.HIdef_jones[flag] = logHImassExpected_jones[flag] - self.logMHI[flag]        
        self.HIdef_flag = flag

        #flag = ~np.isnan(self.cat['t']) & (self.cat['t'] < 3)
        #self.HIdef_jones_bytype[flag] = 6.44 + 1.04*(2*logD25kpc[flag])- self.logMHI[flag]        
        #flag = ~np.isnan(self.cat['t']) & (self.cat['t'] > 5)
        #self.HIdef_jones_bytype[flag] = 7.53 + .81*(2*logD25kpc[flag]) - self.logMHI[flag]        
        #flag = ~np.isnan(self.cat['t']) & (self.cat['t'] <= 5) & (self.cat['t'] >= 3)
        #self.HIdef_jones_bytype[flag] = 7.14 + .93*(2*logD25kpc[flag]) - self.logMHI[flag]        

        # calculate HI def by type for hyperleda galaxies with Ttype
        
        # boselli & gavazzi prescription
        # use value for Sb
        c = 7.82
        d = 0.62
        logh2MHIref = (c + 2*d*self.logD25kpc)
        self.HIdef_bos = logh2MHIref - self.logMHI

        # write out HI def table

        colnames = ['AGC','RA','DEC',\
                    'petroR90_r','D25_arcsec','D25_kpc',\
                    'logMH','logMH_err','HIdef_Toribio','HIdef_Jones','HIdef_Boselli','HIdef_flag']
        columns = [self.tab['AGC'],self.tab['RAdeg_Use'],self.tab['DECdeg_Use'],\
                   self.tab['petroR90_r'],self.radius,self.logD25kpc,\
                   self.tab['logMH'],self.tab['siglogMH'],self.HIdef,self.HIdef_jones,self.HIdef_bos,flag]
        #for i,c in enumerate(columns):
        #    print(len(c))
        HIdef_table = Table(columns,names=colnames)
        dateTimeObj = datetime.now()        
        myDate = dateTimeObj.strftime("%d-%b-%Y")
        fname = homedir+'/research/APPSS/tables/A100-HIdef-'+myDate+'.fits'
        HIdef_table.write(fname,format='fits',overwrite=True)

        pass
    def print_table1(self,nlines=10,filename=None,papertableflag=True):
        '''write out latex version of table 1 '''
        if filename is None:
            fname=latextablepath+'table1.tex'
        else:
            fname = filename 
        outfile = open(fname,'w')
        
        outfile.write('\\begin{table*}%[ptbh!]\n')
        outfile.write('\\begin{center}\n')
        outfile.write('\\scriptsize\n')
        outfile.write('\\setlength\\tabcolsep{3.0pt} \n')
        outfile.write('\\tablenum{1} \n')
        outfile.write('\\caption{Basic Optical Properties of Cross-listed objects in the ALFALFA-SDSS Catalog.\label{tab:catalog1}  } \n')
        outfile.write('\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\n')
        outfile.write('\\hline \n')
        outfile.write('\\toprule \n')
        outfile.write('AGC &	Flag &	SDSS objID  & RA &	DEC &	V$_{helio}$ &	D &	$\sigma_D$  &	Ext$_g$	& Ext$_i$	& expAB$_r$  & $\sigma_{expAB_r}$ &	cmodel$_i$ & $\sigma_{cmodel_i}$ \\\\ \n')
        outfile.write('& & & J2000 & J2000 & $km~s^{-1}$ & Mpc & Mpc & mag & mag & & & mag & mag \\\\ \n')
        outfile.write('(1) & (2) & (3) & (4) & (5) & (6) & (7) & (8) & (9) & (10) & (11) & (12) & (13) & (14)  \\\\ \n')
        outfile.write('\\midrule \n')
        outfile.write('\\hline \n')
        for i in range(nlines): # print first N lines of data
            # AGC photflag sdss_objid wiseobjid RA DEC vhelio D D_err Ext_g Ext_i expABr err cmodelI err
            
            ##
            ## REMOVING WISE ID (JULY 16,2020)
            ##
            #s = '{0:d} & {1:d} & {2:d} & {3:d}& {4:9.6f}&{5:9.5f} & {6:d} & {7:.1f} & {8:.1f} &  {9:.2f} & {10:.2f}& {11:.2f}& {12:.2f}& {13:.2f} &{14:.2f}\\\\ \n'.format(self.tab['AGC'][i],self.tab['sdssPhotFlag'][i],self.tab['objID_1'][i],self.tab['unwise_objid'][i],self.tab['RAdeg_Use'][i],self.tab['DECdeg_Use'][i],self.tab['Vhelio'][i],self.tab['Dist'][i],self.tab['sigDist'][i],self.tab['extinction_g'][i],self.tab['extinction_i'][i],self.tab['expAB_r'][i],self.expAB_r_err[i],self.tab['cModelMag_i'][i],self.tab['cModelMagErr_i'][i])
            s = '{0:d} & {1:d} & {2:d} & {3:9.6f}&{4:9.5f} & {5:d} & {6:.1f} & {7:.1f} &  {8:.2f} & {9:.2f}& {10:.2f}& {11:.2f}& {12:.2f} &{13:.2f}\\\\ \n'.format(self.tab['AGC'][i],self.tab['sdssPhotFlag'][i],self.tab['objID_1'][i],self.tab['RAdeg_Use'][i],self.tab['DECdeg_Use'][i],self.tab['Vhelio'][i],self.tab['Dist'][i],self.tab['sigDist'][i],self.tab['extinction_g'][i],self.tab['extinction_i'][i],self.tab['expAB_r'][i],self.expAB_r_err[i],self.tab['cModelMag_i'][i],self.tab['cModelMagErr_i'][i])
            if papertableflag:
                # replace nans with \\nodata
                s=s.replace('nan','\\nodata')

            outfile.write(s)

        outfile.write('\\bottomrule \n')
        outfile.write('\\hline \n')
        outfile.write('\\end{tabular} \n')
        outfile.write('\\end{center} \n')
        outfile.write('\\tablecomments{Table 1 is published in its entirety in the machine-readable format.  A portion is shown here for guidance regarding its form and content.}')        
        outfile.write('\\end{table*} \n')
        outfile.close()
    def print_table2(self,nlines=10,filename=None,papertableflag=True):
        '''
        write out latex version of table 2

        changes from old verion:
        - remove Cluver stellar mass (2)
        - add error SFR22
        - add err SFRUV
        - add err SFRUVcorr

        net gain of one column
        '''
        if filename is None:
            fname=latextablepath+'table2.tex'
        else:
            fname = filename 
        outfile = open(fname,'w')
        outfile.write('\\begin{sidewaystable*}%[ptbh!]%\\tiny \n')
        outfile.write('\\begin{center}\n')
        outfile.write('\\tablewidth{0.5\\textwidth} \n')
        outfile.write('\\scriptsize \n')
        outfile.write('%\\footnotesize \n')
        outfile.write('\\setlength\\tabcolsep{1.0pt} \n')
        outfile.write('\\tablenum{2}\n')
        outfile.write('\\caption{Derived Properties of Cross-listed objects in the ALFALFA-SDSS Catalog.\\label{tab:catalog2}  }\n')
        outfile.write('\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\n')
        outfile.write('\\hline\n')
        outfile.write('\\toprule\n')
        #outfile.write('AGC & $\\gamma_g$ & $\\sigma_{\\gamma_g}$ & $\\gamma_i$ & $\\sigma_{\\gamma_i}$ & M$_{icorr}$ &	$\\sigma_{M_{icorr}}$ &	(g-i)$_{corr}$	& $\\sigma_{(g-i)_{corr}}$ & log M$_{\\star, Taylor}$ &	$\\sigma_{log M_{\\star,Taylor}}$  & log M$_{\\star, McGaugh}$ &	$\\sigma_{log M_{\\star, McGaugh}}$ & SFR$_{22}$ & $\\sigma_{log SFR_{22}}$ & ${SFR_{NUV}}$ & $\\sigma_{log SFR_{NUV}}$ & SFR$_{NUVIR}$ & $\\sigma_{log SFR_{UVIR}}$ & M$_{HI}$ & $\\sigma_{M_{HI}}$  \\\\\n')
        outfile.write('AGC & $\\gamma_g$ &  $\\gamma_i$  & M$_{icorr}$ &	$\\rm \\sigma_{M_{icorr}}$ &	(g-i)$_{corr}$	& $\\sigma_{(g-i)_{corr}}$ & log M$_{\\star}$ &	$\\rm \\sigma_{log M_{\\star}}$  & log M$_{\\star}$ &	$\\rm \\sigma_{log M_{\\star}}$ & log M$_{\\star}$& $\\rm \\sigma_{log M_{\\star}}$ & logSFR$_{22}$ & $\\rm \\sigma_{log SFR_{22}}$  & logSFR$\\rm _{NUVcor}$ & $\\rm \\sigma_{log SFR_{NUVcor}}$  & logSFR& $\\rm \\sigma_{logSFR}$ & M$_{HI}$ & $\\rm \\sigma_{M_{HI}}$  \\\\\n')
        outfile.write('&   & &  & &	& & Taylor & Taylor  & McGaugh & McGaugh &  GSWLC &GSWLC& & &  & & GSWLC & GSWLC &  &   \\\\\n')
        #outfile.write(' & mag & mag & mag & mag & mag & mag & $log(M_\\odot)$ & $log(M_\\odot)$ & $log(M_\\odot)$ & $log(M_\\odot)$& $log(M_\\odot)$& $log(M_\\odot)$ & $\\rm log(M_\\odot~yr^{-1})$ & $\\rm log(M_\\odot yr^{-1})$ & $\\rm log(M_\\odot~yr^{-1})$  & $\\rm log(M_\\odot) yr^{-1}$ &  $log(M_\\odot~yr^{-1})$&  $log(M_\\odot~yr^{-1})$ & $log(M_\\odot)$ & $log(M_\\odot)$    \\\\\n')

        ## removing units from error columns to decrease table width
        outfile.write(' & mag & mag & mag & mag & mag & mag & $log(M_\\odot)$ &  & $log(M_\\odot)$ & & $log(M_\\odot)$&  & $\\rm log(M_\\odot~yr^{-1})$ &  & $\\rm log(M_\\odot~yr^{-1})$  &  &  $log(M_\\odot~yr^{-1})$&  & $log(M_\\odot)$ &     \\\\\n')        
        outfile.write('(1) & (2) & (3) & (4) & (5) & (6) & (7) & (8) & (9) & (10) & (11) & (12) & (13) & (14) & (15) & (16) & (17) & (18) & (19) &(20) &(21)  \\\\\n')
        outfile.write('\\midrule\n')
        outfile.write('\\hline\n')
        for i in range(nlines):
            try:
                j = self.agcdict2[self.tab['AGC'][i]]
            except KeyError:
                print(self.tab['AGC'][i])
            #print(self.tab['AGC'][i],j,self.tab2['AGC'][j])
            s=' {0:d} & {1:.2f} & {2:.2f} & {3:.2f} & {4:.2f}& {5:.2f}  & {6:.2f} & {7:.2f} & {8:.2f} & {9:.2f}& {10:.2f}&{11:.2f} &{12:.2f} &{13:.2f} &{14:.2f} &{15:.2f} &{16:.2f}&{17:.2f}&{18:.2f}&{19:.2f}&{20:.2f}  \\\\ \n'.format(self.tab['AGC'][i],self.tab['gamma_g'][i],self.tab['gamma_i'][i],self.tab['absMag_i_corr'][i],self.absMag_i_corr_err[i],self.tab['gmi_corr'][i],self.gmi_corr_err[i],self.tab['logMstarTaylor'][i],self.logMstarTaylor_err[i],self.tab['logMstarMcGaugh'][i],self.logMstarMcGaugh_err[i],self.gsw_mstar[i],self.gsw_mstar_err[i],self.sfr22[i],self.sfr22_err[i], self.sfrnuvir[i],self.sfrnuvir_err[i],self.gsw_sfr[i],self.gsw_sfr_err[i],self.tab['logMH'][i],self.tab['siglogMH'][i])
            if papertableflag:
                # replace nans with \\nodata
                s=s.replace('nan','\\nodata')
            outfile.write(s)
        outfile.write('\\bottomrule\n')
        outfile.write('\\hline\n')
        outfile.write('\\end{tabular}\n')
        outfile.write('\\end{center} \n')
        outfile.write('\\tablecomments{Table 2 is published in its entirety in the machine-readable format.  A portion is shown here for guidance regarding its form and content.}')
        
        outfile.write('\\end{sidewaystable*} \n')
        outfile.close()
    def write_full_tables(self):
        # table to assist with UAT summer research
        # will add columns that should be useful for lots of projects
        #
        # basically all of table 2:
        # a100
        # GSWLC mass, sfr
        # logMstarTaylor, err
        # logMstarMcGaugh, err
        # logSFR22, err
        # logSFRNUV, err
        # log SFR_NUVIR, err
        # logMHI, err
        # g,i
        #
        # plus RA, DEC, vhelio, D(err)
        dateTimeObj = datetime.now()
        myDate = dateTimeObj.strftime("%d-%b-%Y")

        tab1 = Table([self.tab['AGC'],self.tab['sdssPhotFlag'],self.tab['objID_1'],\
                      self.tab['RAdeg_Use'],self.tab['DECdeg_Use'],self.tab['Vhelio'],\
                      self.tab['Dist'],self.tab['sigDist'],\
                      self.tab['extinction_g'],self.tab['extinction_i'],\
                      self.tab['expAB_r'],self.expAB_r_err,\
                      self.tab['cModelMag_i'],self.tab['cModelMagErr_i']],
                     names=['AGC','sdssPhotFlag','sdss_objid',\
                            'RA','DEC','Vhelio',\
                            'Dist','sigDist',\
                            'extinction_g','extinction_i','expAB_r','expAB_r_err',\
                            'cModelMag_i','cModelMagErr_i'])
        tab1.write(tablepath+'durbala2020-table1.'+myDate+'.fits',format='fits',overwrite=True)


        tab2 = Table([self.tab['AGC'],self.tab['gamma_g'],self.tab['gamma_i'],self.tab['absMag_i_corr'],\
                      self.absMag_i_corr_err,self.tab['gmi_corr'],self.gmi_corr_err,\
                      self.tab['logMstarTaylor'],self.logMstarTaylor_err,\
                      self.tab['logMstarMcGaugh'],self.logMstarMcGaugh_err,\
                      self.gsw_mstar,self.gsw_mstar_err,\
                      self.sfr22,self.sfr22_err, self.sfrnuvir,self.sfrnuvir_err,\
                      self.gsw_sfr,self.gsw_sfr_err,\
                      self.tab['logMH'],self.tab['siglogMH']], \
                     names=['AGC','gamma_g','gamma_i','absMag_i_corr',\
                            'absMag_i_corr_err','gmi_corr','gmi_corr_err',\
                            'logMstarTaylor','logMstarTaylor_err',\
                            'logMstarMcGaugh','logMstarMcGaugh_err',\
                            'logMstarGSWLC','logMstarGSWLC_err',\
                            'logSFR22','logSFR22_err', 'logSFRNUVIR','logSFRNUVIR_err',\
                            'logSFRGSWLC','logSFRGSWLC_err',\
                            'logMH','logMH_err'])

        tab2.write(tablepath+'durbala2020-table2.'+myDate+'.fits',format='fits',overwrite=True)

        # machine readable tables for AAS journal
        #tab1.write(latextablepath+'durbala2020-table1.'+myDate+'.csv',format='csv',overwrite=True)        
        #tab2.write(latextablepath+'durbala2020-table2.'+myDate+'.csv',format='csv',overwrite=True)
        #ascii.write(tab1,latextablepath+'durbala2020-table1.'+myDate+'.txt',format='cds',overwrite=True)        
        #ascii.write(tab2,latextablepath+'durbala2020-table2.'+myDate+'.txt',format='cds',overwrite=True)

        # write out full latex tables for AAS journal
        self.print_table1(nlines=len(self.tab),filename=latextablepath+'table1_long.'+myDate+'.tex',papertableflag=False)
        self.print_table2(nlines=len(self.tab),filename=latextablepath+'table2_long.'+myDate+'.tex',papertableflag=False)        
        shutil.copy(latextablepath+'table1_long.'+myDate+'.tex',latextablepath+'table1_long.tex')
        shutil.copy(latextablepath+'table2_long.'+myDate+'.tex',latextablepath+'table2_long.tex')        
        pass
if __name__ == '__main__':
    t = latextable()
    #t.calculate_errors()
    #t.clean_arrays()
    #t.print_table1()
    #t.print_table2()
    #t.write_full_tables()
