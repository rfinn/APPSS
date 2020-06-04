#!/usr/bin/env python
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table
from datetime import datetime

homedir = os.getenv('HOME')
tablepath = homedir+'/research/APPSS/tables/'
latextablepath = homedir+'/research/APPSS/latex-tables/'


class latextable():
    def __init__(self):
        # read in data table
        self.tab = fits.getdata(tablepath+'a100-sdss-wise.fits')
        # the next table has the NSA data appended to the end columns
        # it contains the full NSA, but the first rows should be line-matched to the a100-sdss-wise
        self.tab2 = fits.getdata(tablepath+'full-a100-sdss-wise-nsa.fits')

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
        self.gammag_err[zero_err_flag] = np.zeros(np.sum(zero_err_flag))
        # gamma_i
        self.gammai_err = 0.15*np.sqrt(self.tab['cModelMagErr_i']**2 + self.mw_ext_i_err**2 + (5/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2)
        zero_err_flag = self.tab['absMag_i_corr'] >= -17
        self.gammai_err[zero_err_flag] = np.zeros(np.sum(zero_err_flag))

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
        
        # logMstarTaylor
        self.logMstarTaylor_err = np.sqrt(0.49*self.gmi_corr_err**2 + 0.16*self.absMag_i_corr_err**2)
        
        # McGaugh
        self.absMag_W1_err = np.sqrt(self.tab['w1_mag_err']**2 +  (5/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2)
        self.logMstarMcGaugh_err = 0.4*self.absMag_W1_err
        # SFR NUV_corr

        
        # SFR 22
        self.logSFR22_KE_err = np.sqrt(0.16*self.tab['w4_mag_err']**2 + (2/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2)

        self.logSFR_NUV_KE_err = np.sqrt(0.16*self.tab2['SERSIC_ABSMAG'][:,1]**2 + (2/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2)

        # hold on to your hat for this one...

        #A = 1/22*np.power(10,-1*(self.tab['w4_mag']+6.62)/2.5) + 2.26/0.23*np.power(10,-1*self.tab2['SERSIC_ABSMAG'][:,1]/2.5)
        A_w4 =  1/22*np.power(10,-1*(self.tab['w4_mag']+6.62)/2.5)
        A_nuv = 2.26/0.23*np.power(10,-1*self.tab2['SERSIC_ABSMAG'][:,1]/2.5)
        A = A_w4 + A_nuv

        sigsq_d = (2/(self.tab['Dist']*np.log(10)))**2*self.tab['sigdist']**2
        sigsq_nuv = A_nuv**2*1./self.tab2['SERSIC_AMIVAR'][:,1]
        sigsq_w4 = A_w4**2*self.tab['w4_mag_err']**2
        self.logSFR_NUVIR_KE_err = np.sqrt(sigsq_d + (1./A/2.5)**2*(sigsq_nuv + sigsq_w4))
        self.sigsq_nuv = sigsq_nuv
        self.A_w4 = A_w4
        self.A_nuv = A_nuv
    def clean_sfrs(self):
        '''
        remove bogus values from SFR estimates
        need to make a flag
        '''
        self.sfr22flag = (self.tab['w4_mag'] > 0) & (~np.isnan(self.tab['w4_mag']))
        self.sfr22 = -99*np.ones(len(self.tab),'f')
        self.sfr22_err = -99*np.ones(len(self.tab),'f')                
        self.sfr22[self.sfr22flag] = self.tab['logSFR22_KE'][self.sfr22flag]
        self.sfr22_err[self.sfr22flag] = self.logSFR22_KE_err[self.sfr22flag]

        self.sfrnuvflag = self.tab2['SERSIC_ABSMAG'][:,1] < 0.
        self.sfrnuv = -99*np.ones(len(self.tab),'f')
        self.sfrnuv_err = -99*np.ones(len(self.tab),'f')                

        self.sfrnuv[self.sfrnuvflag] = self.tab2['logSFR_NUV_KE'][self.sfrnuvflag]
        self.sfrnuv_err[self.sfrnuvflag] = self.logSFR_NUV_KE_err[self.sfrnuvflag]

        self.sfrnuvir = -99*np.ones(len(self.tab),'f')
        self.sfrnuvir_err = -99*np.ones(len(self.tab),'f')                
        flag = self.sfrnuvflag & self.sfr22flag
        
        self.sfrnuvir[flag] = self.tab2['logSFR_NUVIR_KE'][flag]
        self.sfrnuvir_err[flag] = self.logSFR_NUVIR_KE_err[flag]
        inf_flag = np.isinf(self.sfrnuvir_err)
        print('galaxies with UV+IR sfr err = inf')
        printindices = np.arange(len(inf_flag))[inf_flag]
        for i in printindices:
            print('{0:02d} {1:.2f} {2:.2f} {3:.2f} {4:.2f} {5:.2f} {6:.2f} {7:.2f} {8:.2e} {9:.2e} {10:.2e}'.format(i,self.sfr22[i],self.sfr22_err[i],self.sfrnuv[i],self.sfrnuv_err[i],self.sfrnuvir[i],self.sfrnuvir_err[i],self.tab2['SERSIC_ABSMAG'][:,1][i],1./np.sqrt(self.tab2['SERSIC_AMIVAR'][:,1][i]),self.A_w4[i],self.A_nuv[i]))

    def print_table1(self):
        outfile = open(latextablepath+'table1.tex','w')
        outfile.write('\\begin{table*}%[ptbh!]\n')
        outfile.write('\\begin{center}\n')
        outfile.write('\\scriptsize\n')
        outfile.write('\\setlength\\tabcolsep{3.0pt} \n')
        outfile.write('\\tablenum{1} \n')
        outfile.write('\\caption{Basic Optical Properties of Cross-listed objects in the $\\alpha.100$-SDSS Catalog\label{tab:catalog1}} \n')
        outfile.write('\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\n')
        outfile.write('\\hline \n')
        outfile.write('\\toprule \n')
        outfile.write('AGC &	Flag &	SDSS objID & unWISE object ID & RA &	DEC &	V$_{helio}$ &	D &	$\sigma_D$  &	Ext$_g$	& Ext$_i$	& expAB$_r$  & $\sigma_{expAB_r}$ &	cmodel$_i$ & $\sigma_{cmodel_i}$ \\\\ \n')
        outfile.write('& & & & J2000 & J2000 & $\\kms$ & Mpc & Mpc & mag & mag & & & mag & mag \\\\ \n')
        outfile.write('(1) & (2) & (3) & (4) & (5) & (6) & (7) & (8) & (9) & (10) & (11) & (12) & (13) & (14) & (15) \\\\ \n')
        outfile.write('\\midrule \n')
        outfile.write('\\hline \n')
        for i in range(25): # print first N lines of data
            # AGC photflag sdss_objid wiseobjid RA DEC vhelio D D_err Ext_g Ext_i expABr err cmodelI err
            s = '{0:d} & {1:d} & {2:d} & {3:d}& {4:9.6f}&{5:9.5f} & {6:d} & {7:.1f} & {8:.1f} &  {9:.2f} & {10:.2f}& {11:.2f}& {12:.2f}& {13:.2f} &{14:.2f}\\\\ \n'.format(self.tab['AGC'][i],self.tab['sdssPhotFlag'][i],self.tab['objID_1'][i],self.tab['unwise_objid'][i],self.tab['RAdeg_Use'][i],self.tab['DECdeg_Use'][i],self.tab['Vhelio'][i],self.tab['Dist'][i],self.tab['sigDist'][i],self.tab['extinction_g'][i],self.tab['extinction_i'][i],self.tab['expAB_r'][i],self.expAB_r_err[i],self.tab['cModelMag_i'][i],self.tab['cModelMagErr_i'][i])
            outfile.write(s)

        outfile.write('\\bottomrule \n')
        outfile.write('\\hline \n')
        outfile.write('\\end{tabular} \n')
        outfile.write('\\end{center} \n')
        outfile.write('\\end{table*} \n')
        outfile.close()
    def print_table2(self):
        '''
        changes from old verion:
        - remove Cluver stellar mass (2)
        - add error SFR22
        - add err SFRUV
        - add err SFRUVcorr

        net gain of one column
        '''
        outfile = open(latextablepath+'table2.tex','w')
        outfile.write('\\begin{sidewaystable*}%[ptbh!]%\\tiny \n')
        outfile.write('\\begin{center}\n')
        outfile.write('\\tablewidth{0.5\\textwidth} \n')
        outfile.write('\\scriptsize \n')
        outfile.write('%\\footnotesize \n')
        outfile.write('\\setlength\\tabcolsep{1.0pt} \n')
        outfile.write('\\tablenum{2}\n')
        outfile.write('\\caption{Derived Properties of Cross-listed objects in the $\\alpha.100$-SDSS Catalog\\label{tab:catalog2}}\n')
        outfile.write('\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\n')
        outfile.write('\\hline\n')
        outfile.write('\\toprule\n')
        #outfile.write('AGC & $\\gamma_g$ & $\\sigma_{\\gamma_g}$ & $\\gamma_i$ & $\\sigma_{\\gamma_i}$ & M$_{icorr}$ &	$\\sigma_{M_{icorr}}$ &	(g-i)$_{corr}$	& $\\sigma_{(g-i)_{corr}}$ & log M$_{\\star, Taylor}$ &	$\\sigma_{log M_{\\star,Taylor}}$  & log M$_{\\star, McGaugh}$ &	$\\sigma_{log M_{\\star, McGaugh}}$ & SFR$_{22}$ & $\\sigma_{log SFR_{22}}$ & ${SFR_{NUV}}$ & $\\sigma_{log SFR_{NUV}}$ & SFR$_{NUVIR}$ & $\\sigma_{log SFR_{UVIR}}$ & M$_{HI}$ & $\\sigma_{M_{HI}}$  \\\\\n')
        outfile.write('AGC & $\\gamma_g$ &  $\\gamma_i$  & M$_{icorr}$ &	$\\rm \\sigma_{M_{icorr}}$ &	(g-i)$_{corr}$	& $\\sigma_{(g-i)_{corr}}$ & log M$_{\\star}$ &	$\\rm \\sigma_{log M_{\\star}}$  & log M$_{\\star}$ &	$\\rm \\sigma_{log M_{\\star}}$ & logSFR$_{22}$ & $\\rm \\sigma_{log SFR_{22}}$ & $\\rm {SFR_{NUV}}$ & $\\rm \\sigma_{log SFR_{NUV}}$ & logSFR$\\rm _{NUVIR}$ & $\\rm \\sigma_{log SFR_{UVIR}}$ & M$_{HI}$ & $\\rm \\sigma_{M_{HI}}$  \\\\\n')
        outfile.write('&   & &  & &	& & Taylor & Taylor  & McGaugh & McGaugh & & &  & &  &  & &   \\\\\n')
        outfile.write(' & mag & mag & mag & mag & mag & mag & $log(M_\\odot)$ & $log(M_\\odot)$ & $log(M_\\odot)$ & $log(M_\\odot)$ & $\\rm log(M_\\odot~yr^{-1})$ & $\\rm log(M_\\odot yr^{-1})$ & $\\rm log(M_\\odot~yr^{-1})$ & $\\rm (M_\\odot ~yr^{-1})$ & $\\rm log(M_\\odot~ yr^{-1})$ & $\\rm log(M_\\odot) yr^{-1}$ & $log(M_\\odot)$ & $log(M_\\odot)$ \\\\\n')
        outfile.write('(1) & (2) & (3) & (4) & (5) & (6) & (7) & (8) & (9) & (10) & (11) & (12) & (13) & (14) & (15) & (16) & (17) & (18) & (19) \\\\\n')
        outfile.write('\\midrule\n')
        outfile.write('\\hline\n')
        for i in range(25):
            try:
                j = self.agcdict2[self.tab['AGC'][i]]
            except KeyError:
                print(self.tab['AGC'][i])
            #print(self.tab['AGC'][i],j,self.tab2['AGC'][j])
            s=' {0:d} & {1:.2f} & {2:.2f} & {3:.2f} & {4:.2f}& {5:.2f}  & {6:.2f} & {7:.2f} & {8:.2f} & {9:.2f}& {10:.2f}&{11:.2f} &{12:.2f} &{13:.2f} &{14:.2f} &{15:.2f} &{16:.2f} &{17:.2f} & {18:.2f}  \\\\ \n'.format(self.tab['AGC'][i],self.tab['gamma_g'][i],self.tab['gamma_i'][i],self.tab['absMag_i_corr'][i],self.absMag_i_corr_err[i],self.tab['gmi_corr'][i],self.gmi_corr_err[i],self.tab['logMstarTaylor'][i],self.logMstarTaylor_err[i],self.tab['logMstarMcGaugh'][i],self.logMstarMcGaugh_err[i],self.sfr22[i],self.sfr22_err[i], self.sfrnuv[i],self.sfrnuv_err[i], self.sfrnuvir[i],self.sfrnuvir_err[i],self.tab['logMH'][i],self.tab['siglogMH'][i])
            outfile.write(s)
        outfile.write('\\bottomrule\n')
        outfile.write('\\hline\n')
        outfile.write('\\end{tabular}\n')
        outfile.write('\\end{center} \n')
        outfile.write('\\end{sidewaystable*} \n')
        outfile.close()
    def print_table2_old(self):
        outfile = open(tablepath+'table2.tex','w')
        outfile.write('\\begin{sidewaystable*}%[ptbh!]%\\tiny \n')
        outfile.write('\\begin{center}\n')
        outfile.write('\\tablewidth{0.5\\textwidth} \n')
        outfile.write('\\scriptsize \n')
        outfile.write('%\\footnotesize \n')
        outfile.write('\\setlength\\tabcolsep{1.0pt} \n')
        outfile.write('\\tablenum{2}\n')
        outfile.write('\\caption{Derived Properties of Cross-listed objects in the $\\alpha.100$-SDSS Catalog\\label{tab:catalog}}\n')
        outfile.write('\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\n')
        outfile.write('\\hline\n')
        outfile.write('\\toprule\n')
        outfile.write('AGC & $\\gamma_g$ & $\\sigma_{\\gamma_g}$ & $\\gamma_i$ & $\\sigma_{\\gamma_i}$ & M$_{icorr}$ &	$\\sigma_{M_{icorr}}$ &	(g-i)$_{corr}$	& $\\sigma_{(g-i)_{corr}}$ & log M$_{\\star, Taylor}$ &	$\\sigma_{log M_{\\star,Taylor}}$ & log M$_{\\star, Cluver}$  &	$\\sigma_{log M_{\\star, Cluver}}$	& log M$_{\\star, McGaugh}$ &	$\\sigma_{log M_{\\star, McGaugh}}$ & SFR$_{22}$ &  ${SFR_{NUV}}$ & SFR$_{NUVIR}$ & M$_{HI}$ & $\\sigma_{M_{HI}}$  \\\\\n')
        outfile.write(' & mag & mag & mag & mag & mag & mag & mag & mag & $log(M_\\odot)$ & $log(M_\\odot)$ & $log(M_\\odot)$ & $log(M_\\odot)$ & $log(M_\\odot)$ & $log(M_\\odot)$ & $(M_\\odot) yr^{-1}$ & $(M_\\odot) yr^{-1}$ & $(M_\\odot) yr^{-1}$ & $log(M_\\odot)$ & $log(M_\\odot)$ \\\\\n')
        outfile.write('(1) & (2) & (3) & (4) & (5) & (6) & (7) & (8) & (9) & (10) & (11) & (12) & (13) & (14) & (15) & (16) & (17) & (18) & (19)& (20) \\\\\n')
        outfile.write('\\midrule\n')
        outfile.write('\\hline\n')
        for i in range(25):
            try:
                j = self.agcdict2[self.tab['AGC'][i]]
                sfruv = self.tab2['logSFR_NUV_KE'][j]
                sfruvir = self.tab2['logSFR_NUVIR_KE'][j]
            except KeyError:
                print(self.tab['AGC'][i])
                sfruv=0
                sfruvir=0
            #print(self.tab['AGC'][i],j,self.tab2['AGC'][j])
            s=' {0:d} & {1:.2f} & {2:.2f} & {3:.2f} & {4:.2f}& {5:.2f}  & {6:.2f} & {7:.2f} & {8:.2f} & {9:.2f}& {10:.2f}&{11:.2f} &{12:.2f} &{13:.2f} &{14:.2f} &{15:.2f} &{16:.2f} &{17:.2f} & {18:.2f} \\\\ \n'.format(self.tab['AGC'][i],self.tab['gamma_g'][i],0,self.tab['gamma_i'][i],0,self.tab['absMag_i_corr'][i],0,self.tab['gmi_corr'][i],0,self.tab['logMstarTaylor'][i],0,self.tab['logMstarCluver'][i],0,self.tab['logMstarMcGaugh'][i],0,self.tab['logSFR22_KE'][i],sfruv,sfruvir,self.tab['logMH'][i],self.tab['siglogMH'][i])
            outfile.write(s)
        outfile.write('\\bottomrule\n')
        outfile.write('\\hline\n')
        outfile.write('\\end{tabular}\n')
        outfile.write('\\end{center} \n')
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
        tab1 = Table([self.tab['AGC'],self.tab['sdssPhotFlag'],self.tab['objID_1'],self.tab['unwise_objid'],self.tab['RAdeg_Use'],self.tab['DECdeg_Use'],self.tab['Vhelio'],self.tab['Dist'],self.tab['sigDist'],self.tab['extinction_g'],self.tab['extinction_i'],self.tab['expAB_r'],self.expAB_r_err,self.tab['cModelMag_i'],self.tab['cModelMagErr_i']], \
                     names=['AGC','sdssPhotFlag','sdss_objid','unwise_objid','RA','DEC','Vhelio','Dist','sigDist','extinction_g','extinction_i','expAB_r','expAB_r_err','cModelMag_i','cModelMagErr_i'])
        tab1.write(tablepath+'durbala2020-table1.'+myDate+'.fits',format='fits')

        tab2 = Table([self.tab['AGC'],self.tab['gamma_g'],self.tab['gamma_i'],self.tab['absMag_i_corr'],self.absMag_i_corr_err,self.tab['gmi_corr'],self.gmi_corr_err,self.tab['logMstarTaylor'],self.logMstarTaylor_err,self.tab['logMstarMcGaugh'],self.logMstarMcGaugh_err,self.sfr22,self.sfr22_err, self.sfrnuv,self.sfrnuv_err, self.sfrnuvir,self.sfrnuvir_err,self.tab['logMH'],self.tab['siglogMH']],\
                     names=['AGC','gamma_g','gamma_i','absMag_i_corr','absMag_i_corr_err','gmi_corr','gmi_corr_err','logMstarTaylor','logMstarTaylor_err','logMstarMcGaugh','logMstarMcGaugh_err','logSFR22','logSFR22_err', 'logSFRNUV','logSFRNUV_err', 'logSFRNUVIR','logSFRNUVIR_err','logMH','logMH_err'])
        tab2.write(tablepath+'durbala2020-table2.'+myDate+'.fits',format='fits')
        pass
if __name__ == '__main__':
    t = latextable()
    t.calculate_errors()
    t.clean_sfrs()
    t.print_table1()
    t.print_table2()
    t.write_full_tables()
