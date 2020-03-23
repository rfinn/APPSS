#!/usr/bin/env python
import numpy as np
import os
from astropy.io import fits


homedir = os.getenv('HOME')
tablepath = homedir+'/research/APPSS/tables/'


class latextable():
    def __init__(self):
        # read in data table
        self.tab = fits.getdata(tablepath+'a100-sdss-wise.fits')
        self.tab2 = fits.getdata(tablepath+'full-a100-sdss-wise-nsa.fits')
        self.agcdict2=dict((a,b) for a,b in zip(self.tab2['AGC'],np.arange(len(self.tab2['AGC']))))
        pass
    def print_table1(self):
        outfile = open(tablepath+'table1.tex','w')
        outfile.write('\\begin{table*}%[ptbh!]\n')
        outfile.write('\\begin{center}\n')
        outfile.write('\\scriptsize\n')
        outfile.write('\\setlength\\tabcolsep{3.0pt} \n')
        outfile.write('\\tablenum{1} \n')
        outfile.write('\\caption{Basic Optical Properties of Cross-listed objects in the $\\alpha.100$-SDSS Catalog\label{tab:catalog}} \n')
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
            s = '{0:d} & {1:d} & {2:d} & {3:d}& {4:9.6f}&{5:9.5f} & {6:d} & {7:.1f} & {8:.1f} &  {9:.2f} & {10:.2f}& {11:.2f}& {12:.2f}& {13:.2f} &{14:.2f}\\\\ \n'.format(self.tab['AGC'][i],self.tab['photFlag_gi'][i],self.tab['objID_1'][i],self.tab['unwise_objid'][i],self.tab['RAdeg_Use'][i],self.tab['DECdeg_Use'][i],self.tab['Vhelio'][i],self.tab['Dist'][i],self.tab['sigDist'][i],self.tab['extinction_g'][i],self.tab['extinction_i'][i],self.tab['expAB_r'][i],0,self.tab['cModelMag_i'][i],self.tab['cModelMagErr_i'][i])
            outfile.write(s)

        outfile.write('\\bottomrule \n')
        outfile.write('\\hline \n')
        outfile.write('\\end{tabular} \n')
        outfile.write('\\end{center} \n')
        outfile.write('\\end{table*} \n')
        outfile.close()
    def print_table2(self):
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
if __name__ == '__main__':
    t = latextable()
    t.print_table1()
    t.print_table2()
