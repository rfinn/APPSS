#!/usr/bin/env python

'''
writing this for APPSS analysis

match a100 with many other catalogs

generate plots for paper

'''

import numpy as np
from matplotlib import pyplot as plt

import os
from astropy.io import fits
from astropy.io import ascii
from astropy import constants as c
from astropy import units as u
from astropy.table import Table, join
from astropy.coordinates import SkyCoord
from scipy.stats import ks_2samp
plt.rcParams.update({'font.size': 14})

import sys

## some plot parameters

# max and min for g-i plots
gimin = -.5
gimax =2

def ks(x,y,run_anderson=True):
    #D,pvalue=ks_2samp(x,y)
    D,pvalue=ks_2samp(x,y)
    print('KS Test (median of bootstrap):')
    print('D = %6.2f'%(D))
    print('p-value = %3.2e (prob that samples are from same distribution)'%(pvalue))
    if run_anderson:
        anderson(x,y)
    return D,pvalue

def anderson(x,y):
    t=anderson_ksamp([x,y])
    print('Anderson-Darling test Test:')
    print('D = %6.2f'%(t[0]))
    print('p-value = %3.2e (prob that samples are from same distribution)'%(t[2]) )
    return t[0],t[2]


def colormass(x1,y1,x2,y2,name1,name2, figname, hexbinflag=False,contourflag=False, \
             xmin=7.9, xmax=11.6, ymin=-1.2, ymax=1.2, contour_bins = 40, ncontour_levels=5,\
             xlabel='$\log_{10}(M_\star/M_\odot) $', ylabel='$(g-i)_{corrected} $', color2='c',\
             nhistbin=50, alphagray=.1):
    fig = plt.figure(figsize=(8,8))
    nrow = 4
    ncol = 4
    
    # for purposes of this plot, only keep data within the 
    # window specified by [xmin:xmax, ymin:ymax]
    
    keepflag1 = (x1 >= xmin) & (x1 <= xmax) & (y1 >= ymin) & (y1 <= ymax)
    keepflag2 = (x2 >= xmin) & (x2 <= xmax) & (y2 >= ymin) & (y2 <= ymax)
    
    x1 = x1[keepflag1]
    y1 = y1[keepflag1]
    
    x2 = x2[keepflag2]
    y2 = y2[keepflag2]
    
    ax1 = plt.subplot2grid((nrow,ncol),(1,0),rowspan=nrow-1,colspan=ncol-1, fig=fig)
    if hexbinflag:
        #t1 = plt.hist2d(x1,y1,bins=100,cmap='gray_r')
        #H, xbins,ybins = np.histogram2d(x1,y1,bins=20)
        #extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        #plt.contour(np.log10(H.T+1),  10, extent = extent, zorder=1,colors='k')
        #plt.hexbin(xvar2,yvar2,bins='log',cmap='Blues', gridsize=100)

        plt.hexbin(x1,y1,bins='log',cmap='gray_r', gridsize=75,label=name1)
    else:
        plt.plot(x1,y1,'k.',alpha=alphagray,label=name1, zorder=2)
    if contourflag:
        H, xbins,ybins = np.histogram2d(x2,y2,bins=contour_bins)
        extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        plt.contour((H.T), levels=ncontour_levels, extent = extent, zorder=1,colors=color2, label='__nolegend__')
        #plt.legend()
    else:
        plt.plot(x2,y2,'c.',color=color2,alpha=.3, label=name2)
        
        
        #plt.legend()
    #sns.kdeplot(agc['LogMstarTaylor'][keepagc],agc['gmi_corrected'][keepagc])#,bins='log',gridsize=200,cmap='blue_r')
    #plt.colorbar()
    
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(xlabel,fontsize=22)
    plt.ylabel(ylabel,fontsize=22)
    #plt.axis([7.9,11.6,-.05,2])
    ax2 = plt.subplot2grid((nrow,ncol),(0,0),rowspan=1,colspan=ncol-1, fig=fig, sharex = ax1, yticks=[])
    t = plt.hist(x1, normed=True, bins=nhistbin,color='k',histtype='step',lw=1.5, label=name1)
    t = plt.hist(x2, normed=True, bins=nhistbin,color=color2,histtype='step',lw=1.5, label=name2)
    #plt.legend()
    ax2.legend(fontsize=10,loc='upper left')
    ax2.xaxis.tick_top()
    ax3 = plt.subplot2grid((nrow,ncol),(1,ncol-1),rowspan=nrow-1,colspan=1, fig=fig, sharey = ax1, xticks=[])
    t=plt.hist(y1, normed=True, orientation='horizontal',bins=nhistbin,color='k',histtype='step',lw=1.5, label=name1)
    t=plt.hist(y2, normed=True, orientation='horizontal',bins=nhistbin,color=color2,histtype='step',lw=1.5, label=name2)
    
    plt.yticks(rotation='horizontal')
    ax3.yaxis.tick_right()
    plt.savefig(figname)

    print('############################################################# ')
    print('KS test comparising galaxies within range shown on the plot')
    print('')
    print('STELLAR MASS')
    t = ks(x1,x2,run_anderson=False)
    print('')
    print('COLOR')
    t = ks(y1,y2,run_anderson=False)


def sfrmass(x1,y1,x2,y2,name1,name2, figname, hexbinflag=False,contourflag=False, \
             xmin=7.9, xmax=11.6, ymin=-1.2, ymax=1.2, contour_bins = 40, ncontour_levels=5,\
             xlabel='$\log_{10}(M_\star/M_\odot) $', ylabel='$\log_{10}(M_\odot/yr)$', color2='c',\
             nhistbin=50, alphagray=.1,plotline=True):
    fig = plt.figure(figsize=(8,8))
    nrow = 4
    ncol = 4
    
    # for purposes of this plot, only keep data within the 
    # window specified by [xmin:xmax, ymin:ymax]
    
    keepflag1 = (x1 >= xmin) & (x1 <= xmax) & (y1 >= ymin) & (y1 <= ymax)
    keepflag2 = (x2 >= xmin) & (x2 <= xmax) & (y2 >= ymin) & (y2 <= ymax)
    
    x1 = x1[keepflag1]
    y1 = y1[keepflag1]
    
    x2 = x2[keepflag2]
    y2 = y2[keepflag2]
    
    ax1 = plt.subplot2grid((nrow,ncol),(1,0),rowspan=nrow-1,colspan=ncol-1, fig=fig)
    if hexbinflag:
        #t1 = plt.hist2d(x1,y1,bins=100,cmap='gray_r')
        #H, xbins,ybins = np.histogram2d(x1,y1,bins=20)
        #extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        #plt.contour(np.log10(H.T+1),  10, extent = extent, zorder=1,colors='k')
        #plt.hexbin(xvar2,yvar2,bins='log',cmap='Blues', gridsize=100)

        plt.hexbin(x1,y1,bins='log',cmap='gray_r', gridsize=75,label=name1)
    else:
        plt.plot(x1,y1,'k.',alpha=alphagray,label=name1, zorder=2)
    if contourflag:
        H, xbins,ybins = np.histogram2d(x2,y2,bins=contour_bins)
        extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        plt.contour((H.T), levels=ncontour_levels, extent = extent, zorder=1,colors=color2, label='__nolegend__')
        #plt.legend()
    else:
        plt.plot(x2,y2,'c.',color=color2,alpha=.3, label=name2)
    if plotline:
        xl = np.linspace(xmin,xmax,100)
        slope=.75
        xref, yref = 9.45,-.44
        yl = yref + slope*(xl - xref)
        plt.plot(xl,yl,'k--')

        #plt.legend()
    #sns.kdeplot(agc['LogMstarTaylor'][keepagc],agc['gmi_corrected'][keepagc])#,bins='log',gridsize=200,cmap='blue_r')
    #plt.colorbar()
    
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(xlabel,fontsize=22)
    plt.ylabel(ylabel,fontsize=22)
    #plt.axis([7.9,11.6,-.05,2])
    ax2 = plt.subplot2grid((nrow,ncol),(0,0),rowspan=1,colspan=ncol-1, fig=fig, sharex = ax1, yticks=[])
    t = plt.hist(x1, normed=True, bins=nhistbin,color='k',histtype='step',lw=1.5, label=name1)
    t = plt.hist(x2, normed=True, bins=nhistbin,color=color2,histtype='step',lw=1.5, label=name2)
    #plt.legend()
    ax2.legend(fontsize=10, loc='upper left')
    ax2.xaxis.tick_top()
    ax3 = plt.subplot2grid((nrow,ncol),(1,ncol-1),rowspan=nrow-1,colspan=1, fig=fig, sharey = ax1, xticks=[])
    t=plt.hist(y1, normed=True, orientation='horizontal',bins=nhistbin,color='k',histtype='step',lw=1.5, label=name1)
    t=plt.hist(y2, normed=True, orientation='horizontal',bins=nhistbin,color=color2,histtype='step',lw=1.5, label=name2)
    
    plt.yticks(rotation='horizontal')
    ax3.yaxis.tick_right()
    plt.savefig(figname)

    print('############################################################# ')
    print('KS test comparising galaxies within range shown on the plot')
    print('')
    print('STELLAR MASS')
    t = ks(x1,x2,run_anderson=False)
    print('')
    print('COLOR')
    t = ks(y1,y2,run_anderson=False)


    
def colormag(mag, color, ab, ylabel):
    #plt.plot(mag, color,  'k.', alpha=.05)
    limits = [-23.9, -14.1, -.5, 1.9]
    flag = ab > 0.8
    #plt.hexbin(mag[flag],color[flag], extent=limits, cmap='gray_r', vmin=0,vmax=45)
    plt.plot(mag[flag],color[flag],'k.',alpha=.2, markersize=3, label='$B/A > 0.8$')
    flag = ab < 0.3
    #plt.hexbin(mag[flag],color[flag], extent=limits, cmap='Purples_r', vmin=0,vmax=45)
    plt.plot(mag[flag],color[flag],'c.',alpha=.2,markersize=3,label='$B/A < 0.3$')
    plt.axis(limits)
    plt.xticks(np.arange(-23,-14,2))
    plt.gca().invert_xaxis()
    plt.title(ylabel, fontsize=14)

class matchedcats():
    '''
    This is what I've done so far using Mary's matched catalogs for
    a100 - NSA
    a100 - S4G
    '''
    def __init__(self,a100sdss=None, a100nsa=None,a100gsw=None,a100s4g=None):
        self.a100sdss = fits.getdata(a100sdss)

        self.a100nsa = fits.getdata(a100nsa)
        self.a100gsw = fits.getdata(a100gsw)
        self.a100s4g = fits.getdata(a100s4g)
    def figure1(self):
        plt.figure(figsize=(10,4))
        plt.subplots_adjust(wspace=.0,bottom=.2)
        # 
        plt.subplot(1,3,1)
        photflag = self.a100sdss['photFlag_gi'] == 1
        colormag(self.a100sdss['absMag_i'][photflag], self.a100sdss['gmi_no_int'][photflag], self.a100sdss['expAB_g'][photflag],'${uncorrected}$')
        plt.ylabel('$g-i$',fontsize=16)
        plt.legend(loc='lower left',markerscale=6)
        #plt.gca()
        
        plt.subplot(1,3,2)
        photflag = self.a100sdss['photFlag_gi'] == 1
        colormag(self.a100sdss['I_Shao'][photflag], self.a100sdss['gmi_Shao'][photflag],
        self.a100sdss['expAB_g'][photflag],'${Shao\ et\ al. \ 2007}$')
        plt.yticks(())
        ax = plt.gca()

        plt.text(.5,-.25,'$M_i$',fontsize=16, transform = ax.transAxes)

        plt.subplot(1,3,3)
        photflag = self.a100sdss['photFlag_gi'] == 1
        colormag(self.a100sdss['absMag_i_corr'][photflag], self.a100sdss['gmi_corr'][photflag],
        self.a100sdss['expAB_g'][photflag],'${this \ work}$')
        plt.yticks(())
        
        plt.savefig('Figure1.pdf')
        
    def figa_nsa(self):
        # correct to H0=70
        x = np.log10(self.a100nsa.SERSIC_MASS/.7**2)
        #x = np.log10(self.a100nsa.SERSIC_MASS)
        y = ((self.a100nsa.SERSIC_ABSMAG[:,3] - self.a100nsa.EXTINCTION[:,3]) - (self.a100nsa.SERSIC_ABSMAG[:,5] - self.a100nsa.EXTINCTION[:,5]))
        nsa_mass_flag =  ( self.a100nsa.SERSIC_MASS > 1000.) 
        # require phot error < 0.05 for abs mag
        ivar = 1./.05**2
        nsa_phot_flag = np.ones(len(nsa_mass_flag),'bool')

        # require low error in both g and i bands
        for i in np.arange(3,6):
            nsa_phot_flag = nsa_phot_flag & (self.a100nsa.SERSIC_AMIVAR[:,i] > (1./.05**2))
    
        # flag1 = (self.a100nsa.matchFlag == 3) & (self.a100nsa.photFlag_gi == 1)
        nsa_flag = nsa_mass_flag & nsa_phot_flag
        flag1 = (self.a100nsa.a100Flag == 1) & (self.a100nsa.nsaFlag ==1) & nsa_flag #& photflagnsa  & (gmi_corr_nsa > -1) & (gmi_corr_nsa < 3.)#
        # x1 = self.a100nsa.LogMstarTaylor_2[flag1]
        print('number of galaxies in NSA + A100 sample = ',sum(flag1))
        x1 = x[flag1]
        y1 = y[flag1]
        
        flag2 = ((self.a100nsa.a100Flag ==0) & (self.a100nsa.nsaFlag==1))& nsa_flag #& (logstellarmassTaylor_nsa > 7) & (gmi_corr_nsa > -1) & (gmi_corr_nsa < 3.)#& (self.a100nsa.photFlag_gi == 1) 
        x2 = x[flag2]
        y2 = y[flag2]
        print(len(x2), sum(flag2))
        contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(10,500,12)
        #print(contour_levels)
        colormass(x1,y1, x2, y2, 'A100+NSA', 'NSA only', 'a100-nsa-color-mass-2.pdf', \
                  hexbinflag=True,contourflag=True,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=5., xmax=12, ymin=gimin,ymax=gimax,\
                  xlabel='$NSA \ \log_{10}(M_\star/M_\odot)$', ylabel='$ NSA \ (M_g - M_i)$', color2='r')
        return flag1
    def figb_nsa(self):
        flag1 = (self.a100nsa.a100Flag == 1) & (self.a100nsa.nsaFlag == 1) & (self.a100nsa.photFlag_gi == 1)
        # x1 = self.a100nsa.LogMstarTaylor_2[flag1]
        x1 = self.a100nsa.logMstarTaylor[flag1]
        y1 = self.a100nsa.gmi_corr[flag1]

        flag2 = (self.a100nsa.a100Flag ==1) & (self.a100nsa.nsaFlag == 0) & (self.a100nsa.photFlag_gi == 1) 
        x2 = self.a100nsa.logMstarTaylor[flag2]
        y2 = self.a100nsa.gmi_corr[flag2]
        print(len(x2), sum(flag2))
        contour_levels = np.linspace(1,100,12)
        colormass(x1,y1, x2, y2, 'A100+NSA', 'A100 only', 'a100-nsa-color-mass-1.pdf', \
                  hexbinflag=True, contourflag=True,contour_bins=30,  ncontour_levels=contour_levels,\
                  color2='b',xmin=5.,xmax=12, ymin=gimin,ymax=gimax)

    def figa_gswlc(self):
        # figure a is catalog specific quantities
        
        
        gsw_flag = (self.a100gsw.logMstar > 0) & (self.a100gsw.photFlag_gi_2 == 1)
        flag1 = (self.a100gsw.a100Flag == 1) & (self.a100gsw.gswFlag ==1) & gsw_flag 
        x1 = self.a100gsw.logMstar[flag1]
        y1 = self.a100gsw.gmi_corr_2[flag1]

        flag2 = (self.a100gsw.a100Flag ==0) & (self.a100gsw.gswFlag == 1) & (self.a100gsw.photFlag_gi_2 == 1) 
        x2 = self.a100gsw.logMstar[flag2]
        y2 = self.a100gsw.gmi_corr_2[flag2]
        
        contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(2,500,12)
        #print(contour_levels)
        colormass(x1,y1, x2, y2, 'A100+GSWLC', 'GSWLC only', 'a100-gswlc-color-mass-2.pdf', \
                  hexbinflag=True,contourflag=True,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=5.,xmax=12,ymin=gimin,ymax=gimax, \
                  xlabel='$GSWLC \ \log_{10}(M_\star/M_\odot)$', ylabel='$  \ (M_g - M_i)_{corrected}$', color2='r')
        return flag1
    def figb_gswlc(self):
        # in both
        # AND
        # in A100 but not in GSWLC
        flag1 = (self.a100gsw.a100Flag == 1) & (self.a100gsw.gswFlag == 1) & (self.a100gsw.photFlag_gi_1 == 1)
        # x1 = self.a100nsa.LogMstarTaylor_2[flag1]
        x1 = self.a100gsw.logMstarTaylor_1[flag1]
        y1 = self.a100gsw.gmi_corr_1[flag1]

        flag2 = (self.a100gsw.a100Flag ==1) & (self.a100gsw.gswFlag == 0) & (self.a100gsw.photFlag_gi_1 == 1) 
        x2 = self.a100gsw.logMstarTaylor_1[flag2]
        y2 = self.a100gsw.gmi_corr_1[flag2]
        print(len(x2), sum(flag2))
        contour_levels = np.linspace(2,100,12)
        colormass(x1,y1, x2, y2, 'A100+GSWLC', 'A100 only', 'a100-gswlc-color-mass-1.pdf', \
                  hexbinflag=True, contourflag=True,contour_bins=30, ncontour_levels=contour_levels,\
                  color2='b',xmin=5.,xmax=12,ymin=gimin,ymax=gimax)
    def figa_s4g(self):
        x = self.a100s4g.mstar
        y = self.a100s4g.bvtc
        #x = a100s4g.mabs
        #y = fullsdss['g']-fullsdss['r']
        #y = a100s4g.mag1 - a100s4g.mag2
        photflag = self.a100s4g.photFlag_gi == 1
        flag1 = (self.a100s4g.a100Flag == 1) & (self.a100s4g.s4gFlag == 1) #& photflag
        print('number with both = ',sum(flag1))
        x1 = x[flag1]
        y1 = y[flag1]

        flag2 = (self.a100s4g.a100Flag == 0) & (self.a100s4g.s4gFlag == 1) #& photflag
        print('number with S4G only = ',sum(flag2))
        x2 = x[flag2]
        y2 = y[flag2]
        print(len(x2), sum(flag2))
        #contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(1,100,12)
        print(contour_levels)
        colormass(x1,y1, x2, y2, 'A100+S4G', 'S4G only', 'a100-s4g-color-mass-1.pdf', \
                  hexbinflag=False,contourflag=False,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=5,xmax=13,ymin=gimin, ymax=gimax,nhistbin=15,alphagray=.5,\
                  xlabel='$S4G \ M_{ABS}$', ylabel='$ Leda \ (B-V)$', color2='r')
    def figb_s4g(self):
        x = self.a100s4g.logMstarTaylor
        y = self.a100s4g.gmi_corr

        flag1 = (self.a100s4g.a100Flag == 1) & (self.a100s4g.s4gFlag == 1) & (self.a100s4g.photFlag_gi == 1)
        # x1 = self.a100nsa.LogMstarTaylor_2[flag1]
        x1 = x[flag1]
        y1 = y[flag1]

        flag2 = (self.a100s4g.a100Flag == 1) & (self.a100s4g.s4gFlag == 0)& (self.a100s4g.photFlag_gi == 1)
        x2 = x[flag2]
        y2 = y[flag2]
        #print(len(x2), sum(flag2))
        #contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(2,200,12)
        print(contour_levels)
        colormass(x1,y1, x2, y2, 'A100+S4G', 'A100 only', 'a100-s4g-color-mass-2.pdf', \
                  hexbinflag=False,contourflag=True,contour_bins=30, ncontour_levels=contour_levels,\
                  xmin=5.,xmax=12,ymin=gimin,ymax=gimax,nhistbin=20,alphagray=.5,color2='b')

    def mstar(self):
        # compare different estimates of stellar mass with our estimate from Taylor

        # 3 panel plot
        # panel one - NSA np.log10(SERSIC_MASS) vs logMstarTaylor
        # panel two - GSWLC Mass vs logMstarTaylor
        # panel three - S4G mstar vs logMstarTaylor

        cats = [self.a100nsa, self.a100gsw, self.a100s4g]
        xvar = ['logMstarTaylor','logMstarTaylor_1','logMstarTaylor']
        yvar = ['SERSIC_MASS','logMstar','mstar']
        flags = ['photFlag_gi','photFlag_gi_1','photFlag_gi']
        survey = ['$\log_{10}(NSA \ SERSIC\_MASS/M_\odot) $', \
                  '$\log_{10}(GSWLC \ Mstar/M_\odot )$', \
                  '$\log_{10}(S4G \ Mstar/M_\odot )$']
        survey = ["NSA \n" r"$\rm \log_{10}(M_\star/M_\odot) $", \
                  "GSWLC \n" r"$\rm \log_{10}(M_\star/M_\odot) $", \
                  "S4G \n " r"$\rm \log_{10}(M_\star/M_\odot) $"]
        survey_name = [r'$\rm NSA $', \
                  r'$\rm GSWLC  $', \
                  r'$\rm S4G  $']
        plt.figure(figsize=(8,6))
        plt.subplots_adjust(hspace=.0,wspace=.5,left=.15,top=.95)
        xmin=5.5
        xmax=12.5
        xl = np.linspace(xmin,xmax,100)
        for i in np.arange(3):
            for j in np.arange(2):
                plt.subplot(3,2,2*i+1+j)
                if j == 0:
                    ymin=xmin
                    ymax=xmax
                    if i == 0:
                        y = np.log10(cats[i][yvar[i]]/.7**2)
                    else:
                        y = cats[i][yvar[i]]
                elif j == 1:
                    ymin=-1.2
                    ymax=1.2
                    if i == 0:
                        y = np.log10(cats[i][yvar[i]]/.7**2) - (cats[i][xvar[i]])
                    else:
                        y = (cats[i][yvar[i]]) - (cats[i][xvar[i]])
                flag = cats[i][flags[i]] == 1

                x = cats[i][xvar[i]]
                flag = flag & (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
                
                if i < 2:
                    plt.hexbin(x[flag],y[flag],cmap='gray_r', gridsize=40,vmin=1,vmax=40)
                else:
                    plt.plot(x[flag],y[flag],'k.',alpha=.3)
                #if i == 1:
                #    plt.text(-.35,.5,r'$\rm  \log_{10}(M_\star/M_\odot)$',transform=plt.gca().transAxes,rotation=90,verticalalignment='center',fontsize=16)
                if i == 2:
                    plt.xlabel(r'$\rm Taylor \ \log_{10}(M_\star/M_\odot)$')
                if i < 2:
                    plt.xticks([])
                if j == 0:
                    plt.ylabel(survey[i])
                    plt.text(6,11.3,'N = %i'%(sum(flag)),fontsize=12)
                    plt.plot(xl,xl,'k--')
                elif j == 1:
                    plt.axhline(y=0, ls='--',color='k')
                    plt.yticks(np.arange(-1,2,1))
                    plt.ylabel(survey_name[i]+" - Taylor \n"+r"$\rm \Delta \log_{10}(M_\star/M_\odot)$")
                    s = 'mean(std) = %.1f(%.1f)'%(np.mean(y[flag]),np.std(y[flag]))
                    plt.text(6,-1,s,fontsize=12)
                plt.axis([xmin,xmax,ymin,ymax])

        plt.savefig('mstar-comparison.pdf')
    def sfms(self):
        # star-forming main sequence
        
        gsw_flag = (self.a100gsw.logMstar > 0) & (self.a100gsw.photFlag_gi_2 == 1)
        flag1 = (self.a100gsw.a100Flag == 1) & (self.a100gsw.gswFlag ==1) & gsw_flag 
        x1 = self.a100gsw.logMstar[flag1]
        y1 = self.a100gsw.logSFR[flag1]

        flag2 = (self.a100gsw.a100Flag ==0) & (self.a100gsw.gswFlag == 1) & (self.a100gsw.photFlag_gi_2 == 1) 
        x2 = self.a100gsw.logMstar[flag2]
        y2 = self.a100gsw.logSFR[flag2]
        
        contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(2,500,12)
        #print(contour_levels)
        sfrmass(x1,y1, x2, y2, 'A100+GSWLC', 'GSWLC only', 'a100-gswlc-color-mass-2.pdf', \
                  hexbinflag=True,contourflag=True,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=7., xmax=12,ymin=-3,ymax=2, \
                  xlabel='$GSWLC \ \log_{10}(M_\star/M_\odot)$', ylabel='$\log_{10}(SFR (M_\odot/yr))$',  color2='r')
        # add reference line at log(sSFR) = -10
        plt.savefig('sfms.pdf')
        return flag1
    def ssfrmstar(self):
        # star-forming main sequence
        
        gsw_flag = (self.a100gsw.logMstar > 0) & (self.a100gsw.photFlag_gi_2 == 1)
        flag1 = (self.a100gsw.a100Flag == 1) & (self.a100gsw.gswFlag ==1) & gsw_flag 
        x1 = self.a100gsw.logMstar[flag1]
        y1 = self.a100gsw.logSFR[flag1]-x1

        flag2 = (self.a100gsw.a100Flag ==0) & (self.a100gsw.gswFlag == 1) & (self.a100gsw.photFlag_gi_2 == 1) 
        x2 = self.a100gsw.logMstar[flag2]
        y2 = self.a100gsw.logSFR[flag2]-x2
        
        contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(2,500,12)
        #print(contour_levels)
        sfrmass(x1,y1, x2, y2, 'A100+GSWLC', 'GSWLC only', 'a100-gswlc-color-mass-2.pdf', \
                  hexbinflag=True,contourflag=True,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=7., xmax=12,ymin=-13.8,ymax=-8.1, \
                  xlabel='$GSWLC \ \log_{10}(M_\star/M_\odot)$', ylabel=r'$\rm \log_{10}(sSFR/yr^{-1})$',  color2='r',plotline=False)
        # add reference line at log(sSFR) = -10
        plt.savefig('ssfr-mstar-gswlc.pdf')
        return flag1
    def ssfrcolor(self):
        # sSFR vs color

        
        gsw_flag = (self.a100gsw.logMstar > 0) & (self.a100gsw.photFlag_gi_2 == 1)
        flag1 = (self.a100gsw.a100Flag == 1) & (self.a100gsw.gswFlag ==1) & gsw_flag 
        y1 = self.a100gsw.logSFR[flag1]-self.a100gsw.logMstar[flag1]
        x1 = self.a100gsw.gmi_corr_2[flag1]

        flag2 = (self.a100gsw.a100Flag ==0) & (self.a100gsw.gswFlag == 1) & (self.a100gsw.photFlag_gi_2 == 1) 
        y2 = self.a100gsw.logSFR[flag2]-self.a100gsw.logMstar[flag2]
        x2 = self.a100gsw.gmi_corr_2[flag2]
        contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(2,500,12)
        #print(contour_levels)
        sfrmass(x1,y1, x2, y2, 'A100+GSWLC', 'GSWLC only', 'a100-gswlc-color-mass-2.pdf', \
                  hexbinflag=True,contourflag=True,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=-.5, xmax=2,ymin=-13.8,ymax=-8.1, \
                  xlabel='$(g-i)_{corrected}$', ylabel=r'$\rm \log_{10}(sSFR/yr^{-1})$',  color2='r',plotline=False)
        # add reference line at log(sSFR) = -10
        plt.savefig('ssfr-color-gswlc.pdf')
        return flag1
    def ssfrHIfrac(self):
        # star-forming main sequence
        
        gsw_flag = (self.a100gsw.logMstar > 0) & (self.a100gsw.photFlag_gi_2 == 1)
        flag1 = (self.a100gsw.a100Flag == 1) & (self.a100gsw.gswFlag ==1) & gsw_flag 
        x1 = self.a100gsw.logMstar[flag1]
        y1 = self.a100gsw.logSFR[flag1]-x1


        x2 = self.a100gsw.logMstar[flag1]
        y2 = self.a100gsw.logMH[flag1]-x2

        plt.figure(figsize=(6,4))
        plt.subplots_adjust(bottom=.175, left=.175)
        #plt.plot(y2,y1,'k.',alpha=.2)
        plt.hexbin(y2,y1,bins='log',cmap='gray_r', gridsize=75,label='A100+GSWLC')
        
        plt.ylabel(r'$\rm \log_{10}(sSFR/yr^{-1})$',fontsize=16)
        plt.xlabel(r'$\rm log_{10}(M_{HI}/M_\star) $',fontsize=16)
        plt.savefig('ssfr-HIfrac.pdf')
        return flag1     
    def ssfr(self):
        gsw_flag = (self.a100gsw.logMstar > 0) & (self.a100gsw.photFlag_gi_2 == 1)
        flag1 = (self.a100gsw.a100Flag == 1) & (self.a100gsw.gswFlag ==1) & gsw_flag 
        x1 = self.a100gsw.logMstar[flag1]
        y1 = self.a100gsw.logSFR[flag1]
        ssfr1 = y1 - x1
        name1 = 'A100+GSWLC'
        
        flag2 = (self.a100gsw.a100Flag ==0) & (self.a100gsw.gswFlag == 1) & (self.a100gsw.photFlag_gi_2 == 1) 
        x2 = self.a100gsw.logMstar[flag2]
        y2 = self.a100gsw.logSFR[flag2]
        ssfr2 = y2-x2
        color2='r'
        name2 = 'GSWLC only'
        
        nhistbin = 50
        
        plt.figure()
        plt.subplots_adjust(bottom=.15)
        t=plt.hist(ssfr1, normed=True, bins=nhistbin,color='k',histtype='step',lw=1.5, label=name1)
        t=plt.hist(ssfr2, normed=True, bins=nhistbin,color=color2,histtype='step',lw=1.5, label=name2)
        plt.xlabel(r'$\rm \log_{10}(sSFR/yr^{-1}) $')
        plt.ylabel('Normalized Distribution')
        plt.xlim(-14,-7.5)
        plt.legend(loc='upper left')
        plt.savefig('ssfr.pdf')
    def plotelbaz(self):
        xe=np.arange(8.5,11.5,.1)
        xe=10.**xe
        ye=(.08e-9)*xe
        plt.plot(np.log10(xe),np.log10(ye),'k-',lw=1,label='$Elbaz+2011$')
        plt.plot(np.log10(xe),np.log10(2*ye),'k:',lw=1,label='$2 \ SFR_{MS}$')
        # using our own MS fit for field galaxies
        # use stellar mass between 9.5 and 10.5

    def compare_mstar_taylor_cluver(self):
        plt.figure(figsize=(6,4))
        plt.subplots_adjust(bottom=.15)
        w1snr = np.abs(self.a100sdss['w1_nanomaggies']*np.sqrt(self.a100sdss['w1_nanomaggies_ivar']))
        flag = w1snr > 5
        x = self.a100sdss['logMstarTaylor']
        y = self.a100sdss['logMstarCluver']
        #plt.plot(x[flag],y[flag],'k.')
        xmin=7.5
        xmax=11.5
        plt.hexbin(x[flag],y[flag],extent=[xmin,xmax,xmin,xmax],cmap='gray_r')
        plt.xlabel('logMstar Taylor',fontsize=12)
        plt.ylabel('logMstar WISE Cluver',fontsize=12)
        xl = np.linspace(xmin,xmax,100)
        plt.plot(xl,xl,'r--',label='1:1')
        plt.plot(xl,xl+.25,'r--',c='.5',label='logM Taylor + .25')
        plt.legend()
        s = 'N = %i'%(sum(flag))
        plt.text(10.5,8,s,horizontalalignment='left')

        plt.savefig('mstar-taylor-cluver.pdf')
        plt.savefig('mstar-taylor-cluver.png')
    def compare_mstar_taylor_mcgaugh(self):
        plt.figure(figsize=(8,4))
        plt.subplots_adjust(bottom=.15)
        w1snr = np.abs(self.a100sdss['w1_nanomaggies']*np.sqrt(self.a100sdss['w1_nanomaggies_ivar']))
        flag = w1snr > 5
        ngal = sum(flag)
        x = self.a100sdss['logMstarTaylor']
        y = self.a100sdss['logMstarMcGaugh']
        #plt.plot(x[flag],y[flag],'k.')
        xmin=7.5
        xmax=11.5
        plt.hexbin(x[flag],y[flag],extent=[xmin,xmax,xmin,xmax],cmap='gray_r')
        plt.xlabel('logMstar Taylor',fontsize=12)
        plt.ylabel('logMstar WISE McGaugh',fontsize=12)
        xl = np.linspace(xmin,xmax,100)
        plt.plot(xl,xl,'r--',label='1:1')
        plt.plot(xl,xl+.3,'r--',c='.5',label='logM Taylor + .3')
        plt.legend()

        s = 'N = %i'%(sum(flag))
        plt.text(10.5,8,s,horizontalalignment='left')
        plt.savefig('mstar-taylor-mcgaugh.pdf')
        plt.savefig('mstar-taylor-mcgaugh.png')
    def compare_sfr_gsw_wise(self):
        # join a100sdss and a100gsw tables using AGC number
        self.a100sdssgsw = join(self.a100sdss,self.a100gsw,keys='AGC')


        plt.figure(figsize=(8,3))
        plt.subplots_adjust(bottom=.3, wspace=.3)
        yvar = [self.a100sdssgsw['logSFR12'], self.a100sdssgsw['logSFR22']]
        ylabels = ['logSFR12 Cluver', 'logSFR22 Cluver']
        for i in range(len(yvar)):
                
            plt.subplot(1,2,i+1)
            # plot 12um SFR vs GSWLC SFR
            x = self.a100sdssgsw['logSFR']
            y = yvar[i]
            xmin=-1.5
            xmax=1.5
            flag = np.ones(len(x),'bool')
            plt.hexbin(x[flag],(y[flag]),extent=[xmin,xmax,1,4],cmap='gray_r')
            plt.xlabel('logSFR GSWLC',fontsize=12)
            plt.ylabel(ylabels[i],fontsize=12)
            xl = np.linspace(xmin,xmax,100)
            plt.plot(xl,xl,'r--',label='1:1')
            plt.legend()
        plt.savefig('sfr-gsw-wise.pdf')
        plt.savefig('sfr-gsw-wise.png')        

    def sfr_mstar_wise(self):
        plt.figure(figsize=(6,4))
        plt.subplots_adjust(bottom=.15)
        w1snr = np.abs(self.a100sdss['w1_nanomaggies']*np.sqrt(self.a100sdss['w1_nanomaggies_ivar']))
        w12snr = np.abs(self.a100sdss['w3_nanomaggies']*np.sqrt(self.a100sdss['w3_nanomaggies_ivar']))
        flag = (w1snr > 5) & (w12snr > 5)
        x = self.a100sdss['logMstarMcGaugh']
        y = self.a100sdss['logSFR12']
        #plt.plot(x[flag],y[flag],'k.')
        xmin=7.5
        xmax=11.5
        ymin = -1
        ymax=5
        plt.hexbin(x[flag],y[flag],extent=[xmin,xmax,ymin,ymax],cmap='gray_r')
        plt.xlabel('logMstar Cluver',fontsize=12)
        plt.ylabel('logSFR 12um',fontsize=12)
        #xl = np.linspace(xmin,xmax,100)
        #plt.plot(xl,xl,'r--',label='1:1')
        #plt.plot(xl,xl+.4,'r--',c='.5',label='logM Taylor + .4')
        self.plotelbaz()
        plt.legend()
        s = 'N = %i'%(sum(flag))
        plt.text(.9,.1,s,horizontalalignment='right',transform = plt.gca().transAxes)
        plt.savefig('sfr-mstar-wise.pdf')
        plt.savefig('sfr-mstar-wise.png')
    def wise_colors(self):
        # x is 4.6-12
        # y is 3.4-4.6
        # cluver+2014, figure 5, shows different regions for spirals, ellip, starburst, etc
        # references color-color diagram of jarrett+2011

        x = self.a100sdss['w2_mag'] - self.a100sdss['w3_mag']
        y = self.a100sdss['w1_mag'] - self.a100sdss['w2_mag']
        w1snr = np.abs(self.a100sdss['w1_nanomaggies']*np.sqrt(self.a100sdss['w1_nanomaggies_ivar']))
        w2snr = np.abs(self.a100sdss['w2_nanomaggies']*np.sqrt(self.a100sdss['w2_nanomaggies_ivar']))
        w3snr = np.abs(self.a100sdss['w3_nanomaggies']*np.sqrt(self.a100sdss['w3_nanomaggies_ivar']))

        flag = (w1snr > 5) & (w2snr > 5) & (w3snr > 5)
        xmin=-1
        xmax=5
        ymin = -.5
        ymax=1.5
        plt.figure(figsize=(6,4))
        plt.subplots_adjust(bottom=.15,left=.15)

        plt.hexbin(x[flag],y[flag],extent=[xmin,xmax,ymin,ymax],cmap='gray_r')
        plt.plot(x[flag],y[flag],'k.',alpha=.1)
        plt.xlabel('W2-W3',fontsize=12)
        plt.ylabel('W1-W2',fontsize=12)
        plt.axis([-1,5,-.5,1.5])
        plt.axhline(y=.76,ls='--',color='c')

    
def paperplots():
    p.figure1()

    p.mstar()
    
    p.figa_nsa()
    p.figb_nsa()

    p.figa_gswlc()
    p.figb_gswlc()
    
    p.figa_s4g()
    p.figb_s4g()

    p.ssfrmstar()
    p.ssfrcolor()
    p.ssfrHIfrac()
    
if __name__ == '__main__':
    homedir = os.getenv('HOME')
    table_path = homedir+'/github/appss/tables/'
    a100 = table_path+'a100-sdss.fits'
    a100 = table_path+'a100-sdss-wise.fits'
    a100nsa = table_path+'a100-nsa.fits'
    a100gsw = table_path+'a100-gswlcA2.fits'
    a100s4g = table_path+'a100-s4g.fits'
    p = matchedcats(a100sdss=a100, a100nsa=a100nsa,a100gsw=a100gsw,a100s4g=a100s4g)
