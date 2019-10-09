#!/usr/bin/env python

'''
writing this for APPSS analysis

match a100 with many other catalogs

generate plots for paper

'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt


import os
from astropy.io import fits
from astropy.io import ascii
from astropy import constants as c
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

plt.rcParams.update({'font.size': 14})

import sys



def colormass(x1,y1,x2,y2,name1,name2, figname, hexbinflag=False,contourflag=False, \
             xmin=7.9, xmax=11.6, ymin=-0.05, ymax=2., contour_bins = 40, ncontour_levels=5,\
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
    ax2.legend(fontsize=10)
    ax2.xaxis.tick_top()
    ax3 = plt.subplot2grid((nrow,ncol),(1,ncol-1),rowspan=nrow-1,colspan=1, fig=fig, sharey = ax1, xticks=[])
    t=plt.hist(y1, normed=True, orientation='horizontal',bins=nhistbin,color='k',histtype='step',lw=1.5, label=name1)
    t=plt.hist(y2, normed=True, orientation='horizontal',bins=nhistbin,color=color2,histtype='step',lw=1.5, label=name2)
    
    plt.yticks(rotation='horizontal')
    ax3.yaxis.tick_right()
    plt.savefig(figname)
    
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
        plt.figure(figsize=(10,3))
        plt.subplots_adjust(wspace=.0)
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
        
    def figure2a(self):
        # correct to H0=70
        x = np.log10(self.a100nsa.SERSIC_MASS/.7**2)
        y = ((self.a100nsa.SERSIC_ABSMAG[:,3] - self.a100nsa.EXTINCTION[:,3]) - (self.a100nsa.SERSIC_ABSMAG[:,5] - self.a100nsa.EXTINCTION[:,5]))
        nsa_mass_flag =  ( self.a100nsa.SERSIC_MASS > 1000.) 
        # require phot error < 0.05 for abs mag
        ivar = 1./.05**2
        nsa_phot_flag = np.ones(len(nsa_mass_flag),'bool')
        for i in np.arange(3,6):
            nsa_phot_flag = nsa_phot_flag & (self.a100nsa.SERSIC_AMIVAR[:,i] > (1./.05**2))
    
        # flag1 = (self.a100nsa.matchFlag == 3) & (self.a100nsa.photFlag_gi == 1)
        nsa_flag = nsa_mass_flag & nsa_phot_flag
        flag1 = (self.a100nsa.a100Flag & self.a100nsa.nsaFlag) & nsa_flag #& photflagnsa  & (gmi_corr_nsa > -1) & (gmi_corr_nsa < 3.)#
        # x1 = self.a100nsa.LogMstarTaylor_2[flag1]
        x1 = x[flag1]
        y1 = y[flag1]
        
        flag2 = (~self.a100nsa.a100Flag & self.a100nsa.nsaFlag)& nsa_flag #& (logstellarmassTaylor_nsa > 7) & (gmi_corr_nsa > -1) & (gmi_corr_nsa < 3.)#& (self.a100nsa.photFlag_gi == 1) 
        x2 = x[flag2]
        y2 = y[flag2]
        print(len(x2), sum(flag2))
        contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(10,500,12)
        print(contour_levels)
        colormass(x1,y1, x2, y2, 'A100+NSA', 'NSA only', 'a100-nsa-color-mass-2.pdf', \
                  hexbinflag=True,contourflag=True,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=5., ymin=-.5,xmax=12, \
                  xlabel='$NSA \ \log_{10}(M_\star/M_\odot)$', ylabel='$ NSA \ (M_g - M_i)$', color2='r')

    def figure2b(self):
        flag1 = (self.a100nsa.a100Flag & self.a100nsa.nsaFlag) & (self.a100nsa.photFlag_gi == 1)
        # x1 = self.a100nsa.LogMstarTaylor_2[flag1]
        x1 = self.a100nsa.logMstarTaylor[flag1]
        y1 = self.a100nsa.gmi_corrected[flag1]

        flag2 = (self.a100nsa.a100Flag & ~self.a100nsa.nsaFlag) & (self.a100nsa.photFlag_gi == 1) 
        x2 = self.a100nsa.logMstarTaylor[flag2]
        y2 = self.a100nsa.gmi_corrected[flag2]
        print(len(x2), sum(flag2))
        colormass(x1,y1, x2, y2, 'A100+NSA', 'A100 only', 'a100-nsa-color-mass-1.pdf', \
                  hexbinflag=True, contourflag=False,color2='b',xmin=5., ymin=-.5,xmax=12)
    def figa_s4g(self):
        x = self.a100s4g.mstar
        y = self.a100s4g.bvtc
        #x = a100s4g.mabs
        #y = fullsdss['g']-fullsdss['r']
        #y = a100s4g.mag1 - a100s4g.mag2
        photflag = self.a100s4g.photFlag_gi == 1
        flag1 = (self.a100s4g.a100Flag & self.a100.s4gFlag) & photflag
        print('number with both = ',sum(flag1))
        x1 = x[flag1]
        y1 = y[flag1]

        flag2 = (~self.a100s4g.a100Flag & self.a100.s4gFlag) & photflag
        print('number with S4G only = ',sum(flag2))
        x2 = x[flag2]
        y2 = y[flag2]
        print(len(x2), sum(flag2))
        #contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(1,100,12)
        print(contour_levels)
        colormass(x1,y1, x2, y2, 'A100+S4G', 'S4G only', 'a100-s4g-color-mass-1.pdf', \
                  hexbinflag=False,contourflag=False,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=5, ymin=-1,xmax=13,ymax=3,nhistbin=15,alphagray=.5,\
                  xlabel='$S4G \ M_{ABS}$', ylabel='$ Leda \ (B-V)$', color2='r')
    def figb_s4g(self):
        x = self.a100s4g.logMstarTaylor
        y = self.a100s4g.gmi_corrected

        flag1 = (self.a100s4g.a100Flag & self.a100.s4gFlag) & (self.a100s4g.photFlag_gi == 1)
        # x1 = self.a100nsa.LogMstarTaylor_2[flag1]
        x1 = x[flag1]
        y1 = y[flag1]

        flag2 = (self.a100s4g.a100Flag & ~self.a100.s4gFlag)& (a100s4g.photFlag_gi == 1)
        x2 = x[flag2]
        y2 = y[flag2]
        #print(len(x2), sum(flag2))
        #contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(1,100,12)
        print(contour_levels)
        colormass(x1,y1, x2, y2, 'A100+S4G', 'A100 only', 'a100-s4g-color-mass-2.pdf', \
                  hexbinflag=False,contourflag=False,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=5., ymin=-.5,xmax=12,nhistbin=20,alphagray=.5,color2='b')

        
    


if __name__ == '__main__':
    homedir = os.getenv('HOME')
    table_path = homedir+'/github/appss/tables/'
    a100 = table_path+'a100-sdss.fits'
    a100nsa = table_path+'a100-nsa.fits'
    a100gsw = table_path+'a100-gswlcA2.fits'
    a100s4g = table_path+'a100-s4g.fits'
    p = matchedcats(a100sdss=a100, a100nsa=a100nsa,a100gsw=a100gsw,a100s4g=a100s4g)
    p.figure1()
