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
from astropy.coordinates import SkyCoord, Angle
from scipy.stats import ks_2samp
from scipy.optimize import curve_fit, least_squares
from scipy.stats import binned_statistic
plt.rcParams.update({'font.size': 14})

import sys

#####################################
## COLORBLIND FRIENDLY COLOR PALETTE
#####################################
colorblind1='#F5793A' # orange
colorblind2 = '#85C0F9' # light blue
colorblind3='#0F2080' # dark blue

# using colors from matplotlib default color cycle
mycolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#colorblind1=mycolors[1] # orange
#colorblind2 = mycolors[9]# light blue
#colorblind3=mycolors[0] # dark blue
#colorblind2='#ababab' # grey
# from matplotlib
# dk blue, orange, grey, dark grey, light blue, rust, 
newcolors = ['#006BA4','#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

###############################
## some plot parameters
###############################
# max and min for g-i plots
gimin = -.5
gimax =2

#################################
### LIMIT TO OVERLAP REGION (NSA, GSWCLC)
#################################
ramin = 140.
ramax = 230.
decmin = 0.
decmax = 35.
zmax = 0.05
vmax = 15000

#################################
### SETUP PATHS
#################################

homedir = os.getenv("HOME")
if homedir.find('Users') > -1:
    # running on macbook
    tabledir = homedir+'/github/APPSS/tables/'
else:
    tabledir = homedir+'/research/APPSS/tables/'

#################################
### GET SALIM MAIN SEQUENCE FITS, SALIM+2018 ###
#################################
# all SF galaxies according to BPT
t = np.loadtxt(homedir+'/research/GSWLC/salim2018_ms_v1.dat')
log_mstar1 = t[:,0]
log_ssfr1 = t[:,1]
# everything with sSFR > -11
t = np.loadtxt(homedir+'/research/GSWLC/salim2018_ms_v2.dat')
log_mstar2 = t[:,0]
log_ssfr2 = t[:,1]

# my version, binned median for GSWLC galaxies with vr < 15,0000 
t = Table.read(homedir+'/research/APPSS/GSWLC2-median-ssfr-mstar-vr15k.dat',format='ipac')
log_mstar2 = t['med_logMstar']
log_ssfr2 = t['med_logsSFR']

#################################
### FUNCTIONS
#################################

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

def fitZPoffset(x,zp):
    return x+zp

def fitline(x,m,b):
    return m*x+b

def fitline_error(x,m,b,y):
    pass
def fitparab(x,a,b,c):
    return a*x**2+b*x+c

    
def ratioerror(a,b):
    # compute error in ratio
    # assuming errors are the sqrt of counts
    # f = a/b
    # err_f = (1/b)**2 err_a**2 + (a/b**2)**2 err_b**2
    # err_f = (1/b)**2 a + (a/b**2)**2 b
    
    return a/b,a/b*(1/b+a/b**2)
    
def plotelbaz():
    xe=np.arange(8.5,11.5,.1)
    xe=10.**xe
    ye=(.08e-9)*xe
    plt.plot(np.log10(xe),np.log10(ye),'k-',lw=1,label='$Elbaz+2011$')
    plt.plot(np.log10(xe),np.log10(2*ye),'k:',lw=1,label='$2 \ SFR_{MS}$')
    # using our own MS fit for field galaxies
    # use stellar mass between 9.5 and 10.5
def plotsalim07(useOne=False,plotsalim18=True):
    #plot the main sequence from Salim+07 for a Chabrier IMF
    lmstar=np.arange(8.5,11.5,0.1)
    #use their equation 11 for pure SF galaxies
    lssfr = -0.35*(lmstar - 10) - 9.83
    #use their equation 12 for color-selected galaxies including
    #AGN/SF composites.  This is for log(Mstar)>9.4
    #lssfr = -0.53*(lmstar - 10) - 9.87

    lsfr = lmstar + lssfr -.3
    sfr = 10.**lsfr

    ## USE SALIM+2018 FIT
    if useOne:
        lmstar = log_mstar1
        lssfr = log_ssfr1
    else:
        lmstar = log_mstar2
        lssfr = log_ssfr2


    if plotsalim18:
        lsfr = log_mstar1+log_ssfr1
        plt.plot(log_mstar1, lsfr, 'w-', lw=5)
        plt.plot(log_mstar1, lsfr, c='gray',ls='-', lw=3, label='Salim+2018')
    lsfr = log_mstar2+log_ssfr2
    plt.plot(log_mstar2, lsfr, 'w-', lw=5)
    plt.plot(log_mstar2, lsfr, c=mycolors[3],ls='--', lw=3, label='GSWLC-2 med')
    #plt.plot(lmstar, lsfr-np.log10(5.), 'w--', lw=4)
    #plt.plot(lmstar, lsfr-np.log10(5.), c=colorblind1,ls='--', lw=2)

def plotsalimssfr(useOne=False,yoffset=None,plotsalim18=True):
    #plot the main sequence from Salim+07 for a Chabrier IMF

    lmstar=np.arange(8.5,11.5,0.1)

    #use their equation 11 for pure SF galaxies
    lssfr = -0.35*(lmstar - 10) - 9.83
    # shift SFR down by 0.3 dex
    lssfr = lssfr - 0.3
    
    ## USE SALIM+2018 FIT
    if useOne:
        lmstar = log_mstar1
        lssfr = log_ssfr1
    else:
        lmstar = log_mstar2
        lssfr = log_ssfr2

    #use their equation 12 for color-selected galaxies including
    #AGN/SF composites.  This is for log(Mstar)>9.4
    #lssfr = -0.53*(lmstar - 10) - 9.87

    #lsfr = lmstar + lssfr -.3
    #sfr = 10.**lsfr
    if plotsalim18:
        lssfr=log_ssfr1
        #if yoffset is not None:
        #    lssfr = lssfr + yoffset
        
        plt.plot(log_mstar1, lssfr, 'w-', lw=5)
        plt.plot(log_mstar1, lssfr, c='gray',ls='-', lw=3, label='Salim+2018')
    lssfr = log_ssfr2
    if yoffset is not None:
        lssfr = lssfr + yoffset
    
    plt.plot(log_mstar2, lssfr, 'w-', lw=5)
    plt.plot(log_mstar2, lssfr, c=mycolors[3],ls='--', lw=3, label='GSWLC-2 med')
    #plt.plot(lmstar, lsfr-np.log10(5.), 'w--', lw=4)
    #plt.plot(lmstar, lsfr-np.log10(5.), c=colorblind1,ls='--', lw=2)
        
def plothuangssfr(useOne=False):
    #plot the main sequence from Salim+07 for a Chabrier IMF
    yoffset=-.4
    lmstar=np.linspace(8.5,9.5,100)
    lssfr = -0.149*lmstar - 8.207+yoffset
    plt.plot(lmstar, lssfr, 'w-', lw=5)
    plt.plot(lmstar, lssfr, c=mycolors[2],ls='-', lw=3, label='_nolegend_')
    

    lmstar=np.linspace(9.5,11.5,100)
    lssfr = -0.759*lmstar - 2.402+yoffset
    plt.plot(lmstar, lssfr, 'w-', lw=5)
    plt.plot(lmstar, lssfr, c=mycolors[2],ls='-', lw=3, label='Huang+2012')
    #plt.plot(lmstar, lssfr-np.log10(5.), 'w--', lw=4)
    #plt.plot(lmstar, lssfr-np.log10(5.), c=colorblind1,ls='--', lw=2)


def colormass(x1,y1,x2,y2,name1,name2, figname, hexbinflag=False,contourflag1=False,contourflag=False, \
             xmin=7.9, xmax=11.6, ymin=-1.2, ymax=1.2, contour_bins = 40, ncontour_levels=5,\
              xlabel='$Taylor \ \log_{10}(M_\star (M_\odot)) $', ylabel='$(g-i)_{corrected} $', color1=colorblind3, color2=colorblind1,\
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
    if contourflag1:
        H, xbins,ybins = np.histogram2d(x1,y1,bins=contour_bins)
        extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        CS = plt.contour((H.T), levels=ncontour_levels, extent = extent, zorder=10,colors=color1, label='__nolegend__')
        # trying to get name of contour into the legend
        CS.collections[0].set_label(name1)
        #plt.legend()
    elif hexbinflag:
        #t1 = plt.hist2d(x1,y1,bins=100,cmap='gray_r')
        #H, xbins,ybins = np.histogram2d(x1,y1,bins=20)
        #extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        #plt.contour(np.log10(H.T+1),  10, extent = extent, zorder=1,colors='k')
        #plt.hexbin(xvar2,yvar2,bins='log',cmap='Blues', gridsize=100)

        #plt.hexbin(x1,y1,bins='log',cmap='gray_r', gridsize=75,label=name1)
        plt.hexbin(x1,y1,bins='log',cmap='Purples', gridsize=75,label=name1)
    else:
        plt.plot(x1,y1,'k.',color=color1,alpha=alphagray,label=name1, zorder=1)
    if contourflag:
        H, xbins,ybins = np.histogram2d(x2,y2,bins=contour_bins)
        extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        CS = plt.contour((H.T), levels=ncontour_levels, extent = extent, zorder=1,colors=color2, label='__nolegend__')
        CS.collections[0].set_label(name2)
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
    t = plt.hist(x1, normed=True, bins=nhistbin,color=color1,histtype='step',lw=1.5, label=name1+' ({:d})'.format(sum(keepflag1)))
    t = plt.hist(x2, normed=True, bins=nhistbin,color=color2,histtype='step',lw=1.5, label=name2+' ({:d})'.format(sum(keepflag2)))
    #plt.legend()
    ax2.legend(fontsize=10,loc='upper left')
    ax2.xaxis.tick_top()
    ax3 = plt.subplot2grid((nrow,ncol),(1,ncol-1),rowspan=nrow-1,colspan=1, fig=fig, sharey = ax1, xticks=[])
    t=plt.hist(y1, normed=True, orientation='horizontal',bins=nhistbin,color=color1,histtype='step',lw=1.5, label=name1)
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


def sfrmass(x1,y1,x2,y2,name1,name2, figname, hexbinflag=False,contourflag1=False,contourflag=False, \
             xmin=7.9, xmax=11.6, ymin=-1.2, ymax=1.2, contour_bins = 40, ncontour_levels=5,\
             xlabel='$\log_{10}(M_\star/M_\odot) $', ylabel='$\log_{10}(M_\odot/yr)$', color2='c',\
             color1='k',nhistbin=50, alphagray=.1,plotmsline=True,plotssfrline=False,plotsalim18=True):
    fig = plt.figure(figsize=(8,8))
    plt.subplots_adjust(left=.15)
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
    if contourflag1:
        H, xbins,ybins = np.histogram2d(x1,y1,bins=contour_bins)
        extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        CS = plt.contour((H.T), levels=ncontour_levels, extent = extent, zorder=10,colors=color1, label='_nolegend_')
        #CS.collections[0].set_label(name1)
        #plt.legend()
    elif hexbinflag:
        #t1 = plt.hist2d(x1,y1,bins=100,cmap='gray_r')
        #H, xbins,ybins = np.histogram2d(x1,y1,bins=20)
        #extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        #plt.contour(np.log10(H.T+1),  10, extent = extent, zorder=1,colors='k')
        #plt.hexbin(xvar2,yvar2,bins='log',cmap='Blues', gridsize=100)

        #plt.hexbin(x1,y1,bins='log',cmap='gray_r', gridsize=75,label=name1)
        plt.hexbin(x1,y1,bins='log',cmap='gray_r', gridsize=75,label='_nolegend_')
    else:
        #plt.plot(x1,y1,'k.',color=color1,alpha=alphagray,label=name1, zorder=1)
        plt.plot(x1,y1,'k.',color=color1,alpha=alphagray,label='_nolegend_', zorder=1)        
    if contourflag:
        H, xbins,ybins = np.histogram2d(x2,y2,bins=contour_bins)
        extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        CS = plt.contour((H.T), levels=ncontour_levels, extent = extent, zorder=10,colors=color2, label='_nolegend_')
        #CS.collections[0].set_label(name2)
        #plt.legend()
    else:
        plt.plot(x2,y2,'c.',color=color2,alpha=.3, label='_nolegend_')
    if plotmsline:
        xl = np.linspace(xmin,xmax,100)
        slope=.75
        xref, yref = 9.45,-.44
        yl = yref + slope*(xl - xref)
        #plt.plot(xl,yl,'k--')
        #plotsalim07()
        plotsalim07(useOne=True,plotsalim18=plotsalim18)
        plt.legend()
    if plotssfrline:
        '''
        xl = np.linspace(xmin,xmax,100)
        slope=-.5
        xref, yref = 9.,-9.7
        yl = yref + slope*(xl - xref)
        plt.plot(xl,yl,'k-',lw=2,label='GSWLC ref')
        '''
        plotsalimssfr(plotsalim18=plotsalim18)
        #plothuangssfr()        
        plt.legend()
    #sns.kdeplot(agc['LogMstarTaylor'][keepagc],agc['gmi_corrected'][keepagc])#,bins='log',gridsize=200,cmap='blue_r')
    #plt.colorbar()


    
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(xlabel,fontsize=22)
    plt.ylabel(ylabel,fontsize=22)
    #plt.axis([7.9,11.6,-.05,2])
    ax2 = plt.subplot2grid((nrow,ncol),(0,0),rowspan=1,colspan=ncol-1, fig=fig, sharex = ax1, yticks=[])
    t = plt.hist(x1, normed=True, bins=nhistbin,color=color1,histtype='step',lw=1.5, label=name1+' ({:d})'.format(sum(keepflag1)))
    t = plt.hist(x2, normed=True, bins=nhistbin,color=color2,histtype='step',lw=1.5, label=name2+' ({:d})'.format(sum(keepflag2)))
    #plt.legend()
    ax2.legend(fontsize=10, loc='upper left')
    ax2.xaxis.tick_top()
    ax3 = plt.subplot2grid((nrow,ncol),(1,ncol-1),rowspan=nrow-1,colspan=1, fig=fig, sharey = ax1, xticks=[])
    t=plt.hist(y1, normed=True, orientation='horizontal',bins=nhistbin,color=color1,histtype='step',lw=1.5, label=name1)
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
    return ax1,ax2,ax3
def sfrmass_simple(x1,y1,x2,y2,name1,name2, figname, hexbinflag=False,contourflag1=False,contourflag=False, \
             xmin=7.9, xmax=11.6, ymin=-1.2, ymax=1.2, contour_bins = 40, ncontour_levels=5,\
             xlabel='$\log_{10}(M_\star/M_\odot) $', ylabel='$\log_{10}(M_\odot/yr)$', color2='c',\
             color1='k',nhistbin=50, alphagray=.1,plotmsline=True,plotssfrline=False):
    ## JUST PLOTS THE CENTRAL PANEL
    fig = plt.figure(figsize=(8,8))
    
    # for purposes of this plot, only keep data within the 
    # window specified by [xmin:xmax, ymin:ymax]
    
    keepflag1 = (x1 >= xmin) & (x1 <= xmax) & (y1 >= ymin) & (y1 <= ymax)
    keepflag2 = (x2 >= xmin) & (x2 <= xmax) & (y2 >= ymin) & (y2 <= ymax)
    
    x1 = x1[keepflag1]
    y1 = y1[keepflag1]
    
    x2 = x2[keepflag2]
    y2 = y2[keepflag2]
    

    if contourflag1:
        H, xbins,ybins = np.histogram2d(x1,y1,bins=contour_bins)
        extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        CS = plt.contour((H.T), levels=ncontour_levels, extent = extent, zorder=10,colors=color1, label='__nolegend__')
        #plt.legend()
        CS.collections[0].set_label(name1)
    elif hexbinflag:
        #t1 = plt.hist2d(x1,y1,bins=100,cmap='gray_r')
        #H, xbins,ybins = np.histogram2d(x1,y1,bins=20)
        #extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        #plt.contour(np.log10(H.T+1),  10, extent = extent, zorder=1,colors='k')
        #plt.hexbin(xvar2,yvar2,bins='log',cmap='Blues', gridsize=100)

        #plt.hexbin(x1,y1,bins='log',cmap='gray_r', gridsize=75,label=name1)
        plt.hexbin(x1,y1,bins='log',cmap='gray_r', gridsize=75,label=name1)
    else:
        plt.plot(x1,y1,'k.',color=color1,alpha=alphagray,label=name1, zorder=1)
    if contourflag:
        H, xbins,ybins = np.histogram2d(x2,y2,bins=contour_bins)
        extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        CS = plt.contour((H.T), levels=ncontour_levels, extent = extent, zorder=10,colors=color2, label=name2)
        CS.collections[0].set_label(name2)
        #plt.legend()
    else:
        plt.plot(x2,y2,'c.',color=color2,alpha=.3, label=name2)
    if plotmsline:
        xl = np.linspace(xmin,xmax,100)
        slope=.75
        xref, yref = 9.45,-.44
        yl = yref + slope*(xl - xref)
        #plt.plot(xl,yl,'k--')
        plotsalim07()
        plotsalim07(useOne=True)        
        plt.legend()
    if plotssfrline:
        '''
        xl = np.linspace(xmin,xmax,100)
        slope=-.5
        xref, yref = 9.,-9.7
        yl = yref + slope*(xl - xref)
        plt.plot(xl,yl,'k-',lw=2,label='GSWLC ref')
        '''
        plotsalimssfr()
        #plothuangssfr()        
        plt.legend()
    #sns.kdeplot(agc['LogMstarTaylor'][keepagc],agc['gmi_corrected'][keepagc])#,bins='log',gridsize=200,cmap='blue_r')
    #plt.colorbar()


    
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(xlabel,fontsize=22)
    plt.ylabel(ylabel,fontsize=22)
    #plt.axis([7.9,11.6,-.05,2])
    plt.savefig(figname)



    
def colormag(mag, color, ab, ylabel):
    #plt.plot(mag, color,  'k.', alpha=.05)
    limits = [-23.9, -14.1, -.5, 1.9]
    flag = ab > 0.8
    #plt.hexbin(mag[flag],color[flag], extent=limits, cmap='gray_r', vmin=0,vmax=45)
    plt.plot(mag[flag],color[flag],'k.',alpha=.2, markersize=3, label='$b/a > 0.8$',color='k')
    flag = ab < 0.3
    #plt.hexbin(mag[flag],color[flag], extent=limits, cmap='Purples_r', vmin=0,vmax=45)
    plt.plot(mag[flag],color[flag],'c.',color=colorblind2, alpha=.2,markersize=3,label='$b/a < 0.3$')
    plt.axis(limits)
    plt.xticks(np.arange(-23,-14,2))
    plt.gca().invert_xaxis()
    plt.title(ylabel, fontsize=14)

########################################
### CLASS DEFINITIONS
########################################

class matchedcats():
    '''
    This is what I've done so far using Mary's matched catalogs for
    a100 - NSA
    a100 - S4G
    '''
    def __init__(self,a100sdss=None, a100nsa=None,a100gsw=None,a100s4g=None,allcats=None):
        self.a100sdss = fits.getdata(a100sdss)

        self.a100nsa = fits.getdata(a100nsa)
        self.a100gsw = fits.getdata(a100gsw)
        self.a100s4g = fits.getdata(a100s4g)
        self.allcats = fits.getdata(allcats)
        self.overlap_flag_all = self.get_overlap_a100(cat=self.allcats)
    def get_overlap_gswlc(self):
        self.overlap_flag_gswlc = (self.a100gsw['RA_2'] > ramin) &\
          (self.a100gsw['RA_2']< ramax) &\
          (self.a100gsw['DEC_2'] > decmin) &\
          (self.a100gsw['DEC_2'] < decmax) &\
          (self.a100gsw['Z']*3.e5 < vmax) 
    def get_overlap_nsa(self):
        self.overlap_flag_nsa = (self.a100nsa['RA'] > ramin) &\
          (self.a100nsa['RA']< ramax) &\
          (self.a100nsa['DEC'] > decmin) &\
          (self.a100nsa['DEC'] < decmax) &\
          (self.a100nsa['Z']*3.e5 < vmax) 
    def get_overlap_a100(self,cat):
        return (cat['RAdeg_OC'] > ramin) &\
          (cat['RAdeg_OC'] < ramax) &\
          (cat['DECdeg_OC'] > decmin) &\
          (cat['DECdeg_OC'] < decmax) &\
          (cat['Vhelio'] < vmax) 
    def figure1(self):
        plt.figure(figsize=(10,4))
        plt.subplots_adjust(wspace=.0,bottom=.2)
        # 
        plt.subplot(1,3,1)
        photflag = self.a100sdss['photFlag_gi'] == 1
        colormag(self.a100sdss['absMag_i'][photflag], self.a100sdss['gmi_no_int'][photflag], self.a100sdss['expAB_g'][photflag],'${uncorrected}$')
        plt.ylabel('$g-i$',fontsize=16)
        leg=plt.legend(loc='lower left',markerscale=6)
        for lh in leg.legendHandles:
            print("I'm trying...")
            lh._legmarker.set_alpha(1)
            lh.set_alpha(1)
        #plt.gca()
        
        plt.subplot(1,3,2)
        photflag = self.a100sdss['photFlag_gi'] == 1
        colormag(self.a100sdss['I_Shao'][photflag], self.a100sdss['gmi_Shao'][photflag],self.a100sdss['expAB_g'][photflag],'${Shao\ et\ al. \ 2007}$')
        plt.yticks(())
        ax = plt.gca()

        plt.text(.5,-.25,'$M_i$',fontsize=16, transform = ax.transAxes)

        plt.subplot(1,3,3)
        photflag = self.a100sdss['photFlag_gi'] == 1
        colormag(self.a100sdss['absMag_i_corr'][photflag], self.a100sdss['gmi_corr'][photflag],
        self.a100sdss['expAB_g'][photflag],'${this \ work}$')
        plt.yticks(())
        
        plt.savefig('fig1.pdf')
        plt.savefig('fig1.png')        
        
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
        colormass(x1,y1, x2, y2, 'ALFALFA+NSA', 'NSA only', 'a100-nsa-color-mass-2.pdf', \
                  hexbinflag=True,contourflag=True,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=6., xmax=12, ymin=gimin,ymax=gimax,\
                  xlabel='$NSA \ \log_{10}(M_\star (M_\odot))$', ylabel='$ NSA \ (M_g - M_i)$', color2=colorblind1)
        plt.savefig('a100-nsa-color-mass-2.png')
        plt.savefig('fig6a.pdf')        
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
        contour_levels = np.linspace(2,400,12)
        colormass(x1,y1, x2, y2, 'ALFALFA+NSA', 'ALFALFA only', 'a100-nsa-color-mass-1.pdf', \
                  hexbinflag=False, contourflag1=True,contourflag=False,contour_bins=30,  ncontour_levels=contour_levels,\
                  color2=colorblind2,xmin=6.,xmax=12, ymin=gimin,ymax=gimax)
        plt.savefig('a100-nsa-color-mass-1.png')
        plt.savefig('fig6b.pdf')                
    def figa_gswlc(self):
        # figure a is catalog specific quantities
        
        # here I am requiring that is has a valid stellar mass from GSWLC
        # AND reliable SDSS photometry.
        # but why do we need reliable sdss photometry for this?
        # Answer: because we are using our corrected g-i color
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
        colormass(x1,y1, x2, y2, 'ALFALFA+GSWLC-2', 'GSWLC-2 only', 'a100-gswlc-color-mass-2.pdf', \
                  hexbinflag=True,contourflag=True,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=6.,xmax=12,ymin=gimin,ymax=gimax, \
                  xlabel='$GSWLC-2 \ \log_{10}(M_\star (M_\odot))$', ylabel='$  \ (g - i)_{corrected}$', color2=colorblind1)
        plt.savefig('a100-gswlc-color-mass-2.png')
        plt.savefig('fig8a.pdf')        
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
        #print(len(x2), sum(flag2))
        contour_levels = np.linspace(2,200,12)
        colormass(x1,y1, x2, y2, 'ALFALFA+GSWLC-2', 'ALFALFA only', 'a100-gswlc-color-mass-1.pdf', \
                  hexbinflag=False, alphagray=.08, contourflag1=True,contourflag=False,contour_bins=30, ncontour_levels=contour_levels,\
                  color2=colorblind2,xmin=6.,xmax=12,ymin=gimin,ymax=gimax)
        plt.savefig('a100-gswlc-color-mass-1.png')
        plt.savefig('fig8b.pdf')                
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
        #print(len(x2), sum(flag2))
        #contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(1,100,12)
        #print(contour_levels)
        colormass(x1,y1, x2, y2, 'ALFALFA+S4G', 'S4G only', 'a100-s4g-color-mass-1.pdf', \
                  hexbinflag=False,contourflag=False,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=6,xmax=13,ymin=gimin, ymax=gimax,nhistbin=10,alphagray=.5,\
                  xlabel='$S4G \ \log_{10}(M_\star/M_\odot)$', ylabel='$ Leda \ (B-V)$', color2=colorblind1)
        plt.savefig('a100-s4g-color-mass-1.png')
        plt.savefig('fig7a.pdf')                
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
        contour_levels = np.linspace(2,400,10)
        print(contour_levels)
        colormass(x1,y1, x2, y2, 'ALFALFA+S4G', 'ALFALFA only', 'a100-s4g-color-mass-2.pdf', \
                  hexbinflag=False,contourflag=True,contour_bins=30, ncontour_levels=contour_levels,\
                  xmin=6.,xmax=12,ymin=gimin,ymax=gimax,nhistbin=20,alphagray=.5,color2=colorblind2)
        plt.savefig('a100-s4g-color-mass-2.png')
        plt.savefig('fig7b.pdf')                        
    def mstar(self):
        # compare different estimates of stellar mass with our estimate from Taylor

        # 3 panel plot
        # panel one - NSA np.log10(SERSIC_MASS) vs logMstarTaylor
        # panel two - GSWLC Mass vs logMstarTaylor
        # panel three - S4G mstar vs logMstarTaylor

        cats = [self.a100nsa, self.a100gsw, self.a100s4g,self.a100nsa,self.a100nsa]

        xvar = ['logMstarTaylor','logMstarTaylor_1','logMstarTaylor','logMstarTaylor','logMstarTaylor']
        yvar = ['SERSIC_MASS','logMstar','mstar','logMstarCluver','logMstarMcGaugh']
        flags = ['photFlag_gi','photFlag_gi_1','photFlag_gi','photFlag_gi','photFlag_gi']
        survey = ['$\log_{10}(NSA \ SERSIC\_MASS/M_\odot) $', \
                  '$\log_{10}(GSWLC-2 \ Mstar/M_\odot )$', \
                  '$\log_{10}(S4G \ Mstar/M_\odot )$', \
                  '$\log_{10}(Cluver \ Mstar/M_\odot )$',\
                  '$\log_{10}(McGaugh \ Mstar/M_\odot )$']
        survey = ["NSA \n" r"$\rm \log_{10}(M_\star/M_\odot) $", \
                  "GSWLC-2 \n" r"$\rm \log_{10}(M_\star/M_\odot) $", \
                  "S4G \n " r"$\rm \log_{10}(M_\star/M_\odot) $", \
                  "Cluver \n " r"$\rm \log_{10}(M_\star/M_\odot) $",\
                  "McGaugh \n " r"$\rm \log_{10}(M_\star/M_\odot) $"
        ]
        survey_name = [r'$\rm NSA $', \
                       r'$\rm GSWLC-2  $', \
                       r'$\rm S4G  $',\
                       r'$\rm Cluver $', \
                       r'$\rm McGaugh $']
        plt.figure(figsize=(8,8))
        plt.subplots_adjust(hspace=.0,wspace=.5,left=.15,top=.95)
        xmin=5.5
        xmax=12.5
        xl = np.linspace(xmin,xmax,100)
        for i in range(len(cats)):
            for j in np.arange(2):
                plt.subplot(len(cats),2,2*i+1+j)
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
                
                if i != 2:
                    plt.hexbin(x[flag],y[flag],cmap='gray_r', gridsize=40,vmin=1,vmax=40)
                else:
                    plt.plot(x[flag],y[flag],'k.',alpha=.3)
                #if i == 1:
                #    plt.text(-.35,.5,r'$\rm  \log_{10}(M_\star/M_\odot)$',transform=plt.gca().transAxes,rotation=90,verticalalignment='center',fontsize=16)
                if i == len(cats)-1:
                    plt.xlabel(r'$\rm Taylor \ \log_{10}(M_\star/M_\odot)$')
                if i < len(cats)-1:
                    plt.xticks([])
                else:
                    plt.xticks(fontsize=11)
                if j == 0:
                    plt.ylabel(survey[i],fontsize=11)
                    plt.text(6,11.3,'N = %i'%(sum(flag)),fontsize=11)
                    plt.plot(xl,xl,'k--')
                    plt.yticks(fontsize=11)
                elif j == 1:
                    plt.axhline(y=0, ls='--',color='k')
                    plt.yticks(np.arange(-1,2,1),fontsize=11)
                    plt.ylabel(survey_name[i]+" - Taylor \n"+r"$\rm \Delta \log_{10}(M_\star/M_\odot)$",fontsize=11)
                    s = 'mean(std) = %.2f(%.2f)'%(np.mean(y[flag]),np.std(y[flag]))
                    plt.text(6,-1,s,fontsize=12)
                plt.axis([xmin,xmax,ymin,ymax])

        plt.savefig('mstar-comparison.pdf')
        plt.savefig('mstar-comparison.png')        
    def compare_sfrs(self):
        # compare IR vs UV vs IR+UV
        # compare distribution of SFRs
        plt.figure()
        mybins = np.linspace(-5,1,20)
        flag = (self.a100nsa.w4_mag > 0) & (self.a100nsa.SERSIC_ABSMAG[:,1] < 0)#(self.a100nsa.finn_index != 999999) & self.a100nsa.nsaFlag
        #flag = self.a100nsa.nsaFlag#& (self.a100nsa.w4_nanomaggies*np.sqrt(self.a100nsa.w4_nanomaggies_ivar) > 5)
        #flag = np.ones(len(self.a100nsa),'bool')
        
        print('number to plot = ',sum(flag))
        plt.hist(self.a100nsa.logSFR_NUV_KE[flag],histtype='step',color='b',bins=mybins,label='UV')
        
        plt.hist(self.a100nsa.logSFR_NUVIR_KE[flag],histtype='step',color=colorblind2,bins=mybins,label='NUVcorr')
        #flag = self.a100nsa.w4_mag > 0
        plt.hist(self.a100nsa.logSFR22_KE[flag],histtype='step',color=colorblind3,bins=mybins,label='IR')

        plt.legend(loc='upper left')

    def sfrmstar_gswlc(self):
        # star-forming main sequence
        ### LIMIT TO OVERLAP REGION ###
        ramin = 140.
        ramax = 230.
        decmin = 0.
        decmax = 35.
        zmax = 0.05
        vmax = 15000

        # keep overlap region
        # define flag based on GSWLC RA, DEC and redshift
        # GSWLC RA = RA_2
        # GSWLC_DEC = DEC_2
        # GSWLC redshift = Z

        
        ## July 27, 2020
        ## not sure why this changed, but GSWLC is RA_1
        # GSWLC RA = RA_1
        # GSWLC_DEC = DEC_1
        # GSWLC redshift = Z
        overlapFlag = (self.a100gsw['RA_1'] > ramin) &\
          (self.a100gsw['RA_1']< ramax) &\
          (self.a100gsw['DEC_1'] > decmin) &\
          (self.a100gsw['DEC_1'] < decmax) &\
          (self.a100gsw['Z']*3.e5 < vmax) 


        gsw_flag = (self.a100gsw.logMstar > 0) #& (self.a100gsw.photFlag_gi_2 == 1)
        flag1 = (self.a100gsw.a100Flag == 1) & (self.a100gsw.gswFlag ==1) & gsw_flag & overlapFlag
                                                       
        x1 = self.a100gsw.logMstar[flag1]
        y1 = self.a100gsw.logSFR[flag1]

        flag2 = (self.a100gsw.a100Flag ==0) & (self.a100gsw.gswFlag == 1)& gsw_flag & overlapFlag# & (self.a100gsw.photFlag_gi_2 == 1) 
        x2 = self.a100gsw.logMstar[flag2]
        y2 = self.a100gsw.logSFR[flag2]
        
        contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(2,500,12)
        #print(contour_levels)
        
        ax1,ax2,ax3 = sfrmass(x1,y1, x2, y2, 'ALFALFA+GSWLC-2', 'GSWLC-2 only', 'sfrmstar-gswlc.pdf', \
                  hexbinflag=False,contourflag=True,contour_bins=40, \
                  ncontour_levels=contour_levels,\
                  xmin=6., xmax=12,ymin=-3,ymax=2, \
                  xlabel='$GSWLC-2 \ \log_{10}(M_\star (M_\odot))$', \
                ylabel='$GSWLC-2 \ \log_{10}(SFR \ (M_\odot~yr^{-1}))$',  color2=colorblind1, color1=colorblind3)
        # add reference line at log(sSFR) = -10
        
        plt.savefig('sfrmstar-gswlc.pdf')
        plt.savefig('sfrmstar-gswlc.png')
        plt.savefig('fig9a.pdf')        
        #return flag1
    def sfrmstar_a100(self,correctMass=False,useTaylor=False):
        # star-forming main sequence

        cat = self.allcats
        if useTaylor:
            mass_key = 'logMstarTaylor_1'
        else:
            mass_key = 'logMstarMcGaugh'
        # keep overlap region
        keepa100 = self.get_overlap_a100(cat)
        # cull a100

        if useTaylor:
            a100_flag =  (cat['SERSIC_ABSMAG'][:,1] < 0)  & keepa100 & (cat['photFlag_gi_1'] == 1)
        else:
            a100_flag = (cat['SERSIC_ABSMAG'][:,1] < 0)  & keepa100 &(cat['w1_mag'] > 0)
        # could add flag that keeps SFR22  in SFR22 > 0 (and not require NUV detection for these)
        print('number with a100_flag = ',sum(a100_flag))
        flag1 = (cat['a100Flag'] == 1) & (cat['gswFlag'] ==1) & a100_flag 

        flag2 = (cat['a100Flag'] ==1) & (cat['gswFlag'] == 0) & a100_flag
        
        print("number of those NOT in GSWLC = ",sum(flag2))

        x1 = cat[mass_key][flag1]
        y1 = cat['logSFR_NUVIR_KE'][flag1]

        x2 = cat[mass_key][flag2]
        y2 = cat['logSFR_NUVIR_KE'][flag2]
        print('number in flag2 = ',sum(flag2))

        if correctMass & (not useTaylor):
            x1 = self.correctMcGaughMass(x1)
            x2 = self.correctMcGaughMass(x2)            

        ## CUT BOTH SAMPLES AT sSFR > -11.5
        
        #flag1 = (y1-x1) > -11.5
        #flag2 = (y2-x2) > -11.5

        #x1 = x1[flag1]
        #y1 = y1[flag1]
        
        #x2 = x2[flag2]
        #y2 = y2[flag2]
        
        contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(2,500,12)
        #print(contour_levels)

        if correctMass and (not useTaylor):
            xlabel = '$McGaugh_{corrected} \ \log_{10}(M_\star (M_\odot))$'
            outfile1 = 'sfrmstar-a100-correctedMstar.pdf'            
        elif useTaylor:
            xlabel = '$Taylor \ \log_{10}(M_\star (M_\odot))$'
            outfile1 = 'sfrmstar-a100-Taylor.pdf'            
        else:
            xlabel = '$McGaugh \ \log_{10}(M_\star (M_\odot))$'
            outfile1 = 'sfrmstar-a100.pdf'            
        ax1,ax2,ax3 = sfrmass(x1,y1, x2, y2, 'ALFALFA+GSWLC-2', 'ALFALFA only', outfile1, \
                hexbinflag=False,alphagray=.05,contourflag1=True,contourflag=False,\
                contour_bins=40, ncontour_levels=contour_levels,\
                xmin=6., xmax=12,ymin=-3,ymax=2, \
                xlabel=xlabel, \
                ylabel=r'$\rm \log_{10}(SFR_{NUVcor} \ (M_\odot~yr^{-1}))$', \
                              color2=colorblind2,color1=colorblind3,plotmsline=True,plotsalim18=False)


        # add the median for a100 galaxies with ssfr > -11
        # for a direct comparison with GSWLC
        lmstar = cat[mass_key]
        sfr = cat['logSFR_NUVIR_KE']
        ssfr = sfr - lmstar
        newflag = (flag1 | flag2)  & (ssfr > -11) & (lmstar > 7.5) & (lmstar < 11)
        nbins=20
        ybin,xbin_edges,binnumber = binned_statistic(lmstar[newflag],sfr[newflag],bins=nbins,statistic='median')
        ybin_err,xbin_edges,binnumber = binned_statistic(lmstar[newflag],sfr[newflag],bins=nbins,statistic='std')
        ybin_err = ybin_err/np.sqrt(sum(newflag)/nbins)
        xbin = 0.5*(xbin_edges[:-1]+xbin_edges[1:])
        #ax1.plot(xbin,ybin,'w-',lw=4,label='_nolegend_')
        ax1.plot(xbin,ybin,'w-',lw=5,label='_nolegend_')        
        ax1.plot(xbin,ybin,'k-',lw=3,ls='-.',label='ALFALFA med')
        #ax1.fill_between(xbin,ybin+ybin_err,ybin-ybin_err,color='k',label='A100 med',zorder=10)
        #ax1.errorbar(xbin,ybin,yerr=ybin_err,fmt="none",ec='c',lw=2,label='_nolegend_')        
        ax1.legend()

        
        if useTaylor:
            print('saving taylor files')
            plt.savefig('sfrmstar-a100-Taylor.pdf')
            plt.savefig('sfrmstar-a100-Taylor.png')
            plt.savefig('fig9b.pdf')                        
        elif correctMass:
            plt.savefig('sfrmstar-a100-correctedMstar.pdf')
            plt.savefig('sfrmstar-a100-correctedMstar.png')
        else:
            plt.savefig('sfrmstar-a100.pdf')
            plt.savefig('sfrmstar-a100.png')
        plt.savefig('fig9b.pdf')                        
            
        #return flag1
    def ssfrmstar_gswlc(self,masslimit=0,ssfrlimit=-11.5,plotssfrline=False):
        # star-forming main sequence
        
        gsw_flag = (self.a100gsw.logMstar > masslimit) & (self.a100gsw.photFlag_gi_2 == 1)
        flag1 = (self.a100gsw.a100Flag == 1) & (self.a100gsw.gswFlag ==1) & gsw_flag 
        x1 = self.a100gsw.logMstar[flag1]
        y1 = self.a100gsw.logSFR[flag1]-x1

        flag2 = (self.a100gsw.a100Flag ==0) & (self.a100gsw.gswFlag == 1) & (self.a100gsw.photFlag_gi_2 == 1) & (self.a100gsw.logMstar > masslimit) 
        x2 = self.a100gsw.logMstar[flag2]
        y2 = self.a100gsw.logSFR[flag2]-x2

        ## APPLY SSFR > -11.5 CUT
        flag1 = y1 > ssfrlimit
        flag2 = y2 > ssfrlimit

        x1 = x1[flag1]
        y1 = y1[flag1]
        
        x2 = x2[flag2]
        y2 = y2[flag2]

        #self.ssfrmstar_simple(x1,y1,x2,y2)
        contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(2,500,12)
        #print(contour_levels)
        outfile1 = 'ssfr-mstar-gswlc.pdf'
        ax1,ax2,ax3 = sfrmass(x1,y1, x2, y2, 'ALFALFA+GSWLC-2', 'GSWLC-2 only', outfile1, \
                  hexbinflag=False,contourflag=True,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=6., xmax=12,ymin=-12,ymax=-8.1, alphagray=.08,\
                  xlabel='$GSWLC-2 \ \log_{10}(M_\star (M_\odot))$', ylabel=r'$\rm GSWLC-2 \ \log_{10}(sSFR \ (yr^{-1}))$',  color2=colorblind1,color1=colorblind3,plotmsline=False,plotssfrline=plotssfrline)
        # add reference line at log(sSFR) = -10
        plt.savefig('ssfr-mstar-gswlc.pdf')
        plt.savefig('ssfr-mstar-gswlc.png')
        plt.savefig('fig10a.pdf')        
        return flag1
    def ssfrmstar_a100(self,ssfrlimit=-11.5,correctMass=False,useTaylor=False):
        # star-forming main sequence
        # use a100 values for SFR and Mstar


        # keep overlap region
        keepa100 = self.get_overlap_a100(self.allcats)
        
        #a100_flag = (self.allcats['w4_mag'] > 0)& (self.allcats['SERSIC_ABSMAG'][:,1] < 0) & (self.allcats.photFlag_gi_2 == 1)

        # all must have W4 detection, UV detection, and in overlap region
        a100_flag_allcats = (self.allcats['logMstarMcGaugh'] > 0) & (self.allcats['logSFR_NUVIR_KE'] > -5) & keepa100  & (self.allcats.photFlag_gi_1 == 1)
        if useTaylor:
            a100_flag_allcats = (self.allcats['logSFR_NUVIR_KE'] > -5) & keepa100  & (self.allcats.photFlag_gi_1 == 1)            
        
        flag1 = (self.allcats.a100Flag == 1) & (self.allcats.gswFlag ==1) & a100_flag_allcats 
        print("number of a100 with W4 and NUV detections, in GSWLC overlap = ",sum(a100_flag_allcats))
        flag2 = (self.allcats.a100Flag ==1) & (self.allcats.gswFlag == 0) & a100_flag_allcats
        print("number of those NOT in GSWLC = ",sum(flag2))


        if useTaylor:
            x1 = self.allcats['logMstarTaylor_1']
        else:
            x1 = self.allcats['logMstarMcGaugh']

        print('number in flag2 = ',sum(flag2))
        if correctMass & (not useTaylor):
            x1 = self.correctMcGaughMass(x1)

            xlabel = '$McGaugh_{corrected} \ \log_{10}(M_\star (M_\odot))$'
            outfile1 = 'ssfrmstar-a100-correctedMstar.pdf'
            outfile2 = 'ssfrmstar-a100-correctedMstar.png'
        elif useTaylor:
            xlabel = '$Taylor \ \log_{10}(M_\star (M_\odot))$'
            outfile1 = 'ssfrmstar-a100-Taylor.pdf'
            outfile2 = 'ssfrmstar-a100-Taylor.png'                        
        else:
            xlabel = '$McGaugh \ \log_{10}(M_\star (M_\odot))$'
            outfile1 = 'ssfrmstar-a100.pdf'
            outfile2 = 'ssfrmstar-a100.png'                        
        y1 = self.allcats['logSFR_NUVIR_KE']-x1

        
        x2 = x1[flag2]
        y2 = y1[flag2]        
        
        x1 = x1[flag1]
        y1 = y1[flag1]        
        ## CUT BOTH SAMPLES AT sSFR > -11.5
        
        flag1 = y1 > ssfrlimit
        flag2 = y2 > ssfrlimit

        x1 = x1[flag1]
        y1 = y1[flag1]
        
        x2 = x2[flag2]
        y2 = y2[flag2]
        
        #contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(2,200,12)
        #print(contour_levels)
        #outfile1 = 'ssfr-mstar-a100.pdf'
        ax1, ax2, ax3 = sfrmass(x1,y1, x2, y2, 'ALFALFA+GSWLC-2', 'ALFALFA only', outfile1, \
                  hexbinflag=False,alphagray=.05,contourflag1=True,contourflag=False,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=6., xmax=12,ymin=-12,ymax=-8.1, \
                                xlabel=xlabel, ylabel=r'$\rm \log_{10}(sSFR_{NUVcor} \ (yr^{-1}))$',  color2=colorblind2,color1=colorblind3,plotmsline=False,plotssfrline=True,plotsalim18=False)
        # add the median for a100 galaxies with ssfr > -11
        # for a direct comparison with GSWLC
        a100_flag_allcats = (self.allcats['logSFR_NUVIR_KE'] > -5) & keepa100  & (self.allcats.photFlag_gi_1 == 1)
        if useTaylor:
            lmstar = self.allcats['logMstarTaylor_1']
        elif correctMass: 
            lmstar = self.correctMcGaughMass(self.allcats['logMstarMcGaugh'])
        else:
            lmstar = self.allcats['logMstarMcGaugh']
                                             
        ssfr = self.allcats['logSFR_NUVIR_KE']-lmstar
        newflag =  (self.allcats.a100Flag == 1) & a100_flag_allcats  & (ssfr > -11) & (lmstar > 7.5) & (lmstar < 11)
        nbins=20
        ybin,xbin_edges,binnumber = binned_statistic(lmstar[newflag],ssfr[newflag],bins=nbins,statistic='mean')
        ybin_err,xbin_edges,binnumber = binned_statistic(lmstar[newflag],ssfr[newflag],bins=nbins,statistic='std')
        ybin_err = ybin_err/np.sqrt(sum(newflag)/nbins)
        xbin = 0.5*(xbin_edges[:-1]+xbin_edges[1:])
        #ax1.plot(xbin,ybin,'w-',lw=4,label='_nolegend_')
        #ax1.plot(xbin,ybin,'k-',lw=2,label='A100 med')
        ax1.plot(xbin,ybin,'w-',lw=5,label='_nolegend_')        
        ax1.plot(xbin,ybin,'k-',lw=3,ls='-.',label='ALFALFA med')
        
        #ax1.fill_between(xbin,ybin+ybin_err,ybin-ybin_err,color='.5',label='A100 med',zorder=10)
        #ax1.errorbar(xbin,ybin,yerr=ybin_err,fmt="none",ec='c',lw=2,label='_nolegend_')        
        ax1.legend()

        
        plt.savefig(outfile1)
        plt.savefig(outfile2)
        plt.savefig('fig10b.pdf')        
        return flag1
    def ssfrmstar_HIfrac_a100(self,ssfrlimit=-11.5,correctMass=False):
        # star-forming main sequence
        # use a100 values for SFR and Mstar


        # keep overlap region
        keepa100 = self.get_overlap_a100(self.allcats)
        
        #a100_flag = (self.allcats['w4_mag'] > 0)& (self.allcats['SERSIC_ABSMAG'][:,1] < 0) & (self.allcats.photFlag_gi_2 == 1)

        # all must have W4 detection, UV detection, and in overlap region
        a100_flag_allcats = (self.allcats['logMstarMcGaugh'] > 0) & (self.allcats['logSFR_NUVIR_KE'] > -5) & keepa100  & (self.allcats.photFlag_gi_1 == 1)
        
        flag1 = (self.allcats.a100Flag == 1) & (self.allcats.gswFlag ==1) & a100_flag_allcats 
        print("number of a100 with W4 and NUV detections, in GSWLC overlap = ",sum(a100_flag_allcats))
        flag2 = (self.allcats.a100Flag ==1) & (self.allcats.gswFlag == 0) & a100_flag_allcats
        print("number of those NOT in GSWLC = ",sum(flag2))

        x1 = self.allcats['logMstarMcGaugh']
        color1 = self.allcats['logMH']-self.allcats['logMstarMcGaugh']

        print('number in flag2 = ',sum(flag2))
        if correctMass:
            x1 = self.correctMcGaughMass(x1)

            xlabel = '$McGaugh_{corrected} \ \log_{10}(M_\star/M_\odot)$'
            outfile1 = 'ssfrmstar-a100-correctedMstar.pdf'
            outfile2 = 'ssfrmstar-a100-correctedMstar.png'                        
        else:
            xlabel = '$McGaugh \ \log_{10}(M_\star/M_\odot)$'
            outfile1 = 'ssfrmstar-a100.pdf'
            outfile2 = 'ssfrmstar-a100.png'                        
        y1 = self.allcats['logSFR_NUVIR_KE']-x1

        
        x2 = x1[flag2]
        y2 = y1[flag2]        
        color2 = color1[flag2]        
        x1 = x1[flag1]
        y1 = y1[flag1]
        color1 = color1[flag1]
        ## CUT BOTH SAMPLES AT sSFR > -11.5
        
        flag1 = y1 > ssfrlimit
        flag2 = y2 > ssfrlimit

        x1 = x1[flag1]
        y1 = y1[flag1]
        color1 = color1[flag1]        
        x2 = x2[flag2]
        y2 = y2[flag2]
        color2 = color2[flag2]                
        
        contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(2,500,12)
        #print(contour_levels)
        plt.figure(figsize=(10,6))
        allax = []
        plt.subplot(1,2,1)
        plt.scatter(x1,y1,c=color1,s=5,alpha=.5,vmin=-1.5,vmax=1.5)
        allax.append(plt.gca())
        plt.title('ALFALFA+GSWLC-2')
        plt.ylabel('$sSFR$')
        plt.axis([7,12,-12,-8])
        plt.subplot(1,2,2)
        plt.scatter(x2,y2,c=color2,s=5,alpha=.5,vmin=-1.5,vmax=1.5)
        allax.append(plt.gca())
        plt.axis([7,12,-12,-8])        
        plt.title('ALFALFA only')        
        plt.xlabel('$log (M_\star)$')

        plt.colorbar(label='$log(M_{HI}/M_\star)$',ax=allax)
        ax1 = plt.gca()
        lmstar = self.allcats['logMstarMcGaugh']
        ssfr = self.allcats['logSFR_NUVIR_KE']-lmstar
        newflag =  (self.allcats.a100Flag == 1) & a100_flag_allcats  & (ssfr > -11) & (lmstar > 7.5) & (lmstar < 11)
        nbins=20
        ybin,xbin_edges,binnumber = binned_statistic(lmstar[newflag],ssfr[newflag],bins=nbins,statistic='mean')
        ybin_err,xbin_edges,binnumber = binned_statistic(lmstar[newflag],ssfr[newflag],bins=nbins,statistic='std')
        ybin_err = ybin_err/np.sqrt(sum(newflag)/nbins)
        xbin = 0.5*(xbin_edges[:-1]+xbin_edges[1:])
        #ax1.plot(xbin,ybin,'w-',lw=4,label='_nolegend_')
        #ax1.plot(xbin,ybin,'k-',lw=2,label='A100 med')
        ax1.plot(xbin,ybin,'w-',lw=5,label='_nolegend_')        
        ax1.plot(xbin,ybin,'k-',lw=3,ls='-.',label='ALFALFA med')
        
        #ax1.fill_between(xbin,ybin+ybin_err,ybin-ybin_err,color='.5',label='A100 med',zorder=10)
        #ax1.errorbar(xbin,ybin,yerr=ybin_err,fmt="none",ec='c',lw=2,label='_nolegend_')        
        ax1.legend()

        
        plt.savefig(outfile1)
        plt.savefig(outfile2)
        plt.figure(figsize=(10,6))
        allax=[]
        plt.subplot(1,2,1)
        plt.scatter(color1,y1,c=x1,s=5,alpha=.5,vmin=8,vmax=11)
        #plt.plot(xbin,ybin,'w-',lw=5,label='_nolegend_')        
        #plt.plot(xbin,ybin,'k-',lw=3,ls='-.',label='A100 med')
        plt.axis([-2,2,-12,-8])                
        allax.append(plt.gca())
        plt.ylabel('$sSFR$')        
        plt.subplot(1,2,2)
        plt.scatter(color2,y2,c=x2,s=5,alpha=.5,vmin=8,vmax=11)
        #plt.plot(xbin,ybin,'w-',lw=5,label='_nolegend_')        
        #plt.plot(xbin,ybin,'k-',lw=3,ls='-.',label='A100 med')
        
        allax.append(plt.gca())
        plt.axis([-2,2,-12,-8])                        
        plt.xlabel('$log (M_{HI}/M_\star)$')

        plt.colorbar(label='$log(M_\star)$',ax=allax)
        return flag1
    def ssfr_HIfrac_a100(self,ssfrlimit=-11.5,correctMass=False):
        # star-forming main sequence
        # use a100 values for SFR and Mstar

        ### LIMIT TO OVERLAP REGION ###
        keepa100 = self.get_overlap_a100(self.allcats)

        
        #a100_flag = (self.allcats['w4_mag'] > 0)& (self.allcats['SERSIC_ABSMAG'][:,1] < 0) & (self.allcats.photFlag_gi_2 == 1)

        # all must have W4 detection, UV detection, and in overlap region
        a100_flag_allcats = (self.allcats['logMstarMcGaugh'] > 0) & (self.allcats['logSFR_NUVIR_KE'] > -5) & keepa100  & (self.allcats.photFlag_gi_1 == 1)
        
        flag1 = (self.allcats.a100Flag == 1) & (self.allcats.gswFlag ==1) & a100_flag_allcats 
        print("number of a100 with W4 and NUV detections, in GSWLC overlap = ",sum(a100_flag_allcats))
        flag2 = (self.allcats.a100Flag ==1) & (self.allcats.gswFlag == 0) & a100_flag_allcats
        print("number of those NOT in GSWLC = ",sum(flag2))

        x0 = self.allcats['logMstarMcGaugh']
        x1 = self.allcats['logMH']- x0


        print('number in flag2 = ',sum(flag2))
        if correctMass:
            x1 = self.allcats['logMH']-self.correctMcGaughMass(x0)

            xlabel = '$\log_{10}(M_{HI}/M_\star(corr))$'
            outfile1 = 'ssfr-HIfrac-a100-correctedMstar.pdf'
            outfile2 = 'ssfr-HIfrac-a100-correctedMstar.png'                        
        else:
            xlabel = '$\log_{10}(M_{HI}/M_\star)$'
            outfile1 = 'ssfr-HIfrac-a100.pdf'
            outfile2 = 'ssfr-HIfrac-a100.png'                        
        y1 = self.allcats['logSFR_NUVIR_KE']-x0

        
        x2 = x1[flag2]
        y2 = y1[flag2]        
        
        x1 = x1[flag1]
        y1 = y1[flag1]        
        ## CUT BOTH SAMPLES AT sSFR > -11.5
        
        flag1 = y1 > ssfrlimit
        flag2 = y2 > ssfrlimit

        x1 = x1[flag1]
        y1 = y1[flag1]
        
        x2 = x2[flag2]
        y2 = y2[flag2]
        
        contour_levels = np.logspace(.7,5.5,15)
        contour_levels = np.linspace(2,500,12)
        #print(contour_levels)
        outfile1 = 'ssfr-mstar-a100.pdf'
        ax1, ax2, ax3 = sfrmass(x1,y1, x2, y2, 'ALFALFA+GSWLC-2', 'ALFALFA only', outfile1, \
                  hexbinflag=False,alphagray=.05,contourflag1=True,contourflag=False,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=-3, xmax=3,ymin=-12,ymax=-8.1, \
                  xlabel=xlabel, ylabel=r'$\rm \log_{10}(sSFR/yr^{-1})$',  color2=colorblind2,color1=colorblind3,plotmsline=False,plotssfrline=True)
        # add the median for a100 galaxies with ssfr > -11
        # for a direct comparison with GSWLC
        lmstar = self.allcats['logMstarMcGaugh']
        ssfr = self.allcats['logSFR_NUVIR_KE']-lmstar
        newflag =  (self.allcats.a100Flag == 1) & a100_flag_allcats  & (ssfr > -11) & (lmstar > 7.5) & (lmstar < 11)
        nbins=20
        ybin,xbin_edges,binnumber = binned_statistic(lmstar[newflag],ssfr[newflag],bins=nbins,statistic='mean')
        ybin_err,xbin_edges,binnumber = binned_statistic(lmstar[newflag],ssfr[newflag],bins=nbins,statistic='std')
        ybin_err = ybin_err/np.sqrt(sum(newflag)/nbins)
        xbin = 0.5*(xbin_edges[:-1]+xbin_edges[1:])
        #ax1.plot(xbin,ybin,'w-',lw=4,label='_nolegend_')
        #ax1.plot(xbin,ybin,'k-',lw=2,label='A100 med')
        ax1.plot(xbin,ybin,'w-',lw=5,label='_nolegend_')        
        ax1.plot(xbin,ybin,'k-',lw=3,ls='-.',label='ALFALFA med')
        
        #ax1.fill_between(xbin,ybin+ybin_err,ybin-ybin_err,color='.5',label='A100 med',zorder=10)
        #ax1.errorbar(xbin,ybin,yerr=ybin_err,fmt="none",ec='c',lw=2,label='_nolegend_')        
        ax1.legend()

        
        plt.savefig(outfile1)
        plt.savefig(outfile2)
        return flag1
    def correctMcGaughMass(self,mass):
        correctFlag = (mass > 7.5) & (mass < 11)
        mass[correctFlag] = 0.0647*mass[correctFlag]**2-0.195*mass[correctFlag]+5.32
        return mass
    
        
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
        sfrmass(x1,y1, x2, y2, 'ALFALFA+GSWLC-2', 'GSWLC-2 only', 'a100-gswlc-color-mass-2.pdf', \
                  hexbinflag=True,contourflag=True,contour_bins=40, ncontour_levels=contour_levels,\
                  xmin=-.5, xmax=2,ymin=-13.8,ymax=-8.1, \
                  xlabel='$(g-i)_{corrected}$', ylabel=r'$\rm \log_{10}(sSFR/yr^{-1})$',  color2=colorblind1,plotline=False)
        # add reference line at log(sSFR) = -10
        plt.savefig('ssfr-color-gswlc.pdf')
        plt.savefig('ssfr-color-gswlc.png')        
        return flag1
    def ssfrHIfrac(self):
        '''plot sSFR vs HI mass fraction'''

        
        gsw_flag = (self.a100gsw.logMstar > 0) & (self.a100gsw.photFlag_gi_2 == 1)
        flag1 = (self.a100gsw.a100Flag == 1) & (self.a100gsw.gswFlag ==1) & gsw_flag 
        x1 = self.a100gsw.logMstar[flag1]
        y1 = self.a100gsw.logSFR[flag1]-x1


        x2 = self.a100gsw.logMstar[flag1]
        y2 = self.a100gsw.logMH[flag1]-x2

        plt.figure(figsize=(6,4))
        plt.subplots_adjust(bottom=.175, left=.175)
        #plt.plot(y2,y1,'k.',alpha=.2)
        plt.hexbin(y2,y1,bins='log', gridsize=75,label='ALFALFA+GSWLC-2')
        
        plt.ylabel(r'$\rm \log_{10}(sSFR/yr^{-1})$',fontsize=16)
        plt.xlabel(r'$\rm log_{10}(M_{HI}/M_\star) $',fontsize=16)
        plt.savefig('ssfr-HIfrac.pdf')
        plt.savefig('ssfr-HIfrac.png')        
        return flag1     
    def ssfrHIfraca100(self,colordens=False):
        '''plot sSFR vs HI mass fraction for a100 sample only'''

        a100_flag_allcats = (self.allcats['logSFR_NUVIR_KE'] > -5)   & (self.allcats.photFlag_gi_1 == 1)            
        


        xlabel = '$Taylor \ \log_{10}(M_\star (M_\odot))$'
        outfile1 = 'ssfr-HIfrac-a100-Taylor.pdf'
        outfile2 = 'ssfr-HIfrac-a100-Taylor.png'

        mstar = self.allcats['logMstarTaylor_1']        
        sfr = self.allcats['logSFR_NUVIR_KE']

        ## CUT BOTH SAMPLES AT sSFR > -11.5
        ssfrlimit = -15
        ssfr = sfr - mstar
        hifrac = self.allcats['logMH']-mstar

        flag1 = ssfr > ssfrlimit

        mstar = mstar[flag1]
        ssfr = ssfr[flag1]
        hifrac = hifrac[flag1]
        
        plt.figure(figsize=(6,4))
        plt.subplots_adjust(bottom=.175, left=.175)
        #plt.plot(y2,y1,'k.',alpha=.2)
        if colordens:
            plt.hexbin(hifrac,ssfr,bins='log', gridsize=75,label='ALFALFA+GSWLC-2')
        else:
            plt.scatter(hifrac,ssfr,c=mstar, alpha=.2,s=10)
        
        cb = plt.colorbar()
        plt.axis([-2,2,-14,-8])
        plt.ylabel(r'$\rm \log_{10}(sSFR/yr^{-1})$',fontsize=16)
        plt.xlabel(r'$\rm log_{10}(M_{HI}/M_\star) $',fontsize=16)
        plt.savefig(outfile1)
        plt.savefig(outfile2)        

    def ssfr(self):
        gsw_flag = (self.a100gsw.logMstar > 0) & (self.a100gsw.photFlag_gi_2 == 1)
        flag1 = (self.a100gsw.a100Flag == 1) & (self.a100gsw.gswFlag ==1) & gsw_flag 
        x1 = self.a100gsw.logMstar[flag1]
        y1 = self.a100gsw.logSFR[flag1]
        ssfr1 = y1 - x1
        name1 = 'ALFALFA+GSWLC-2'
        
        flag2 = (self.a100gsw.a100Flag ==0) & (self.a100gsw.gswFlag == 1) & (self.a100gsw.photFlag_gi_2 == 1) 
        x2 = self.a100gsw.logMstar[flag2]
        y2 = self.a100gsw.logSFR[flag2]
        ssfr2 = y2-x2
        color2=colorblind1
        name2 = 'GSWLC-2 only'
        
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
        plt.savefig('ssfr.png')        
    def plotelbaz(self):
        xe=np.arange(8.5,11.5,.1)
        xe=10.**xe
        ye=(.08e-9)*xe
        plt.plot(np.log10(xe),np.log10(ye),'k-',lw=1,label='$Elbaz+2011$')
        plt.plot(np.log10(xe),np.log10(2*ye),'k:',lw=1,label='$2 \ SFR_{MS}$')
        # using our own MS fit for field galaxies
        # use stellar mass between 9.5 and 10.5
    def plotsalim07(self):
        #plot the main sequence from Salim+07 for a Chabrier IMF

        lmstar=np.arange(8.5,11.5,0.1)

        #use their equation 11 for pure SF galaxies
        lssfr = -0.35*(lmstar - 10) - 9.83

        #use their equation 12 for color-selected galaxies including
        #AGN/SF composites.  This is for log(Mstar)>9.4
        #lssfr = -0.53*(lmstar - 10) - 9.87

        lsfr = lmstar + lssfr -.3
        sfr = 10.**lsfr

        plt.plot(lmstar, lsfr, 'w-', lw=4)
        plt.plot(lmstar, lsfr, c='salmon',ls='-', lw=2, label='$Salim+07$')
        plt.plot(lmstar, lsfr-np.log10(5.), 'w--', lw=4)
        plt.plot(lmstar, lsfr-np.log10(5.), c='salmon',ls='--', lw=2)
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
            plt.xlabel('logSFR GSWLC-2',fontsize=12)
            plt.ylabel(ylabels[i],fontsize=12)
            xl = np.linspace(xmin,xmax,100)
            plt.plot(xl,xl,'r--',label='1:1')
            plt.legend()
        plt.savefig('sfr-gsw-wise.pdf')
        plt.savefig('sfr-gsw-wise.png')        

    def sfr_mstar_wise(self):
        plt.figure(figsize=(10,8))
        plt.subplots_adjust(bottom=.15)
        w1snr = np.abs(self.a100sdss['w1_nanomaggies']*np.sqrt(self.a100sdss['w1_nanomaggies_ivar']))
        w12snr = np.abs(self.a100sdss['w3_nanomaggies']*np.sqrt(self.a100sdss['w3_nanomaggies_ivar']))
        flag = (w1snr > 5) & (w12snr > 5)
        x = self.a100sdss['logMstarMcGaugh']
        y = self.a100sdss['logSFR12']
        #plt.plot(x[flag],y[flag],'k.')
        xmin=8.5
        xmax=11.5
        ymin = -1
        ymax=2
        #plt.hexbin(x[flag],y[flag],extent=[xmin,xmax,ymin,ymax],cmap='gray_r')
        #plt.plot(x[flag],y[flag],'k.',alpha=.1)#extent=[xmin,xmax,ymin,ymax],cmap='gray_r')        
        plt.hexbin(x[flag],self.a100sdss['logSFR22_KE'][flag],extent=[xmin,xmax,ymin,ymax],cmap='gray_r')        
        plt.xlabel('logMstar Cluver',fontsize=12)
        plt.ylabel('logSFR 12um',fontsize=12)
        #xl = np.linspace(xmin,xmax,100)
        #plt.plot(xl,xl,'r--',label='1:1')
        #plt.plot(xl,xl+.4,'r--',c='.5',label='logM Taylor + .4')
        self.plotsalim07()
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

    def wise_gsw_mstar(self):
        # x is 4.6-12
        # y is 3.4-4.6
        # cluver+2014, figure 5, shows different regions for spirals, ellip, starburst, etc
        # references color-color diagram of jarrett+2011

        x = self.a100gsw['logMstar']
        y = self.a100gsw['logMstarCluver']
        y = self.a100gsw['logMstarMcGaugh']
        xmin=7
        xmax=12
        ymin=xmin
        ymax=xmax
        plt.figure(figsize=(10,8))
        plt.subplots_adjust(bottom=.15,left=.15)
        flag = np.ones(len(x),'bool')
        #plt.hexbin(x[flag],y[flag],extent=[xmin,xmax,ymin,ymax],cmap='gray_r')
        plt.plot(x[flag],y[flag],'k.',alpha=.1,label='ALFALFA+WISE+GSWLC-2')
        plt.xlabel('logMstar GSWLC-2',fontsize=20)
        plt.ylabel('logMStar McGaugh',fontsize=20)
        plt.axis([xmin,xmax,ymin,ymax])
        xl = np.linspace(xmin,xmax,20)
        plt.plot(xl,xl,'r-',label='1:1')
        plt.plot(xl,xl+.3,'r--',label='1:1+0.3')        
        #plt.axhline(y=.76,ls='--',color='c')
        plt.legend()
    def wise_gsw_sfr(self):
        # x is 4.6-12
        # y is 3.4-4.6
        # cluver+2014, figure 5, shows different regions for spirals, ellip, starburst, etc
        # references color-color diagram of jarrett+2011
        w4snr = np.abs(self.a100gsw['w4_nanomaggies']*np.sqrt(self.a100gsw['w4_nanomaggies_ivar']))
        flag = w4snr > 5
        x = self.a100gsw['logSFR']
        y = self.a100gsw['logSFR22_KE']

        xmin=-2
        xmax=2
        ymin=xmin
        ymax=xmax
        plt.figure(figsize=(10,8))
        plt.subplots_adjust(bottom=.15,left=.15)
        #flag = np.ones(len(x),'bool')
        #plt.hexbin(x[flag],y[flag],extent=[xmin,xmax,ymin,ymax],cmap='gray_r')
        plt.plot(x[flag],y[flag],'k.',alpha=.1,label='ALFALFA+WISE+GSWLC-2')
        plt.xlabel('logSFR GSWLC-2',fontsize=20)
        plt.ylabel('logSFR 22 Kennicutt & Evans',fontsize=20)
        plt.axis([xmin,xmax,ymin,ymax])
        xl = np.linspace(xmin,xmax,20)
        plt.plot(xl,xl,'r-',label='1:1')
        plt.plot(xl,xl-.3,'r--',label='1:1-0.3')        
        #plt.axhline(y=.76,ls='--',color='c')
        plt.legend()
    def detection_fractions(self,overlap=False):
        # make a histogram showing fraction of a100 detected
        # as a function of stellar mass
        #
        # GSWLC
        # NSA
        # W1
        # W4
        # NUV
        #
        # use the a100 catalog that is matched to all the others
        # a100-sdss-wise-nsa-GSWLC2.fits
        #
        # this is read in as allcats

        cat = self.allcats
        overlapFlag = self.get_overlap_a100(cat)

        flags = [(self.allcats['nsaFlag']==1) & (self.allcats['SERSIC_ABSMAG'][:,1] < 0.),\
                 self.allcats['gswFlag']==1,\
                 self.allcats['w1_mag'] > 0.,\
                 self.allcats['w4_mag'] > 0.,\
                 self.allcats['logMstar'] > 0,\
                 self.allcats['a100Flag']==1]
        baseflag = self.allcats['photFlag_gi_1'] == 1  & (self.allcats['logMstarTaylor_1'] > 6)& (self.allcats['logMstarTaylor_1'] < 11.5)
        if overlap:
            baseflag = baseflag & overlapFlag
        ylabels = ['NUV/NSA','GSWLC-2','W1','W4','GSWLC-2 logMstar','A100']
        symbols = ['o','s','^','D','*','v','D']
        fig,ax = plt.subplots(1,1,figsize=(8,6))
        #plt.subplots_adjust(bottom=.15,left=.12)
        
        x = self.allcats['logMstarTaylor_1']
        mybins = np.linspace(min(x),max(x),20)
        t= np.histogram(x[baseflag],bins=mybins)
        ytot = t[0]
        xtot = t[1]
        # calculate the position of the bin centers        
        xplt = 0.5*(xtot[0:-1]+xtot[1:])        
        for i,y in enumerate(flags[0:-2]):
            t = np.histogram(x[flags[i]& baseflag],bins=mybins)
            frac,yerr = ratioerror(t[0],ytot)
            
            #if i == 0:
            #    ax.plot(xplt,frac,'ko',color=mycolors[i],marker=symbols[i],markersize=16,label=ylabels[i],mfc='none')
            #else:
            ax.plot(xplt,frac,'ko',color=mycolors[i],marker=symbols[i],markersize=8+i,label=ylabels[i])
            plt.errorbar(xplt,frac,yerr=yerr,color=mycolors[i])

        ax.legend(loc='lower right')
        #plt.subplots_adjust(right=.8)
        plt.xlabel(r'$Taylor \ \log_{10}(M_\star/M_\odot) $',fontsize=18)
        plt.ylabel(r'$Fraction\  of \ ALFALFA-SDSS\ Galaxies $',fontsize=18)
        plt.axhline(y=1,ls=':',color='k')
        plt.ylim(-.05,1.09)
        
        output1 = 'a100_detection_frac.png'
        output2 = 'a100_detection_frac.pdf'
        output2 = 'fig2a.pdf'
        if overlap:
            output1 = 'a100_detection_frac_overlap.png'
            output2 = 'a100_detection_frac_overlap.pdf'        
            output2='fig2b.pdf'
        plt.savefig(output1)
        plt.savefig(output2)

    def compare_sfr_uvir_mstar(self):
        plt.figure(figsize=(8,6))
        limits=[7.5,11.5,-2,1.2]
        x = self.allcats['logMstarTaylor_1']
        flag = (self.allcats['logSFR_NUV_KE'] > -99) & (self.allcats['w4_mag'] > 0)& (self.allcats['SERSIC_ABSMAG'][:,1] < 0) 
        y = self.allcats['logSFR_NUV_KE']-self.allcats['logSFR22_KE']
        plt.hexbin(x[flag],y[flag], gridsize=50,extent=limits)

        plt.xlabel('$Taylor \ \log_{10}(M_\star/M_\odot)$',fontsize=20)
        plt.ylabel('$\log_{10}(SFR_{NUV}/SFR_{IR})$',fontsize=20)
        plt.axhline(y=0,ls='--',color='k')
        plt.axis(limits)
        plt.title(r'$ALFALFA \ \alpha.100 \ Sample$',fontsize=20)
        plt.savefig(homedir+'/research/APPSS/plots/ratio-sfr-uvir-vs-mstar.pdf')
        plt.savefig(homedir+'/research/APPSS/plots/ratio-sfr-uvir-vs-mstar.png')
        
    
class calibsfr():
    def __init__(self):
        # read in catalog that matches a100+NSA+GSWLC
        self.cat = fits.getdata(tabledir+'/a100-sdss-wise-nsa-gswlcA2.fits')

        self.gswflag = self.cat['logMstar'] > 0
    def compare_mstar(self):

        # plot UV+IR vs GSWLC SFR
        # logSFR22_KE
        # logSFR_NUV_KE
        # logSFR_NUVIR_KE
        ycols = ['logMstarTaylor_1','logMstarMcGaugh', 'logMstarCluver']
        ylabels = ['logMstar Taylor','logMstar McGaugh','logMstar Cluver']
        # GSWLC value is logSFR
        x = self.cat['logMstar']
        flag = (self.cat['logSFR'] > -99) & (self.cat['w1_mag'] > 0)
        
        plt.figure(figsize=(12,7))
        plt.subplots_adjust(wspace=.4,bottom=.2)
        xmin=7
        xmax=12
        for i in range(len(ycols)):
            plt.subplot(2,3,i+1)
            #plt.scatter(x[flag],self.cat[ycols[i]][flag],label=ycols[i],s=5)
            plt.hexbin(x[flag],self.cat[ycols[i]][flag],cmap='gray_r',vmin=0,vmax=50,extent=(xmin,xmax,xmin,xmax),gridsize=75)

            plt.ylabel(ylabels[i])

            xl = np.linspace(xmin,xmax,100)
            plt.plot(xl,xl,'k-',lw=2,label='1:1')
            #plt.plot(xl,xl+.3,'k:',lw=1,label='1:1+.3')
            #plt.plot(xl,xl-.3,'k--',lw=1,label='1:1-.3')
            # fit offset
            # initial guess
            p0 = np.array([1,0])
            flag2 = flag & (x > 8) & (x < 12) & (self.cat[ycols[i]] > 8) & (self.cat[ycols[i]] < 12)
            popt,pcov = curve_fit(fitZPoffset,x[flag2],self.cat[ycols[i]][flag2],p0=p0, method='dogbox')
            plt.plot(x[flag2],fitZPoffset(x[flag2],*popt),'r-',label='fit: m=1,b=%.2f'%tuple(popt))
            
            plt.axis([xmin,xmax,xmin,xmax])
            plt.legend(loc='upper left',fontsize=10)

            # plot residuals
            plt.subplot(2,3,i+1+3)
            residual = self.cat[ycols[i]] - fitZPoffset(x,*popt)
            plt.ylabel(ylabels[i]+' - fit')
            plt.hexbin(x[flag],residual[flag],cmap='gray_r',vmin=0,vmax=50,extent=(xmin,xmax,-1,1),gridsize=75)
            plt.text(7.75,-.75,'$\sigma = {:.2f}$'.format(np.std(residual[flag2])))
            plt.axhline(y=0,c='k',lw=2)
            plt.xlabel('GSWLC-2 logMstar')
        plt.savefig(homedir+'/research/APPSS/plots/GSWLC-mstar-comparison.pdf')
        plt.savefig(homedir+'/research/APPSS/plots/GSWLC-mstar-comparison.png')

    def fit_mstar(self):

        # plot UV+IR vs GSWLC SFR
        # logSFR22_KE
        # logSFR_NUV_KE
        # logSFR_NUVIR_KE
        xcols = ['logMstarTaylor_1','logMstarMcGaugh','logMstarCluver']
        xlabels = ['$Taylor \ \log_{10}(M_\star/M_\odot)$','$McGaugh \ \log_{10}(M_\star/M_\odot)$','$Cluver \ \log_{10}(M_\star/M_\odot)$']
        ylabels = ['GSWLC-2 logMstar ']
        ylabels = ['$GSWLC-2 \ \log_{10}(M_\star/M_\odot)  $']        
        # GSWLC value is logSFR
        y = self.cat['logMstar']
        flag = (self.cat['logSFR'] > -99) & (self.cat['w1_mag'] > 0)
        
        plt.figure(figsize=(12,7))
        plt.subplots_adjust(wspace=.45,bottom=.2)
        xmin=7
        xmax=12
        for i in range(len(xcols)):
            x = self.cat[xcols[i]]
            plt.subplot(2,3,i+1)
            func = fitline
            #if i == 1:
            #    func = fitparab # fit a parabola to McGaugh stellar mass
            #plt.scatter(x[flag],self.cat[ycols[i]][flag],label=ycols[i],s=5)
            plt.hexbin(x[flag],y[flag],cmap='gray_r',vmin=0,vmax=50,extent=(xmin,xmax,xmin,xmax),gridsize=75)

            plt.ylabel(ylabels[0])

            xl = np.linspace(xmin,xmax,100)
            plt.plot(xl,xl,'k-',lw=3,label='1:1',color=colorblind2)
            #plt.plot(xl,xl+.3,'k:',lw=1,label='1:1+.3')
            #plt.plot(xl,xl-.3,'k--',lw=1,label='1:1-.3')
            # fit offset
            flag2 = flag & (x > 8) & (x < 12) & (y > 8) & (y < 12)
            popt,pcov = curve_fit(func,x[flag2],y[flag2])
            #if i == 1:
            #    s = 'a=%.2e,b=%.2e,c=%.2e'%tuple(popt)
            #else:
            #    s = 'a=%.2e,b=%.2e'%tuple(popt)
            s = 'y = %.3f x + %.3e'%tuple(popt)
            plt.plot(xl,func(xl,*popt),'w-',label='_nolegend_',color='w',lw=3)            
            plt.plot(xl,func(xl,*popt),'r--',label=s,color=colorblind1,lw=3)
            
            plt.axis([xmin,xmax,xmin,xmax])
            plt.legend(loc='lower right',fontsize=10)

            # plot residuals
            plt.subplot(2,3,i+1+3)
            residual = y - func(x,*popt)
            #plt.ylabel(ylabels[0]+' - fit')
            plt.ylabel('$residual$')            
            plt.hexbin(x[flag],residual[flag],cmap='gray_r',vmin=0,vmax=50,extent=(xmin,xmax,-1,1),gridsize=75)
            plt.text(7.75,-.75,'$\sigma = {:.2f}$'.format(np.std(residual[flag2])))
            plt.axhline(y=0,c=colorblind2,lw=3)
            plt.xlabel(xlabels[i])
        plt.savefig(homedir+'/research/APPSS/plots/wise-mstar-fit2gswlc.pdf')
        plt.savefig(homedir+'/research/APPSS/plots/wise-mstar-fit2gswlc.png')
        plt.savefig(homedir+'/research/APPSS/plots/fig3.pdf')


    def fit2_mstar_taylor(self):
        # fit unwise stellar masses to taylor
        # this might be better because we have taylor stellar masses
        # for most of a100, whereas GSWLC is going to be biased toward
        # higher stellar mass
        xcols = ['logMstarMcGaugh','logMstarCluver']
        xlabels = ['logMstar McGaugh','logMstar Cluver']
        ylabels = ['logMstar Taylor']        
        # GSWLC value is logSFR
        y = self.cat['logMstarTaylor_1']
        flag =  (self.cat['w1_mag'] > 0)
        plt.figure(figsize=(10,7))
        plt.subplots_adjust(wspace=.45,bottom=.2)
        xmin=7
        xmax=12
        for i in range(len(xcols)):
            x = self.cat[xcols[i]]
            plt.subplot(2,2,i+1)
            func = fitline
            if i == 0:
                func = fitparab # fit a parabola to McGaugh stellar mass
            #plt.scatter(x[flag],self.cat[ycols[i]][flag],label=ycols[i],s=5)
            plt.hexbin(x[flag],y[flag],cmap='gray_r',vmin=0,vmax=50,extent=(xmin,xmax,xmin,xmax),gridsize=75)

            plt.ylabel(ylabels[0])

            xl = np.linspace(xmin,xmax,100)
            plt.plot(xl,xl,'k-',lw=2,label='1:1')
            #plt.plot(xl,xl+.3,'k:',lw=1,label='1:1+.3')
            offset=-.3
            if i == 1:
                plt.plot(xl,xl-.3,'r--',lw=1,label='1:1+{:.1f}'.format(offset))
            # fit offset
            flag2 = flag & (x > 8) & (x < 12) & (y > 8) & (y < 12)
            popt,pcov = curve_fit(func,x[flag2],y[flag2])
            if i == 0:
                s = 'a=%.2f,b=%.2f,c=%.2f'%tuple(popt)
            else:
                s = 'a=%.2f,b=%.2f'%tuple(popt)
            plt.plot(xl,func(xl,*popt),'r-',label=s)
            
            plt.axis([xmin,xmax,xmin,xmax])
            plt.legend(loc='lower right',fontsize=10)

            # plot residuals
            plt.subplot(2,2,i+1+2)
            if i == 1:
                residual = y - (x + offset)
            else:
                residual = y - func(x,*popt)
            plt.ylabel(ylabels[0]+' - fit')
            plt.hexbin(x[flag],residual[flag],cmap='gray_r',vmin=0,vmax=50,extent=(xmin,xmax,-1,1),gridsize=75)
            plt.text(7.75,-.75,'$\sigma = {:.2f}$'.format(np.std(residual[flag2])))
            plt.axhline(y=0,c='k',lw=2)
            plt.xlabel(xlabels[i])
        plt.savefig(homedir+'/research/APPSS/plots/wise-mstar-fit2taylor.pdf')
        plt.savefig(homedir+'/research/APPSS/plots/wise-mstar-fit2taylor.png')



    def compare_sfr(self):

        # logSFR22_KE
        # logSFR_NUV_KE
        # logSFR_NUVIR_KE
        ycols = ['logSFR22_KE','logSFR_NUV_KE','logSFR_NUVIR_KE']
                            
        # GSWLC value is logSFR
        x = self.cat['logSFR']
        flag = (self.cat['logSFR'] > -99) & (self.cat['w4_mag'] > 0)
        
        plt.figure(figsize=(12,3.5))
        plt.subplots_adjust(wspace=.3,bottom=.2)
        xmin=-2.5
        xmax=2.5
        for i in range(len(ycols)):
            plt.subplot(1,3,i+1)
            #plt.scatter(x[flag],self.cat[ycols[i]][flag],label=ycols[i],s=5)
            plt.hexbin(x[flag],self.cat[ycols[i]][flag],cmap='gray_r',vmin=0,vmax=50,extent=(xmin,xmax,xmin,xmax),gridsize=75)
            plt.xlabel('logSFR GSWLC-2')
            plt.ylabel(ycols[i])

            xl = np.linspace(-2,2,100)
            plt.plot(xl,xl,'k-',lw=2,label='1:1')
            plt.plot(xl,xl+.3,'k:',lw=1,label='1:1+.3')
            plt.plot(xl,xl-.3,'k--',lw=1,label='1:1-.3')
            flag2 = flag & (x > -2) & (x < 2) & (self.cat[ycols[i]] > -2) & (self.cat[ycols[i]] < 2)
            popt,pcov = curve_fit(fitZPoffset,x[flag2],self.cat[ycols[i]][flag2])
            plt.plot(x[flag2],fitZPoffset(x[flag2],*popt),'r-',label='fit: m=1,b=%.2f'%tuple(popt))
            #c = np.polyfit(x[flag],self.cat[ycols[i]][flag],1)            
            #plt.plot(xl,np.polyval(c,xl),'r-',label='ZP offset')
            #s = '{:.2f},{:.2f}'.format(c[0],c[1])
            #plt.text(2,-2,s,horizontalalignment='right')
            plt.axis([-2.5,2.5,-2.5,2.5])
            plt.legend(loc='upper left',fontsize=10)
        plt.savefig(homedir+'/research/APPSS/plots/GSWLC-SFR-comparison.pdf')

        plt.savefig(homedir+'/research/APPSS/plots/GSWLC-SFR-comparison.png')
    def fit_sfr(self,norder=1,snr_cut=5):
        
        # logSFR22_KE
        # logSFR_NUV_KE
        # logSFR_NUVIR_KE
        xcols = ['logSFR22_KE','logSFR_NUV_KE','logSFR_NUVIR_KE']
        xlabels = ['$log_{10}(SFR_{22})$','$log_{10}(SFR_{NUV})$','$log_{10}(SFR_{NUVcor})$']        
        ylabels = ['$GSWLC-2 \ log_{10}(SFR) $']
        # GSWLC value is logSFR
        y = self.cat['logSFR']
        flag = (self.cat['logSFR'] > -99) & (self.cat['w4_mag'] > 0)& (self.cat['SERSIC_ABSMAG'][:,1] < 0) & (self.cat['gswFlag'] == 1)

        # sn cut on w4
        plt.figure(figsize=(12,7))
        plt.subplots_adjust(wspace=.45,bottom=.2)
        xmin=-2.5
        xmax=2.5
        hexbinmax = 30

        for i in range(len(xcols)):
            x = self.cat[xcols[i]]
            #func = fitZPoffset
            if norder == 0:
                print('fitting offset only')
                func = fitZPoffset
                p0 = np.array([2])                
            elif norder == 1:
                func = fitline
                p0 = np.array([2,2])
            else:
                func = fitparab
                p0 = np.array([2,2,2])
            plt.subplot(2,3,i+1)
            #plt.scatter(x[flag],self.cat[ycols[i]][flag],label=ycols[i],s=5)
            plt.hexbin(x[flag],y[flag],cmap='gray_r',vmin=0,vmax=hexbinmax,extent=(xmin,xmax,xmin,xmax),gridsize=75)



            xl = np.linspace(-2,2,100)
            plt.plot(xl,xl,'k-',lw=3,label='1:1',color=colorblind2)
            #plt.plot(xl,xl+.3,'k:',lw=1,label='1:1+.3')
            #plt.plot(xl,xl-.3,'k--',lw=1,label='1:1-.3')
            minfit  = -2
            maxfit = 2
            if i == 0:
                minfit = minfit
                maxfit = maxfit
                flag = flag & (np.abs(self.cat['w4_nanomaggies']*np.sqrt(self.cat['w4_nanomaggies_ivar'])) > snr_cut)
            elif i == 1:
                minfit = minfit
                maxfit = maxfit
                flag = flag & (np.abs(self.cat['SERSIC_ABSMAG'][:,1]*np.sqrt(self.cat['SERSIC_AMIVAR'][:,1])) > snr_cut) 
            elif i == 2:
                minfit = minfit
                maxfit = maxfit
                flag = flag & (np.abs(self.cat['w4_nanomaggies']*np.sqrt(self.cat['w4_nanomaggies_ivar'])) > snr_cut) & \
                    (np.abs(self.cat['SERSIC_ABSMAG'][:,1]*np.sqrt(self.cat['SERSIC_AMIVAR'][:,1])) > snr_cut) 
            flag2 = flag & (x > minfit) & (x < maxfit) & (y > minfit) & (y < maxfit)




            popt,pcov = curve_fit(func,x[flag2],y[flag2],p0=p0)#,method='dogbox')
            #s = 'fit: a=1,b=%.2f'%tuple(popt)
            
            if len(popt) == 1:
                s = 'y=x+%.2f'%tuple(popt)
            elif len(popt) == 2:
                s = 'a=%.2f,b=%.2f'%tuple(popt)
            elif len(popt) == 3:
                s = 'a=%.2f,b=%.2f,c=%.2f'%tuple(popt)
            else:
                s = 'oops'
            #plt.plot(xl,func(xl,*popt),'w-',label='_nolegend_',color='w',lw=3)            
            plt.plot(xl,func(xl,*popt),'r--',label=s,color=colorblind1,lw=3)
                
            #plt.plot(xl,func(xl,*popt),'r-',label=s,color=colorblind1,lw=3)
            #c = np.polyfit(x[flag],self.cat[ycols[i]][flag],1)            
            #plt.plot(xl,np.polyval(c,xl),'r-',label='ZP offset')
            #s = '{:.2f},{:.2f}'.format(c[0],c[1])
            #plt.text(2,-2,s,horizontalalignment='right')
            plt.axis([-2.5,2.5,-2.5,2.5])
            plt.legend(loc='lower right',fontsize=10)
            plt.ylabel(ylabels[0])
            plt.subplot(2,3,i+1+3)
            residual = y - func(x,*popt)
            plt.ylabel('residuals')
            plt.hexbin(x[flag],residual[flag],cmap='gray_r',vmin=0,vmax=hexbinmax,extent=(xmin,xmax,-2,2),gridsize=75)
            plt.text(-2,-2,'$\sigma = {:.2f}$'.format(np.std(residual[flag2])))
            plt.axhline(y=0,c=colorblind2,lw=3)
            plt.xlabel(xlabels[i])
            
        plt.savefig(homedir+'/research/APPSS/plots/GSWLC-SFR-fit.pdf')
        plt.savefig(homedir+'/research/APPSS/plots/GSWLC-SFR-fit.png')
        plt.savefig(homedir+'/research/APPSS/plots/fig4.pdf')        
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
    table_path = tabledir#homedir+'/github/appss/tables/'
    a100 = table_path+'a100-sdss.fits'
    a100 = table_path+'a100-sdss-wise.fits'
    a100nsa = table_path+'a100-nsa.fits'
    a100gsw = table_path+'a100-gswlcA2.fits'
    a100s4g = table_path+'a100-s4g.fits'
    allcats = table_path+'a100-sdss-wise-nsa-gswlcA2.fits'
    p = matchedcats(a100sdss=a100, a100nsa=a100nsa,a100gsw=a100gsw,a100s4g=a100s4g,allcats=allcats)

    fp = calibsfr()
