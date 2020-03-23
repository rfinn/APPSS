#!/usr/bin/python

import numpy as np
import os
import sys

from astropy.io import fits, ascii
from astropy.table import Table, join, hstack, Column, MaskedColumn
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.visualization import simple_norm

from astroquery.skyview import SkyView

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from urllib.parse import urlencode
from urllib.request import urlretrieve

import pandas as pd

homedir = os.getenv('HOME')
sys.path.append(homedir+'/github/Virgo/programs/')
from mksupersample import getlegacy, getlegacyimages

## LEGACY SURVEY
legacy_pixel_scale = 0.262 # arcsec/pixel, don't actually use this
image_size = 60 # image size to download from legacy survey, in pixels
default_image_size = image_size


class duplicates():
    def __init__(self):
        # read in a100-sdss-wise table
        self.cat = fits.getdata(homedir+'/research/APPSS/tables/a100-sdss-wise.fits')

        # identify a100 galaxies with multiple matches

        unique, arindex, counts = np.unique(self.cat['AGC'],return_counts=True, return_index=True)
        print('number of AGC with multiple unWISE matches = ',sum(counts > 1))
        #print('')
        #print('AGC number of sources with multiple unWISE matches = ')
        #print(unique[counts > 1])
        self.doubles = unique[counts > 1]
        self.sep_unwise = np.sqrt((self.cat['ra_2']-self.cat['RAdeg_Use'])**2 + (self.cat['dec_2']-self.cat['DECdeg_Use'])**2)
    def addgals(self,w,ra1,dec1,ra2,dec2,jpegflag=True):
        c1 = SkyCoord(ra1*u.deg,dec1*u.deg,frame='icrs')
        c2 = SkyCoord(ra2*u.deg,dec2*u.deg,frame='icrs')
        cats = [c1,c2]
        symbols=['co','b*','r+']
        edgecolors = ['c','w','r']
        symbols=['co','r^','yD','gs']
        edgecolors = ['c','b','r','xkcd:goldenrod', 'g']
        edgecolors = ['c','r','y', 'g']
        facecolors = ['None','None','None','None','None']
        sizes = [14,14,14,16,18]
        text_offsets = [(10,14),(10,7),(10,0),(10,-7),(10,-14)]
        
        for i,c in enumerate(cats):
            px,py = w.wcs_world2pix(c.ra.deg,c.dec.deg,1)
            galnumber = np.arange(len(c.ra.deg))
            #print('number of galaxies in catalog = ',len(c.ra.deg))
            # only keep objects on image
            keepflag = (px > 0) & (py > 0) & (px < image_size) & (py < image_size)
            if jpegflag:
                plt.plot(px[keepflag],image_size - py[keepflag],symbols[i],mec=edgecolors[i],mfc=facecolors[i],markersize=sizes[i])
            else:
                plt.plot(px[keepflag],py[keepflag],symbols[i],mec=edgecolors[i],mfc=facecolors[i],markersize=sizes[i])
            # label points
            #print('number of galaxies in FOV = ',sum(keepflag))

    def densearray(self,outfile_string='test',agcflag=False,onlyflag=True,startindex=None,endindex=None):
        plt.figure(figsize=(12,7))
        plt.subplots_adjust(bottom=.05,left=.05,top=.9,right=.95,hspace=.01,wspace=.01)

        nsubplot = 1
        nrow=5
        ncol=10
        if endindex is not None:
            maxcount = endindex-startindex+1
        else:
            maxcount = nrow*ncol+1
        if startindex is not None:
            i = startindex
        else:
            i = 0
        while nsubplot < maxcount:
            jpgflag=True
            #print(i,nsubplot,maxcount)
            plt.subplot(nrow,ncol,nsubplot)
            #print('flag index = ',i)
            #try:
            massflag=False
            # get ra and dec
            dindex = np.arange(len(self.cat))[self.cat['AGC'] == self.doubles[i]]
            ra1 = self.cat['RAdeg_Use'][dindex]
            dec1 = self.cat['DECdeg_Use'][dindex]
            ra2 = self.cat['ra_2'][dindex]
            dec2 = self.cat['dec_2'][dindex]

            w = getlegacy(ra1[0], dec1[0],jpeg=jpgflag,imsize=image_size)

            if w is None:
                jpgflag=False
                print('trouble in paradise',i)
                print('maybe coords are outside Legacy Survey?')
                print(ra1[0],dec1[0])
                # try to get 2MASS J image
                # check to see if 2MASS image exists
                gra = '%.5f'%(ra1[0]) # accuracy is of order .1"
                gdec = '%.5f'%(dec1[0])
                galpos = gra+'-'+gdec
                rootname = 'cutouts/DSS2-'+str(galpos)+'-'+str(image_size)+'-1arcsecpix'     
                
                fits_name = rootname+'.fits'
                if not(os.path.exists(fits_name)):
                    print('downloading DSS2 Image ')                    
                    #
                    c = SkyCoord(ra=ra1[0]*u.deg,dec=dec1[0]*u.deg)
                    x = SkyView.get_images(position=c,survey=['DSS2 Red'],pixels=[60,60])
                    # save fits image
                    fits.writeto(fits_name, x[0][0].data, header=x[0][0].header)
                else:
                    print('using 2mass image ',fits_name)
                im, h = fits.getdata(fits_name,header=True)
                w = WCS(h)
                norm = simple_norm(im,stretch='asinh',percent=99.5)
                plt.imshow(im,origin='upper',cmap='gray_r', norm=norm)
                # pixel scale is 1 arcsec
                # therefore, to show a 60x60 arcsec image, want to set boundary to center-30:center+30
                im_nrow,im_ncol=im.shape
            
                massflag=True

            if massflag:
                text_color='k'
            else:
                text_color='0.7'
            plt.text(.05,.85,'AGC '+str(self.doubles[i]),fontsize=8,c=text_color, transform=plt.gca().transAxes)
            # remove ticks for internal images
            #print(nsubplot,np.mod(nsubplot,ncol))
            # adjust ticksize of outer left and bottom images
            if massflag:
                plt.axis([int(im_nrow/2-image_size/2),int(im_nrow/2+image_size/2),int(im_ncol/2-image_size/2),int(im_ncol/2+image_size/2)])
            else:
                plt.xticks(np.arange(0,image_size,20),fontsize=8)
                plt.yticks(np.arange(0,image_size,20),fontsize=8)

                    #plt.axis([20,80,20,80])
            if (nsubplot < (nrow-1)*(ncol)):
                plt.xticks([],[])
            if (np.mod(nsubplot,ncol) > 1) | (np.mod(nsubplot,ncol) == 0) :
                #print('no y labels')
                plt.yticks([],[])

            print('jpegflag = ',jpgflag)
            self.addgals(w,ra1,dec1,ra2,dec2,jpegflag=jpgflag)
            
            i = i + 1
            nsubplot += 1

    def plot_all(self,startgal=None):
        plt.close('all')
        flag = np.ones_like(self.doubles, dtype='bool')
        #print('LENGTH OF GALIDS IN FOV = ',len(self.galids_in_fov))
        #self.plotimages(flag,outfile_string='All Galaxies',agcflag=False,onlyflag=True)
        ngal = len(self.doubles)
        ngalperplot = 50
        nplots = np.floor(ngal/ngalperplot)
        #galids_in_fov = []
        if (ngal/ngalperplot - nplots) > 0:
            nplots += 1
        nplots = int(nplots)
        endindex = None
        if startgal is None:
            allplots = [i for i in range(nplots)]
        else:
            first_plot = int(np.floor(startgal/ngalperplot))
            allplots = [i for i in range(first_plot,nplots)]
        for i in allplots:
        #for i in range(1):
            plt.close('all')
            startindex = i*ngalperplot
            s1 = '%04d'%(startindex)
            n2 = startindex+49
            if n2 > (ngal-1):
                n2 = ngal-1
                endindex=n2
                print('MAKING LAST PLOT')
            s2 = '%04d'%(n2)
            print(s1,s2)

            self.densearray(outfile_string='All-Galaxies',agcflag=False,onlyflag=True,startindex = startindex, endindex=endindex)

            plt.savefig('plots/gcutouts-'+s1+'-'+s2+'.pdf')
            plt.savefig('plots/gcutouts-'+s1+'-'+s2+'.png')
    def plotpositions(self):
        # get list of RA and DECs
        ra1 = []
        dec1 = []
        ra2 = []
        dec2 = []
        for i in range(len(self.doubles)):
            dindex = np.arange(len(self.cat))[self.cat['AGC'] == self.doubles[i]]
            ra1 = ra1+self.cat['RAdeg_Use'][dindex].tolist()
            dec1 = dec1+self.cat['DECdeg_Use'][dindex].tolist()
            ra2 = ra2+self.cat['ra_2'][dindex].tolist()
            dec2 = dec2+self.cat['dec_2'][dindex].tolist()
        print(ra1)
        ra1 = np.array(ra1)
        dec1 = np.array(dec1)
        ra2 = np.array(ra2)
        dec2 = np.array(dec2)
        plt.figure()
        plt.plot(ra1,dec1,'bs',label='AGC')
        plt.plot(ra2,dec2,'ro',markersize=10,mfc='none',label='unWISE')
        plt.legend()
        plt.xlabel('RA (deg)')
        plt.ylabel('DEC (deg)')        
        plt.savefig('plots/positions.pdf')
                
    def plotone(self,i,dssflag=False,imsize=None,plotsingle=True):
        ra1 = self.cat['RAdeg_Use']
        dec1 = self.cat['DECdeg_Use']
        ra2 = self.cat['ra_2']
        dec2 = self.cat['dec_2']
        
        if plotsingle:
            plt.figure(figsize=(4,4))
        flag = np.ones_like(ra1, dtype='bool')
        agcflag=False
        onlyflag=False
        ra = ra1[i]
        dec = dec1[i]
        if dssflag:
            w = None
            jpegflag = False
        else:
            w = getlegacy(ra,dec,agcflag=agcflag,onlyflag=onlyflag,imsize=imsize)
            jpegflag = True
        if w is None:
            jpegflag = False
            if imsize is not None:
                image_size=imsize
            else:
                image_size = default_image_size
            print('trouble in paradise',i)
            print('maybe coords are outside Legacy Survey?')
            print(ra,dec)
            # try to get 2MASS J image
            # check to see if 2MASS image exists
            gra = '%.5f'%(ra) # accuracy is of order .1"
            gdec = '%.5f'%(dec)
            galpos = gra+'-'+gdec
            rootname = 'cutouts/DSS2-'+str(galpos)+'-'+str(image_size)+'-1arcsecpix'     

            fits_name = rootname+'.fits'
            if not(os.path.exists(fits_name)):
                #print('downloading 2MASS J image ')
                print('downloading DSS2 Image ')                    
                #
                c = SkyCoord(ra=ra*u.deg,dec=dec*u.deg)
                x = SkyView.get_images(position=c,survey=['DSS2 Red'],pixels=[60,60])
                # save fits image
                fits.writeto(fits_name, x[0][0].data, header=x[0][0].header)
            else:
                print('using DSS2 image ',fits_name)
            im, h = fits.getdata(fits_name,header=True)
            w = WCS(h)
            norm = simple_norm(im,stretch='asinh',percent=99.5)
            plt.imshow(im,origin='upper',cmap='gray_r', norm=norm)

        self.addgals(w,ra1,dec1,ra2,dec2,jpegflag=jpegflag)
        text_color='0.7'
        plt.text(.05,.85,'Gal '+str(i),fontsize=8,c=text_color, transform=plt.gca().transAxes)
            # remove ticks for internal images
            #print(nsubplot,np.mod(nsubplot,ncol))
            # adjust ticksize of outer left and bottom images

    def plot_widesep(self):
        flag = self.sep_unwise > 1 # galaxies with > 1 arcsec separation
        widesep = np.arange(len(self.cat))[flag]
        for i in widesep:
            self.plotone(i)
if __name__ == '__main__':
    d = duplicates()
    
