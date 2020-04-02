#!/usr/bin/env/ python

import numpy as np
from matplotlib import pyplot as plt
#import warnings
#warnings.filterwarnings('ignore')
from astropy.io import fits, ascii
from astropy.coordinates import Angle
import astropy.units as u


nsa = fits.getdata('/home/rfinn/research/NSA/nsa_v1_0_1.fits')
gsw = ascii.read('/home/rfinn/research/GSWLC/GSWLC-A2.dat')
s4g = ascii.read('/home/rfinn/research/APPSS/tables/spitzer.s4gcat_5173.tbl')
a100 = fits.getdata('/home/rfinn/research/APPSS/tables/a100-sdss.fits')

# keep cz < 15000 km/s
nsa = nsa[nsa['Z']*3.e5 < 15000]
gsw = gsw[gsw['Z']*3.e5 < 15000]
s4g = s4g[s4g['vopt'] < 15000]


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)#, projection='lambert')

all_ra = [nsa['RA'],gsw['RA'],a100['RAdeg_Use'],s4g['ra']]
all_dec = [nsa['DEC'],gsw['DEC'],a100['DECdeg_Use'],s4g['dec']]
labels = ['NSA','GSWLC','A100','S4G']
myorder = [0,1,3]
#myorder = [1]
mycolors = ['k','0.7','b','r']
for i in myorder:
    if i == 3:
        # s4g angles are already in degrees
        ra = Angle(all_ra[i])
        #ra = ra.wrap_at(180*u.degree)
        dec = Angle(all_dec[i])
    else:
        ra = Angle(all_ra[i]*u.degree)
        #ra = ra.wrap_at(180*u.degree)
        dec = Angle(all_dec[i]*u.degree)  
    if i == 3:
        ax.scatter(ra.degree, dec.degree,label=labels[i],s=5,c=mycolors[i])
    else:
        flag = np.random.randint(0,len(ra),int(len(ra)/2))
        ax.scatter(ra.degree[flag], dec.degree[flag],label=labels[i],alpha=.6,s=3,c=mycolors[i])
# A100 polygons
a100_vert_spring = np.array([[110,0],
                              [113,18],
                              [128,18],
                              [129,20],
                              [137,20.1],
                              [137, 23.7],
                              [127, 23.70],
                              [110, 23.8],
                              [111.4,32.4],
                              [138, 32],
                              [139,36.3],
                              [233, 36.3],
                              [233, 32],
                              [248, 32],
                              [250, 24],
                              [233, 24],
                              [233, 18.2],
                              [249, 18.2],
                              [249, -.5],
                              [110,0]])
a100_vert_fall_1 = np.array([[0,-.2],
                             [0,36.3],
                             [48, 36.3],
                             [49, 13.6],
                             [38.7, 13.4],
                             [39, 10.2],
                             [47.3,10.2],
                             [48, -.2],
                             [0,-.2]
                         ])
a100_vert_fall_2 = np.array([[326,-.2],
                         [326,36.3],
                          [360, 36.3],
                          [360, -.2],
                          [327,-.2]
                         ])
allvert = [a100_vert_spring,a100_vert_fall_1,a100_vert_fall_2]
for i in range(len(allvert)):
    if i == 0:
        mylabel='A100'
    else:
        mylabel='_nolegend_'
    plt.plot(allvert[i][:,0],allvert[i][:,1],'b-',lw=2.5,label=mylabel)
    pass

comparison_vert = np.array([[140,0],
                         [140,36.],
                          [230, 36],
                          [230, 0],
                          [140,0]
                         ])
xcomp = np.array([140,230])
ycomp1 = np.array([0,0])
ycomp2 = np.array([36.,36.])
plt.fill_between(xcomp,ycomp2,y2=ycomp1,facecolor="none",hatch='X',edgecolor='b',alpha=1,label='Overlap')
#plt.plot(comparison_vert[:,0],comparison_vert[:.1],'c-')
# plot NSA
#ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
#plt.axis([-40.,40,0,45])
ax.grid(True)
plt.legend(markerscale=2,ncol=2,fontsize=12)
plt.xlabel('$RA \ (deg)$',fontsize=20)
plt.ylabel('$DEC \ (deg)$',fontsize=20)
plt.savefig('surveys-skyplot-1panel.pdf')
