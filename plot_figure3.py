'''
NS: Plot:
        -  bathymetry of reference 
        -  smoothed bathymetry 
        - difference between the reference and the smoothed bathymetry
        - histograms of the slopes on the domain for the reference and the smoothed bathymetry
    Both configurations used combination of advective scheme RSUP5 and 200 sigma-levels.
    
'''

import matplotlib
matplotlib.use('Agg') 
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors   as colors
import matplotlib.ticker   as ticker
from  matplotlib.cm import ScalarMappable
from  matplotlib.colors import ListedColormap, BoundaryNorm
from netCDF4 import Dataset
import sys
sys.path.append('/home2/datahome/nschifan/Python_Modules_p3/')
import R_tools as tools
import R_tools_fort as toolsF
import gsw as gsw
import obsfit1d as fit
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
from croco_simulations_jonathan_hist import Croco
matplotlib.rcParams.update({'font.size': 28})

# ------------ parameters ------------ 
name_pathdata_jon = ['RREXNUM200_RSUP5_NOFILT_T','RREXNUMS200_RSUP5_NOFILT_T']
name_nc_jon       = ['rrexnum200-rsup5-nofilt','rrexnums200-rsup5-nofilt']
title_name        = ['exp200-rsup5','exp200-rsup5-smooth']
name_exp_grd_jon  = ['rrex200-up5','rrex200-up5']
nbr_levels_jon    = ['200','200']
name_exp_jon      = ['rrexnum200','rrexnum200']
time              = ['40']   # --> Outputs from CROCO to read
ndfiles           = 1        # read only the first time-step 
nt = len(time)*ndfiles       # total number of time-steps considered, here 1
var_list          = ['zeta'] # variables to read from CROCO

# --- plot options ---
fs         = 28
lw_c       = 0.5
rho0       = 1027.4
jsec       = 250
extent     = [-37.5,-21.2,53,62.5]
lon_0,lat_0= -32,57.5 # centre of the map 
# ------------ bin for slope ------
minc      = 1e-4
maxc      = 0.2
nce       = 1e-4      
nbin      = 50         
h_e    = np.arange(minc,maxc,nce) 
h_c    = 0.5*(h_e[1:]+h_e[:-1])   # bin center  
# ------------ norm for bathymetry (smoothed and reference) -------------
cmap_h         = plt.cm.jet_r 
norm_h         = colors.Normalize(vmin=0,vmax=4000)
levels_h       = np.arange(0,5000,200)
cbticks_h      = [1000,2000,3000,4000]  
cblabel_h      = 'h [m]'
cblabel_hs     = r'$h_S$ [m]'
# ------------ norm difference of bathymetries ----
cmap_hdiff    = plt.cm.RdBu_r
norm_hdiff    = colors.Normalize(vmin=-150,vmax=150)
levels_hdiff  = np.arange(-150,150,5)
cbticks_hdiff = [-150,0,150] 
cblabel_hdiff = r'h-$h_S$ [m]'

# --- read variables ---
for exp in range(len(name_exp_jon)):
    name_exp      = name_exp_jon[exp]                  # name file netcdf
    name_pathdata = name_pathdata_jon[exp]             # folder where are netcdf
    name_exp_grd  = name_exp_grd_jon[exp]              # folder where grid data
    name_nc       = name_nc_jon[exp]                   # name of output netcdf
    nbr_levels    = nbr_levels_jon[exp]
    data = Croco(name_exp,nbr_levels,time[0],name_exp_grd,name_pathdata)
    data.get_grid()
    data.get_outputs(0,var_list,get_date=False)
    [z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
    # --- compute slope gradient ---
    dhdx = tools.u2rho((data.h[1:,:]-data.h[:-1,:]))*data.pm
    dhdy = tools.v2rho((data.h[:,1:]-data.h[:,:-1]))*data.pn 
    gradh        = np.ravel(np.sqrt(dhdx**2 + dhdy**2))
    if exp==0:
        print('--- compute slopes for referenced bathymetry ---')
        print(np.nanmax(gradh))
        h            = data.h
        gradh200_bin = stats.binned_statistic(gradh,gradh,statistic='count',bins=h_e)[0] 
    else:
        print('--- compute slopes for smoothed bathymetry ---')
        print(np.nanmax(gradh))
        hs            = data.h
        gradh200s_bin = stats.binned_statistic(gradh,gradh,statistic='count',bins=h_e)[0]       

# --- make plot ---
line   = 0
column = 0
figure = plt.figure(figsize=(20,20))
gs     = gridspec.GridSpec(4,2,height_ratios=[1,0.05,1,0.05],wspace=0.3,hspace=0.6)

ax = plt.subplot(gs[0,0]) # ---> Horizontal map of reference bathymetry
plt.gca().set_aspect('equal', adjustable='box')
ax.set_title('a)',fontsize=fs)
ctf1   = ax.contourf(h.T,levels=levels_h,cmap=cmap_h,norm=norm_h,extend='max')
ct     = ax.contour(h.T,levels=levels_h,colors='k',linewidths=lw_c)
ax.set_xticks([0,250,500,750,1000],['0','200','400','600','800'])
ax.set_yticks([0,250,500,750],['0','200','400','600'])
plt.xlabel(r'km in $\xi$-direction',fontsize=fs)
plt.ylabel(r'km in $\eta$-direction',fontsize=fs)
ax.axhline(y=250,c='k',lw=2.5,linestyle='dashed')


ax = plt.subplot(gs[1,0]) # ---> colorbar of referenced bathymetry
cb     = plt.colorbar(ctf1,ax,orientation='horizontal',ticks=[1000,2000,3000,4000])
cb.set_label(cblabel_h,fontsize=fs,labelpad=10) 
cb.ax.tick_params(labelsize=fs-2)


ax = plt.subplot(gs[0,1]) # ---> Horizontal map of smoothed bathymetry
plt.gca().set_aspect('equal', adjustable='box')
ax.set_title('b)',fontsize=fs)
ctf2   = ax.contourf(hs.T,levels=levels_h,cmap=cmap_h,norm=norm_h,extend='max')
ct     = ax.contour(hs.T,levels=levels_h,colors='k',linewidths=lw_c)
ax.set_xticks([0,250,500,750,1000],['0','200','400','600','800'])
ax.set_yticks([0,250,500,750],['0','200','400','600'])
plt.xlabel(r'km in $\xi$-direction',fontsize=fs)
plt.ylabel(r'km in $\eta$-direction',fontsize=fs)
ax.axhline(y=250,c='k',lw=2.5,linestyle='dashed')

ax = plt.subplot(gs[1,1]) # ---> colorbar of smoothed bathymetry
cb     = plt.colorbar(ctf2,ax,orientation='horizontal',ticks=[1000,2000,3000,4000])
cb.set_label(cblabel_hs,fontsize=fs,labelpad=10) 
cb.ax.tick_params(labelsize=fs-2)


ax = plt.subplot(gs[2,0]) # ---> Difference between bathymetry of reference and smoothed (h - hs)
plt.gca().set_aspect('equal', adjustable='box')
ax.set_title('c)',fontsize=fs)
ctf3   = ax.contourf((h-hs).T,levels=levels_hdiff,cmap=cmap_hdiff,norm=norm_hdiff,extend='both')
ct     = ax.contour(h.T,levels=levels_h,colors='k',linewidths=lw_c)
ax.set_xticks([0,250,500,750,1000],['0','200','400','600','800'])
ax.set_yticks([0,250,500,750],['0','200','400','600'])
plt.xlabel(r'km in $\xi$-direction',fontsize=fs)
plt.ylabel(r'km in $\eta$-direction',fontsize=fs)

ax = plt.subplot(gs[3,0]) # ---> colorbar  h - hs
cb     = plt.colorbar(ctf3,ax,orientation='horizontal',ticks=cbticks_hdiff)
cb.set_label(cblabel_hdiff,fontsize=fs,labelpad=10) #-60
cb.ax.tick_params(labelsize=fs-2)


ax = plt.subplot(gs[2:,1]) # ---> histogram of slopes for bathymetry of reference and smoothed
ax.set_title('d)',fontsize=fs)
plt.bar(h_c,gradh200_bin,width=nce,color='k',edgecolor='k',label=title_name[0]) #, label='tracer 1')
plt.bar(h_c,gradh200s_bin,width=nce,color='r',edgecolor='r',label=title_name[1],alpha=0.4)
plt.ylabel('Occurence',fontsize=fs)
plt.xlabel('Topographic slope [%]',fontsize=fs)
plt.yscale('log')
plt.xlim(0,0.22)
plt.xticks([0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20],['0','2','4','6','8','10','12','14','16','18','20'])
ax.tick_params(labelsize=fs)
plt.legend(fontsize=fs)


#ax = plt.subplot(gs[4,0]) # ---> vertical transect of bathymetry of reference 
#plt.title('e)',fontsize=fs)
#lonsec = 0.5*(data.lonr[1:,jsec]+data.lonr[:-1,jsec])
#lonsec = np.tile(lonsec,(z_w.shape[-1],1)).T
#zsec = 0.5*(z_w[1:,jsec,:]+z_w[:-1,jsec,:])
#plt.fill_between(data.lonr[:,jsec],-3100,-h[:,jsec],fc='lightgray',ec='k',alpha=0.5)
#plt.ylim(-3100,0)
#plt.xlim(lonsec[50,0],lonsec[750,0])
#ax.set_xticks([-35,-30],['35','30'])
#plt.ylabel('z [m]',fontsize=fs)
#plt.xlabel('Longitude [$^{\circ}$W]',fontsize=fs)
#ax.tick_params(labelsize=fs)


#ax = plt.subplot(gs[4,1]) # ---> vertical transect of smoothed bathymetry  
#plt.title('f)',fontsize=fs)
#lonsec = 0.5*(data.lonr[1:,jsec]+data.lonr[:-1,jsec])
#lonsec = np.tile(lonsec,(z_w.shape[-1],1)).T
#zsec = 0.5*(z_w[1:,jsec,:]+z_w[:-1,jsec,:])
#plt.fill_between(data.lonr[:,jsec],-3100,-hs[:,jsec],fc='lightgray',ec='k',alpha=0.5)
#plt.ylim(-3100,0)
#plt.xlim(lonsec[50,0],lonsec[750,0])
#ax.set_yticks([])
#ax.set_xticks([-35,-30],['35','30'])
#plt.xlabel('Longitude [$^{\circ}$W]',fontsize=fs)
#ax.tick_params(labelsize=fs)


plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/All_simu/bathy_and_slope_Jon_exp200_s200.png',bbox_inches='tight')
plt.close()










