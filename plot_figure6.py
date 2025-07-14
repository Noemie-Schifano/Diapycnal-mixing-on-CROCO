'''
NS : make comparison between diffusivities from in-situ measurements and CROCO
     In-situ measurements are :
               - RREX15 and RREX17 microstructure 
               - OVIDE microstructure 
     CROCO, open a NetCDF (extract ) with:
               - Parameterized mixing K_KPP (named "AKt" in CROCO outputs) 
               - Effective mixing parameterized online K_eff 
     NB: extraction from CROCO is made at an interpollated section (see blue line on Fig. 1 in article)
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
from netCDF4 import Dataset
import matplotlib.gridspec as gridspec
import scipy.interpolate as itp 
import scipy.io as sio 
import scipy.stats as stats
from rrex_cv import RREX_class

# ------ parameters ------ 
path_data    = '/home/datawork-lops-osi/cvic/Data_obs'
file_kunze   = path_data+'strainfineoutall.nc' 
file_ovide   = path_data+'dissMean_Ovide08_001.mat'
name_exp     = 'rrexnum200-rsup5'
file_croco   = '/home/datawork-lops-rrex/nschifan/DIAGS/'+name_exp+'_Keff_AKt_rrex-middleline.nc'
rho0  = 1025.
dmin  = -500 # discard profiles where h<dmin -- continental shelf

# - cross-ridge stations from west to east - 
list_vmp_ovid08 = [24,23,22,21,20,19,18,17,16]#,13,12,11]
list_vmp_rrex15 = [14,39,40,41,38,15]
list_vmp_rrex17 = [17,18,19,20,22,16,23,24,13,12,11,10,9,8]
list_kunze      = [5208,5207,5206,5205,5204,5203,5202,5201,5343,350,351,352,5233,5398,5399,5160,5159]


# --- plot options ---
plot_map_stations  = False # station numbers  
plot_map_epsz      = False # map with scattered vertically-integrated dissipation   
plot_section_eps   = False # cross-ridge section of dissipation from rrex,ovide,kunze 
plot_section_kappa = True  # cross-ridge section of diffusivity from rrex,ovide,kunze 
fs                 = 28
lw                 = 1.5 
ms                 = 12
cf                 = colors.to_rgba('teal')
cmap               = plt.cm.Reds
cmapr              = cmap.reversed()
colors_point       = [cmapr(i) for i in np.linspace(0,1,3)]
lw_c               = 0.4  # coastline linewidth 
region             = 'RREX'
if region == 'RREX':
    # - inset - 
    lon_0i,lat_0i = -32,57.5

# - bathymetry - 
cmap_h    = plt.cm.gray_r
norm_h         = colors.Normalize(vmin=0,vmax=4000)
levels_h  = np.arange(0,5000,500)
cbticks_h = [1000,2000,3000,4000] 

# - diffusivity -
cmap_kappa     = plt.cm.CMRmap_r 
pmin,pmax,pint = -6,-4,0.2 # power min,max,interval
norm_kappa = colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),ncolors=cmap_kappa.N,clip=True)
cbticks_kappa = np.logspace(pmin,pmax,pmax-pmin+1)
logfmt  = ticker.LogFormatterMathtext(10,labelOnlyBase=False) # nice writting in colorbar

# --> depth binning for depth 
dz    = 100.                              # [m] 
zbine = np.arange(0,-4000-dz,-dz)[::-1]   # bin edges (shallowest bin edge is the surface)  
zbinc = 0.5*(zbine[1:]+zbine[:-1])        # bin centres 
nz    = zbinc.shape[0]


def closest(lst, k):     
     # find in the list lst the indx of the closest value of K 
     return int(min(range(len(lst)), key = lambda i: abs(lst[i]-k)))


# ------ read data ------ 
print(' ... read data ... ')

############### RREX15 ###################################################
rrex15 = RREX_class(year='2015',list_vmp=list_vmp_rrex15)
rrex15.get_bathy(resolution='lr')
rrex15.interpolate_bathy()
rrex15.get_dist() 
rrex15.get_dissipation(zbine)
rrex15.get_ctd_bvf()
rrex15.N2[rrex15.N2<0] = np.nan
rrex15.get_diffusivity() 

############### RREX17 ###################################################
rrex17 = RREX_class(year='2017',list_vmp=list_vmp_rrex17)
rrex17.get_bathy(resolution='lr')
rrex17.interpolate_bathy()
rrex17.get_dist() 
rrex17.get_dissipation(zbine)
rrex17.get_ctd_bvf()
rrex17.N2[rrex17.N2<0] = np.nan
rrex17.get_diffusivity() 

############### OVIDE ####################################################
ovid08 = RREX_class(year='2008',list_vmp=list_vmp_ovid08)
ovid08.get_bathy(resolution='lr')
ovid08.interpolate_bathy()
ovid08.get_dist() 
ovid08.get_dissipation(zbine)
ovid08.get_ctd_bvf()
ovid08.N2[ovid08.N2<0] = np.nan
ovid08.get_diffusivity() 

############### CROCO ####################################################
croco1           = RREX_class(year='2015',list_vmp=[0]) 
nc               = Dataset(file_croco,'r') 
croco1.AKt       = nc.variables['AKt_z'][:] 
croco1.Keff      = nc.variables['Keff_z'][:]
croco1.z         = nc.variables['z'][:] 
croco1.AKt_hab   = nc.variables['AKt_hab'][:] 
croco1.Keff_hab  = nc.variables['Keff_hab'][:]
croco1.hab       = nc.variables['hab'][:] 
croco1.h         = -nc.variables['h'][:] 
croco1.lonvmp    = nc.variables['lon'][:] 
croco1.lonctd    = nc.variables['lon'][:] 
croco1.latvmp    = nc.variables['lat'][:] 
croco1.latctd    = nc.variables['lat'][:] 
nc.close() 
croco1.nvmp              = croco1.latvmp.shape[0] 
croco1.nctd              = croco1.latvmp.shape[0] 
croco1.hvmp              = np.copy(croco1.h) 
zindmin                  = np.nanargmin(abs(croco1.z+30)) 
croco1.AKt[:,:,zindmin:] = np.nan # sort of MLD 
croco1.get_dist()

# --> replacing zero values by nan 
AKti            = np.copy(croco1.AKt)
Keffi           = np.copy(croco1.Keff)
AKti[AKti==0]   = np.nan
Keffi[Keffi==0] = np.nan

# --> bin CROCO data as in-situ data 
croco1.kappa = np.zeros((croco1.nvmp,nz))
croco1.kappa_keff = np.zeros((croco1.nvmp,nz))
for i in range(croco1.nctd):
    croco1.kappa[i,:]      = stats.binned_statistic(croco1.z,np.nanmean(AKti[:,i,:],axis=0),
                                                             statistic=np.nanmean,bins=zbine)[0] 
    croco1.kappa_keff[i,:] = stats.binned_statistic(croco1.z,np.nanmean(Keffi[:,i,:],axis=0),
                                                              statistic=np.nanmean,bins=zbine)[0]
croco1.zc_eps   = zbinc 
croco1.ze_eps   = zbine
croco1.dz_eps   = dz   
croco1.eps      = np.copy(croco1.kappa)*np.nan 
croco1.eps_keff = np.copy(croco1.kappa_keff)*np.nan


############### COMMON DIAGNOSTICS #######################################
print(' ... perform some diagnostics to the different datasets ... ') 
# --> regrid in height-above-bottom (hab) coordinates 
habe                  = abs(zbine)[::-1]  # hab edges. '::-1' cause it needs to be monotically increasing   
habc                  = 0.5*(habe[1:]+habe[:-1]) # hab centres
rrex15.kappa_hab      = np.nan*np.copy(rrex15.kappa) 
rrex17.kappa_hab      = np.nan*np.copy(rrex17.kappa) 
ovid08.kappa_hab      = np.nan*np.copy(ovid08.kappa) 
croco1.kappa_hab      = np.nan*np.copy(croco1.kappa) 
croco1.kappa_keff_hab = np.nan*np.copy(croco1.kappa_keff)

for i in range(rrex15.nvmp):
    habi   = zbinc-rrex15.sfdvmp[i]
    kappai = rrex15.kappa[i,:] 
    rrex15.kappa_hab[i,:] = stats.binned_statistic(habi[~np.isnan(kappai)],kappai[~np.isnan(kappai)],
                            statistic=np.nanmean,bins=habe)[0]
for i in range(rrex17.nvmp):
    habi   = zbinc-rrex17.sfdvmp[i]
    kappai = rrex17.kappa[i,:] 
    rrex17.kappa_hab[i,:] = stats.binned_statistic(habi[~np.isnan(kappai)],kappai[~np.isnan(kappai)],
                            statistic=np.nanmean,bins=habe)[0]
for i in range(ovid08.nvmp):
    habi   = zbinc+ovid08.sfdctd[i] 
    kappai = ovid08.kappa[i,:] 
    ovid08.kappa_hab[i,:] = stats.binned_statistic(habi[~np.isnan(kappai)],kappai[~np.isnan(kappai)],
                            statistic=np.nanmean,bins=habe)[0]
for i in range(croco1.nvmp):
    habi   = zbinc-croco1.h[i] 
    kappai = croco1.kappa[i,:]
    croco1.kappa_hab[i,:] = stats.binned_statistic(habi[~np.isnan(kappai)],kappai[~np.isnan(kappai)],
                            statistic=np.nanmean,bins=habe)[0]

for i in range(croco1.nvmp):
    habi   = zbinc-croco1.h[i]
    kappai = croco1.kappa_keff[i,:]
    if len(kappai[~np.isnan(kappai)])>0:
        croco1.kappa_keff_hab[i,:] = stats.binned_statistic(habi[~np.isnan(kappai)],kappai[~np.isnan(kappai)],
                                statistic=np.nanmean,bins=habe)[0]
 
   
# --> compute 10th and 90th percentiles and median of the different diffusivities
rrex15.kappa_perc = np.nanpercentile(rrex15.kappa,[10,50,90],axis=0) 
rrex17.kappa_perc = np.nanpercentile(rrex17.kappa,[10,50,90],axis=0) 
ovid08.kappa_perc = np.nanpercentile(ovid08.kappa,[10,50,90],axis=0) 
croco1.kappa_perc = np.nanpercentile(croco1.kappa,[10,50,90],axis=0) 
croco1.kappa_keff_perc = np.nanpercentile(croco1.kappa_keff,[10,50,90],axis=0)

rrex15.kappa_hab_perc = np.nanpercentile(rrex15.kappa_hab,[10,50,90],axis=0) 
rrex17.kappa_hab_perc = np.nanpercentile(rrex17.kappa_hab,[10,50,90],axis=0) 
ovid08.kappa_hab_perc = np.nanpercentile(ovid08.kappa_hab,[10,50,90],axis=0) 
croco1.kappa_hab_perc = np.nanpercentile(croco1.kappa_hab,[10,50,90],axis=0) 
croco1.kappa_keff_hab_perc = np.nanpercentile(croco1.kappa_keff_hab,[10,50,90],axis=0)


############### MAKE PLOT ################################################
print(' ... make plot ... ')
title_name   = ['a)','b) RREX15','c) RREX17','d) OVIDE08','e) $\kappa_{KPP}$','f) $\kappa_{eff}$','g)','h)']
if plot_section_kappa: 
    # ---  plotting options --- 
    dkm       = 20 # column width in km for dissipation profiles and single columns of u from LADCP  
    kappa_ref = -6 # log of reference kappa (where bars start)
    fs_small  = 20
    xlim      = [-500,500] # [km] 
    ylim      = [-3500,0]  # [m]
    # --- centers the plot on the ridge top --- 
    middist = rrex15.hdistcum[np.nanargmax(rrex15.hitp)] 
    rrex15.hdistcum -= middist; rrex15.distcum -= middist
    middist = rrex17.hdistcum[np.nanargmax(rrex17.hitp)] 
    rrex17.hdistcum -= middist; rrex17.distcum -= middist
    middist = ovid08.hdistcum[np.nanargmax(ovid08.hitp)] 
    ovid08.hdistcum -= middist; ovid08.distcum -= middist
    middist = croco1.distcum[np.nanargmax(croco1.h)]  
    croco1.distcum -= middist 
    # ------------ make plot ------------  
    plt.figure(figsize=(20,20))                                                     
    gs = gridspec.GridSpec(7,2,height_ratios=[1,1,1,1,1,1,1],width_ratios=[3,1],wspace=0.1,hspace=0.5)
    ax = plt.subplot(gs[1,0]) # ----------------------------------- vertically averaged / median diffusivity   
    # - median - 
    plt.scatter(rrex15.distcum,np.nanmedian(rrex15.kappa,axis=1),s=ms+60,linewidth=0,marker='v',color='k',label='RREX15') 
    plt.scatter(rrex17.distcum,np.nanmedian(rrex17.kappa,axis=1),s=ms+60,linewidth=0,marker='^',color='k',label='RREX17') 
    plt.scatter(ovid08.distcum,np.nanmedian(ovid08.kappa,axis=1),s=ms+60,linewidth=0,marker='*',color='k',label='OVIDE08') 
    plt.plot(croco1.distcum,np.nanmedian(croco1.kappa,axis=1),color=cf,lw=2*lw,label='$\kappa_{KPP}$') 
    plt.plot(croco1.distcum,np.nanmedian(croco1.kappa_keff,axis=1),color='r',lw=2*lw,label='$\kappa_{eff}$')
    plt.legend(loc='upper center',bbox_to_anchor=(0.5,2.4),prop={'size':fs},framealpha=1,ncol=3,numpoints=1)
    ax.set_yscale('log') 
    ax.set_xlim(xlim)
    ax.set_ylim(1e-6,1e-2) 
    ax.set_yticks([1e-6,1e-5,1e-4,1e-3,1e-2])
    ax.get_yaxis().set_major_formatter(ticker.LogFormatterMathtext(10,labelOnlyBase=False))
    ax.set_xticklabels(())  
    plt.ylabel(r'[m$^2$ s$^{-1}$]',fontsize=fs)
    ax.tick_params(labelsize=fs) 
    ax.set_title(title_name[0],fontsize=fs)
 
    ax = plt.subplot(gs[2,0]) # ----------------------------------- x-z plot rrex15 
    plt.fill_between(rrex15.hdistcum,ylim[0],rrex15.hitp,color='lightgray',linewidth=0,zorder=3) 
    for i in range(rrex15.nvmp):
        rrex15.eps[i,:] = rrex15.kappa[i,:]
        rrex15.plot_fancy_dissipation(ax,i,dkm,eps_ref=kappa_ref,lw=lw,ls=fs_small) 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yticks([0,-1000,-2000,-3000])
    ax.set_xticklabels(())  
    ax.set_ylabel('z [m]',fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.set_title(title_name[1],fontsize=fs)

    ax = plt.subplot(gs[3,0]) # ----------------------------------- x-z plot rrex17 
    plt.fill_between(rrex17.hdistcum,ylim[0],rrex17.hitp,color='lightgray',linewidth=0,zorder=3) 
    for i in range(rrex17.nvmp): 
        rrex17.eps[i,:] = rrex17.kappa[i,:]   
        rrex17.plot_fancy_dissipation(ax,i,dkm,eps_ref=kappa_ref,lw=lw,ls=fs_small) 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticklabels(())  
    ax.set_yticks([0,-1000,-2000,-3000])
    ax.set_ylabel('z [m]',fontsize=fs)
    ax.tick_params(labelsize=fs) 
    ax.set_title(title_name[2],fontsize=fs)

    ax = plt.subplot(gs[4,0]) # ----------------------------------- x-z plot ovide  
    plt.fill_between(ovid08.hdistcum,ylim[0],ovid08.hitp,color='lightgray',linewidth=0,zorder=3) 
    for i in range(ovid08.nvmp): 
        ovid08.eps[i,:] = ovid08.kappa[i,:] 
        ovid08.plot_fancy_dissipation(ax,i,dkm,eps_ref=kappa_ref,lw=lw,ls=fs_small) 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticklabels(())  
    ax.set_yticks([0,-1000,-2000,-3000])
    ax.set_ylabel('z [m]',fontsize=fs)
    ax.tick_params(labelsize=fs)  
    ax.set_title(title_name[3],fontsize=fs)

    ax = plt.subplot(gs[5,0]) # ----------------------------------- x-z plot croco K_KPP
    plt.fill_between(croco1.distcum,ylim[0],croco1.h,color='lightgray',linewidth=0,zorder=3) 
    for i in np.arange(20,croco1.nvmp,40):
        croco1.eps[i,:] = croco1.kappa[i,:] 
        croco1.plot_fancy_dissipation(ax,i,dkm,eps_ref=kappa_ref,lw=lw,ls=fs_small) 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yticks([0,-1000,-2000,-3000])
    ax.set_ylabel('z [m]',fontsize=fs)
    ax.set_xticklabels(())
    ax.tick_params(labelsize=fs)
    ax.set_title(title_name[4],fontsize=fs)

    ax = plt.subplot(gs[6,0]) # ----------------------------------- x-z plot croco K_eff
    plt.fill_between(croco1.distcum,ylim[0],croco1.h,color='lightgray',linewidth=0,zorder=3)
    for i in np.arange(20,croco1.nvmp,40):
        print(i)
        croco1.eps_keff[i,:] = croco1.kappa_keff[i,:]
        croco1.eps_keff[croco1.eps_keff==np.nan]=0
        croco1.plot_fancy_dissipation_keff(ax,i,dkm,eps_ref=kappa_ref,lw=lw,ls=fs_small)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yticks([0,-1000,-2000,-3000])
    ax.set_ylabel('z [m]',fontsize=fs)
    ax.set_xlabel('Distance to the ridge top [km]',fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.set_title(title_name[5],fontsize=fs)


    ax = plt.subplot(gs[0:3,1]) # ----------------------------------- SURFACE mean/median vertical profiles
    # - median profiles - 
    plt.plot(rrex15.kappa_perc[1],zbinc,'k',marker='v',ms=ms,mew=0,lw=lw) 
    plt.plot(rrex17.kappa_perc[1],zbinc,'k',marker='^',ms=ms,mew=0,lw=lw) 
    plt.plot(ovid08.kappa_perc[1],zbinc,'k',marker='*',ms=ms,mew=0,lw=lw) 
    plt.plot(croco1.kappa_perc[1],zbinc,color=cf,marker='+',lw=2*lw) 
    plt.plot(croco1.kappa_keff_perc[1],zbinc,color='r',marker='+',lw=2*lw)
    plt.xscale('log')
    plt.ylim(-1000,0)
    plt.xlim(1e-6,1e-2)
    plt.ylabel('z [m]',fontsize=fs)
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    plt.xlabel(r'$\kappa$ [m$^2$ s$^{-1}$]',fontsize=fs)
    ax.tick_params(labelsize=fs)
    plt.title(title_name[6],fontsize=fs)
                    
    ax = plt.subplot(gs[4:,1]) # ----------------------------------- DEEP mean/median vertical profiles
    # - median profiles - 
    plt.plot(rrex15.kappa_hab_perc[1],habc,'k',marker='v',ms=ms,mew=0,lw=lw) 
    plt.plot(rrex17.kappa_hab_perc[1],habc,'k',marker='^',ms=ms,mew=0,lw=lw) 
    plt.plot(ovid08.kappa_hab_perc[1],habc,'k',marker='*',ms=ms,mew=0,lw=lw) 
    plt.plot(croco1.kappa_hab_perc[1],habc,color=cf,marker='+',lw=2*lw) 
    plt.plot(croco1.kappa_keff_hab_perc[1],habc,color='r',marker='+',lw=2*lw)
    plt.xscale('log')
    plt.ylim(0,1000)
    plt.xlim(1e-6,1e-2) 
    plt.ylabel('Height above bottom [m]',fontsize=fs)
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    plt.xlabel(r'$\kappa$ [m$^2$ s$^{-1}$]',fontsize=fs)
    ax.tick_params(labelsize=fs)
    plt.title(title_name[7],fontsize=fs)
    plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/cross-ridge_diffusivity_rrex_ovide_kunze_'+name_exp+'.pdf',bbox_inches='tight') 


    ### ------------- > plot of a map including: - CROCO grid and bathymetry;
    ###                                          - in-situ measurements location and colorbar of diffusivities;
    ###                                          - slice to compare CROCO with in-situ measurements;  
    fs = 14
    plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(2,3,height_ratios=[0.05,1])
    # ----------------
    ax = plt.subplot(gs[1,:],projection=ccrs.Orthographic(lon_0i,lat_0i))
    extent = [-39,-20,53,63]
    ax.set_extent(extent) 
    gl = ax.gridlines(draw_labels=True,dms=True, x_inline=False, y_inline=False, linewidth=0.5, color='whitesmoke', alpha=0.5, linestyle='--')
    xh,yh   = rrex15.lonh,rrex15.lath
    x15,y15 = rrex15.lonvmp,rrex15.latvmp
    x17,y17 = rrex17.lonvmp,rrex17.latvmp
    x08,y08 = ovid08.lonvmp,ovid08.latvmp
    xc1,yc1 = croco1.lonvmp,croco1.latvmp 
    # --- bathymetry
    ctf = ax.contourf(xh,yh,-rrex15.h,levels=levels_h,cmap=cmap_h,norm=norm_h,extend='max',transform=ccrs.PlateCarree())
    ct  = ax.contour(xh,yh,-rrex15.h,levels=levels_h,colors='k',linewidths=lw_c,transform=ccrs.PlateCarree())
    # --- diffusivityes
    sc  = ax.scatter(x15,y15,c=np.nanmean(rrex15.kappa,axis=1),marker='v',s=4*ms,linewidths=0,
                    norm=norm_kappa,cmap=cmap_kappa,zorder=4,transform=ccrs.PlateCarree())
    sc  = ax.scatter(x17,y17,c=np.nanmean(rrex17.kappa,axis=1),marker='^',s=4*ms,linewidths=0,
                    norm=norm_kappa,cmap=cmap_kappa,zorder=4,transform=ccrs.PlateCarree())
    sc  = ax.scatter(x08,y08,c=np.nanmean(ovid08.kappa,axis=1),marker='*',s=6*ms,linewidths=0,
                    norm=norm_kappa,cmap=cmap_kappa,zorder=4,transform=ccrs.PlateCarree())
    ax.plot(xc1,yc1,color=cf,lw=2*lw,zorder=3,transform=ccrs.PlateCarree())
    # --- plot limit CROCO grid ---
    loncroco = []
    latcroco = []
    for i in range(len(lon_croco)):
        loncroco.append(xh[0,closest(xh[0,:],lon_croco[i])])
        latcroco.append(yh[closest(yh[:,0],lat_croco[i]),0])
    print('--- 	CROCO grid ---')
    ax.plot(loncroco,latcroco,color='azure',linewidth=2,linestyle=(0, (5, 1)),transform=ccrs.PlateCarree(),label='CROCO grid')

    # --- legend axes ---
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = ticker.FixedLocator([-34, -30, -26,-22]) 
    gl.ylocator = LatitudeLocator() 
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': fs, 'color': 'k'}
    gl.ylabel_style = {'size': fs, 'color': 'k'}
    # --- colorbars --- 
    cax    = plt.subplot(gs[0,0])
    cb     = plt.colorbar(sc,cax,orientation='horizontal',format=logfmt,ticks=cbticks_kappa)
    cb.set_label(r'$\langle\kappa\rangle_z$ [m$^2$ s$^{-1}$]',fontsize=fs,labelpad=10)
    cb.ax.tick_params(labelsize=fs-2)
    cax    = plt.subplot(gs[0,2])
    cb     = plt.colorbar(ctf,cax,orientation='horizontal',ticks=cbticks_h)
    cb.set_label('Bathymetry [m]',fontsize=fs,labelpad=10)
    cb.ax.tick_params(labelsize=fs-2)

    plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/map_cross-ridge_diffusivity_rrex_ovide_kunze_'+name_exp+'.pdf',bbox_inches='tight')


