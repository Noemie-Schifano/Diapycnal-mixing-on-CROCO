'''
NS : Horizontal map of the grid with:
            - time-mean currents interpolated at a constant depth "depth"
            - markers to show tracer release locations
            - markers to show in-situ measurements
            - line to show where CROCO data are extracted to compare with in-situ measurements
            - line to show where vertical sections of CROCO outputs are done (after in the article)
''' 

import matplotlib
matplotlib.use('Agg') 
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
import matplotlib.ticker as ticker
import matplotlib.colors   as colors
sys.path.append('/home/datawork-lops-rrex/nschifan/Python_Modules_p3/')
import R_tools as tools 
import R_tools_fort as toolsF
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
from croco_simulations_jonathan_hist_ncra import Croco
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.colors   as colors
from rrex_cv import RREX_class

# ---------- in situ parameters ------
path_data_ins    = '/home/datawork-lops-osi/cvic/Data_obs' #'/Users/cv1m15/Data/'
file_kunze       = path_data_ins+'strainfineoutall.nc'
file_ovide       = path_data_ins+'dissMean_Ovide08_001.mat'
name_exp         = 'rrexnum200-rsup5'
file_croco       = '/home/datawork-lops-rrex/nschifan/DIAGS/'+name_exp+'_AKt_rrex-middleline.nc'
rho0  = 1025.
dmin  = -500 # discard profiles where h<dmin -- continental shelf
#---> cross-ridge stations from west to east  
list_vmp_ovid08 = [26,25,24,23,22,21,20,19,18,17,16,15,14]#,13,12,11]
list_vmp_rrex15 = [14,39,40,41,38,15]
list_vmp_rrex17 = [17,18,19,20,22,16,23,24,13,12,11,10,9,8]
list_kunze      = [5208,5207,5206,5205,5204,5203,5202,5201,5343,350,351,352,5233,5398,5399,5160,5159]

# ------------ CROCO parameters, time-mean ------------ 
name_exp   = 'rrexnum200-rsup5'
file_croco = '/home/datawork-lops-rrex/nschifan/DIAGS/'+name_exp+'_AKt_rrex-middleline.nc'
file_topo  = '/home/datawork-lops-rrex/nschifan/Data_in_situ_Rene/topo15_NorthAtl.nc'
region     = 'rrex'  
path_data  = '/home/datawork-lops-rrex/nschifan/'
file_grd   = path_data+'GRD/rrexnum_grd.nc'
begin      = '10'
date_end   = '30'
file_his   = path_data+'RREXNUM_200_RSUP5_NOFILT_T_copy/HIS/rrexnum200_his_00020-00058.nc'  
file0      = path_data+'HIST/rrex100-up3_avg.00000.nc'  
date       = ['00020-00058'] 
date_real  = [' 09/2008']

# ------------ CROCO parameters, snapshot ------------ 
name_exp    = 'rrexnum200' 
name_exp_path ='rrexnum200_rsup5'
name_pathdata = 'RREXNUM200_RSUP5_NOFILT_T'
name_exp_grd= 'rrex200-up5'
nbr_levels  = '200'
ndfiles     = 1  # number of days per netcdf
time        = ['20']
var_list    = ['u','v','zeta']

# ----------- CROCO, tracer release ---------------
file_ini_tpas     =  '/home/datawork-lops-rrex/nschifan/DIAGS/rrexnum200-rsup5_tracers_release.nc'

# --- plot options --- 
depth = -1000 # [m] depth to show currents  
jsec  = 400
fs    = 28       # fontsize 
ms    = 40      # markersize 
lw    = 1.5
lw_c  = 0.4   # linewidth coast 
my_bbox = dict(fc='w',ec='k',pad=2,lw=0.,alpha=0.5)
res   = 'h' # resolution of the coastline: c (coarse), l (low), i (intermediate), h (high)
proj  = 'lcc'  # map projection. lcc = lambert conformal 
paral = np.arange(0,80,2)
merid = np.arange(0,360,2)
if region == 'rrex':
    lon_0,lat_0   = -32,57.5 # centre of the map 
    Lx,Ly         = 1000e3,800e3 # [km] zonal and meridional 
extent      =  [-40,-19,50,65] #[-37.5,-21.2,53,62.5]
extent_zoom =   [-29.8,-29,58,58.5] 
xticks      = np.arange(-40,-15,5)
yticks      = np.arange(55,65,5)
# --- colorbar and color
#--> bathymetry  
cmap_h    = plt.cm.terrain_r # gray_r
norm_h         = colors.Normalize(vmin=0,vmax=5000)
levels_h  = np.arange(0,5500,500)
cbticks_h = [1000,2000,3000,4000,5000] #[-3000,-2000,-1000] 
cblabel_h      = 'Bathymetry [m]'
# --> tracer colorbar
pmin,pmax,pint = -3,0.1,0.1
cmap_tpas2     = plt.cm.YlOrBr
cblabel_tpas1  = '[tracer 1]'
cmap_tpas1   = plt.cm.Blues
levels_tpas2   = np.power(10,np.arange(pmin,pmax+pint,pint))
levels_tpas1   = levels_tpas2
norm_tpas2     = colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),
                        ncolors=cmap_tpas2.N,clip=True)
cblabel_tpas2  = '[tracer 2]'
cbticks_tpas   = [10**(-3),10**(-2),0.1,1]

##--> marker for tracers
#cmap_ab      = plt.cm.get_cmap('Blues')
#cmap_r       = plt.cm.get_cmap('YlOrBr')
#cf_ab        = cmap_ab(0.8)
#cf_r         = cmap_r(0.8)
#cf_ab        = colors.to_rgba('royalblue')
#cf_r         = colors.to_rgba('chocolate')
#colors_point =[cf_ab,cf_r] 
cf          = colors.to_rgba('teal')

# -------------- Read data from cruise RREX15 ----------------- 
rrex15 = RREX_class(year='2015',list_vmp=list_vmp_rrex15)
rrex15.get_bathy(resolution='lr')
rrex15.interpolate_bathy()
rrex15.get_dist()

# -------------- Read data from cruise RREX17 ----------------- 
rrex17 = RREX_class(year='2017',list_vmp=list_vmp_rrex17)
rrex17.get_bathy(resolution='lr')
rrex17.interpolate_bathy()
rrex17.get_dist()

# -------------- Read data from cruise OVIDE08 ----------------- 
ovid08 = RREX_class(year='2008',list_vmp=list_vmp_ovid08)
ovid08.get_bathy(resolution='lr')
ovid08.interpolate_bathy()
ovid08.get_dist()

# -------------- Read CROCO outputs at the location of the cruise ----------------- 
croco1 = RREX_class(year='2015',list_vmp=[0]) # dumb class to use further functions 
nc               = Dataset(file_croco,'r')
croco1.lonvmp    = nc.variables['lon'][:]
croco1.latvmp    = nc.variables['lat'][:]
nc.close()
croco1.get_dist()

# ------------- Read bathymetry of the North Atlantic ----------
nc = Dataset(file_topo,'r')
lon15 = nc.variables['lon'][:]
lat15 = nc.variables['lat'][:]
z15   = -nc.variables['z'][:]
nc.close()

# --> longitude and latitude of cruise measurements + CROCO section 
x15,y15 = rrex15.lonvmp,rrex15.latvmp
x17,y17 = rrex17.lonvmp,rrex17.latvmp
x08,y08 = ovid08.lonvmp,ovid08.latvmp
xc1,yc1 = croco1.lonvmp,croco1.latvmp

# --> read initial tracer patches concentrations, summed over z-axis
nc_tpas = Dataset(file_ini_tpas,'r')
tpas03_ini = nc_tpas.variables['tpas03_ini'][:]
tpas05_ini = nc_tpas.variables['tpas05_ini'][:]
nc_tpas.close()
# Nan values to plot
tpas03_ini[tpas03_ini<1e-3]=np.nan
tpas05_ini[tpas05_ini<1e-3]=np.nan

# -------------- Read time-mean CROCO outputs and do plot ----------------- 
for dd in date:
    # ------------ read data ------------ 
    data = Croco(name_exp,nbr_levels,time[0],name_exp_grd,name_pathdata)
    data.get_grid()
    for t in [0]:
        data.get_outputs(t,var_list)
        # ------------ processing ------------ 
        [z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'][:,:],data.hc,data.Cs_r,data.Cs_w)
        simul     = data.angle
        u_rho =np.zeros([1002,802,200])
        v_rho =np.zeros([1002,802,200])
        for nl in range (0,100):
            [u_rho[:,:,nl],v_rho[:,:,nl]] = tools.rotuv_rho(simul,data.var['u'][:,:,nl],data.var['v'][:,:,nl])
        u_z = tools.vinterp(u_rho[:,:,:],[depth],z_r,z_w)
        v_z = tools.vinterp(v_rho[:,:,:],[depth],z_r,z_w)


        # ------------ make plot  ------------ 
        f = plt.figure(figsize=(20,20))
        gs = gridspec.GridSpec(3,2,height_ratios=[1,0.03,0.03],hspace=0.2)
        ax = plt.subplot(gs[0,:],projection=ccrs.Orthographic(lon_0,lat_0)) 
        ax.set_extent(extent,ccrs.PlateCarree())                                                                           
        gl = ax.gridlines(draw_labels=True,crs=ccrs.PlateCarree(),xlocs=xticks,ylocs=yticks,
                            x_inline=False, y_inline=False, linewidth=0.8, color='k', alpha=0.7, linestyle='--',zorder=0)
        # xlocs; ylocs
        ax.coastlines('110m')
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='k', facecolor='gray',zorder=4))
        eps = 25
        qnt = -3,0,0.1 #-3,0, 0.1
        # --> bathymetry from CROCO grid
        ctf1  =  ax.contourf(data.lonr,data.latr,data.h,levels=levels_h,cmap=cmap_h,norm=norm_h,zorder=2,alpha=0.7,transform=ccrs.PlateCarree())
        ct  = ax.contour(data.lonr,data.latr,data.h,levels=levels_h,colors='k',linewidths=lw_c,zorder=4,transform=ccrs.PlateCarree())
        # --> bathymetry outside CROCO grid 
        bathym = ax.contour(lon15,lat15,z15,levels=levels_h,colors='k',linewidths=lw_c,zorder=1,transform=ccrs.PlateCarree())
        # --> Quiver currents
        Q =  plt.quiver(data.lonr[::eps,::eps], data.latr[::eps,::eps], u_z[::eps,::eps,t], v_z[::eps,::eps,t], units='width',scale=7,color='steelblue',zorder=5,alpha=0.8,transform=ccrs.PlateCarree()) 
        # --- quiverkey ---
        ax.quiverkey(Q, X=0.8, Y=0.1, U=0.2,label=r'20 cm$s^{-1}$',labelpos='E',color='steelblue',fontproperties={'size':fs},transform=ccrs.PlateCarree())
        # --> area above ridge
        index  = [250,250,450,450,250]
        index2 = [400,500,500,400,400]
        x = [data.lonr[index[0],index2[0]],data.lonr[index[1],index2[1]],data.lonr[index[2],index2[2]],data.lonr[index[3],index2[3]],data.lonr[index[4],index2[4]]] #,data.lonr[index[0],index2[1]]]
        y = [data.latr[index[0],index2[0]],data.latr[index[1],index2[1]],data.latr[index[2],index2[2]],data.latr[index[3],index2[3]],data.latr[index[4],index2[4]]]#,data.latr[index[1],index2[1]]]
        ax.fill(x, y, transform=ccrs.PlateCarree(), color='yellow',fill=False, hatch='\\\\\\\\\\\\',zorder=5)#,alpha=0.5)
        # --> area above abyssal plain
        index = [680,680,880,880,680]
        index2 = [400,500,500,400,400]
        x = [data.lonr[index[0],index2[0]],data.lonr[index[1],index2[1]],data.lonr[index[2],index2[2]],data.lonr[index[3],index2[3]],data.lonr[index[4],index2[4]]] #,data.lonr[index[0],index2[1]]]
        y = [data.latr[index[0],index2[0]],data.latr[index[1],index2[1]],data.latr[index[2],index2[2]],data.latr[index[3],index2[3]],data.latr[index[4],index2[4]]]
        ax.fill(x, y, transform=ccrs.PlateCarree(), color='orange',fill=False, hatch='\\\\\\\\\\\\',zorder=5)#,alpha=0.5)
        # --> tracers release location
        ctf_tpas03 = ax.contourf(data.lonr,data.latr,tpas03_ini,levels=levels_tpas1,cmap=cmap_tpas1,zorder=3,extend='both',norm=colors.LogNorm(),transform=ccrs.PlateCarree())
        ctf_tpas05 = ax.contourf(data.lonr,data.latr,tpas05_ini,levels=levels_tpas1,cmap=cmap_tpas2,zorder=3,extend='both',norm=colors.LogNorm(),transform=ccrs.PlateCarree())
        #index  = [625,375] 
        #index2 = [275,275]
        #legendtra = ['Release tracer 1','Release tracer 2']
        #for point in range(2):
        #    print('tracer ',point, data.lonr[index[point],index2[point]],data.latr[index[point],index2[point]])
        #    plt.scatter(data.lonr[index[point],index2[point]],data.latr[index[point],index2[point]],color = colors_point[point],
        #                                          marker='D',s=15,linewidth=5,label=legendtra[point],zorder=5,transform=ccrs.PlateCarree())

        # --- markers in-situ data ---
        sc  = ax.scatter(x15[x15>np.nanmin(xc1)],y15[x15>np.nanmin(xc1)],marker='v',s=4*ms,linewidths=0,
                    color='k',zorder=4,transform=ccrs.PlateCarree(),label='RREX15')
        sc  = ax.scatter(x17[x17>np.nanmin(xc1)],y17[x17>np.nanmin(xc1)],marker='^',s=4*ms,linewidths=0,
                    color='k',zorder=4,transform=ccrs.PlateCarree(),label='RREX17')
        sc  = ax.scatter(x08[x08>np.nanmin(xc1)],y08[x08>np.nanmin(xc1)],marker='*',s=6*ms,linewidths=0,
                    color='k',zorder=4,transform=ccrs.PlateCarree(),label='OVIDE08')
        ax.plot(xc1,yc1,color=cf,lw=2*lw,zorder=5,transform=ccrs.PlateCarree()) #,label=r'$K_{KPP}$ and $K_{eff}$ used in section 3.1.1')
        #x = [data.lonr[0,245],data.lonr[0,255],data.lonr[-1,255],data.lonr[-1,245],data.lonr[0,245]]
        #y = [data.latr[0,245],data.latr[0,255],data.latr[-1,255],data.latr[-1,245],data.latr[0,245]]
        #ax.fill(x,y,transform=ccrs.PlateCarree(), color='m', alpha=0.5,zorder=5)
        plt.plot(data.lonr[:,250],data.latr[:,250],'k',lw=lw+1,linestyle='dashed', transform=ccrs.PlateCarree(),zorder=5)
        plt.legend(frameon=False,loc='lower left',prop={'size':fs})
       
        # -------- actual colorbar ------- 
        # --> bathymetry
        wi,hi = 0.5,0.01 #0.005
        xc,yc = 0.3,0.1 #0.025
        dl,dr = 0.03,0.02 # left and right offset 
        db,dt = 0.05,0.04 # bottom and top offset
        # - bounding box 'aext' -  
        aext  = plt.axes([xc-dl,yc-db,wi+dl+dr,hi+db+dt],facecolor='w')
        aext.set_xticks(());aext.set_yticks(());aext.patch.set_alpha(0.5)
        plt.setp(aext.spines.values(), linewidth=0.) # no frame lines
        cax    = plt.axes([xc,yc,wi,hi],facecolor='w')
        cb     = plt.colorbar(ctf1,cax,orientation='horizontal',ticks=[1000,2000,3000,4000])#,ticks=cbticks_h)
        cb.set_label(cblabel_h,fontsize=fs,labelpad=-85) #-60
        cb.ax.tick_params(labelsize=fs-2)

        # --- legend axes ---
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = ticker.FixedLocator(xticks)
        gl.ylocator = ticker.FixedLocator(yticks)

        # --- colorbar tpas03 --- 
        aext =  plt.subplot(gs[1,0])
        aext.set_xticks(());aext.set_yticks(());aext.patch.set_alpha(0.5)
        plt.setp(aext.spines.values(), linewidth=0.) # no frame lines
        cb     = plt.colorbar(ctf_tpas03,aext,orientation='horizontal',ticks=cbticks_tpas)
        cb.set_label(cblabel_tpas1,fontsize=fs,labelpad=-100)
        cb.ax.set_xticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','1'])
        cb.ax.tick_params(labelsize=fs)

        # --- colorbar tpas05 --- 
        aext =  plt.subplot(gs[1,1])
        cb     = plt.colorbar(ctf_tpas05,aext,orientation='horizontal',ticks=cbticks_tpas)
        cb.set_label(cblabel_tpas2,fontsize=fs,labelpad=-100)
        cb.ax.set_xticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','1'])
        cb.ax.tick_params(labelsize=fs)

        print('--> savefig')    
        plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/map_u_v_in_situ_Jon_%im_'%(-depth)+dd+'.png',dpi=200,bbox_inches='tight')

