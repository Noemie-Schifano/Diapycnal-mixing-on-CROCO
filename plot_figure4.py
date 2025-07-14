'''
NS : Plot snapshots of horizontal and vertical map of the two tracers at the release time and 15 days afters   
'''

import matplotlib
matplotlib.use('Agg') 
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors   as colors
import matplotlib.ticker   as ticker
from netCDF4 import Dataset
import sys
sys.path.append('/home/datawork-lops-rrex/nschifan/Python_Modules_p3/')
import R_tools as tools
import R_tools_fort as toolsF
from croco_simulations_jonathan_hist import Croco
import gsw as gsw
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)


# ----------- netCDF output --------
file_out      =  '/home/datawork-lops-rrex/nschifan/DIAGS/rrexnum200-rsup5_tracers_release.nc'
# ------------ parameters ------------ 
name_exp      = 'rrexnum200' 
name_exp_path ='rrexnum200_rsup5'
name_pathdata = 'RREXNUM200_RSUP5_NOFILT_T'
name_exp_grd  = 'rrex200-up5'
nbr_levels    = '200'
tname         = ['tpas03','tpas05']
tname_plot    = ['tracer 1','tracer 2']
var_list      = ['zeta','temp','salt','tpas03','tpas05']
time          = ['20','50']
timed         = ['0','15','0','15','0','15','0)','15)']
label_fig     = ['a)','b)','c)','d)','e)','f)','g)','h)']
ndfiles       = 1  # read only the first time of netcdf file

# --- plot options --- 
jsec    = 250
jsecr   = 250
fs      = 28                                   # fontsize 
lw      = 0.3                                  # linewidth
lw_c    = 0.2                                  # linewidth coast 
sig_min,sig_max = 27.9,27.55                   # axis limits 
lon_0,lat_0   = -31.5,58.6                     # centre of the map 
levels_rho_contour = np.arange(26.5,28.4,0.1)
extent     = [-37.5,-21.2,53,62.5]             # °E °N
# --> tracer colorbar
pmin,pmax,pint = -3,0,0.1 
cmap_tpas2     = plt.cm.YlOrBr 
cblabel_tpas1  = '[tracer 1]'
cmap_tpas1   = plt.cm.Blues 
levels_tpas2   = np.power(10,np.arange(pmin,pmax+pint,pint))
levels_tpas1   = levels_tpas2
norm_tpas2     = colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),
                        ncolors=cmap_tpas2.N,clip=True)
cblabel_tpas2  = '[tracer 2]'
cbticks_tpas   = [10**(-3),10**(-2),0.1,1] 


# --> save vertical slice of tracers 
tpas1 =  np.zeros((1000,int(nbr_levels),len(time)))
tpas2 =  np.zeros((1000,int(nbr_levels),len(time)))
# --> save anomaly of potential density referenced at the surface 
sigma    = np.zeros((1002,int(nbr_levels),len(time)*len(tname)))
# --> save horizontal slice of tracers (concentrations summed over all the water column!)
tpas1h    = np.zeros((1002,802,len(time)))
tpas2h    = np.zeros((1002,802,len(time)))
# ------------ read data ------------ 
it=0
for t_nc in range(len(time)):
    for t in range(ndfiles):
        data = Croco(name_exp,nbr_levels,time[t_nc],name_exp_grd,name_pathdata)
        data.get_grid()
        # ------------ read data ------------ 
        data.get_outputs(t,var_list)
        # --> vertical slice of tracers
        tpas1[:,:,it] = data.var[tname[0]][1:-1,jsec,:] 
        tpas2[:,:,it] = data.var[tname[1]][1:-1,jsec,:] 
        # --> horizontal slice of tracers, concentrations summed over all the water column!
        tpas1h[:,:,it] =  np.nansum(data.var[tname[0]],axis=2)
        tpas2h[:,:,it] =  np.nansum(data.var[tname[1]],axis=2)

        if it ==0:
            print(' ... save grid data for plot ... ')
            latr = data.latr
            lonr = data.lonr
            h    = data.h

        print(' ... get vertical levels and density ... ')
        [z_r,z_w]       = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
        p               = gsw.p_from_z(z_r,np.nanmean(data.latr))
        SA              = gsw.SA_from_SP(data.var['salt'],p,np.nanmean(data.lonr),np.nanmean(data.latr))
        rho_po          = gsw.pot_rho_t_exact(SA,data.var['temp'],p,0)
        sigma[:,:,it] = rho_po[:,jsecr,:] - 1000 
        # --> mask concentrations inferior at eps
        eps   = 1e-3
        tpas1[tpas1<eps]=np.nan
        tpas2[tpas2<eps]=np.nan
        tpas1h[:,:,it] =  np.ma.masked_array(tpas1h[:,:,it],tpas1h[:,:,it]<=eps)
        tpas2h[:,:,it] =  np.ma.masked_array(tpas2h[:,:,it],tpas2h[:,:,it]<=eps)

        it+=1

# --> arrange sigma for the index in the plot 
sigma[:,:,2] = sigma[:,:,0]
sigma[:,:,3] = sigma[:,:,1]


print( '--- MAKE PLOT ---')
# ------------ make plot  ------------ 
count_y=0 #  column
last_line=0
plt.figure(figsize=(20,13))
gs = gridspec.GridSpec(3,4,height_ratios=[1,1,0.05],hspace=0.4) 
for it in range(len(time)*len(tname)):
    ax = plt.subplot(gs[0,count_y])
    ax.set_facecolor('lightgray')
    #------------------------------------ vertical section with tracer concentration 
    if count_y<2:
        lonsec = 0.5*(data.lonr[1:,jsec]+data.lonr[:-1,jsec]) 
        lonsec = np.tile(lonsec,(z_w.shape[-1],1)).T  
        zsec = 0.5*(z_w[1:,jsec,:]+z_w[:-1,jsec,:]) 
        tpas = tpas1
        if it==0:
            ctf1 = plt.pcolormesh(lonsec,zsec,tpas[:,:,it],norm=norm_tpas2,cmap=cmap_tpas1)
            plt.title(label_fig[it]+' tracer 1, day='+timed[it],fontsize=fs)
        else:
            ctf = plt.pcolormesh(lonsec,zsec,tpas[:,:,it],norm=norm_tpas2,cmap=cmap_tpas1)
            plt.title(label_fig[it]+' tracer 1, day='+timed[it],fontsize=fs)
        lon_tile = np.tile(data.lonr[:,jsec],(z_r.shape[-1],1)).T
        ct  = plt.contour(lon_tile,z_r[:,jsec,:],sigma[:,:,it],levels=levels_rho_contour,colors='k',linewidths=lw)
        plt.fill_between(data.lonr[:,jsec],-3100,-data.h[:,jsec],fc='gray',ec='k',alpha=0.5)
    else:
        lonsec = 0.5*(data.lonr[1:,jsecr]+data.lonr[:-1,jsecr]) 
        lonsec = np.tile(lonsec,(z_w.shape[-1],1)).T  
        zsec = 0.5*(z_w[1:,jsec,:]+z_w[:-1,jsec,:]) 
        tpas = tpas2
        if (it-2)==0:
            ctf2 = plt.pcolormesh(lonsec,zsec,tpas[:,:,it-2],norm=norm_tpas2,cmap=cmap_tpas2)
        else:
            ctf = plt.pcolormesh(lonsec,zsec,tpas[:,:,it-2],norm=norm_tpas2,cmap=cmap_tpas2)
        plt.title(label_fig[it]+' tracer 2, day='+timed[it],fontsize=fs)
        lon_tile = np.tile(data.lonr[:,jsecr],(z_r.shape[-1],1)).T
        ct  = plt.contour(lon_tile,z_r[:,jsecr,:],sigma[:,:,it],levels=levels_rho_contour,colors='k',linewidths=lw)
        plt.fill_between(data.lonr[:,jsecr],-3100,-data.h[:,jsecr],fc='gray',ec='k',alpha=0.5)
    plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs-2)
    if count_y>0:
        ax.set_yticklabels([])
    else:
        plt.ylabel('z [m]',fontsize=fs)
    plt.ylim(-3100,0)
    ax.set_xticks([-36,-30])
    ax.set_xticklabels(['36°W','30°W'])
    count_y+=1
    ax.tick_params(labelsize=fs)

count_y=0
for it in range(len(time)*len(tname)):
    ax = plt.subplot(gs[1,count_y])   
    plt.gca().set_aspect('equal', adjustable='box')
    #------------------------------------ horizontal section with tracer concentration summed over all the water column
    if count_y<2:
        tpas=tpas1h
        if it==0:
            ctf01 = ax.contourf(tpas[:,:,it].T,levels=levels_tpas1,cmap=cmap_tpas1,extend='max',zorder=4,norm=colors.LogNorm())
            plt.title(label_fig[it+4]+' tracer 1, day='+timed[it],fontsize=fs)
        else:
            ax.contourf(tpas[:,:,it].T,levels=levels_tpas1,cmap=cmap_tpas1,extend='max',zorder=4,norm=colors.LogNorm()) # colourful filled contours
            plt.title(label_fig[it+4]+' tracer 1, day='+timed[it],fontsize=fs)
    else:
        tpas=tpas2h
        if (it-2)==0:
            ctf02 = ax.contourf(tpas[:,:,it-2].T,levels=levels_tpas1,cmap=cmap_tpas2,extend='max',zorder=4,norm=colors.LogNorm())
        else:
            ax.contourf(tpas[:,:,it-2].T,levels=levels_tpas1,cmap=cmap_tpas2,extend='max',zorder=4,norm=colors.LogNorm()) # colourful filled contours
        plt.title(label_fig[it+4]+' tracer 2, day='+timed[it],fontsize=fs)
    bathy = ax.contour(h.T,levels=[1000,2000,3000,4000],colors='k',linewidths=lw_c)
    if count_y>0:
        ax.set_yticklabels([])
    else:
        ax.set_yticks([0,250,500,750],['0','200','400','600'])
        plt.ylabel(r'km in $\eta$-direction',fontsize=fs)
    ax.set_xticks([0,250,500,750,1000],['0','200','400','600','800'])
    plt.xlabel(r'km in $\xi$-direction',fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.axhline(y=jsec,c='k',lw=1,linestyle='dashed')
    count_y+=1

    #ax = plt.subplot(gs[1,count_y],projection=ccrs.Orthographic(lon_0,lat_0))
    ##------------------------------------ horizontal section with tracer concentration summed over all the water column
    #if count_y<2:
    #    tpas=tpas1h
    #    if it==0:
    #        ctf01 = ax.contourf(lonr,latr,tpas[:,:,it],levels=levels_tpas1,cmap=cmap_tpas1,extend='max',zorder=4,norm=colors.LogNorm(),transform=ccrs.PlateCarree())
    #        plt.title(label_fig[it+4]+' tracer 1, day='+timed[it],fontsize=fs)
    #    else:
    #        ax.contourf(lonr,latr,tpas[:,:,it],levels=levels_tpas1,cmap=cmap_tpas1,extend='max',zorder=4,norm=colors.LogNorm(),transform=ccrs.PlateCarree()) # colourful filled contours
    #        plt.title(label_fig[it+4]+' tracer 1, day='+timed[it],fontsize=fs)
    #else:
    #    tpas=tpas2h
    #    if (it-2)==0:
    #        ctf02 = ax.contourf(lonr,latr,tpas[:,:,it-2],levels=levels_tpas1,cmap=cmap_tpas2,extend='max',zorder=4,norm=colors.LogNorm(),transform=ccrs.PlateCarree())
    #    else:
    #        ax.contourf(lonr,latr,tpas[:,:,it-2],levels=levels_tpas1,cmap=cmap_tpas2,extend='max',zorder=4,norm=colors.LogNorm(),transform=ccrs.PlateCarree()) # colourful filled contours
    #    plt.title(label_fig[it+4]+' tracer 2, day='+timed[it],fontsize=fs)
    #ax.set_extent(extent)
    #gl = ax.gridlines(draw_labels=True,dms=True, x_inline=False, y_inline=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    #ax.coastlines('110m', alpha=0.1)
    #bathy = ax.contour(lonr,latr,h,levels =[1000,2000,3000,4000] , colors='k',linewidths= lw_c,transform=ccrs.PlateCarree())
    #gl.top_labels    = False
    #gl.right_labels  = False
    #if count_y>0:
    #    gl.left_labels = False
    ## --- legend axes ---
    #gl.xlocator = ticker.FixedLocator([-36, -33,-30, -27,-24]) 
    #gl.ylocator = ticker.FixedLocator([54,55.5,57,58.5,60,61.5])
    #gl.xformatter = LongitudeFormatter()
    #gl.yformatter = LatitudeFormatter()
    #gl.xlabel_style = {'size': fs}
    #gl.ylabel_style = {'size': fs}
    #ax.tick_params(labelsize=fs)
    #ax.set_extent(extent)
    #ax.coastlines('110m', alpha=0.1)

    #plt.xlabel('Longitude [$^{\circ}$E]',fontsize=fs-2)
    #plt.ylabel('Latitude [$^{\circ}$E]',fontsize=fs-2)
    #ax.tick_params(labelsize=fs)       
    #plt.plot(lonr[:,jsec],latr[:,jsec],'k--',lw=1, transform=ccrs.PlateCarree())
    #count_y+=1

    #plt.plot(lonr[:,0],latr[:,0],color='gray',lw=0.5, transform=ccrs.PlateCarree())
    #plt.plot(lonr[:,-1],latr[:,-1],color='gray',lw=0.5, transform=ccrs.PlateCarree())
    #plt.plot(lonr[0,:],latr[0,:],color='gray',lw=0.5, transform=ccrs.PlateCarree())
    #plt.plot(lonr[-1,:],latr[-1,:],color='gray',lw=0.5, transform=ccrs.PlateCarree())


# --- colorbar 1 --- 
aext =  plt.subplot(gs[2,1])
aext.set_xticks(());aext.set_yticks(());aext.patch.set_alpha(0.5)
plt.setp(aext.spines.values(), linewidth=0.) # no frame lines
cb     = plt.colorbar(ctf1,aext,orientation='horizontal',extend='both',ticks=cbticks_tpas)
cb.set_label(cblabel_tpas1,fontsize=fs,labelpad=-80)
cb.ax.set_xticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','1'])
cb.ax.tick_params(labelsize=fs)

# --- colorbar 2 --- 
aext =  plt.subplot(gs[2,2])
cb     = plt.colorbar(ctf2,aext,orientation='horizontal',extend='both',ticks=cbticks_tpas)
cb.set_label(cblabel_tpas2,fontsize=fs,labelpad=-80)
cb.ax.set_xticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','1'])
cb.ax.tick_params(labelsize=fs)

plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/vertical_horizontal_slice_'+name_exp_path+'_jsec%.3i.png'%jsec,dpi=180,bbox_inches='tight')
plt.close()


# ---------------- read netCDF file with initial tracer patch ---------
nc = Dataset(file_out,'w')
nc.createDimension('nlon',np.shape(tpas1h[:,:,0])[0])
nc.createDimension('nlat',np.shape(tpas1h[:,:,0])[1])
var = nc.createVariable('tpas03_ini','f',('nlon','nlat'))
var.long_name = 'tpas03 release, sum over z-axis'
var = nc.createVariable('tpas05_ini','f',('nlon','nlat'))
var.long_name = 'tpas05 release, sum over z-axis'
nc.variables['tpas03_ini'][:]         = tpas1h[:,:,0]
nc.variables['tpas05_ini'][:]         = tpas2h[:,:,0]
nc.close()


