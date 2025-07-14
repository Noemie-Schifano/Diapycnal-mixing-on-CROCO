'''
NS 2022/12/08: plot vertical sclice of 1 tracer for all config 
'''

import matplotlib
matplotlib.use('Agg') 
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors   as colors
import matplotlib.ticker   as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
# ------------ parameters ------------ 
name_exp_jon  = ['rrexnum50','rrexnum50','rrexnum50','rrexnum100','rrexnum100','rrexnum100','rrexnum200','rrexnum200','rrexnum200','rrexnum100']
name_pathdata = ['RREXNUM50_NOFILT_T','RREXNUM50_RSUP5_NOFILT_T','RREXNUM50_RSVWENO5_NOFILT_T','RREXNUM100_NOFILT_T','RREXNUM100_RSUP5_NOFILT_T',
                                  'RREXNUM100_RSVWENO5_NOFILT_T','RREXNUM200_NOFILT_T','RREXNUM200_RSUP5_NOFILT_T','RREXNUM200_RSVWENO5_NOFILT_T','RREXNUM100_UP3']
name_exp_grd  = ['rrex50','rrex50','rrex50','rrex100-up3','rrex100-up5','rrex100-weno5','rrex200-up3','rrex200-up5','rrex200-weno5','rrex100-up3']
title_exp     =  ['a) exp50-rsup3','b) exp50-rsup5','c) exp50-weno5','e) exp100-rsup3','f) exp100-rsup5','g) exp100-weno5','h) exp200-rsup3','i) exp200-rsup5','j) exp200-weno5']
nbr_levels    = ['50','50','50','100','100','100','200','200','200','100']
time          = ['40']   # select NetCDF output from CROCO
ndfiles       = 1        # select the time-step of the netcdf for the plot

# --> Choice of tracer patch
choose_tracer = ['ridge','plain'] 
choice        = choose_tracer[1]
if choice=='plain':
    tname       = 'tpas03'      # Name in outputs from CROCO
    tname_plot  = 'tracer 1'    # Name in the plot (and article)
    # -- plot options
    cmap_tpas   = plt.cm.Blues  
    xmin,xmax   = -32,-27 #-32.5,-26.5
    jsec        = 250           # \\\ VERTICAL SLICE
    cblabel_tpas= '[tracer 1]'
    xticks      = [-32,-30,-28]
    xticksl     = ['32','30','28']
else:
    tname       = 'tpas05'
    tname_plot  = 'tracer 2'
    # -- plot options
    cmap_tpas   = plt.cm.YlOrBr
    xmin,xmax   = -34,-30
    jsec        = 250           # \\\ VERTICAL SLICE
    cblabel_tpas= '[tracer 2]'
    xticks      = [-34,-32,-30]
    xticksl     = ['34','32','30']

# --> variables to read in CROCO
var_list = ['zeta','temp','salt',tname] 

# --- plot options --- 
fs      = 42       # fontsize 
lw      = 0.3      # linewidth
ms      = 20       # markersize 
lw_c    = 0.2      # linewidth coast 
# --> colorbar for tracer patch
eps=1e-4
pmin,pmax,pint = -3,0,0.1 
levels_tpas    = np.power(10,np.arange(pmin,pmax+pint,pint))
norm_tpas      = colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),
                        ncolors=cmap_tpas.N,clip=True)
cbticks_tpas   = np.logspace(pmin,pmax,pmax-pmin+1)
levels_rho_contour = np.arange(26.5,28.4,0.1)

# ------------ read data ------------ 
for t_nc in range(len(time)):
    count_NL=0   
    # --- save tracer, density anomaly, and depth for:
    # --> configurations with 50 sigma-levels
    tpas_50      = np.zeros((1000,50,3))
    sigma50      = np.zeros((1002,802,50))
    z_r50        = np.zeros((1002,802,50))
    z_w50        = np.zeros((1002,802,51))
    # --> configurations with 100 sigma-levels
    tpas_100     = np.zeros((1000,100,3))
    sigma100     = np.zeros((1002,802,100))
    z_r100       = np.zeros((1002,802,100))
    z_w100       = np.zeros((1002,802,101))
    # --> configurations with 200 sigma-levels
    tpas_200     = np.zeros((1000,200,3))
    sigma200     = np.zeros((1002,802,200))
    z_r200       = np.zeros((1002,802,200))

    z_w200       = np.zeros((1002,802,201))
    # --- Loop on configurations ---
    for exp in range(len(name_exp_jon)):
        data = Croco(name_exp_jon[exp],nbr_levels[exp],time[t_nc],name_exp_grd[exp],name_pathdata[exp])
        data.get_grid()
        for t in range(ndfiles): 
            # ------------ read data ------------ 
            data.get_outputs(t,var_list)
            print(' ... Read tracer patch ... ')
            if exp==(len(name_exp_jon)-1):
                tpas_100_up3 = np.nansum(data.var[tname][1:-1,jsec-5:jsec+5,:],axis=1)
            else:
                # Y-axis-averaged of the tracer concentration over 10 sections around the vertical section "jsec" 
                if count_NL==0: # -- 50 sigma-levels
                    tpas_50[:,:,exp]   = np.nansum(data.var[tname][1:-1,jsec-5:jsec+5,:],axis=1)
                if count_NL==1: # -- 100 sigma-levels
                    tpas_100[:,:,exp-3] = np.nansum(data.var[tname][1:-1,jsec-5:jsec+5,:],axis=1)
                if count_NL==2: # -- 200 sigma-levels
                    tpas_200[:,:,exp-6] = np.nansum(data.var[tname][1:-1,jsec-5:jsec+5,:],axis=1)
                if exp ==0: # --- exp50-rsup3, 50 sigma-levels
                    print(' ... get vertical levels and potential density anomaly for 50 sigma-levels... ')
                    [z_r50,z_w50] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
                    p       = gsw.p_from_z(z_r50,np.nanmean(data.latr))
                    SA      = gsw.SA_from_SP(data.var['salt'],p,np.nanmean(data.lonr),np.nanmean(data.latr))
                    CT      = gsw.CT_from_pt(SA,data.var['temp']) 
                    rho_po  = gsw.rho(SA,CT,0) 
                    rho_po  = gsw.pot_rho_t_exact(SA,data.var['temp'],p,0)
                    sigma50 = rho_po - 1000
                if exp ==3: # --- rrex100-rsup3, 100 sigma-levels
                    print(' ... get vertical levels and potential density anomaly for 100 sigma-levels... ')
                    [z_r100,z_w100] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
                    p       = gsw.p_from_z(z_r100,np.nanmean(data.latr))
                    SA      = gsw.SA_from_SP(data.var['salt'],p,np.nanmean(data.lonr),np.nanmean(data.latr))
                    CT      = gsw.CT_from_pt(SA,data.var['temp'])
                    rho_po  = gsw.rho(SA,CT,0)
                    rho_po  = gsw.pot_rho_t_exact(SA,data.var['temp'],p,0)
                    sigma100[:,:,:] = rho_po - 1000
                if exp ==6: # --- rrex200-rsup3, 200 sigma-levels
                    print(' ... get vertical levels and potential density anomaly for 200 sigma-levels... ')
                    [z_r200,z_w200] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
                    p       = gsw.p_from_z(z_r200,np.nanmean(data.latr))
                    SA      = gsw.SA_from_SP(data.var['salt'],p,np.nanmean(data.lonr),np.nanmean(data.latr))
                    CT      = gsw.CT_from_pt(SA,data.var['temp'])
                    rho_po  = gsw.rho(SA,CT,0)
                    rho_po  = gsw.pot_rho_t_exact(SA,data.var['temp'],p,0)
                    sigma200[:,:,:] = rho_po - 1000
            tt = (t+1)+ int(time[t_nc])-10
            print('time:',tt)

        count= exp+1
        if (count%3)==0:
            count_NL+=1
    # Mask small concentrations for the plot
    tpas_100_up3[tpas_100_up3<eps]=np.nan
    tpas_50[tpas_50<eps]=np.nan
    tpas_100[tpas_100<eps]=np.nan
    tpas_200[tpas_200<eps]=np.nan
    print( '--- MAKE PLOT ---')
    # ------------ make plot  ------------ 
    count_x=0 #  line
    count_y=1 #  column
    last_line=0
    fig = plt.figure(figsize=(40,20))  
    gs = gridspec.GridSpec(4,4,hspace=0.55,height_ratios=[1,1,1,0.1]) 
    # rrex100-up3
    ax = plt.subplot(gs[1,0])
    ax.set_facecolor('lightgray')
    lonsec = 0.5*(data.lonr[1:,jsec]+data.lonr[:-1,jsec]) 
    lonsec = np.tile(lonsec,(z_w100.shape[-1],1)).T  
    zsec = 0.5*(z_w100[1:,jsec,:]+z_w100[:-1,jsec,:])    
    ctf = plt.pcolormesh(lonsec,zsec,tpas_100_up3,norm=norm_tpas,cmap=cmap_tpas)
    lon_tile = np.tile(data.lonr[:,jsec],(z_r100.shape[-1],1)).T
    ct  = plt.contour(lon_tile,z_r100[:,jsec,:],sigma100[:,jsec,:],levels=levels_rho_contour,colors='k',linewidths=lw)
    plt.fill_between(data.lonr[:,jsec],-3100,-data.h[:,jsec],fc='gray',ec='k',alpha=0.5)
    plt.ylabel('z [m]',fontsize=fs)
    ax.set_xticks(xticks,xticksl)
    plt.xlabel('Longitude [$^{\circ}$W]',fontsize=fs)
    ax.tick_params(labelsize=fs)
    plt.title('d) exp100-up3',fontsize=fs)
    plt.ylim(-3100,0)
    plt.xlim(xmin,xmax)

    # other configurations
    for exp in range(len(name_exp_jon)-1):
        print(count_x,count_y)
        ax = plt.subplot(gs[count_x,count_y])
        ax.set_facecolor('lightgray')
        # -------------------------------------- section with tracer concentration
        if count_x==0: # ----------------------- 50 sigma levels
            lonsec = 0.5*(data.lonr[1:,jsec]+data.lonr[:-1,jsec]) 
            lonsec = np.tile(lonsec,(z_w50.shape[-1],1)).T  
            zsec = 0.5*(z_w50[1:,jsec,:]+z_w50[:-1,jsec,:])     
            if exp==0:
                ctf1 = plt.pcolormesh(lonsec,zsec,tpas_50[:,:,exp],norm=norm_tpas,cmap=cmap_tpas)
            else:
                ctf = plt.pcolormesh(lonsec,zsec,tpas_50[:,:,exp],norm=norm_tpas,cmap=cmap_tpas)
            lon_tile = np.tile(data.lonr[:,jsec],(z_r50.shape[-1],1)).T
            ct  = plt.contour(lon_tile,z_r50[:,jsec,:],sigma50[:,jsec,:],levels=levels_rho_contour,colors='k',linewidths=lw)
            plt.fill_between(data.lonr[:,jsec],-3100,-data.h[:,jsec],fc='gray',ec='k',alpha=0.5)
            ax.set_xticklabels([])
            #if (count_y>0):
            #    ax.set_yticklabels([])
            #else:
            #    plt.ylabel('z [m]',fontsize=fs)
        if count_x==1: # ----------------------- 100 sigma levels
            lonsec = 0.5*(data.lonr[1:,jsec]+data.lonr[:-1,jsec]) 
            lonsec = np.tile(lonsec,(z_w100.shape[-1],1)).T  
            zsec = 0.5*(z_w100[1:,jsec,:]+z_w100[:-1,jsec,:])    
            ctf = plt.pcolormesh(lonsec,zsec,tpas_100[:,:,exp-3],norm=norm_tpas,cmap=cmap_tpas)
            lon_tile = np.tile(data.lonr[:,jsec],(z_r100.shape[-1],1)).T
            ct  = plt.contour(lon_tile,z_r100[:,jsec,:],sigma100[:,jsec,:],levels=levels_rho_contour,colors='k',linewidths=lw)
            plt.fill_between(data.lonr[:,jsec],-3100,-data.h[:,jsec],fc='gray',ec='k',alpha=0.5)
            #if count_y>0:
            #    ax.set_yticklabels([])
            #else:
            #    plt.ylabel('z [m]',fontsize=fs)
        if count_x==2: # ----------------------- 200 sigma levels
            lonsec = 0.5*(data.lonr[1:,jsec]+data.lonr[:-1,jsec]) 
            lonsec = np.tile(lonsec,(z_w200.shape[-1],1)).T  
            zsec = 0.5*(z_w200[1:,jsec,:]+z_w200[:-1,jsec,:])    
            ctf = plt.pcolormesh(lonsec,zsec,tpas_200[:,:,exp-6],norm=norm_tpas,cmap=cmap_tpas)
            lon_tile = np.tile(data.lonr[:,jsec],(z_r200.shape[-1],1)).T
            ct  = plt.contour(lon_tile,z_r200[:,jsec,:],sigma200[:,jsec,:],levels=levels_rho_contour,colors='k',linewidths=lw)
            plt.fill_between(data.lonr[:,jsec],-3100,-data.h[:,jsec],fc='gray',ec='k',alpha=0.5)
        if count_y>1:
            ax.set_yticklabels([])
        elif (count_y==1)and(count_x==1):
            ax.set_yticklabels([])
        else:
            plt.ylabel('z [m]',fontsize=fs)
        plt.ylim(-3100,0)
        plt.xlim(xmin,xmax)
        if last_line>5:
            ax.set_xticks(xticks,xticksl)
            plt.xlabel('Longitude [$^{\circ}$W]',fontsize=fs)
        else:
            ax.set_xticklabels([])
        ax.tick_params(labelsize=fs)        
        plt.title(title_exp[exp],fontsize=fs)
        last_line+=1
        count_y+=1
        if (count_y%4)==0:
           count_x += 1
           count_y = 1

    # --> colorbar
    cax    = plt.subplot(gs[3,2])
    cb     = plt.colorbar(ctf1,cax,orientation='horizontal',extend="both",ticks=cbticks_tpas)
    cb.set_label(cblabel_tpas,fontsize=fs,labelpad=10)
    cb.ax.tick_params(labelsize=fs)
    cb.ax.set_xticklabels([r'$10^{-3}$',r'$10^{-2}$', r'$10^{-1}$','1'])
    plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/All_simu/All_simu_vertical_slice_'+tname+'_Jon_jsec%.3i_days_%.2i.png'%(jsec,tt),dpi=180,bbox_inches='tight')
    plt.close()


