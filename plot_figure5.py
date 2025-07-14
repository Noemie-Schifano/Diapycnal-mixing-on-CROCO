'''
NS : For one configuration, plot vertical section at jsec of:
          - horizontal (u,v) and vertical (w) eulerian currents;
          - Stratification N2;
          - Vertical shear of horizontal velocity S2;
          - Richardson number Ri;
          - Paramterized mixing (K_KPP, named "AKt" in CROCO outputs);
          - Effective mixing diagnosed online K_eff;
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
from croco_simulations_jonathan_hist import Croco
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)


# ------------ parameters ------------ 
name_exp_jon      = ['rrexnum50','rrexnum100','rrexnum200','rrexnum200','rrexnum200']
name_pathdata_jon = ['RREXNUM50_RSUP5_NOFILT_T','RREXNUM100_RSUP5_NOFILT_notides_T','RREXNUM200_RSVWENO5_NOFILT_T','RREXNUM200_RSUP5_NOFILT_T','RREXNUMS200_RSUP5_NOFILT_T']
name_exp_grd_jon  = ['rrex50','rrex100-up5','rrex200-weno5','rrex200-up5','rrex200-up5']
name_nc_jon       = ['rrexnum50-rsup5-nofilt','rrexnum100-rsup5-nofilt-notides','rrexnum200-rsvweno5-nofilt','rrexnum200-rsup5-nofilt','rrexnums200-rsup5-nofilt']
# --> select the simulation "exp" 
exp           = 3
name_exp      = name_exp_jon[exp]                  # name file netcdf
name_pathdata = name_pathdata_jon[exp]             # folder where are netcdf
name_exp_grd  = name_exp_grd_jon[exp]              # folder where grid data
name_nc       = name_nc_jon[exp]                   # name of output netcdf
nbr_levels  = '200'
time        = ['60'] 
ndfiles     = 1                                     # read only the first snapshot
nt          = len(time)*ndfiles
dt          = 3600*24
var_list     = ['temp','zeta','salt','AKt','u','v']
var_list_avg = ['w']

# --- plot options ---
jsec        = 250
fs          = 28
lw          = 1.5

# ---> define plots options for each variable 
# --- AKt ---
pmin,pmax,pint = -5,-3,0.1
cmap_AKt        = plt.cm.Reds
levels_AKt   = np.power(10,np.arange(pmin,pmax+pint,pint))
norm_AKt     = colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),
                        ncolors=cmap_AKt.N,clip=True)
cbticks_AKt  = np.logspace(pmin,pmax,pmax-pmin+1)
logfmt       = ticker.LogFormatterMathtext(10,labelOnlyBase=False)

# --- Keff ---
cmap_Keff      = plt.cm.Reds
levels_Keff   = np.power(10,np.arange(pmin,pmax+pint,pint))
norm_Keff     = colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),
                        ncolors=cmap_Keff.N,clip=True)
cbticks_Keff  = np.logspace(pmin,pmax,pmax-pmin+1)

# --- Ri ---
cmap_Ri     = plt.cm.RdBu_r #seismic
pmin,pmax,pint  = -1000,1000,1
levels_Ri      = None #np.power(10,np.arange(pmin,pmax+pint,pint))
                                                   # linscale=0.03
norm_Ri        = colors.SymLogNorm(linthresh=0.03, linscale=1,
                                              vmin=pmin, vmax=pmax)
cbticks_Ri     =  [pmin,0,0.7,pmax]

# --- N2 ---
pmin,pmax,pint = -9,-5,0.1
cmap_N2        = plt.cm.jet
levels_N2      = np.power(10,np.arange(pmin,pmax+pint,pint))
norm_N2        = colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),
                        ncolors=cmap_N2.N,clip=True)
cbticks_N2     = np.logspace(pmin,pmax,pmax-pmin+1)

# --- u ---
cmap_u     = plt.cm.RdBu_r #seismic
pmin,pmax = -0.2,0.2  # -0.2,0.2,1
levels_u      = None #np.power(10,np.arange(pmin,pmax+pint,pint))
norm_u        = colors.Normalize(vmin=pmin,vmax=pmax)
cbticks_u     =  [pmin,0,pmax]
levels_u      = None

# --- S2 ---
pmin,pmax,pint = -9,-4,0.1 
cmap_S2        = plt.cm.inferno 
cbticks_S2     = np.logspace(pmin,pmax,pmax-pmin+1)
logfmt         = ticker.LogFormatterMathtext(10,labelOnlyBase=False) 
levels_S2      =  np.power(10,np.arange(pmin,pmax+pint,pint))
norm_S2        = colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),
                                                          ncolors=cmap_S2.N,clip=True)

# --- w ---
cmap_w        = plt.cm.RdBu_r
pmin,pmax     = -0.005,0.005
norm_w        = colors.Normalize(vmin=pmin,vmax=pmax)
cbticks_w     =  [pmin,0,pmax]

# --- total variables ---
cmap_var     = [cmap_u,cmap_u,cmap_w,cmap_N2,cmap_S2,cmap_Ri,cmap_AKt,cmap_Keff]
cbticks_var  = [cbticks_u,cbticks_u,cbticks_w,cbticks_N2,cbticks_S2,cbticks_Ri,cbticks_AKt,cbticks_Keff]
norm_var     = [norm_u,norm_u,norm_w,norm_N2,norm_S2,norm_Ri,norm_AKt,norm_Keff]
cblabel_var  = [r'u [m s$^{-1}$]',r'v [m s$^{-1}$]',r'w [m s$^{-1}$]',r'$N^2$ [s$^{-2}$]',
                r'S$^{2}$ [$s^{-2}$]',r'$R_{i}$',r'$\kappa_{KPP}$ [m$^2$ s$^{-1}$]',r'$\kappa_{eff}$ [m$^2$ s$^{-1}$]']
title_name   = ['a)','b)','c)','d)','e)','f)','g)','h)']

# --- plot map ---
extent             =   [-37.5,-21.2,53,62.5]
levels_rho_contour = np.arange(26.5,28.4,0.1)
                                              
def plot_vertical_slice(ax,var,sigma,lonsec,lon_tile,zsec,zr,norm_var,cmap_var,lonr,h,jsec,fs,line,column,title_name,cbticks_var,cblabel_var,levels_rho_contour):  
    # make a vertical section at y=jsec of the field var
    print(np.shape(var),np.shape(lonsec),np.shape(zsec))
    ctf = plt.pcolormesh(lonsec,zsec,var,norm=norm_var,cmap=cmap_var)
    plt.contour(lon_tile,zr,sigma,levels=levels_rho_contour,colors='k',linewidths=lw,alpha=0.5)
    plt.fill_between(lonr,-3100,-h,fc='gray',ec='k',alpha=0.5)
    if column==0:
        plt.ylabel('z [m]',fontsize=fs)
    else:
        ax.set_yticklabels([])
    if line<3:
        ax.set_xticklabels([])
    else:
        ax.set_xticks([-35,-30,-25],['35','30','25'])
        plt.xlabel('Longitude [$^{\circ}$W]',fontsize=fs)
    plt.ylim(-3100,0)
    plt.xlim(lonsec[1,0],lonsec[-2,0])
    ax.tick_params(labelsize=fs)
    plt.title(title_name,fontsize=fs)
    if (line==1)and(column==1): # --> Ri
        plt.contour(lonsec,zsec,var,levels=[0.7],colors='yellow',linewidths=0.3)
        cb = plt.colorbar(ctf,location='right', orientation='vertical',ticks=cbticks_var)
        cb.ax.set_yticklabels([r'-10$^3$','0',r'$Ri_{c}$=0.7',r'10$^3$'])
        #plt.contour(lonsec,zsec,var,levels=[0.25,0.7],colors='yellow',linewidths=0.3)
        cb.ax.axhline(y=0.7, color='yellow', linewidth=2)
        #cb.ax.plot([0.7], color='yellow', lw=2, transform=cb.ax.transAxes)
        #cb.ax.add_patch(plt.Rectangle((0, 0.58), 1, 0.04, transform=cb.ax.transAxes, color='yellow', lw=0))
    elif ((line==2)or(line==3))and(column==1):
        cb =  plt.colorbar(ctf,location='right',orientation='vertical',format=logfmt,ticks=cbticks_var)
        cb.ax.set_yticklabels([r'10$^{-5}$',r'10$^{-4}$',r'10$^{-3}$'])
    elif (line==2)and(column==0): # --> w
        cb =  plt.colorbar(ctf,location='right',orientation='vertical',format=logfmt,ticks=cbticks_var)
        cb.ax.set_yticklabels([r'-5.10$^{-3}$','0',r'5.10$^{-3}$'])
    elif (line==3)and(column==0): # --> N2
        cb =  plt.colorbar(ctf,location='right',orientation='vertical',format=logfmt,ticks=cbticks_var)
        cb.ax.set_yticklabels([r'10$^{-9}$',r'10$^{-8}$',r'10$^{-7}$',r'10$^{-6}$',r'10$^{-5}$'])
    elif (line==0)and(column==1): # --> S2    
        cb =  plt.colorbar(ctf,location='right',orientation='vertical',format=logfmt,ticks=cbticks_var)
        cb.ax.set_yticklabels([r'10$^{-9}$',r'10$^{-8}$',r'10$^{-7}$',r'10$^{-6}$',r'10$^{-5}$',r'10$^{-4}$'])
    else:
        cb =  plt.colorbar(ctf,location='right',orientation='vertical',ticks=cbticks_var)
    cb.set_label(cblabel_var,fontsize=fs,labelpad=10)
    cb.ax.tick_params(labelsize=fs)


# --- read variables ---
for t_nc in range(len(time)):
    N2   = np.zeros((1002,int(nbr_levels)))
    data = Croco(name_exp,nbr_levels,time[t_nc],name_exp_grd,name_pathdata)
    data.get_grid()
    dsurf   = 1./np.transpose(np.tile(data.pm*data.pn,(int(nbr_levels),1,1)),(1,2,0))   # horizontal surface area
    dsurf_w = 1./np.transpose(np.tile(data.pm*data.pn,(int(nbr_levels)-1,1,1)),(1,2,0)) # horizontal surface area at w-points
    for t in range(ndfiles):
        print('=====================  time index %.4i ====================='%t)
        print('    ---> read outputs ')
        data.get_outputs(t,var_list)
        AKt  = np.asfortranarray(tools.w2rho(tools.rho2u(data.var['AKt']))[:,jsec,:])
        [z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
        print(' ... compute N2 with gsw ... ')
        p       = gsw.p_from_z(z_r,np.nanmean(data.latr))
        SA      = gsw.SA_from_SP(data.var['salt'],p,np.nanmean(data.lonr),np.nanmean(data.latr))
        CT      = gsw.CT_from_pt(SA,data.var['temp'])
        for i in range(np.shape(z_r)[0]):
            [N2i,pmid] = gsw.Nsquared(SA[i,jsec,:],CT[i,jsec,:],p[i,jsec,:],data.latr[i,jsec]) 
            N2[i,:-1]      = N2i
        N2[:,-1]=np.nan
        print('--> N2',np.shape(N2))
        # --> compute density anomaly 
        rho_po  = gsw.pot_rho_t_exact(SA,data.var['temp'],p,0)
        sigma   = rho_po - 1000
        print(' ... compute Ri ... ')
        # --- compute dudz and dvdz ---
        u   = np.asfortranarray(data.var['u'])
        v   = np.asfortranarray(tools.rho2u(tools.v2rho_3d(data.var['v'])))
        print('--> u',np.shape(u))
        print('--> v',np.shape(v))
        z_ru   = tools.rho2u(z_r)
        z_wu   = tools.rho2u(z_w)
        dz_w   = np.diff(z_ru,axis=-1)
        dudz_w = np.diff(u,axis=-1)/dz_w
        dvdz_w = np.diff(v,axis=-1)/dz_w
        dudz   =  tools.vinterp(dudz_w,z_ru,z_wu[:,:,1:-1],z_ru)[:,jsec,:]
        dvdz   =  tools.vinterp(dvdz_w,z_ru,z_wu[:,:,1:-1],z_ru)[:,jsec,:]
        # --- compute S2 ---
        S2  = dudz**2 + dvdz**2
        # --- compute Ri ---
        Ri  = tools.rho2u(N2)/S2
        print(' ... compute Keff ... ')
        Keff    = tools.rho2u(data.get_diffusivity(t,'diffusivity'))[:,jsec,:]
        # --- compute w ---
        w = toolsF.get_wvlcty(data.var['u'],data.var['v'],z_r,z_w,data.pm,data.pn)
        print(np.shape(w))


# --- make plot ---
print(' --- make plot --- ')
var_to_plot = [u[:,jsec,:],v[:,jsec,:],tools.rho2u(w)[:,jsec,:],tools.rho2u(N2),S2,Ri,AKt,Keff]
line=0
column=0
plt.figure(figsize=(20,20))      
gs = gridspec.GridSpec(4,2,wspace=0.2,hspace=0.3)
lonsec = 0.5*(data.lonr[1:,jsec]+data.lonr[:-1,jsec]) 
lonsec = np.tile(lonsec,(z_r.shape[-1],1)).T  
zsec = 0.5*(z_r[1:,jsec,:]+z_r[:-1,jsec,:])      
lon_tile = np.tile(data.lonr[:,jsec],(z_r.shape[-1],1)).T
for var in range(len(var_to_plot)):
    print(line,column)
    ax = plt.subplot(gs[line,column]) 
    print('-->',cblabel_var[var])
    plot_vertical_slice(ax,var_to_plot[var],sigma[:,jsec,:],lonsec,lon_tile,zsec,z_r[:,jsec,:],norm_var[var],cmap_var[var],data.lonr[:,jsec],data.h[:,jsec],jsec,
                                                              fs,line,column,title_name[var],cbticks_var[var],cblabel_var[var],levels_rho_contour)
    if line<3:
        line+=1
    else:
        line=0
        column=1
plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/All_simu/Dynamic_totale_'+name_nc+'section_'+str(jsec)+'_Jon.png',dpi=200,bbox_inches='tight')
plt.close()

