'''
NS : Needs "precompute_figure7.py" to run before 
    Compute statistics (10th and 90th percentiles + median value) of diffusivities :   
                 (1) parameterized diffusivity K_KPP, called "AKt" in CROCO 
                 (2) Effective diffusivity diagnosed online K_eff
     Considering all the time of the simulation.
     Diffusivities are re-gridded in the height-above-the-bottom (hab) coordinates, 
                      and statistics on diffusivities are made for each bin of hab.
     This is done for each of the 9 configuration we used.
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
sys.path.append('/home2/datahome/nschifan/Python_Modules_p3/')
import R_tools as tools
import R_tools_fort as toolsF
import obsfit1d as fit
from croco_simulations_jonathan_hist import Croco

# ------------ parameters ------------ 
name_exp_jon      = ['rrexnum50','rrexnum50','rrexnum50','rrexnum100','rrexnum100','rrexnum100','rrexnum200','rrexnum200','rrexnum200']
name_pathdata_jon = ['RREXNUM50_NOFILT_T','RREXNUM50_RSUP5_NOFILT_T','RREXNUM50_RSVWENO5_NOFILT_T','RREXNUM100_NOFILT_T',
                                        'RREXNUM100_RSUP5_NOFILT_T','RREXNUM100_RSVWENO5_NOFILT_T','RREXNUM200_NOFILT_T',
                                                              'RREXNUM200_RSUP5_NOFILT_T','RREXNUM200_RSVWENO5_NOFILT_T']
name_exp_grd_jon  = ['rrex50','rrex50','rrex50','rrex100-up3','rrex100-up5','rrex100-weno5','rrex200-up3','rrex200-up5','rrex200-weno5']
name_nc_jon       = ['rrexnum50-nofilt','rrexnum50-rsup5-nofilt','rrexnum50-rsvweno5-nofilt','rrexnum100-nofilt','rrexnum100-rsup5-nofilt',
                                    'rrexnum100-rsvweno5-nofilt','rrexnum200-nofilt','rrexnum200-rsup5-nofilt','rrexnum200-rsvweno5-nofilt']
name_x_jon        = [ ['exp50-rsup3','exp100-rsup3','exp200-rsup3'],
                      ['exp50-rsup5','exp100-rsup5','exp200-rsup5'],
                      ['exp50-rsweno5','exp100-rsweno5','exp200-rsweno5'] ]
nbr_levels_jon    = ['50','50','50','100','100','100','200','200','200']
time              = ['20','22','24','26','28','30','32','34','36','38','40',
                          '42','44','46','48','50','52','54','56','58','60',
                          '62','64','66','68','70','72','74','76','78','80']
ndfiles           = 2 # number of files per netcdf
nt                = len(time)*ndfiles
var_list          = ['zeta','AKt'] # variables to read

# --- save statistics of diffusivities in a NetCDF file --- 
file_diag = '/home/datawork-lops-rrex/nschifan/DIAGS/Keff_AKt_ridge_plain_hab.nc'

# --- plot options ---
fs = 14
lw = 2.5
cf = colors.to_rgba('black')     # color for the percentiles (10th and 90th) for the barplot
cfr= colors.to_rgba('gold')    # color of the median value of the effective diffusivity K_eff above the ridge area
cfp= colors.to_rgba('orange')    # color of the median value of the effective diffusivity K_eff above the abyssal plain area
cfparam= colors.to_rgba('teal')  # color of the median value of the parameterized diffusivity K_KPP 

# --- bin of height-above-bottom (hab)  --- 
dz    = 30.                      # [m]
habe  = np.arange(0,5000+dz,dz)
habc  = 0.5*(habe[1:]+habe[:-1]) # hab centres
nhab  = habc.shape[0]

# --- create areas ---
#     indexs of [lon_min, lon_max, lat_min, lat_max]
# --> area ridge 
points_r = [250,450,400,500]
# --> area abyssal plain
points_p = [680,880,400,500]

# ---> statistics from "precompute_figure7.py"
nc  = Dataset(file_diag,'r')
Keffr_all50  = nc.variables['Keffr_all50'][:]
Keffr_all100 = nc.variables['Keffr_all100'][:]
Keffr_all200 = nc.variables['Keffr_all200'][:]
AKtr_all50  = nc.variables['AKtr_all50'][:]
AKtr_all100 = nc.variables['AKtr_all100'][:]
AKtr_all200 = nc.variables['AKtr_all200'][:]
# --> Effective diffusivity diagnosed online above the abyssal plain area
Keffp_all50  = nc.variables['Keffp_all50'][:]
Keffp_all100 = nc.variables['Keffp_all100'][:]
Keffp_all200 = nc.variables['Keffp_all200'][:]
AKtp_all50  = nc.variables['AKtp_all50'][:]
AKtp_all100 = nc.variables['AKtp_all100'][:]
AKtp_all200 = nc.variables['AKtp_all200'][:]
nc.close()


def plot_k(ax,k,kparam,hab,cf,cf2,lw,fs,line,column,name_title,number_title,nbr_levels):
    # Bar-plot that show for each bin of height above the bottom, the statistics on diffusiviies:
    #                               (1) parameterized diffusivity K_KPP, called "AKt" in CROCO 
    #                               (2) Effective diffusivity diagnosed online K_eff
    # --> Highlight the median values in the plot
    #plt.plot(AKt[1,:],hab,color=cf2,linewidth=lw+1,label=labelmedian)
    plt.plot(kparam[0],hab,color=cfparam,linestyle='dashed',linewidth=lw+1,alpha=0.5,label=r'$K_{KPP}$, exp50-rsup5')
    plt.plot(kparam[1],hab,color=cfparam,linestyle='solid',linewidth=lw+1,alpha=0.5,label=r'$K_{KPP}$, exp100-rsup5')
    plt.plot(kparam[2],hab,color=cfparam,linestyle='dotted',linewidth=lw+1,alpha=0.5,label=r'$K_{KPP}$, exp200-rsup5')

    plt.plot(k[0][1,:],hab,color=cf,linestyle='dashed',linewidth=lw+1,label=r'$K_{eff}$, exp50-rsup5')
    plt.plot(k[1][1,:],hab,color=cf,linestyle='solid',linewidth=lw+1,label=r'$K_{eff}$, exp100-rsup5')
    plt.plot(k[2][1,:],hab,color=cf,linestyle='dotted',linewidth=lw+1,label=r'$K_{eff}$, exp200-rsup5')
   
    # --> percentiles for exp100-rsup5
    print(k[1][1,10],k[1][0,10],k[1][2,10])
    ax.fill_betweenx(hab,k[1][0,:],k[1][2,:],color=cf,alpha=0.25)

    plt.legend(loc='upper right',prop={'size':fs},ncol=1)
    #plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.385),prop={'size':fs-3},ncol=1)
    plt.xscale('log')
    plt.xlim(1e-6,1)
    plt.axvline(x=1e-1,c='k',alpha=0.2,lw=0.3)
    plt.axvline(x=1e-2,c='k',alpha=0.2,lw=0.3)
    plt.axvline(x=1e-3,c='k',alpha=0.2,lw=0.3)
    plt.axvline(x=1e-4,c='k',alpha=0.2,lw=0.3)
    plt.axvline(x=1e-5,c='k',alpha=0.2,lw=0.3)
    ax.set_xticks([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1],fontsize=fs)
    plt.xlabel('$\kappa$ [m$^2$ s$^{-1}$]',fontsize=fs)
    plt.ylim(0,1000)
    if column==0:
        plt.ylabel('hab [m]',fontsize=fs)
    plt.title(number_title,fontsize=fs)
    ax.tick_params(labelsize=fs)


print("--- Make plot ---")
# --> Organize index to make the plot in a more friendly way
rsup3   = [0,3,6]
rsup5   = [1,4,7]
rsweno5 = [2,5,8]
scheme  = [rsup3,rsup5,rsweno5]
name_scheme = ['RSUP3','RSUP5','RSWENO5']


# --- make figure ---
adv = 1
fig = plt.figure(figsize=(10,10))
gs  = gridspec.GridSpec(1,2,wspace=0.3) 
#fig.suptitle('Ridge                                                                                 Abyssal plain', horizontalalignment='center', fontsize=fs)

ax00 = plt.subplot(gs[0,0]) # ---> ridge, sigma 
exp = scheme[adv]
k   = [Keffr_all50[1,:,:],Keffr_all100[1,:,:],Keffr_all200[1,:,:]]
# --> just median value for AKt
kparam = [AKtr_all50[1,1,:],AKtr_all100[1,1,:],AKtr_all200[1,1,:]]
print(np.shape(k))
hab = habc
plot_k(ax00,k,kparam,hab,cfr,cfparam,lw,fs,0,0,name_x_jon[adv],'a) Ridge',nhab)


ax01 = plt.subplot(gs[0,1]) # ---> abyssal plain
exp = scheme[adv]
k   = [Keffp_all50[1,:,:],Keffp_all100[1,:,:],Keffp_all200[1,:,:]]
# --> just median value for AKt
kparam = [AKtp_all50[1,1,:],AKtp_all100[1,1,:],AKtp_all200[1,1,:]]
hab = habc
plot_k(ax01,k,kparam,hab,cfp,cfparam,lw,fs,0,1,name_x_jon[adv],'b) Abyssal plain',nhab)
    
plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/All_simu/Keff_avg_ridge_abyssal_plain_Jon_hab_corrected.pdf',bbox_inches='tight')
plt.close()



