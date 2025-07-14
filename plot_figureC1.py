'''
NS : Compare configurations exp100-rsup3, exp100-rsup3-filterd and exp100-c4
     and plot for one tracer patch:
        (1) K_eff                  : The effective diapycnale diffusivity computed online and weighted by one tracer concentration
        (2) K_fit -named Kzh here- : The offline effective diapycnale diffusivity as defined in Holmes et al. 2019 (following Ledwell's 1D model 1991)
        (3) K_tracer               : The offline effective diapycnale diffusivity as defined in Ruan and Ferrari 2021 (following Taylor diffusivity 1922)
        (4) K_KPP -named AKt here- : The parameterized vertical diffusivity from KPP parameterization

               --> Need 'compute_k_croco.py' to run before
'''


import matplotlib
matplotlib.use('Agg') 
import numpy as np
import scipy.stats as stats
import scipy.interpolate as itp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors   as colors
import matplotlib.ticker   as ticker
from netCDF4 import Dataset
import obsfit1d as fit
import matplotlib.patches as mpatches


# -------- parameters -----
# --> Choice of tracer patches
tpas_list   = ['tpas03','tpas05']    # name in CROCO outputs
num_tpas    = [0,1]                  # associated index in NetCDF from 'compute_k_croco.py'
nbr_exp     = 9                      # number of configurations compared
ntpas       = len(tpas_list)         # number of tracer patchs, here 2
nt          = 30                     # time-length of variables in NetCDF from 'compute_k_croco.py'

name_exp_jon    = ['num100-nofilt','num100-filt','num100-c4']
name_x_jon      =  ['exp100-rsup3','exp100-rsup3-filt','exp100-c4']
name_x_jon_oneline=[ None,'exp100-rsup3',None,None,
                     None,'exp100-rsup3-filt',None,None,
                     None,'exp100-c4',None,None]
name_y          =  [r'$\kappa_{fit}$',r'$\kappa_{tr}$',r'$\kappa_{eff}$',r'$\kappa_{KPP}$'] 
fs       = 14
xname    = ['100','200']#,'300']
lw       = 2
dz       = 1e-1
end      = 6
ms       = 100
cf0 = colors.to_rgba('olive')
cf1 = colors.to_rgba('mediumpurple')
cf2 = colors.to_rgba('thistle')
cf3 = colors.to_rgba('sienna')
cf4 = colors.to_rgba('goldenrod')

cf2,cf3,cf4=colors.to_rgba('teal'),colors.to_rgba('crimson'),colors.to_rgba('chocolate')
cf5 = colors.to_rgba('midnightblue')
cf6 = colors.to_rgba('sienna')
kmin,kmax = 7e-7,1e-1
      # Kzh, Ktr, Keff, AKt
cf = [cf1,cf4,cf3,cf2,
      cf1,cf4,cf3,cf2,
      cf1,cf4,cf3,cf2 ]


cfw = [cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2,
       cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2,
       cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2]


# --- function for the boxplot ---
def plot_k_oneline(ax,K,name_x_jon,name_y,fs,cf,cfw):    
    K2      = np.copy(K)
    K2[:,0] = np.nan*np.ones(np.shape(K2[:,0]))
    K2[:,4] = np.nan*np.ones(np.shape(K2[:,0]))
    K2[:,8] = np.nan*np.ones(np.shape(K2[:,0]))
    data = [K2[:,expk] for expk in range(np.shape(K2)[-1])]
    # --- deal with Nan issues when using boxplot
    mask = ~np.isnan(data)
    filtered_data = [d[m] for d, m in zip(data, mask)]
    flierprop = dict( markersize=2)
    bp = ax.boxplot(filtered_data,medianprops=dict(color='k'),flierprops=flierprop,patch_artist = True, showfliers=False,zorder=1)
    for patch, colors in zip(bp['boxes'], cf):
        patch.set_facecolor(colors)
        patch.set(color = colors)
    for moustache, colors in zip(bp['whiskers'], cfw):
        moustache.set(color = colors)
    for cap, colors in zip(bp['caps'], cfw):
        cap.set(color = colors)
    [ax.axvline(x, color = 'grey', linestyle='--') for x in [4.5,8.5]] #,12.5,16.5,20.5,24.5,28.5,32.5]]
    # --> markers for K_fit
    ax.scatter(np.arange(1,12,4),[K[0,ikfit] for ikfit in range(0,np.shape(K)[-1],4)],marker='o',s=ms,color=cfw[0],zorder=2)
    plt.legend(name_y,fontsize=fs)
    ax.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2], bp["boxes"][3]], name_y, loc='upper left', fontsize = fs-1,ncol=2)
    return

 
# ---> variables to save
k0 = np.zeros((nt,4*len(name_exp_jon)))
k1 = np.zeros((nt,4*len(name_exp_jon)))

# --- read diffusivities ---
for it in range(len(tpas_list)):
    print('it :',it)
    AKt_alli = np.zeros((nt,nbr_exp))
    Ktr_alli = np.zeros((nt,nbr_exp))
    Kzh_alli = np.zeros((nt,nbr_exp))
    Keff_alli= np.zeros((nt,nbr_exp))
    for exp in range(len(name_exp_jon)):
        file_diag_k = '/home/datawork-lops-rrex/nschifan/DIAGS/rrex'+name_exp_jon[exp]+'hist_diffusivities.nc'
        print(' ... read data '+ name_exp_jon[exp]+ ' '+ tpas_list[it] +' ... ')
        nc = Dataset(file_diag_k,'r')
        # --> read tracer 
        tpas = num_tpas[it]
        if it==0: 
            AKt_alli[it*nt:(it+1)*nt,exp] = nc.variables['AKt_avg'][tpas,0:(nt)].T
            Ktr_alli[it*nt:(it+1)*nt,exp] = nc.variables['Ktr'][tpas,0:(nt)].T       
            Kzh_alli[it*nt:(it+1)*nt,exp]  = nc.variables['Kzh'][tpas,nt-1]*np.ones(np.shape(Kzh_alli[it*nt:(it+1)*nt,exp]))
            Keff_alli[it*nt:(it+1)*nt,exp]    = nc.variables['Keff_avg'][tpas,0:(nt)].T
        else:
            itt = it-1
            AKt_alli[itt*nt:(itt+1)*nt,exp] = nc.variables['AKt_avg'][tpas,0:(nt)].T
            Ktr_alli[itt*nt:(itt+1)*nt,exp] = nc.variables['Ktr'][tpas,0:(nt)].T
            Kzh_alli[itt*nt:(itt+1)*nt,exp]  = nc.variables['Kzh'][tpas,nt-1]*np.ones(np.shape(Kzh_alli[itt*nt:(itt+1)*nt,exp]))
            Keff_alli[itt*nt:(itt+1)*nt,exp]    = nc.variables['Keff_avg'][tpas,0:(nt)].T
        nc.close()

    # --- replace negative values by nan
    AKt_alli[AKt_alli<=0]=np.nan
    Ktr_alli[Ktr_alli<=0]=np.nan
    Kzh_alli[Kzh_alli<=0]=np.nan
    Keff_alli[Keff_alli<=0]=np.nan

    # --- plot all in one line-graph
    # --> sort data in a friendly-way for boxplot   
    if it==0:
        print('list_k')
        list_k  = [ Kzh_alli[:,0],Ktr_alli[:,0],Keff_alli[:,0],AKt_alli[:,0],
                    Kzh_alli[:,1],Ktr_alli[:,1],Keff_alli[:,1],AKt_alli[:,1],
                    Kzh_alli[:,2],Ktr_alli[:,2],Keff_alli[:,2],AKt_alli[:,2]]         
        for ik in range(len(list_k)):
            print(np.shape(list_k[ik]))
            k0[:,ik] = list_k[ik]

    else:
        list_k1  = [ Kzh_alli[:,0],Ktr_alli[:,0],Keff_alli[:,0],AKt_alli[:,0],
                    Kzh_alli[:,1],Ktr_alli[:,1],Keff_alli[:,1],AKt_alli[:,1],
                    Kzh_alli[:,2],Ktr_alli[:,2],Keff_alli[:,2],AKt_alli[:,2]]
        for ik in range(len(list_k1)):        
            print(np.shape(list_k1[ik]))
            k1[:,ik] = list_k1[ik]


# --- make plot ---
plt.figure(figsize=(10,5))
gs = gridspec.GridSpec(1,2)
# --- tracer abyssal plain
ax =  plt.subplot(gs[0])
ax.set_title('a) tracer 1',fontsize=fs)
plot_k_oneline(ax,k0,name_x_jon,name_y,fs,cf,cfw)
plt.yscale('log')
ax.set_xticklabels(name_x_jon_oneline,fontsize=fs)
ax.xaxis.set_major_locator(ticker.FixedLocator([1,2,3,4,5,6,7,8, 9,10,11,12]))
plt.axhline(y=1e-6,c='k',linewidth=lw,alpha=0.2)
plt.axhline(y=1e-5,c='k',linewidth=lw,alpha=0.2)
plt.axhline(y=1e-4,c='k',linewidth=lw,alpha=0.2)
plt.axhline(y=1e-3,c='k',linewidth=lw,alpha=0.2)
plt.axhline(y=1e-2,c='k',linewidth=lw,alpha=0.2)
ax.yaxis.set_ticks_position('both')
plt.xticks(rotation=90)
plt.ylim(7e-7,1e-1)
plt.ylabel(r'Diffusivity [m$^2$ s$^{-1}$]',fontsize=fs)
ax.tick_params(labelsize=fs)
# --- tracer ridge 
ax =  plt.subplot(gs[1])
ax.set_title('b) tracer 2',fontsize=fs)
plot_k_oneline(ax,k1,name_x_jon,name_y,fs,cf,cfw)
plt.yscale('log')
ax.set_xticklabels(name_x_jon_oneline,fontsize=fs)
ax.xaxis.set_major_locator(ticker.FixedLocator([1,2,3,4,5,6,7,8,9,10,11,12]))
plt.axhline(y=1e-6,c='k',linewidth=lw-1,alpha=0.2)
plt.axhline(y=1e-5,c='k',linewidth=lw-1,alpha=0.2)
plt.axhline(y=1e-4,c='k',linewidth=lw-1,alpha=0.2)
plt.axhline(y=1e-3,c='k',linewidth=lw-1,alpha=0.2)
plt.axhline(y=1e-2,c='k',linewidth=lw-1,alpha=0.2)
ax.yaxis.set_ticks_position('both')
plt.xticks(rotation=90)
plt.ylim(kmin,kmax)
ax.set_yticklabels([])
ax.tick_params(labelsize=fs)
plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/All_simu/All_simu_k_Jon_exp100-c4-filt-nofilt.pdf',bbox_inches='tight')
plt.close()

