'''
NS : Compare configurations exp200-rsup5 using smooth bathymetry or reference bathymetry
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

# --- Choice of tracer ------
choose_tracer=['ridge','plain']
choice       = choose_tracer[0]
if choice=='plain':
    tpas_list   = ['tpas03'] # corresponding name of tracer in CROCO outputs
    num_tpas    = [0]        # index of this tracer in the output netcdf from 'compute_k_croco.py'
else:
    tpas_list   = ['tpas05']
    num_tpas    = [1]
# --------- parameters ------
nbr_exp     = 2                 # number of CROCO configuration
ntpas       = len(tpas_list)    # number of tracers (here 1)
nt          = 30                #time-length of variables from 'compute_k_croco.py'
# --> name of NetCDF output from 'compute_k_croco.py for choosen configuration
expc       = 'smooth_topo'

# --- plot options ---
if expc =='smooth_topo':
    name_exp_jon    = ['num200-rsup5-nofilt','nums200-rsup5-nofilt']
    name_x_jon      = ['exp200-rsup5','exp200-rsup5-smooth']
    name_x_jon_oneline = [None,'exp200-rsup5',None,None,None,None,'exp200-rsup5-smooth',None]
name_y          =  [r'$\kappa_{fit}$',r'$\kappa_{tr}$',r'$\kappa_{eff}$',r'$\kappa_{KPP}$'] 
fs       = 16
xname    = ['100','200']
lw       = 2
dz       = 1e-1
end      = 6
ms       = 200
cf0 = colors.to_rgba('olive')
cf1 = colors.to_rgba('mediumpurple')
cf2,cf3,cf4=colors.to_rgba('teal'),colors.to_rgba('crimson'),colors.to_rgba('chocolate')
cf5 = colors.to_rgba('midnightblue')
cf6 = colors.to_rgba('sienna')
kmin,kmax = 1e-6,3e-4



cf = [cf1,cf4,cf3,cf2,
      cf1,cf4,cf3,cf2]

cfw = [cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2,
       cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2]


# --> save variables 
AKt_alli = np.zeros((ntpas*nt,nbr_exp))
Ktr_alli = np.zeros((ntpas*nt,nbr_exp))
Kzh_alli = np.zeros((ntpas*nt,nbr_exp))
Keff_alli= np.zeros((ntpas*nt,nbr_exp))

# --- function for boxplot ---
def plot_k_oneline(ax,K,name_x_jon,name_y,fs,cf,cfw):    
    #data = [K[:,expk] for expk in range(np.shape(K)[-1])]
    K2 = np.copy(K)
    K2[:,0] = np.nan*np.ones(np.shape(K2[:,0]))
    K2[:,4] = np.nan*np.ones(np.shape(K2[:,0]))
    data = [K2[:,expk] for expk in range(np.shape(K2)[-1])]
    #data = [K[:,expk] for expk in range(1,4)]+[K[:,expk] for expk in range(5,8)]
    # --- deal with Nan issues when using boxplot
    mask = ~np.isnan(data)
    filtered_data = [d[m] for d, m in zip(data, mask)]
    flierprop = dict( markersize=2)
    bp = ax.boxplot(filtered_data,medianprops=dict(color='k'),flierprops=flierprop,
                      patch_artist = True, showfliers=False,zorder=1)
    for patch, colors in zip(bp['boxes'], cf):
        patch.set_facecolor(colors)
        patch.set(color = colors)
    for moustache, colors in zip(bp['whiskers'], cfw):
        moustache.set(color = colors)
    for cap, colors in zip(bp['caps'], cfw):
        cap.set(color = colors)
    [ax.axvline(x, color = 'grey', linestyle='--') for x in [4.5]]
    # --> markers for K_fit
    ax.scatter(np.arange(1,8,4),[K[0,ikfit] for ikfit in range(0,np.shape(K)[-1],4)],marker='o',s=ms,color=cfw[0],zorder=2)
    plt.legend(name_y,fontsize=fs)
    ax.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2], bp["boxes"][3]], name_y, loc='upper right', fontsize = fs-1,ncol=2)
    return



# --- read diffusivities ---
for it in range(len(tpas_list)):
    for exp in range(len(name_exp_jon)):
        file_diag_k = '/home/datawork-lops-rrex/nschifan/DIAGS/rrex'+name_exp_jon[exp]+'hist_diffusivities.nc'
        print(' ... read data '+ name_exp_jon[exp]+ ' '+ tpas_list[it] +' ... ')
        nc = Dataset(file_diag_k,'r')
        # all tracers together
        tpas = num_tpas[it]
        AKt_alli[it*nt:(it+1)*nt,exp] = nc.variables['AKt_avg'][tpas,0:(nt)].T#9:(nt+9)].T
        Ktr_alli[it*nt:(it+1)*nt,exp] = nc.variables['Ktr'][tpas,0:(nt)].T       
        Kzh_alli[it*nt:(it+1)*nt,exp] = nc.variables['Kzh'][tpas,nt-1]*np.ones(np.shape(Kzh_alli[it*nt:(it+1)*nt,exp]))
        Keff_alli[it*nt:(it+1)*nt,exp]= nc.variables['Keff_avg'][tpas,0:(nt)].T
        nc.close()

# --- replace negative values by nan
AKt_alli[AKt_alli<0]=np.nan
Ktr_alli[Ktr_alli<0]=np.nan
Kzh_alli[Kzh_alli<=0]=np.nan
Keff_alli[Keff_alli<0]=np.nan

# --- plot all in one line-graph
# --> sort data in a friendly-way for boxplot
k0 = np.zeros((ntpas*nt,4*len(name_exp_jon)))
list_k  = [ Kzh_alli[:,0],Ktr_alli[:,0],Keff_alli[:,0],AKt_alli[:,0],
            Kzh_alli[:,1],Ktr_alli[:,1],Keff_alli[:,1],AKt_alli[:,1]]

#k0 = np.zeros((ntpas*nt,4*len(name_exp_jon)))
#list_k  = [ Ktr_alli[:,0],Keff_alli[:,0],AKt_alli[:,0],
#            Ktr_alli[:,1],Keff_alli[:,1],AKt_alli[:,1]]

for ik in range(len(list_k)):
    print(np.shape(list_k[ik]))
    k0[:,ik] = list_k[ik]

# --> make plot 
plt.figure(figsize=(40,20))
fig,ax = plt.subplots()
#ax.xaxis.set_major_locator(ticker.FixedLocator([1,2,3,4,5,6,7,8]))
plot_k_oneline(ax,k0,name_x_jon,name_y,fs,cf,cfw)
plt.yscale('log')
ax.set_xticklabels(name_x_jon_oneline,fontsize=fs)
ax.xaxis.set_major_locator(ticker.FixedLocator([1,2,3,4,5,6,7,8]))
plt.axhline(y=1e-5,c='k',linewidth=lw-1,alpha=0.2)
plt.axhline(y=1e-4,c='k',linewidth=lw-1,alpha=0.2)
ax.yaxis.set_ticks_position('both')
plt.ylim(kmin,kmax)
plt.ylabel(r'Diffusivity [m$^2$ s$^{-1}$]',fontsize=fs)
ax.tick_params(labelsize=fs-1)
plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/All_simu/All_simu_k_Jon'+expc+tpas_list[0]+'.pdf',bbox_inches='tight')
plt.close()



