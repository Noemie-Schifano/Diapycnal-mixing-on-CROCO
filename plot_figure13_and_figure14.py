'''
NS : Plot for one tracer patch and all the configurations:
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
nbr_exp     = 10                # number of CROCO configuration
ntpas       = len(tpas_list)    # nunber of tracers, here ntpas=1
nt          = 30 #46                # time-length of variables from 'compute_k_croco.py'
# --> name of NetCDF output from 'compute_k_croco.py for each configuration
name_exp_jon    = ['num50-nofilt','num50-rsup5-nofilt','num50-rsvweno5-nofilt','num100-up3-nofilt','num100-nofilt','num100-rsup5-nofilt',
                             'num100-rsvweno5-nofilt','num200-nofilt','num200-rsup5-nofilt','num200-rsvweno5-nofilt']

# --- plot options ---
name_x_jon      =  ['exp50-rsup3','exp50-rsup5','exp50-rsvweno5','exp100-up3','exp100-rsup3','exp100-rsup5','exp100-rsvweno5','exp200-rsup3','exp200-rsup5','exp200-rsvweno5']
name_x_jon_oneline=[ None,'exp50-rsup3',None,None,
                     None,'exp50-rsup5',None,None,
                     None,'exp50-weno5',None,None,
                     None,'exp100-up3' ,None,None,
                     None,'exp100-rsup3',None,None,
                     None,'exp100-rsup5',None,None,
                     None,'exp100-weno5',None,None,
                     None,'exp200-rsup3',None,None,
                     None,'exp200-rsup5',None,None,
                     None,'exp200-weno5',None,None]
name_y          =  [r'$\kappa_{fit}$',r'$\kappa_{tr}$',r'$\kappa_{eff}$',r'$\kappa_{KPP}$'] 
fs       = 10
xname    = ['100','200']#,'300']
lw       = 2
dz       = 1e-1
end      = 6
cf0 = colors.to_rgba('olive')
cf1 = colors.to_rgba('mediumpurple')
cf2,cf3,cf4=colors.to_rgba('teal'),colors.to_rgba('crimson'),colors.to_rgba('chocolate')
cf5 = colors.to_rgba('midnightblue')
cf6 = colors.to_rgba('sienna')
kmin,kmax = 7e-6,7e-3 
      # Kzh, Ktr, Keff, AKt
cf = [cf1,cf4,cf3,cf2,
      cf1,cf4,cf3,cf2,
      cf1,cf4,cf3,cf2,
      cf1,cf4,cf3,cf2,
      cf1,cf4,cf3,cf2,
      cf1,cf4,cf3,cf2,
      cf1,cf4,cf3,cf2,
      cf1,cf4,cf3,cf2,
      cf1,cf4,cf3,cf2,
      cf1,cf4,cf3,cf2 ]


cfw = [cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2,
       cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2,
       cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2,
       cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2,
       cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2,
       cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2,
       cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2,
       cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2,
       cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2,
       cf1,cf1,cf4,cf4,cf3,cf3,cf2,cf2]


# --> save variables 
AKt_alli = np.zeros((ntpas*nt,nbr_exp))
Ktr_alli = np.zeros((ntpas*nt,nbr_exp))
Kzh_alli = np.zeros((ntpas*nt,nbr_exp))
Keff_alli= np.zeros((ntpas*nt,nbr_exp))

# --- function for the plot ----
def plot_k_oneline(ax,K,name_x_jon,name_y,fs,cf,cfw):    
    print('check K',np.shape(K)[-1],np.shape(K[:,0]))
    data = [K[:,expk] for expk in range(np.shape(K)[-1])]
    print(np.shape(data))
    # --- deal with Nan issues when using boxplot
    mask = ~np.isnan(data)
    filtered_data = [d[m] for d, m in zip(data, mask)]
    flierprop = dict( markersize=2)
    # --> make boxplot for each diffusivity and each configuration
    bp = ax.boxplot(filtered_data,medianprops=dict(color='k'),flierprops=flierprop,patch_artist = True, showfliers=False,zorder=1)
    for patch, colors in zip(bp['boxes'], cf):
        patch.set_facecolor(colors)
        patch.set(color = colors)
    for moustache, colors in zip(bp['whiskers'], cfw):
        moustache.set(color = colors)
    for cap, colors in zip(bp['caps'], cfw):
        cap.set(color = colors)
    [ax.axvline(x, color = 'grey', linestyle='--') for x in [4.5,8.5,12.5,16.5,20.5,24.5,28.5,32.5,36.5]]
    # --> markers for K_fit
    ax.scatter(np.arange(1,40,4),[K[0,ikfit] for ikfit in range(0,np.shape(K)[-1],4)],marker='o',color=cfw[0],zorder=2)
    #print([K[0,ikfit] for ikfit in range(0,np.shape(K)[-1],4)])
    plt.legend(name_y,fontsize=fs)
    ax.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2], bp["boxes"][3]], name_y, loc='upper right', fontsize = fs-1,ncol=4)
    return

   
# --- read diffusivities ---
# --> Automatization of tracer hcoice
for it in range(len(tpas_list)):
    # --> Loop on configration choice
    for exp in range(len(name_exp_jon)):
        # --> output NetCDF from 'compute_k_corco.py'
        file_diag_k      = '/home/datawork-lops-rrex/nschifan/DIAGS/rrex'+name_exp_jon[exp]+'hist_diffusivities.nc'
        file_diag_kfit50 = '/home/datawork-lops-rrex/nschifan/DIAGS/rrex'+name_exp_jon[exp]+'_Kfit_hist_diffusivities.nc'
        print(' ... read data '+ name_exp_jon[exp]+ ' '+ tpas_list[it] +' ... ')
        nc = Dataset(file_diag_k,'r')
        tpas = num_tpas[it]
        AKt_alli[it*nt:(it+1)*nt,exp] = nc.variables['AKt_avg'][tpas,0:(nt)].T
        Ktr_alli[it*nt:(it+1)*nt,exp] = nc.variables['Ktr'][tpas,0:(nt)].T       
        # For Kfit -named Kzh here-, only consider the last 10 days
        #Kzh_alli[it*nt:(it+1)*nt,exp]  = nc.variables['Kzh'][tpas,0:(nt)].T 
        if exp>2:
            Kfit  = nc.variables['Kzh'][tpas,nt-1]
        Keff_alli[it*nt:(it+1)*nt,exp]= nc.variables['Keff_avg'][tpas,0:(nt)].T
        nc.close()
        if exp<=2:
            nckfit = Dataset(file_diag_kfit50,'r')
            Kfit   = nckfit.variables['Kzh'][tpas,-1]
            nckfit.close()
        Kzh_alli[it*nt:(it+1)*nt,exp]  = Kfit*np.ones(np.shape(Kzh_alli[it*nt:(it+1)*nt,exp]))

# --- replace negative values by nan
AKt_alli[AKt_alli<=0]=np.nan
Ktr_alli[Ktr_alli<=0]=np.nan
Kzh_alli[Kzh_alli<=0]=np.nan
Keff_alli[Keff_alli<=0]=np.nan


# --- plot all in one line-graph
# --> sort data in a friendly-way for the boxplot
k0 = np.zeros((ntpas*nt,4*len(name_exp_jon)))
list_k  = [ Kzh_alli[:,0],Ktr_alli[:,0],Keff_alli[:,0],AKt_alli[:,0],
            Kzh_alli[:,1],Ktr_alli[:,1],Keff_alli[:,1],AKt_alli[:,1],
            Kzh_alli[:,2],Ktr_alli[:,2],Keff_alli[:,2],AKt_alli[:,2],
            Kzh_alli[:,3],Ktr_alli[:,3],Keff_alli[:,3],AKt_alli[:,3],
            Kzh_alli[:,4],Ktr_alli[:,4],Keff_alli[:,4],AKt_alli[:,4],
            Kzh_alli[:,5],Ktr_alli[:,5],Keff_alli[:,5],AKt_alli[:,5],
            Kzh_alli[:,6],Ktr_alli[:,6],Keff_alli[:,6],AKt_alli[:,6],
            Kzh_alli[:,7],Ktr_alli[:,7],Keff_alli[:,7],AKt_alli[:,7],
            Kzh_alli[:,8],Ktr_alli[:,8],Keff_alli[:,8],AKt_alli[:,8],
            Kzh_alli[:,9],Ktr_alli[:,9],Keff_alli[:,9],AKt_alli[:,9] ]
 
for ik in range(len(list_k)):
    k0[:,ik] = list_k[ik]

# --> make plot 
plt.figure(figsize=(40,40))
fig,ax = plt.subplots()
plot_k_oneline(ax,k0,name_x_jon,name_y,fs,cf,cfw)
plt.yscale('log')
ax.set_xticklabels(name_x_jon_oneline,fontsize=fs)
ax.xaxis.set_major_locator(ticker.FixedLocator([1,2,3,4,5,6,7,8,
                                                9,10,11,12,13,14,15,16,
                                                17,18,19,20,21,22,23,24,
                                                25,26,27,28,29,30,31,32,
                                                33,34,35,36,37,38,39,40]))

plt.axhline(y=1e-5,c='k',linewidth=lw-1,alpha=0.2)
plt.axhline(y=1e-4,c='k',linewidth=lw-1,alpha=0.2)
plt.axhline(y=1e-3,c='k',linewidth=lw-1,alpha=0.2)
ax.yaxis.set_ticks_position('both')
plt.xticks(rotation=90)
plt.ylim(kmin,kmax)
plt.ylabel(r'Diffusivity [m$^2$ s$^{-1}$]',fontsize=fs)
ax.tick_params(labelsize=fs-1)
plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/All_simu/All_simu_k_Jon_'+tpas_list[0]+'.pdf',bbox_inches='tight')
plt.close()





