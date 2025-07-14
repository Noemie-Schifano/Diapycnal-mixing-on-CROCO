'''
NS : For configurations exp50-rsup5 and exp200-rsup5, compare:
         - the tracer distribution in buoyancy space and the fit made by the 1D-model of Ledwell's et al. 1991
         - the time-evolution of all effective and parameterized diapycnal diffusivities (see "compute_k_croco.py")
       --> Need "compute_k_croco.py" tu run before!
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
from collections import OrderedDict

# ------------  parameters ------------
name_exp_jon  = ['rrexnum50-rsup5-nofilt','rrexnum200-rsup5-nofilt']
rhot        =  '27.70'
drho        =  '0.01'
rhoref      =  1027.4
sxy         =  '2'
days        =  '30'
dt          =  3600*24
time        = ['10', '20','30','40','50','60']
ndfiles     = 10  # number of days per netcdf

# ----------- select tracer for analysis --------
choose_tracer=['ridge','plain']
choice       = choose_tracer[1]
if choice=='plain':
    tpas_list   = ['tpas03']   # Tracer name in CROCO outputs
    num_tpas    = [0]          # Index of the tracer in NetCDF from "compute_k_croco.py"
else:
    tpas_list   = ['tpas05']
    num_tpas    = [1]
tid       = num_tpas[0]
var_list  = ['zeta','temp','salt']
var_list += tpas_list 
ntpas     = len(tpas_list)     # number of tracer patches, here 1
ntplot    = 30                 # time-length of the plot

# --- plot options --- 
fs = 14
lw = 1.4
ms = 12      # marker size
tint = 3     # time interval between lines 
coef = 5     # amplification factor for graphic issues 
# --> Color for tracer profil in buoyancy space
cf0 = colors.to_rgba('olive')
# --> Colors for diffusivities
cf1,cf2,cf3,cf4= colors.to_rgba('mediumpurple'),colors.to_rgba('teal'),colors.to_rgba('crimson'),colors.to_rgba('chocolate')

# --- buoyancy bin ---
buoy_bin = 0.08
buoy_e   = np.arange(-49,-45,buoy_bin)*1e-3  #bin edges 
buoy_c200   = 0.5*(buoy_e[1:]+buoy_e[:-1])      #bin centerd
bmin,bmax = -47.5,-46                        #limits for the plot

# ------- read diagnostics from "compute_k_croco.py" ---------
for exp in range(len(name_exp_jon)):
    print(' ... read data ... ') 
    if exp==0:                                                 
        file_diag_k = '/home/datawork-lops-rrex/nschifan/DIAGS/'+name_exp_jon[exp]+'_Kfit_hist_diffusivities.nc'       
        nc = Dataset(file_diag_k,'r') 
        buoy_c50      = nc.variables['buoy_binc'][:]
        nu1           = nc.variables['buoy_avg'][tid,:] 
        sig21         = nc.variables['buoy_var'][tid,:] 
        tpas_bin1     = nc.variables['tpas_bin'][tid,:,:] 
        tpas_mod1     = nc.variables['tpas_mod'][tid,:,:]
        nc.close() 
        nt = nu1.shape[0]
    else:
        file_diag_k = '/home/datawork-lops-rrex/nschifan/DIAGS/'+name_exp_jon[exp]+'hist_diffusivities.nc'
        nc = Dataset(file_diag_k,'r')
        nu2           = nc.variables['buoy_avg'][tid,:]
        sig22         = nc.variables['buoy_var'][tid,:]
        tpas_bin2     = nc.variables['tpas_bin'][tid,:,:]
        tpas_mod2     = nc.variables['tpas_mod'][tid,:,:]
        nc.close()

time = np.arange(nt) 

print( ' ------- make plot ------- ')
plt.figure(figsize=(10,5))
gs = gridspec.GridSpec(1,2,hspace=0.4,wspace=0.4) 
# ----------------------------------------------------------------------------------------- exp50-rsup5 
ax = plt.subplot(gs[0,0]) # -------------------- tracer profiles in buoyancy space  
plt.text(1,-45.6,'a) exp50-rsup5                                                   b) exp200-rsup5',fontsize=fs)
tpas_bin = tpas_bin1  # 3D-solution from CROCO
tpas_mod = tpas_mod1  # 1D-solution according to Ledwell's 1991 and Holmes et al. 2019
buoy_c   = buoy_c50
nu       = nu1
sig2     = sig21
for t in np.arange(0,nt-tint,tint):  
    plt.plot(time[t]+coef*tpas_bin[t,:],1e3*buoy_c,'k',lw=lw) 
    if t!=0: plt.scatter(time[t]+coef*tpas_mod[t,::1],1e3*buoy_c[::1],
                        marker='o',s=ms,ec='gray',fc='none',linewidths=lw) 
t= tint # for labelling 
plt.plot(time[t]+coef*tpas_bin[t,:],1e3*buoy_c,'k',lw=lw,label='3-D solution') 
plt.scatter(time[t]+coef*tpas_mod[t,::1],1e3*buoy_c[::1],
            marker='o',s=ms,ec='gray',fc='none',linewidths=lw,label='1-D fit') 
plt.plot(1e3*nu,color=cf0,lw=2*lw) 
plt.plot(1e3*(nu-np.sqrt(sig2)),color=cf0,ls='--',lw=2*lw) 
plt.plot(1e3*(nu+np.sqrt(sig2)),color=cf0,ls='--',lw=2*lw) 
plt.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),prop={'size':fs},ncol=2) 
plt.ylabel('$b$ [1e$^{-3}$ m s$^{-2}$]',fontsize=fs) 
plt.ylim(bmin,bmax)  
plt.xticks([0,10,20,30],['0','5','10','15'])
plt.xlim(0,ntplot)
ax.tick_params(labelsize=fs) 
plt.xlabel("Time [days]",fontsize=fs)


# ----------------------------------------------------------------------------------------- exp200-rsup5
ax = plt.subplot(gs[0,1]) # -------------------- tracer profiles in buoyancy space
tpas_bin = tpas_bin2 # 3D-solution from CROCO
tpas_mod = tpas_mod2 # 1D-solution according to Ledwell's 1991 and Holmes et al. 2019
buoy_c   = buoy_c200
nu       = nu2
sig2     = sig22
for t in np.arange(0,nt-tint,tint):    
    plt.plot(time[t]+coef*tpas_bin[t,:],1e3*buoy_c,'k',lw=lw)    
    if t!=0: plt.scatter(time[t]+coef*tpas_mod[t,::1],1e3*buoy_c[::1],
                        marker='o',s=ms,ec='gray',fc='none',linewidths=lw)    
t= tint # for labelling 
plt.plot(time[t]+coef*tpas_bin[t,:],1e3*buoy_c,'k',lw=lw,label='3-D solution')    
plt.scatter(time[t]+coef*tpas_mod[t,::1],1e3*buoy_c[::1],
            marker='o',s=ms,ec='gray',fc='none',linewidths=lw,label='1-D fit')    
plt.plot(1e3*nu,color=cf0,lw=2*lw)    
plt.plot(1e3*(nu-np.sqrt(sig2)),color=cf0,ls='--',lw=2*lw)    
plt.plot(1e3*(nu+np.sqrt(sig2)),color=cf0,ls='--',lw=2*lw)    
plt.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),prop={'size':fs},ncol=2)    
#ax.set_yticklabels([])
plt.ylim(bmin,bmax) # in buoyancy space 
plt.xticks([0,10,20,30],['0','5','10','15'])
plt.xlim(0,ntplot)
ax.tick_params(labelsize=fs)    
plt.xlabel("Time [days]",fontsize=fs)

plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/k_dia/exp50-200-rsup5_'+tpas_list[0]+'_time-profiles.pdf',bbox_inches='tight') 
plt.close() 

