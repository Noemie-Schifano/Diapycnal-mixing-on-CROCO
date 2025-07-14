'''
NS : 15/12/2023: for 3 configurations plots vertical section of Keff, sm and alpha_m

'''

import matplotlib
matplotlib.use('Agg') #Choose the backend (needed for plottingteDimension('time',None) inside subprocess)
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
import time as time
import calendar as cal
import datetime as datetime
import obsfit1d as fit
from croco_simulations_jonathan_hist import Croco

# ------------ parameters ------------ 
scheme_to_chose   = ['rsup5','rsvweno5']
choice_scheme     = scheme_to_chose[0]
if choice_scheme == 'rsup5':
    name_pathdata_jon = ['RREXNUM50_RSUP5_NOFILT_T','RREXNUM200_RSUP5_NOFILT_T','RREXNUMS200_RSUP5_NOFILT_T']
    name_nc_jon       = ['rrexnum50-rsup5-nofilt','rrexnum200-rsup5-nofilt','rrexnums200-rsup5-nofilt']
    title_name   = ['a) exp50-rsup5','b) exp200-rsup5','c) exp200-rsup5-smooth',
                    'd) exp50-rsup5','e) exp200-rsup5','f) exp200-rsup5-smooth',
                    'g) exp50-rsup5','h) exp200-rsup5','i) exp200-rsup5-smooth']
else:
    name_pathdata_jon = ['RREXNUM50_RSVWENO5_NOFILT_T','RREXNUM100_RSVWENO5_NOFILT_T','RREXNUM200_RSVWENO5_NOFILT_T']
    name_nc_jon       = ['rrexnum50-rsvweno5-nofilt','rrexnum100-rsvweno5-nofilt','rrexnum200-rsvweno5-nofilt']
    title_name   = ['a) exp50-rsvweno5','b) exp100-rsvweno5','c) exp200-rsvweno5','d) exp50-rsvweno5','e) exp100-rsvweno5','f) exp200-rsvweno5','g) exp50-rsvweno5','h) exp100-rsvweno5','i) exp200-rsvweno5']

name_exp_grd_jon  = ['rrex50','rrex200-up5','rrex200-up5']
nbr_levels_jon    = ['50','200','200']
name_exp_jon      = ['rrexnum50','rrexnum200','rrexnum200']

time        = ['40']#,'22','24','26','28','30','32','34','36','38','40',
                    # '42','44','46','48','50','52','54','56','58','60',
                    # '62','64','66','68','70','72','74','76','78','80',
                    # '82','84','86','88','90','92','94','96','98']
                   # '41','42','43','44','45','46','47','48','49']
ndfiles     = 1 # number of days per netcdf
nt = len(time)*ndfiles
jsec       = 250
fs         = 42
rho0       = 1027.4
var_list   =['temp','zeta','salt']

# ---> define plots options for each variable 
# --- alpham ---
cmapalpham      = plt.cm.PuOr 
cmap_alpham     = cmapalpham.reversed()
#pmin,pmax,pint  = 0.001,0.1,1e-4   #-2,-1,0.01
pmin,pmax,pint  = 0,0.1,1e-4
levels_alpham   = None #np.power(10,np.arange(pmin,pmax+pint,pint))
norm_alpham     = colors.Normalize(vmin=pmin,vmax=pmax)  #colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),ncolors=cmap_alpham.N,clip=True)
cbticks_alpham  = [0,0.05,1e-1]#np.logspace(pmin,pmax,pmax-pmin+1)

# --- sm ---
cmapsm      = plt.cm.PuOr 
cmap_sm     = cmapsm.reversed()
pmin,pmax,pint = -1,1,0.01
levels_sm   = np.power(10,np.arange(pmin,pmax+pint,pint))
norm_sm     = colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),
                        ncolors=cmap_sm.N,clip=True)
cbticks_sm  = [1e-1,1,10]#np.logspace(pmin,pmax,pmax-pmin+1)

# --- Keff ---
pmin,pmax,pint = -5,-3,0.01
cmap_Keff      = plt.cm.Reds
levels_Keff    = np.power(10,np.arange(pmin,pmax+pint,pint))
norm_Keff      = colors.BoundaryNorm(np.logspace(pmin,pmax,int((pmax-pmin)/pint+1)),
                         ncolors=cmap_Keff.N,clip=True)
cbticks_Keff   = [1e-5,1e-4,1e-3] 


# --- total variables for plot ---
cmap_var     = [cmap_Keff,cmap_sm,cmap_alpham]
cbticks_var  = [cbticks_Keff,cbticks_sm,cbticks_alpham]
norm_var     = [norm_Keff,norm_sm,norm_alpham]
cblabel_var  = [r'$\kappa_{eff}$',r'$s_m$ ',r'$\alpha_m$ ']


# --- define variables to save ---
alpham_50 = np.zeros((1001,50))
sm_50     = np.zeros((1001,50))
keff_50   = np.zeros((1001,50))

alpham_100 = np.zeros((1001,100))
sm_100     = np.zeros((1001,100))
keff_100   = np.zeros((1001,100))

alpham_200 = np.zeros((1001,200))
sm_200     = np.zeros((1001,200))
keff_200   = np.zeros((1001,200))

alpham_s200 = np.zeros((1001,200))
sm_s200     = np.zeros((1001,200))
keff_s200   = np.zeros((1001,200))

def plot_vertical_slice(ax,var,lonsec,zsec,norm_var,cmap_var,lonr,h,jsec,fs,line,column,title_name,cbticks_var,cblabel_var):   #data.lonr[:,jsec],-3100,-data.h[:,jsec]
    # make a vertical section at y=jsec of the field var
    print(np.shape(var),np.shape(lonsec),np.shape(zsec))
    ctf = plt.pcolormesh(lonsec,zsec,var,norm=norm_var,cmap=cmap_var)
    plt.fill_between(lonr,-3100,-h,fc='gray',ec='k',alpha=0.5)
    if column==0:
        plt.ylabel('z [m]',fontsize=fs)
    else:
        ax.set_yticklabels([])
    if line<2:
        ax.set_xticklabels([])
    else:
        ax.set_xticks([-35,-30,-25],['35','30','25'])
        plt.xlabel('Longitude [$^{\circ}$W]',fontsize=fs)
    plt.ylim(-3100,0)
    plt.xlim(lonsec[1,0],lonsec[-2,0])
    ax.tick_params(labelsize=fs)
    plt.title(title_name,fontsize=fs)
    if column==2:
        cb =  plt.colorbar(ctf,location='right',extend='max',orientation='vertical',ticks=cbticks_var)
        if line==0:
            cb.ax.set_yticklabels([r'10$^{-5}$',r'10$^{-4}$',r'10$^{-3}$'])
        elif line==2:
            cb.ax.set_yticklabels(['0','0.05','0.1'])
        else:
            cb.ax.set_yticklabels([r'10$^{-1}$','1','10'])
        cb.set_label(cblabel_var,fontsize=fs,labelpad=10)
        cb.ax.tick_params(labelsize=fs)

# --- read variables ---
for exp in range(len(name_exp_jon)):
    name_exp      = name_exp_jon[exp]                  # name file netcdf
    name_pathdata = name_pathdata_jon[exp]             # folder where are netcdf
    name_exp_grd  = name_exp_grd_jon[exp]              # folder where grid data
    name_nc       = name_nc_jon[exp]                   # name of output netcdf
    nbr_levels    = nbr_levels_jon[exp]
    for t_nc in range(len(time)):
        data = Croco(name_exp,nbr_levels,time[t_nc],name_exp_grd,name_pathdata)
        data.get_grid()
        dsurf   = 1./np.transpose(np.tile(data.pm*data.pn,(int(nbr_levels),1,1)),(1,2,0)) # horizontal surface area
        dsurf_w = 1./np.transpose(np.tile(data.pm*data.pn,(int(nbr_levels)-1,1,1)),(1,2,0)) # horizontal surface area at w-points
        for t in range(ndfiles):
            print('=====================  time index %.4i ====================='%t)
            print('    ---> read outputs ')
            data.get_outputs(t,var_list,get_date=False)
            [z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
            dz_r = np.diff(z_w,axis=-1)
            # --- compute grad(rho) with Rtools.F
            [drhodx,drhody,drhodz] = toolsF.rho_grad(data.var['temp'],data.var['salt'],z_r,z_w,rho0,data.pm,data.pn)
            drhodz = 0.5* (drhodz[:,:,1:] + drhodz[:,:,:-1])  
            drhodx = tools.u2rho(drhodx)
            drhody = tools.v2rho(drhody)
            # --- compute Keff, alpham, sm ----
            print('--- grad(rho) has been computed as in CROCO ---')
            # -> 50 sigma levels
            if exp==0:
                lonr50  = data.lonr
                z_r50   = z_r
                h50     = data.h
                keff_50 = data.get_diffusivity(t,'diffusivity',get_date=False)[:,jsec,:]
                alpham_50 = np.maximum( np.abs(drhodx[:,jsec,:]) , np.abs(drhody[:,jsec,:]) ) / np.abs(drhodz[:,jsec,:])
                sm_50    = alpham_50 / data.pm[:,jsec, np.newaxis] / dz_r[:,jsec,:]
            # -> 200 sigma levels
            elif exp==1:
                lonr200  = data.lonr
                z_r200   = z_r
                h200     = data.h
                keff_200 = data.get_diffusivity(t,'diffusivity',get_date=False)[:,jsec,:]
                alpham_200 = np.maximum( np.abs(drhodx[:,jsec,:]) , np.abs(drhody[:,jsec,:]) ) / np.abs(drhodz[:,jsec,:])
                sm_200    = alpham_200 / data.pm[:,jsec, np.newaxis] / dz_r[:,jsec,:]

            # -> 200 sigma levels smoothed
            else:
                lonrs200  = data.lonr
                z_rs200   = z_r
                hs200     = data.h
                keff_s200 = data.get_diffusivity(t,'diffusivity',get_date=False)[:,jsec,:]
                alpham_s200 = np.maximum( np.abs(drhodx[:,jsec,:]) , np.abs(drhody[:,jsec,:]) ) / np.abs(drhodz[:,jsec,:])
                sm_s200    = alpham_s200 / data.pm[:,jsec, np.newaxis] / dz_r[:,jsec,:]


# ---> mask inf values
sm_50[sm_50==np.inf]=9999
sm_200[sm_200==np.inf]=9999
sm_s200[sm_s200==np.inf]=9999

alpham_50[alpham_50==np.inf]=9999
alpham_200[alpham_200==np.inf]=9999
alpham_s200[alpham_s200==np.inf]=9999

smax200  = np.copy(sm_200)
smaxs200 = np.copy(sm_s200)
amax200  = np.copy(alpham_200)
amaxs200 = np.copy(alpham_s200)
smax200[smax200==9999]   =np.nan
smaxs200[smaxs200==9999] =np.nan
amax200[amax200==9999]   =np.nan
amaxs200[amaxs200==9999] =np.nan
print('sm',np.nanmax(smax200),np.nanmax(smaxs200))
print('am',np.nanmax(amax200),np.nanmax(amaxs200))

# --> make plot 
var_to_plot = [keff_50,keff_200,keff_s200,
               sm_50,sm_200,sm_s200,
               alpham_50,alpham_200,alpham_s200]

line   = 0
column = 0
plt.figure(figsize=(40,30))
gs = gridspec.GridSpec(3,3,width_ratios=[1,1,1.25])
for var in range(len(var_to_plot)):
    ax = plt.subplot(gs[line,column])
    print('-->',cblabel_var[line],np.shape(var_to_plot[var]))
    if column==0:
        lonr = lonr50
        z_r  = z_r50
        h    = h50
    elif column==1:
        lonr = lonr200
        z_r  = z_r200
        h    = h200
    else:
        lonr = lonrs200
        z_r  = z_rs200
        h    = hs200
    lonsec = 0.5*(lonr[1:,jsec]+lonr[:-1,jsec]) # 1 ET PAS 2
    lonsec = np.tile(lonsec,(z_r.shape[-1],1)).T  # z_r et pas z_w
    zsec = 0.5*(z_r[1:,jsec,:]+z_r[:-1,jsec,:])     # 0.5*(z_w[1:,jsec,:]+z_w[:-1,jsec,:]) 
    print(line,column)
    plot_vertical_slice(ax,tools.rho2u(var_to_plot[var]),lonsec,zsec,norm_var[line],cmap_var[line],lonr[:,jsec],h[:,jsec],jsec,
                                                           fs,line,column,title_name[var],cbticks_var[line],cblabel_var[line])  
    if column<2:
        column+=1
    else:
        line+=1
        column =0
plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/All_simu/Dynamic_keff_sm_alpham_exp50_200_s200-'+choice_scheme+'_Jon.png',bbox_inches='tight')
plt.close()






















































