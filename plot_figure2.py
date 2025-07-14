'''
NS : plot vertical grid resolution with 50, 100 and 200 sigma-levels 
'''
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors   as colors
import matplotlib.ticker   as ticker
import sys
sys.path.append('/home/datawork-lops-rrex/nschifan/Python_Modules_p3/')
import R_tools as tools
import R_tools_fort as toolsF
from croco_simulations_jonathan_hist import Croco


# ------------ parameters ------------ 
name_exp      = ['rrexnum50','rrexnum100','rrexnum200']
name_pathdata = ['RREXNUM50_NOFILT_T','RREXNUM100_NOFILT_T','RREXNUM200_NOFILT_T']
name_exp_grd  = ['rrex50','rrex100','rrex200']
nbr_levels  = ['50','100','200']
var_list = ['zeta']

# --- plot options --- 
fs            = 28
jsec          = 250
lw_c          = 0.3   # linewidth coast 

cmap_dz    = plt.cm.rainbow
levels_dz  = np.arange(0,75,5) 
norm_dz    = colors.BoundaryNorm(levels_dz,ncolors=cmap_dz.N,clip=True)
cbticks_dz = [0,10,20,30,40,50,60,70]
cblabel_dz = 'dz [m]' 


# ------------ read data ------------ 
dat1 = Croco(name_exp[0],nbr_levels[0],'00',name_exp_grd[0],name_pathdata[0])
dat1.get_grid()
# ------------ processing ------------ 
print(' ... processing ... ') 
dat1.get_outputs(0,var_list)
dat1.get_zlevs() 
dat1.dz = abs(np.diff(dat1.z_w,axis=-1)) 


# ------------ read data ------------ 
dat2 = Croco(name_exp[1],nbr_levels[1],'00',name_exp_grd[1],name_pathdata[1])
dat2.get_grid()
# ------------ processing ------------ 
print(' ... processing ... ')
dat2.get_outputs(0,var_list)
dat2.get_zlevs()
dat2.dz = abs(np.diff(dat2.z_w,axis=-1))


# ------------ read data ------------ 
dat3 = Croco(name_exp[2],nbr_levels[2],'00',name_exp_grd[2],name_pathdata[2])
dat3.get_grid()
# ------------ processing ------------ 
print(' ... processing ... ')
dat3.get_outputs(0,var_list)
dat3.get_zlevs()
dat3.dz = abs(np.diff(dat3.z_w,axis=-1))


# ------------ make plot ------------ 
print(' ... make plot ... ') 
plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(2,3,height_ratios=[1,0.05],hspace=0.6,wspace=0.2)# including colorbars  
ax = plt.subplot(gs[0,0]) # ------------------------ 50 sigma-levels
plt.title('a) '+nbr_levels[0]+r' $s$-levels',fontsize=fs)
lonsec = 0.5*(dat1.lonr[1:,jsec]+dat1.lonr[:-1,jsec])
lonsec = np.tile(lonsec,(dat1.z_w.shape[-1],1)).T
zsec = 0.5*(dat1.z_w[1:,jsec,:]+dat1.z_w[:-1,jsec,:])
ctf = plt.pcolormesh(lonsec,zsec,dat1.dz[1:-1,jsec,:],norm=norm_dz,cmap=cmap_dz)
for k in np.arange(0,dat1.z_w.shape[-1],5):
    plt.plot(dat1.lonr[:,jsec],dat1.z_w[:,jsec,k],'k',lw=lw_c)
plt.fill_between(dat1.lonr[:,jsec],-3100,-dat1.h[:,jsec],fc='lightgray',ec='k',alpha=0.5)
plt.ylim(-3100,0)
plt.xlim(lonsec[50,0],lonsec[750,0])
plt.ylabel('z [m]',fontsize=fs)
ax.set_xticks([-35,-30],['35','30'])
plt.xlabel('Longitude [$^{\circ}$W]',fontsize=fs)
ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[0,1]) # ------------------------ 100 sigma-levels
plt.title('b) '+nbr_levels[1]+r' $s$-levels',fontsize=fs)
lonsec = 0.5*(dat2.lonr[1:,jsec]+dat2.lonr[:-1,jsec])
lonsec = np.tile(lonsec,(dat2.z_w.shape[-1],1)).T
zsec = 0.5*(dat2.z_w[1:,jsec,:]+dat2.z_w[:-1,jsec,:])
ctf = plt.pcolormesh(lonsec,zsec,dat2.dz[1:-1,jsec,:],norm=norm_dz,cmap=cmap_dz)
for k in np.arange(0,dat2.z_w.shape[-1],5):
    plt.plot(dat2.lonr[:,jsec],dat2.z_w[:,jsec,k],'k',lw=lw_c)
plt.fill_between(dat2.lonr[:,jsec],-3100,-dat2.h[:,jsec],fc='lightgray',ec='k',alpha=0.5)
plt.ylim(-3100,0)
plt.xlim(lonsec[50,0],lonsec[750,0])
ax.set_yticks([])
ax.set_xticks([-35,-30],['35','30'])
plt.xlabel('Longitude [$^{\circ}$W]',fontsize=fs)
ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[0,2]) # ------------------------ 200 sigma-levels
plt.title('c) '+nbr_levels[2]+r' $s$-levels',fontsize=fs)
lonsec = 0.5*(dat3.lonr[1:,jsec]+dat3.lonr[:-1,jsec])
lonsec = np.tile(lonsec,(dat3.z_w.shape[-1],1)).T
zsec = 0.5*(dat3.z_w[1:,jsec,:]+dat3.z_w[:-1,jsec,:])
ctf = plt.pcolormesh(lonsec,zsec,dat3.dz[1:-1,jsec,:],norm=norm_dz,cmap=cmap_dz)
for k in np.arange(0,dat3.z_w.shape[-1],5):
    plt.plot(dat3.lonr[:,jsec],dat3.z_w[:,jsec,k],'k',lw=lw_c)
plt.fill_between(dat3.lonr[:,jsec],-3100,-dat3.h[:,jsec],fc='lightgray',ec='k',alpha=0.5)
plt.ylim(-3100,0)
plt.xlim(lonsec[50,0],lonsec[750,0])
ax.set_yticks([])
ax.set_xticks([-35,-30],['35','30'])
plt.xlabel('Longitude [$^{\circ}$W]',fontsize=fs)
ax.tick_params(labelsize=fs)


ax = plt.subplot(gs[1,1]) # ------------------------ colorbar 
cb = plt.colorbar(ctf,cax=ax,orientation='horizontal',ticks=cbticks_dz)
cb.set_label(cblabel_dz,fontsize=fs,labelpad=-87)  #-57)
cb.ax.tick_params(labelsize=fs)


plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/vertical_grid_resolution_Jon.pdf',bbox_inches='tight')




