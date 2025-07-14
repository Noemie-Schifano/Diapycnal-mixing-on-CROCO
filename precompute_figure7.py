'''
NS : Compute statistics (10th and 90th percentiles + median value) of diffusivities :   
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
import dens_po as dens
import obsfit1d as fit
from croco_simulations_jonathan_hist import Croco

# ------------ parameters ------------ 
name_exp_jon      = ['rrexnum50','rrexnum50','rrexnum50','rrexnum100','rrexnum100','rrexnum100','rrexnum200','rrexnum200','rrexnum200']
name_pathdata_jon = ['RREXNUM50_NOFILT_T','RREXNUM50_RSUP5_NOFILT_T','RREXNUM50_RSWENO5_NOFILT_T','RREXNUM100_NOFILT_T',
                                        'RREXNUM100_RSUP5_NOFILT_T','RREXNUM100_RSWENO5_NOFILT_T','RREXNUM200_NOFILT_T',
                                                              'RREXNUM200_RSUP5_NOFILT_T','RREXNUM200_RSWENO5_NOFILT_T']
name_exp_grd_jon  = ['rrex50','rrex50','rrex50','rrex100-up3','rrex100-up5','rrex100-weno5','rrex200-up3','rrex200-up5','rrex200-weno5']
name_nc_jon       = ['rrexnum50-nofilt','rrexnum50-rsup5-nofilt','rrexnum50-rsweno5-nofilt','rrexnum100-nofilt','rrexnum100-rsup5-nofilt',
                                    'rrexnum100-rsweno5-nofilt','rrexnum200-nofilt','rrexnum200-rsup5-nofilt','rrexnum200-rsweno5-nofilt']
name_x_jon        = ['exp50-rsup3','exp50-rsup5','exp50-rsweno5','exp100-rsup3','exp100-rsup5','exp100-rsweno5','exp200-rsup3','exp200-rsup5','exp200-rsweno5']
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
cfr= colors.to_rgba('yellow')    # color of the median value of the effective diffusivity K_eff above the ridge area
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

# --- Save variables for each configuration and for each area (above the ridge and above th abyssal plain ---
# --- Time series of diffusivities sorted as a function of the height above the bottom ---
# --> Effective diffusivity diagnosed online above the ridge area
Keffr_avg50      = [[[] for d in range(nhab)] for exp in range(3)]
Keffr_avg100     = [[[] for d in range(nhab)] for exp in range(3)]
Keffr_avg200     = [[[] for d in range(nhab)] for exp in range(3)]
# --> Effective diffusivity diagnosed online above the abyssal plain area
Keffp_avg50      = [[[] for d in range(nhab)] for exp in range(3)]
Keffp_avg100     = [[[] for d in range(nhab)] for exp in range(3)] 
Keffp_avg200     = [[[] for d in range(nhab)] for exp in range(3)]
# --> Parameterized diffusivity K_KPP (named "AKt" in CROCO), above the ridge area
AKtr_avg50      = [[[] for d in range(nhab)] for exp in range(3)]
AKtr_avg100     = [[[] for d in range(nhab)] for exp in range(3)]
AKtr_avg200     = [[[] for d in range(nhab)] for exp in range(3)]
# --> Parameterized diffusivity K_KPP (named "AKt" in CROCO), above the abyssal plain area
AKtp_avg50      = [[[] for d in range(nhab)] for exp in range(3)]
AKtp_avg100     = [[[] for d in range(nhab)] for exp in range(3)]
AKtp_avg200     = [[[] for d in range(nhab)] for exp in range(3)]
# --- Statistics (10th percentile, 90th percentile and median value) for each bin of height above the bottom
# --> Effective diffusivity diagnosed online above the ridge area
Keffr_all50  = np.zeros((3,3,nhab))
Keffr_all100 = np.zeros((3,3,nhab))
Keffp_all200 = np.zeros((3,3,nhab))
# --> Effective diffusivity diagnosed online above the abyssal plain area
Keffp_all50  = np.zeros((3,3,nhab))
Keffp_all100 = np.zeros((3,3,nhab))
Keffp_all200 = np.zeros((3,3,nhab))
# --> Parameterized diffusivity K_KPP (named "AKt" in CROCO), above the ridge area
AKtr_all50  = np.zeros((3,3,nhab))
AKtr_all100 = np.zeros((3,3,nhab))
AKtr_all200 = np.zeros((3,3,nhab))
# --> Parameterized diffusivity K_KPP (named "AKt" in CROCO), above the abyssal plain area
AKtp_all50  = np.zeros((3,3,nhab))
AKtp_all100 = np.zeros((3,3,nhab))
AKtp_all200 = np.zeros((3,3,nhab))


def define_area(points,lonr,nbr_levels):
    # create an area (0 or 1) of the same dimensions than lonr and the third dimension is 
    #       the number of vertical levels nbr_levels
    # points containes (x,y) coordinates of the limits of the square 
    area = np.zeros((np.shape(lonr)[0],np.shape(lonr)[1],nbr_levels))
    area[points[0]:points[1],points[2]:points[3],:] = 1
    return area        

def plot_k(ax,AKt,hab,kparam,cfparam,cf,cf2,lw,fs,line,column,name_title,number_title,nbr_levels):
    # Bar-plot that show for each bin of height above the bottom, the statistics on diffusiviies:
    #                               (1) parameterized diffusivity K_KPP, called "AKt" in CROCO 
    #                               (2) Effective diffusivity diagnosed online K_eff
    for k in range(nhab-1):
        # k is the index for the "z-direction", not a diffusivity!
        dz = hab[k+1]-hab[k]
        print('hab :',hab[k])
        print('dz  :',dz)
        print('AKt :',AKt[0,k],AKt[1,k],AKt[2,k])
        plt.broken_barh( [(AKt[0,k],AKt[1,k]-AKt[0,k]),
                     (AKt[1,k],AKt[2,k]-AKt[1,k])],
                     (hab[k],dz),
                     facecolors=('none','none'),edgecolors=(cf,cf),linewidth=lw-1)
    # --> legend for the median values
    if line==0:
        if column==0:
            labelmedian = 'median value of $\kappa_{eff}$ above the ridge area'
            labelkpp    = 'median value of $\kappa_{KPP}$ above the ridge area'
        if column==1:
            labelmedian = 'median value of $\kappa_{eff}$ above the abyssal plain area'
            labelkpp    = 'median value of $\kappa_{KPP}$ above the abyssal plain area'
    else: 
        labelmedian = False
        labelkpp    = False
    # --> Highlight the median values in the plot
    plt.plot(AKt[1,:],hab,color=cf2,linewidth=lw+1,label=labelmedian)
    plt.plot(kparam[1,:],hab,color=cfparam,linewidth=lw+1,alpha=0.5,label=labelkpp)
    if line==0:
        plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.385),prop={'size':fs-3},ncol=1)
    plt.xscale('log')
    plt.xlim(1e-6,1)
    if line<2:
        plt.xticks(ticks=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1],labels=None)
        ax.set_xticklabels([])
    else:
        ax.set_xticks([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1],fontsize=fs)
        plt.xlabel('$\kappa_{eff}$ [m$^2$ s$^{-1}$]',fontsize=fs)
    plt.ylim(0,1500)
    plt.ylabel('hab [m]',fontsize=fs)
    plt.title(number_title+name_title,fontsize=fs)
    ax.tick_params(labelsize=fs)


# --- read data ---
print(' ............ time loop ...............................................  ')
for exp in range(len(name_exp_jon)):
    name_exp      = name_exp_jon[exp]                  # name file netcdf
    name_pathdata = name_pathdata_jon[exp]             # folder where are netcdf
    name_exp_grd  = name_exp_grd_jon[exp]              # folder where grid data
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
            dvol   = np.diff(z_w,axis=-1)*dsurf
            dvol_w = np.diff(z_r,axis=-1)*dsurf_w
            tt     = t_nc+t_nc*t
            Keff   = data.get_diffusivity(t,'diffusivity',get_date=False)
            AKt    = data.var['AKt']
            Keff[Keff<=0]=np.nan
            AKt[AKt<=0]=np.nan
            # --- save grid data ---
            if (exp==0)and(tt==0): # ----------------------- for 50 sigma levels
                # define areas 
                [z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
                dvol50   = np.diff(z_w,axis=-1)*dsurf
                dvol_w = np.diff(z_r,axis=-1)*dsurf_w
                h_tile = np.transpose(np.tile(data.h,(50,1,1)),(1,2,0))
                hab50       = h_tile + z_r
            if (exp==3)and(tt==0): # ----------------------- for 100 sigma levels
                # define areas 
                [z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
                dvol100   = np.diff(z_w,axis=-1)*dsurf
                dvol_w = np.diff(z_r,axis=-1)*dsurf_w
                h_tile = np.transpose(np.tile(data.h,(100,1,1)),(1,2,0))
                hab100       = h_tile + z_r
                print('hab',np.shape(hab100))
            if (exp==6)and(tt==0): # ----------------------- for 200 sigma levels
                [z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
                dvol200   = np.diff(z_w,axis=-1)*dsurf
                dvol_w = np.diff(z_r,axis=-1)*dsurf_w
                h_tile = np.transpose(np.tile(data.h,(200,1,1)),(1,2,0))
                hab200       = h_tile + z_r
            # --- Stock diffusivities in height above the bottom space for: 
            #                 (1) parameterized diffusivity K_KPP, called "AKt" in CROCO 
            #                 (2) Effective diffusivity diagnosed online K_eff

            # --> for 50 sigma levels 
            if exp<3:
                hab  = hab50
                dvol = dvol50
                Keffv = Keff 
                AKtv  = AKt 
                # --> above the ridge area
                count = np.zeros(nhab)
                for i in range(points_r[0],points_r[1]+1):
                    for j in range(points_r[2],points_r[3]+1):
                        for d in range(nhab):
                            index = np.where((hab50[i,j,:]>habe[d]) & (hab50[i,j,:]<habe[d+1]))
                            if  index[0].size==0:
                                continue
                            else:
                                for k in range(np.shape(index)[0]):
                                    Keffr_avg50[exp][d].append(Keffv[i,j,index[k][0]])
                                    AKtr_avg50[exp][d].append(AKtv[i,j,index[k][0]])
                # --> above the abyssal plain area
                for i in range(points_p[0],points_p[1]+1):
                    for j in range(points_p[2],points_p[3]+1):
                        count=0
                        for d in range(nhab):
                            index = np.where((hab50[i,j,:]>habe[d]) & (hab50[i,j,:]<habe[d+1]))
                            if  index[0].size==0:
                                continue
                            else:
                                for k in range(np.shape(index)[0]):
                                    Keffp_avg50[exp][d].append(Keffv[i,j,index[k][0]])
                                    AKtp_avg50[exp][d].append(AKtv[i,j,index[k][0]])
            # --> for 100 sigma levels 
            elif (exp>2)and(exp<6):
                hab  = hab100
                dvol = dvol100
                Keffv = Keff
                AKtv  = AKt
                # --> above the ridge area
                for i in range(points_r[0],points_r[1]+1):
                    for j in range(points_r[2],points_r[3]+1):
                        for d in range(nhab):
                            index = np.where((hab100[i,j,:]>habe[d]) & (hab100[i,j,:]<habe[d+1]))
                            if  index[0].size==0:
                                continue
                            else:
                                for k in range(np.shape(index)[0]):
                                    Keffr_avg100[exp-3][d].append(Keffv[i,j,index[k][0]])
                                    AKtr_avg100[exp-3][d].append(AKtv[i,j,index[k][0]])
                # --> above the abyssal plain area
                for i in range(points_p[0],points_p[1]+1):
                    for j in range(points_p[2],points_p[3]+1):
                        count=0
                        for d in range(nhab):
                            index = np.where((hab100[i,j,:]>habe[d]) & (hab100[i,j,:]<habe[d+1]))
                            if  index[0].size==0:
                                continue
                            else:
                                for k in range(np.shape(index)[0]):
                                    Keffp_avg100[exp-3][d].append(Keffv[i,j,index[k][0]])
                                    AKtp_avg100[exp-3][d].append(AKtv[i,j,index[k][0]])
            # --> for 200 sigma levels 
            else:
                hab  = hab200
                dvol = dvol200
                Keffv = Keff
                AKtv  = AKt
                # --> above the ridge area
                for i in range(points_r[0],points_r[1]+1):
                    for j in range(points_r[2],points_r[3]+1):
                        for d in range(nhab):
                            index = np.where((hab200[i,j,:]>habe[d]) & (hab200[i,j,:]<habe[d+1]))
                            if  index[0].size==0:
                                continue
                            else:
                                for k in range(np.shape(index)[0]):
                                    Keffr_avg200[exp-6][d].append(Keffv[i,j,index[k][0]])
                                    AKtr_avg200[exp-6][d].append(AKtv[i,j,index[k][0]])
                # --> above the abyssal plain area
                for i in range(points_p[0],points_p[1]+1):
                    for j in range(points_p[2],points_p[3]+1):
                        for d in range(nhab):
                            index = np.where((hab200[i,j,:]>habe[d]) & (hab200[i,j,:]<habe[d+1]))
                            if  index[0].size==0:
                                continue
                            else:
                                for k in range(np.shape(index)[0]):
                                    Keffp_avg200[exp-6][d].append(Keffv[i,j,index[k][0]])
                                    AKtp_avg200[exp-6][d].append(AKtv[i,j,index[k][0]])

# --- For each configuration, compute for each bin of height above the bottom: the 10th and 90th percentiles + the median value
for exp in range(3):
    for d in range(nhab):
        # --> Effective diffusivity diagnosed online, above the ridge area
        Keffr_all50[exp,:,d] = np.nanpercentile(np.ravel(Keffr_avg50[exp][d]),[10,50,90],axis=0)
        Keffr_all100[exp,:,d] = np.nanpercentile(np.ravel(Keffr_avg100[exp][d]),[10,50,90],axis=0)
        Keffr_all200[exp,:,d] = np.nanpercentile(np.ravel(Keffr_avg200[exp][d]),[10,50,90],axis=0)
        # --> Effective diffusivity diagnosed online, above the abyssal plain area
        Keffp_all50[exp,:,d] = np.nanpercentile(np.ravel(Keffp_avg50[exp][d]),[10,50,90],axis=0)
        Keffp_all100[exp,:,d] = np.nanpercentile(np.ravel(Keffp_avg100[exp][d]),[10,50,90],axis=0)
        Keffp_all200[exp,:,d] = np.nanpercentile(np.ravel(Keffp_avg200[exp][d]),[10,50,90],axis=0)
        # --> Parameterized diffusivity K_KPP (named "AKt" in CROCO), above the ridge area
        AKtr_all50[exp,:,d] = np.nanpercentile(np.ravel(AKtr_avg50[exp][d]),[10,50,90],axis=0)
        AKtr_all100[exp,:,d] = np.nanpercentile(np.ravel(AKtr_avg100[exp][d]),[10,50,90],axis=0)
        AKtr_all200[exp,:,d] = np.nanpercentile(np.ravel(AKtr_avg200[exp][d]),[10,50,90],axis=0)
        # --> Parameterized diffusivity K_KPP (named "AKt" in CROCO), above the abyssal plain area
        AKtp_all50[exp,:,d] = np.nanpercentile(np.ravel(AKtp_avg50[exp][d]),[10,50,90],axis=0)
        AKtp_all100[exp,:,d] = np.nanpercentile(np.ravel(AKtp_avg100[exp][d]),[10,50,90],axis=0)
        AKtp_all200[exp,:,d] = np.nanpercentile(np.ravel(AKtp_avg200[exp][d]),[10,50,90],axis=0)


print("--- Finally make plot ---")
# --> Organize index to make the plot in a more friendly way
rsup3   = [0,3,6]
rsup5   = [1,4,7]
rsweno5 = [2,5,8]
scheme  = [rsup3,rsup5,rsweno5]
name_scheme = ['RSUP3','RSUP5','RSWENO5']


# --- make figure ---
for adv in range(len(scheme)):
    fig = plt.figure(figsize=(10,10))
    gs  = gridspec.GridSpec(3,2,wspace=0.4) 
    fig.suptitle('Ridge                                                                                 Abyssal plain', horizontalalignment='center', fontsize=fs)

    ax00 = plt.subplot(gs[0,0]) # ---> ridge, sigma = 50
    exp = scheme[adv][0]
    k   = Keffr_all50[exp,:,:]
    kparam=AKtr_all50[exp,:,:]
    hab = habc
    plot_k(ax00,k,hab,kparam,cfparam,cf,cfr,lw,fs,0,0,name_x_jon[exp],'a) ',nhab)

    ax10 = plt.subplot(gs[1,0]) # ---> ridge, sigma = 100
    exp = scheme[adv][1]-3
    k   = Keffr_all100[exp,:,:]
    kparam=AKtr_all100[exp,:,:]
    hab = habc
    plot_k(ax10,k,hab,kparam,cfparam,cf,cfr,lw,fs,1,0,name_x_jon[exp+3],'b) ',nhab)
    
    ax20 = plt.subplot(gs[2,0]) # ---> ridge, sigma = 200
    exp = scheme[adv][2]-6
    k = Keffr_all200[exp,:,:]
    kparam=AKtr_all200[exp,:,:]
    hab = habc
    plot_k(ax20,k,hab,kparam,cfparam,cf,cfr,lw,fs,2,0,name_x_jon[exp+6],'c) ',nhab)    

    ax01 = plt.subplot(gs[0,1]) # ---> abyssal plain, sigma = 50
    exp = scheme[adv][0]
    k = Keffp_all50[exp,:,:]
    kparam=AKtp_all50[exp,:,:]
    hab = habc
    plot_k(ax01,k,hab,kparam,cfparam,cf,cfp,lw,fs,0,1,name_x_jon[exp],'d) ',nhab)
    

    ax11 = plt.subplot(gs[1,1]) # ---> abyssal plain, sigma = 100
    exp = scheme[adv][1]-3
    k = Keffp_all100[exp,:,:]
    kparam=AKtp_all100[exp,:,:]
    hab = habc
    plot_k(ax11,k,hab,kparam,cfparam,cf,cfp,lw,fs,1,1,name_x_jon[exp+3],'e) ',nhab)
  
    ax21 = plt.subplot(gs[2,1]) # ---> abyssal plain, sigma = 200
    exp = scheme[adv][2]-6
    k = Keffp_all200[exp,:,:]
    kparam=AKtp_all200[exp,:,:]
    hab = habc
    plot_k(ax21,k,hab,kparam,cfparam,cf,cfp,lw,fs,2,1,name_x_jon[exp+6],'f) ',nhab)
    plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/All_simu/Keff_avg_ridge_abyssal_plain_Jon_hab_'+name_scheme[adv]+'.pdf',bbox_inches='tight')
    plt.close()



# ----------- save parameters in netcdf file ------------ 
print(' ... save in netcdf file ... ')
nc = Dataset(file_diag,'w')
# --- define dimensions ---
nc.createDimension('nperc',3)              # Statistics: 10th and 90th percentiles + median value
nc.createDimension('nhab',nhab)            # Number of bins of height above the bottom
nc.createDimension('nscheme',len(scheme))  # Number of advective scheme combinations: RSUP3, RSUP5 and WENO5
# --- create variables ---
# --> Bins of height above the bottom used
nc.createVariable('hab','f',('nhab'))
# --> Effective diffusivity diagnosed online, above the ridge area
var = nc.createVariable('Keffr_all50','f',('nscheme','nperc','nhab'))
var.long_name = 'Keff, ridge, 50 simga-levels, percentiles [10,50,90]'
var = nc.createVariable('Keffr_all100','f',('nscheme','nperc','nhab'))
var.long_name = 'Keff, ridge, 100 simga-levels, percentiles [10,50,90]'
var = nc.createVariable('Keffr_all200','f',('nscheme','nperc','nhab'))
var.long_name = 'Keff, ridge, 200 simga-levels, percentiles [10,50,90]'
# --> Effective diffusivity diagnosed online, above the abyssal plain area
var = nc.createVariable('Keffp_all50','f',('nscheme','nperc','nhab'))
var.long_name = 'Keff, plain, 50 simga-levels, percentiles [10,50,90]'
var = nc.createVariable('Keffp_all100','f',('nscheme','nperc','nhab'))
var.long_name = 'Keff, plain, 100 simga-levels, percentiles [10,50,90]'
var = nc.createVariable('Keffp_all200','f',('nscheme','nperc','nhab'))
var.long_name = 'Keff, plain, 200 simga-levels, percentiles [10,50,90]'
# --> Parameterized diffusivity K_KPP (named "AKt" in CROCO), above the ridge area
var = nc.createVariable('AKtr_all50','f',('nscheme','nperc','nhab'))
var.long_name = 'AKt, ridge, 50 simga-levels, percentiles [10,50,90]'
var = nc.createVariable('AKtr_all100','f',('nscheme','nperc','nhab'))
var.long_name = 'AKt, ridge, 100 simga-levels, percentiles [10,50,90]'
var = nc.createVariable('AKtr_all200','f',('nscheme','nperc','nhab'))
var.long_name = 'AKt, ridge, 200 simga-levels, percentiles [10,50,90]'
# --> Parameterized diffusivity K_KPP (named "AKt" in CROCO), above the abyssal plain area
var = nc.createVariable('AKtp_all50','f',('nscheme','nperc','nhab'))
var.long_name = 'AKt, plain, 50 simga-levels, percentiles [10,50,90]'
var = nc.createVariable('AKtp_all100','f',('nscheme','nperc','nhab'))
var.long_name = 'AKt, plain, 100 simga-levels, percentiles [10,50,90]'
var = nc.createVariable('AKtp_all200','f',('nscheme','nperc','nhab'))
var.long_name = 'AKt, plain, 200 simga-levels, percentiles [10,50,90]'
# --- save variables ---
# --> Bins of height above the bottom used
nc.variables['hab'][:]     = habc
# --> Effective diffusivity diagnosed online, above the ridge area
nc.variables['Keffr_all50'][:]     = Keffr_all50
nc.variables['Keffr_all100'][:]    = Keffr_all100
nc.variables['Keffr_all200'][:]    = Keffr_all200
# --> Effective diffusivity diagnosed online, above the abyssal plain area
nc.variables['Keffp_all50'][:]     = Keffp_all50
nc.variables['Keffp_all100'][:]    = Keffp_all100
nc.variables['Keffp_all200'][:]    = Keffp_all200
# --> Parameterized diffusivity K_KPP (named "AKt" in CROCO), above the ridge area
nc.variables['AKtr_all50'][:]      = AKtr_all50
nc.variables['AKtr_all100'][:]     = AKtr_all100
nc.variables['AKtr_all200'][:]     = AKtr_all200
# --> Parameterized diffusivity K_KPP (named "AKt" in CROCO), above the abyssal plain area
nc.variables['AKtp_all50'][:]      = AKtp_all50
nc.variables['AKtp_all100'][:]     = AKtp_all100
nc.variables['AKtp_all200'][:]     = AKtp_all200

nc.close()


