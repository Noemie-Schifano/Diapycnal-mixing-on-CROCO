'''
NS O9/11/2023: adapt CV code to mine
CV 2020/10/19: extract diffusivity coefficient AKt in CROCO run at some places, mean, std dev, ... 
''' 
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import scipy.stats as stats
import scipy.interpolate as itp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from netCDF4 import Dataset
import sys
sys.path.append('/home2/datahome/nschifan/Python_Modules_p3/')
import R_tools as tools
from croco_simulations_jonathan_hist import Croco
from distance_sphere_matproof import dist_sphere  

# ------------ parameters ------------ 
name_exp_jon      = ['rrexnum50','rrexnum50','rrexnum50','rrexnum100','rrexnum100','rrexnum100','rrexnum200','rrexnum200','rrexnum200']
name_pathdata_jon = ['RREXNUM50_NOFILT_T','RREXNUM50_RSUP5_NOFILT_T','RREXNUM50_RSWENO5_NOFILT_T','RREXNUM100_NOFILT_T',
                                        'RREXNUM100_RSUP5_NOFILT_T','RREXNUM100_RSWENO5_NOFILT_T','RREXNUM200_NOFILT_T',
                                                              'RREXNUM200_RSUP5_NOFILT_T','RREXNUM200_RSWENO5_NOFILT_T']
name_exp_grd_jon  = ['rrex50','rrex50','rrex50','rrex100-up3','rrex100-up5','rrex100-weno5','rrex200-up3','rrex200-up5','rrex200-up5']
nbr_levels_jon    = ['50','50','50','100','100','100','200','200','200']
name_nc_jon       = ['rrexnum50-rsup3','rrexnum50-rsup5','rrexnum50-rsweno5','rrexnum100-rsup3','rrexnum100-rsup5','rrexnum100-rsweno5','rrexnum200-rsup3','rrexnum200-rsup5','rrexnum200-rsweno5']
time              = [ '20','22','24','26','28','30','32','34','36','38','40',
                           '42','44','46','48','50','52','54','56','58','60',
                                     '62','64','66','68','70','72','74','76']
ndfiles           = 2
var_list          = ['zeta','AKt']
# --> Choose the simulation
exp = 7
name_exp      = name_exp_jon[exp]
name_pathdata = name_pathdata_jon[exp]
name_exp_grd  = name_exp_grd_jon[exp] 
nbr_levels    = nbr_levels_jon[exp]
# --> Patheway and name of output NetCDF file
file_out = '/home/datawork-lops-rrex/nschifan/DIAGS/'+name_nc_jon[exp]+'_Keff_AKt_rrex-middleline.nc'

# - coordinates of the edges of the segments along which data will be interpolated, like CTD stations - 
lonctd = [-37.038,  -31.270, -27.340] # following rrex17  
latctd = [59.428,  58.846,   56.930]  

file_out = '../DIAGS/'+name_nc_jon[exp]+'_Keff_AKt_rrex-middleline.nc' 

# --> binning in depth space
dz      = 5. # [m] grid resolution 
zz      = np.arange(-4000,dz,dz)
hab     = abs(zz[::-1])
nz      = zz.shape[0]
npts_line = 824

# --> create variables to save in NetCDF
AKt_z   = np.zeros((len(time)*ndfiles,npts_line,nz))
AKt_hab = np.zeros((len(time)*ndfiles,npts_line,nz))
Keff_z   = np.zeros((len(time)*ndfiles,npts_line,nz))
Keff_hab = np.zeros((len(time)*ndfiles,npts_line,nz))


# ------------ read data ------------ 
tt=0
for t_nc in range(len(time)):
    print(' history file '+time[t_nc])
    dat1 = Croco(name_exp,nbr_levels,time[t_nc],name_exp_grd,name_pathdata)
    dat1.get_grid()
    if t_nc==0:
        # ------------ processing ------------ 
        nctd   = len(lonctd) 
        ddeg   = abs(dat1.lonr[0,1]-dat1.lonr[0,0]) # model resolution 
        npts   = np.zeros(nctd-1) # number of points per segment between stations 
        for i in range(nctd-1):
            npts[i] = np.ceil(np.max((abs(lonctd[i+1]-lonctd[i])/ddeg,
                                      abs(latctd[i+1]-latctd[i])/ddeg)))
        lonitp = np.concatenate([np.linspace(lonctd[i],lonctd[i+1],int(npts[i]))
                                    for i in range(nctd-1)])
        latitp = np.concatenate([np.linspace(latctd[i],latctd[i+1],int(npts[i]))
                                    for i in range(nctd-1)])
        ij = [] 
        for s in range(len(lonitp)): 
            dist = dist_sphere(latitp[s],lonitp[s],dat1.latr,dat1.lonr) 
            idist = np.nanargmin(dist)
            if idist not in ij: ij.append(idist) 
        npts_line = len(ij) 
        print('dim ',npts_line)
        ii,jj = np.unravel_index(ij,dat1.latr.shape)  

    for t in range(ndfiles): 
        print('time :',tt)
        dat1.get_outputs(t,var_list)
        dat1.get_zlevs()
        for s in range(npts_line):
            # do not use bottom-most AKt that is ==0  
            fitp = itp.interp1d(dat1.z_w[ii[s],jj[s],1:],dat1.var['AKt'][ii[s],jj[s],1:],bounds_error=False) 
            AKt_z[tt,s,:] = fitp(zz)   
            fitp = itp.interp1d(dat1.h[ii[s],jj[s]]+dat1.z_w[ii[s],jj[s],1:],dat1.var['AKt'][ii[s],jj[s],1:],bounds_error=False) 
            AKt_hab[tt,s,:] = fitp(hab)   
            # --- Keff ---
            Keff = dat1.get_diffusivity(t,'diffusivity',get_date=False)
            fitp = itp.interp1d(dat1.z_r[ii[s],jj[s],1:],Keff[ii[s],jj[s],1:],bounds_error=False)
            Keff_z[tt,s,:] = fitp(zz)
            fitp = itp.interp1d(dat1.h[ii[s],jj[s]]+dat1.z_r[ii[s],jj[s],1:],Keff[ii[s],jj[s],1:],bounds_error=False)
            Keff_hab[tt,s,:] = fitp(hab)
    tt+=1
     
print('===================================================') 
print(' ... save in netcdf file ... ') 
print('===================================================') 
nc = Dataset(file_out,'w') 
nc.createDimension('time',len(time)*ndfiles) 
nc.createDimension('station',npts_line) 
nc.createDimension('nz',nz) 
nc.createVariable('lon','f',('station',))  
nc.createVariable('lat','f',('station',))  
nc.createVariable('ii','f',('station',))  
nc.createVariable('jj','f',('station',))  
nc.createVariable('h','f',('station',)) 
nc.createVariable('z','f',('nz',))  
nc.createVariable('hab','f',('nz',))  
nc.createVariable('AKt_z','f',('time','station','nz')) 
nc.createVariable('AKt_hab','f',('time','station','nz')) 
nc.createVariable('Keff_z','f',('time','station','nz'))
nc.createVariable('Keff_hab','f',('time','station','nz'))
nc.variables['lon'][:]     = dat1.lonr[ii,jj] 
nc.variables['lat'][:]     = dat1.latr[ii,jj] 
nc.variables['ii'][:]      = ii
nc.variables['jj'][:]      = jj
nc.variables['h'][:]       = dat1.h[ii,jj] 
nc.variables['z'][:]       = zz 
nc.variables['hab'][:]     = hab 
nc.variables['AKt_z'][:]   = AKt_z 
nc.variables['AKt_hab'][:] = AKt_hab 
nc.variables['Keff_z'][:]   = Keff_z
nc.variables['Keff_hab'][:] = Keff_hab
nc.close() 

    
