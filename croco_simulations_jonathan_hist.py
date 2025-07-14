'''
NS : class with functions associated to croco simulations 
''' 

import numpy as np
from netCDF4 import Dataset 
import calendar as cal
import time as time
import datetime as datetime
import sys
sys.path.append('/home/datawork-lops-rrex/nschifan/Python_Modules_p3/')
import R_tools_fort as toolsF

path_data        = '/home/datawork-lops-megatl/RREXNUM/' # patheway to outputs of CROCO simulation
path_data_100up3 = '/home/datawork-lops-rrex/jgula/'     # adding simulation
path_data_grd    = '/home/datawork-lops-rrex/nschifan/'  # patheway to grid used in the simulation

class Croco(object): 
    def __init__(self,*args,**kwargs):
        print(' ------------ initiate a class with croco simulation ------------ ') 
        self.name_exp    = args[0] 
        self.nbr_levels  = args[1]
        self.time        = args[2]
        self.name_exp_grd= args[3]
        self.name_pathdata=args[4]
        if self.name_pathdata == 'RREXNUM100_UP3':
            self.path_data = path_data_100up3+self.name_pathdata+'/HIS/' 
        else:
            self.path_data   = path_data+self.name_pathdata+'/HIS/'
        self.nfpf = 2  # number of frames per file   
        self.fs   = 1  # file step in days                  
        self.frame_index = []
        for t in range(self.nfpf):
            self.frame_index.append(t)
        return  

    def get_grid(self):
        # --- Read the grid file ---
        print(' ... get grid variables ... ') 
        if (self.name_pathdata=='RREXNUMS200_RSUP5_NOFILT_T')or(self.name_pathdata=='RREXNUMSB200_RSUP5_NOFILT_T'):
            nc = Dataset('/home/datawork-lops-rrex/jgula/INIT_RREXNUMS200/GRD/rrexnums_grd.nc','r')
            print('--> smooth topo grid')
        else:
            nc = Dataset('/home/datawork-lops-rrex/jgula/INIT_RREXNUM'+self.nbr_levels+'/GRD/rrexnum_grd.nc','r')
        self.lonr = np.asfortranarray(nc.variables['lon_rho'][:].T)          
        self.latr = np.asfortranarray(nc.variables['lat_rho'][:].T) 
        self.h    = np.asfortranarray(nc.variables['h'][:].T)
        self.f    = np.asfortranarray(nc.variables['f'][:].T)
        self.pm   = np.asfortranarray(nc.variables['pm'][:].T) 
        self.pn   = np.asfortranarray(nc.variables['pn'][:].T)
        self.angle   = np.asfortranarray(nc.variables['angle'][:].T)
        nc.close()
        # ---> Extract variables that are in daily average outputs
        #nc1= Dataset(path_data+self.name_pathdata+'/HIS/'+self.name_exp+'_avg.00000.nc','r')
        nc1 = Dataset(self.path_data+self.name_exp+'_avg.00000.nc','r')
        self.Cs_r = nc1.Cs_r
        self.Cs_w = nc1.Cs_w
        self.hc   = nc1.hc
        nc1.close()  
        return  

    def get_outputs(self,*args,**kwargs): 
        t = args[0] 
        print('===================================================') 
        print(' ... get output variables at time index %.4i ... '%t) 
        print('===================================================') 
        print(self.path_data)
        print(self.name_exp)
        print(self.time)
        print(' ----'+ self.path_data+self.name_exp+'_his.000'+self.time+'.nc' +' ----')
        print( ' --> temps:'+str(t))
        var_list = args[1]
        nvar = len(var_list) 
        self.var = {} 
        nc = Dataset(self.path_data+self.name_exp+'_his.000'+self.time+'.nc','r') 
        for var_name in var_list:
            print('     --> ',var_name) 
            self.var[var_name] = np.asfortranarray(nc.variables[var_name][self.frame_index[t]].T) 
        self.rho0 = nc.rho0
        self.hc   = nc.hc
        nc.close()     
        return  

    def get_diffusivity(self,*args,**kwargs):
        # !!! Diffusivity is one time-step shifted forward compared to CROCO snapshots
        t = args[0]
        print(self.path_data)
        print(self.name_exp)
        print(self.time)
        print(' ----'+ self.path_data+self.name_exp+'_diags_wdia_his.000'+self.time+'.nc' +' ----')
        print( ' --> temps:'+str(t))
        var_name = args[1]
        if t==0:
            nc    = Dataset(self.path_data+self.name_exp+'_diags_wdia.000'+str(int(int(self.time)-2))+'.nc','r') 
            tdiff = 1
        else:
            nc    = Dataset(self.path_data+self.name_exp+'_diags_wdia.000'+self.time+'.nc','r')
            tdiff = 0
        diff = np.asfortranarray(nc.variables[var_name][self.frame_index[tdiff]].T)
        nc.close() 
        # --- mask diffusivity values that are due to the limits of the method ---
        #     -> when the stratification is nul you divide by a value close to zeros to obtain the diffusivity 
        #     -> this diagnose of diffusivity should be used cautiously when the stratification is low ... 
        diff[diff>10]=np.nan
        # --> diffusivity can not be read for the first time-step
        if (self.time=='20')and(t==0):
            diff_f = np.nan*np.copy(diff)
        else:
            diff_f = diff 
        return diff_f  

    def get_zlevs(self):
        # --- compute the depth at rho points (z_r) and w points (z_w) from: 
        #     --> the free surface zeta 
        #     --> hc: transition depth between the horizontal surfacelevels and the bottom terrain following levels
        #     --> vertical stretching curves at rho points (Cs_r) and at w points (Cs_w)
        print('     --> get vertical levels ')  
        [self.z_r,self.z_w] = toolsF.zlevs(self.h,self.var['zeta'],self.hc,self.Cs_r,self.Cs_w)
        return 
