''' 
CV 2019/05/03: create a class with function associated to RREX 2015 and 2017 
CV 2019/05/10: There are two ways to use it: 
                - OPTION 1: provide a list of VMP stations and get corresponding CTD stations. 
                - OPTION 2: provide a list of CTD stations and get corresponding VMP stations, if any. 
CV 2020/10/09: Adds the OVIDE 2008 VMP data 
'''
import numpy as np
from netCDF4 import Dataset
import scipy.io as sio
import scipy.stats as stats
import scipy.interpolate as itp 
import matplotlib.pyplot as plt 
import gsw as gsw
import time as tm
import datetime as dt
#import sys
#sys.path.append('../../../Python_miscellaneous/')
from pressure_to_depth import pres2depth
from distance_sphere_matproof import dist_sphere
import warnings; warnings.filterwarnings('ignore')


class RREX_class(object):
    def __init__(self,**kwargs):
        self.nanval = 999 # 'nan' value 
        # - read parameters, 2nd arg is defaults value if none is given -
        self.year         = kwargs.get('year',2015)
        print(' ====================================================================================== ')
        print(' ================== initiate a class with RREX '+self.year+' data and methods ================== ')
        print(' ====================================================================================== ')
        self.list_vmp     = kwargs.get('list_vmp',[]) 
        self.list_ctd     = kwargs.get('list_ctd',[]) 
        # - set file names -
        path_data0 = '/data0/project/meddle/cvic/datacvic/' 
        if self.year == '2015':  # load all profiles if no list is provided   
            self.file_pos         = '/home/datawork-lops-osi/cvic/Data_obs/rrex15_pos_VMP'
            self.file_vmp2ctd     = '/home/datawork-lops-osi/cvic/Data_obs/rrex15_vmp_ctd_number.txt'
            self.file_vmp_ind     = path_data0+'RREX/VMP_RREX2015/diss_RREX001.mat' # 1-58
            self.file_vmp         = '/home/datawork-lops-osi/cvic/Data_obs/rrex15_vmp_new.nc'  
            self.file_conv        = '/Users/cv1m15/Data/rrex15_energy_conversion_newmethod_k1eq4.nc'
            self.file_mld         = '/Users/cv1m15/Data/rrex15_vmp_mld.nc'
            self.file_ctd         = '/home/datawork-lops-osi/cvic/Data_obs/rr15_DEPH.nc'
            self.file_ladcp       = '/Users/cv1m15/Data/rrex15_l2_s5_b10_grouped.mat'
            self.list_suspicious  = kwargs.get('list_suspicious',[56]) # potentially dodgy VMP values   
            self.file_aviso       = '/Users/cv1m15/Data/dt_global_allsat_phy_l4_20150601_20170110.nc' 
            self.file_thorpe      = '/Users/cv1m15/Data/rrex15_thorpe_' # see get_thorpe 
            self.file_thorpe_vmp  = '/Users/cv1m15/Data/rrex15_thorpe_from_vmp.nc' # see get_thorpe_vmp 
        elif self.year == '2017': 
            self.file_pos         = '/home/datawork-lops-osi/cvic/Data_obs/rrex17_pos_VMP'
            self.file_vmp2ctd     = '/home/datawork-lops-osi/cvic/Data_obs/rrex17_vmp_ctd_number.txt'
            self.file_vmp_ind     = path_data0+'RREX/VMP_RREX2017/diss_RREX2017_001.mat' # 1-78
            self.file_vmp         = '/home/datawork-lops-osi/cvic/Data_obs/rrex17_vmp_new.nc' 
            self.file_conv        = '/Users/cv1m15/Data/rrex17_energy_conversion_newmethod_k1eq4.nc'
            self.file_mld         = '/Users/cv1m15/Data/rrex17_vmp_mld.nc'
            self.file_ctd         = '/home/datawork-lops-osi/cvic/Data_obs/rr17_DEPH.nc'
            self.file_ladcp       = '/Users/cv1m15/Data/rrex17_l2_s5_b16_grouped.nc' # /!\ netcdf 
            self.list_suspicious  = kwargs.get('list_suspicious',[35]) 
            self.file_aviso       = '/Users/cv1m15/Data/dt_global_allsat_phy_l4_20170831_20180516.nc'
            self.file_thorpe      = '/Users/cv1m15/Data/rrex17_thorpe_' # see get_thorpe 
        elif self.year == '2008': # OVIDE VMP data alone 
            self.file_vmp         = '/home/datawork-lops-osi/cvic/Data_obs/diss_Ovide08_001.mat '
            # --- list mat variables
            print(' --- MAT VARIABLES ---')
            print(sio.whosmat('/home/datawork-lops-osi/cvic/Data_obs/diss_Ovide08_001.mat'))
            print(self.file_vmp)
            self.list_suspicious  = kwargs.get('list_suspicious',self.list_vmp) 
            self.file_ctd         = '/home/datawork-lops-osi/cvic/Data_obs/ovid08_prs.nc'
        # - miscellaneous -  
        self.file_topo_lr   = '/home/datawork-lops-osi/cvic/Data_obs/ETOPO2v2c_f4.nc'
        self.file_topo_hr   = '/Users/cv1m15/Data/reykjanes-ridge.nc' # chunk of SRTM30-PLUS 
        self.file_topo_vhr  = '/Users/cv1m15/Data/topo15_NorthAtl.nc' # chunk of SRTM15-PLUS 
        ###################################################################
        # OPTION 1:  assumes a list of VMP stations has been provided          
        ###################################################################
        if len(self.list_ctd)==0: 
            if len(self.list_vmp)==0:
                exit('ERROR: provide either a list of VMP or CTD stations') 
            else:
                self.nvmp = len(self.list_vmp)
                if self.year in ['2015','2017']: # RREX cruises  
                    self.nctd = self.nvmp # get only corresponding stations 
                    print(' ... RREX '+self.year+': get indices of CTD corresponding to VMP casts ... ')
                    ff       = open(self.file_vmp2ctd,'r')
                    lines    = ff.readlines()
                    npro     = len(lines)-1 # first line is text
                    list_ctd = []
                    for ind_vmp in self.list_vmp: 
                        for ll in np.arange(1,npro+1):
                            col = lines[ll].split()
                            if int(col[0])==ind_vmp: 
                                list_ctd.append(int(col[1]))
                    self.list_ctd = list_ctd
                    # NB: list_ctd contains indices! (Python's indexing starts at 0) 
                elif self.year == '2008': 
                    print(' ... OVIDE '+self.year+' ... ')
                    self.nctd = self.nvmp # get corresponding stations later  
                                                   
                            
        ###################################################################
        # OPTION 2:  assumes a list of CTD stations has been provided          
        ###################################################################
        else:                     
            self.nctd = len(self.list_ctd) 
            print(' ... RREX '+self.year+': get indices of VMP corresponding to CTD casts, if any ... ')
            ff       = open(self.file_vmp2ctd,'r')
            lines    = ff.readlines()
            npro     = len(lines)-1 # first line is text
            list_vmp = [] 
            list_all_ctd = []
            list_all_vmp = []
            for ll in np.arange(1,npro+1):
                list_all_vmp.append(int(lines[ll].split()[0]))
                list_all_ctd.append(int(lines[ll].split()[1]))
            for j in self.list_ctd:
                if j in list_all_ctd:
                    list_vmp.append(list_all_vmp[list_all_ctd.index(j)]) 
                else: 
                    list_vmp.append(self.nanval) 
            self.list_vmp = list_vmp
            self.nvmp = len(self.list_vmp) # takes also into account self.nanval 
        ###################################################################
        # END OF OPTIONS: COMMON PROCESS  
        ###################################################################
        if self.year in ['2015','2017']: 
            print(' ... RREX '+self.year+': get longitude, latitude and seafloor depth at VMP stations ... ')  
            lonvmp = []; latvmp = []; sfdvmp = []
            ff     = open(self.file_pos,'r')
            lines  = ff.readlines()
            for ll in self.list_vmp:
                if ll!=self.nanval:
                    col = lines[ll].split() 
                    lonvmp.append(float(col[0])-float(col[1])/60.) # minus sign because westward  
                    latvmp.append(float(col[2])+float(col[3])/60.)
                    sfdvmp.append(-float(col[4]))
                else: 
                    lonvmp.append(self.nanval) 
                    latvmp.append(self.nanval) 
                    sfdvmp.append(self.nanval) 
            self.lonvmp = np.asarray(lonvmp)
            self.latvmp = np.asarray(latvmp)
            self.sfdvmp = np.asarray(sfdvmp)
            print(' ... RREX '+self.year+': get longitude, latitude and seafloor depth at CTD stations ... ')  
            nc = Dataset(self.file_ctd,'r') 
            self.latctd = nc.variables['LATITUDE'][self.list_ctd] 
            self.lonctd = nc.variables['LONGITUDE'][self.list_ctd] 
            self.sfdctd = -nc.variables['BOTTOM_DEPTH'][self.list_ctd] 
            self.julctd = nc.variables['JULD'][self.list_ctd] 
            dd          = nc.variables['STATION_DATE_BEGIN'][self.list_ctd] 
            enc = 'utf-8' # encoding to convert bytes to str (Python3) 
            self.datectd = []
            for i in range(self.nctd):
                self.datectd.append(str(dd[i,0],enc)+str(dd[i,1],enc)+str(dd[i,2],enc)+str(dd[i,3],enc)+ # year
                    str(dd[i,4],enc)+str(dd[i,5],enc)+  # month 
                    str(dd[i,6],enc)+str(dd[i,7],enc)+  # day 
                    str(dd[i,8],enc)+str(dd[i,9],enc)+  # hour
                    str(dd[i,10],enc)+str(dd[i,11],enc)+  # min
                    str(dd[i,12],enc)+str(dd[i,13],enc))  # sec
    
        elif self.year == '2008': 
            print(' ... OVIDE '+self.year+': get longitude, latitude at VMP stations ... ')  
            lonvmp = []; latvmp = [] 
            m1 = sio.loadmat('/home/datawork-lops-osi/cvic/Data_obs/diss_Ovide08_001.mat',squeeze_me=True)
            for i in range(self.nvmp):
                print(self.file_vmp[:-7]+'%.2i.mat'%(self.list_vmp[i]+1))
                mat = sio.loadmat(self.file_vmp[:-7]+'%.2i.mat'%(self.list_vmp[i]+1),squeeze_me=True) 
                lonvmp.append(mat['lon_vmp'])  
                latvmp.append(mat['lat_vmp'])  
            self.lonvmp = np.asarray(lonvmp)
            self.latvmp = np.asarray(latvmp)

            # - get corresponding CTD stations -
            nc = Dataset(self.file_ctd,'r') 
            latctd = 0.5*(nc.variables['LATITUDE_BEGIN'][:] +nc.variables['LATITUDE_END'][:]) 
            lonctd = 0.5*(nc.variables['LONGITUDE_BEGIN'][:]+nc.variables['LONGITUDE_END'][:]) 
            list_ctd = [] 
            for i in range(self.nvmp): 
                dist_vmp_ctd = dist_sphere(self.latvmp[i],self.lonvmp[i],latctd,lonctd) 
                list_ctd.append(np.nanargmin(dist_vmp_ctd))  
            self.list_ctd = list_ctd 
            self.latctd   = np.asarray(latctd[list_ctd])  
            self.lonctd   = np.asarray(lonctd[list_ctd])  
            self.sfdctd   = nc.variables['BOTTOM_DEPTH'][self.list_ctd].data 
            nc.close() 
            self.sfdvmp   = 0*self.sfdctd # no information in file_vmp   

        ###################################################################
        # END OF COMMON PROCESS 
        ###################################################################
        print('     --> Quick check for consistency: ') 
        print('     VMP and CTD indices, longitude, latitude, seafloor depth, distance between VMP and CTD casts ') 
        for i in range(self.nvmp): 
            print('   %.3i %.3i | %.3f %.3f | %.3f %.3f | %.4i  %.4i | %.3f km'
                  %(self.list_vmp[i],self.list_ctd[i],
                    self.lonvmp[i],self.lonctd[i],
                    self.latvmp[i],self.latctd[i],
                    self.sfdvmp[i],self.sfdctd[i],   
                    dist_sphere(self.latvmp[i],self.lonvmp[i],self.latctd[i],self.lonctd[i])*1e-3))

    def get_dist(self): 
        print(' ... get distance between stations ... ')  
        self.dist = dist_sphere(self.latctd[:-1],self.lonctd[:-1],
                                self.latctd[1:], self.lonctd[1:])
        self.dist = np.concatenate(([0],self.dist))
        self.distcum = np.cumsum(self.dist)*1e-3 # [km] 

    def get_bathy(self,resolution='hr'):
        if   resolution == 'lr' : file_topo = self.file_topo_lr
        elif resolution == 'hr' : file_topo = self.file_topo_hr
        elif resolution == 'vhr': file_topo = self.file_topo_vhr
        print(' ... get bathymetry from ',file_topo,' ...')  
        nc     = Dataset(file_topo,'r')
        try:
            lonh   = nc.variables['x'][:]
            lath   = nc.variables['y'][:]
            self.h = nc.variables['z'][:]
        except: 
            lonh   = nc.variables['lon'][:]
            lath   = nc.variables['lat'][:]
            self.h = nc.variables['z'][:]
        nc.close()
        if resolution == 'lr': # get a subset  
            lonh = lonh[4000:5000] 
            lath = lath[4100:4700]
            self.h = self.h[4100:4700,4000:5000]  
        self.lonh,self.lath = np.meshgrid(lonh,lath) 
        return 

    def interpolate_bathy(self):
        print(' ... interpolate bathymetry on a fine-resolution grid ... ')
        h_nonan = np.copy(self.h.T); h_nonan[np.isnan(h_nonan)]=0 # in srtm15 dataset 
        spline = itp.RectBivariateSpline(self.lonh[0,:],self.lath[:,0],h_nonan,kx=1,ky=1)
        self.hvmp = spline.ev(self.lonvmp,self.latvmp) # bathy at VMP stations 
        self.hctd = spline.ev(self.lonctd,self.latctd) # bathy at CTD stations 
        del(h_nonan)
        ddeg   = self.lonh[0,1]-self.lonh[0,0] # resolution of bathy dataset 
        npts   = np.zeros(self.nctd-1) # number of points per segment between stations 
        for i in range(self.nctd-1):
            npts[i] = np.ceil(np.max((abs(self.lonctd[i+1]-self.lonctd[i])/ddeg,
                                      abs(self.latctd[i+1]-self.latctd[i])/ddeg)))
        lonitp = np.concatenate([np.linspace(self.lonctd[i],self.lonctd[i+1],int(npts[i])) 
                                for i in range(self.nctd-1)])
        latitp = np.concatenate([np.linspace(self.latctd[i],self.latctd[i+1],int(npts[i])) 
                                for i in range(self.nctd-1)])
        self.hitp  = spline.ev(lonitp,latitp)
        distitp    = dist_sphere(latitp[:-1],lonitp[:-1],latitp[1:],lonitp[1:])
        distitp    = np.concatenate(([0],distitp))
        self.hdistcum = np.cumsum(distitp)*1e-3 # [km]  
        return 
   
    def get_dissipation(self,zbine): 
        print(' ... get dissipation ... ')  
        nz  = zbine.shape[0]-1 
        eps = np.zeros((self.nvmp,nz)); eps[:] = np.nan 
        if self.year in ['2015','2017']: 
            nc   = Dataset(self.file_vmp,'r')
            print('   --> apply correction to all profiles, eps>1e-7 is set to nan ')  
            for i in range(self.nvmp):
                if self.list_vmp[i]!=self.nanval: 
                    epsi = nc.variables['eps'][self.list_vmp[i],:] 
                    zi   = nc.variables['z'][self.list_vmp[i],:] 
                    # - suspicious profile(s) - 
                    if self.list_vmp[i] in self.list_suspicious:
                        print('   --> apply correction to suspicious profiles, eps>1e-8 is set to nan ')  
                        epsi[epsi>1e-8] = np.nan 
                    # - bin in depth - 
                    epsi[epsi>1e-7] = np.nan 
                    [eps[i,:],_,_] = stats.binned_statistic(zi,epsi[:],statistic=np.nanmean,bins=zbine)
            nc.close()
        elif self.year == '2008': 
            for i in range(self.nvmp): 
                mat  = sio.loadmat(self.file_vmp[:-7]+'%.2i.mat'%(self.list_vmp[i]+1),squeeze_me=True) 
                pres = mat['Pmean']#'Pres_bin'] 
                epsi = mat['Emean'] 
                epsi0 = mat['Eps'][:,0]
                epsi1 = mat['Eps'][:,1]
                epsi2 = mat['Eps'][:,2]
                epsi3 = mat['Eps'][:,3]
                
                plt.figure(figsize=(10,10))
                plt.plot(np.arange(len(epsi)),epsi,label='Emean')
                plt.plot(np.arange(len(epsi)),epsi0,label='Eps0')
                plt.plot(np.arange(len(epsi)),epsi1,label='Eps1')
                plt.plot(np.arange(len(epsi)),epsi2,label='Eps2')
                plt.plot(np.arange(len(epsi)),epsi3,label='Eps3')
                plt.legend()
                plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/test_ovide_'+str(i)+'.pdf',bbox_inches='tight')

                plt.figure(figsize=(10,10))
                plt.plot(np.arange(len(epsi)),epsi,label='Emean')
                plt.plot(np.arange(len(epsi)),epsi0,label='Eps0')
                plt.ylim(0,1e-8)
                plt.legend()
                plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/test_ovide_zoom_'+str(i)+'.pdf',bbox_inches='tight')



                zi   = gsw.z_from_p(pres,self.latvmp[i]) 
                # - suspicious profile(s) - 
                if self.list_vmp[i] in self.list_suspicious:
                    print('   --> apply correction to suspicious profiles, eps>1e-8 is set to nan ')  
                    epsi[epsi>1e-8] = np.nan 
                [eps[i,:],_,_] = stats.binned_statistic(zi,epsi,statistic=np.nanmean,bins=zbine)
        self.eps    = eps
        self.ze_eps = zbine                       # bin edges 
        self.zc_eps = 0.5*(zbine[1:]+zbine[:-1])  # bin centers 
        self.dz_eps = abs(zbine[1]-zbine[0])      
        return  

    def get_diffusivity(self): 
        ''' needs get_dissipation and get_ctd_bvf before
            gets the diffusivity (kappa) on same bins as epsilon 
        '''
        print(' ... get diffusivity ... ') 
        # 1/ bin stratification 
        N2_bin = np.nan*np.copy(self.eps) 
        for i in range(self.nvmp):
            N2_bin[i,:] = stats.binned_statistic(self.zc_N2[i,:],self.N2[i,:],statistic=np.nanmean,bins=self.ze_eps)[0] 
        # 2/ Osborn 1980 
        gamma = 0.2 # mixing efficiency  
        self.kappa = gamma*self.eps/N2_bin 
        return 

    def get_tidalconversion(self,correction=True):
        print(' ... RREX '+self.year+': get tidal energy conversion ... ')  
        nc = Dataset(self.file_conv,'r')
        if correction:
            self.Eft = nc.variables['Eft_sc'][self.list_vmp] # supercritical-slope correction applied 
            self.Efn = nc.variables['Efn_sc'][self.list_vmp,:]
        else:
            self.Eft = nc.variables['Eft'][self.list_vmp] 
            self.Efn = nc.variables['Efn'][self.list_vmp,:]
        nc.close()
        return 

    def get_mld(self,which='max'):
        print(' ... RREX '+self.year+': get mixed-layer depth ... ') 
        nc  = Dataset(self.file_mld,'r')
        if which=='max':
            self.mld = nc.variables['mld_max'][self.list_vmp]
        elif which=='temp': 
            self.mld = nc.variables['mld_tem'][self.list_vmp]
        elif which=='density': 
            self.mld = nc.variables['mld_sig'][self.list_vmp]
        nc.close()
        return 

    def get_ctd_density(self,which='insitu'): 
        print(' ... RREX '+self.year+': get CTD density ... ') 
        nc = Dataset(self.file_ctd,'r') 
        self.zc_sigma  = -nc.variables['DEPH'][self.list_ctd,:].data 
        self.zc_sigma[self.zc_sigma==-nc.variables['DEPH']._FillValue] = np.nan 
        if which=='insitu': # in situ density anomaly (-1000 kg m-3) 
            self.sigma = nc.variables['SIGI'][self.list_ctd,:].data
        elif which=='potential_0': # potential density referenced to surface  
            self.sigma = nc.variables['SIG0'][self.list_ctd,:].data  
        elif which=='potential_1': # potential density referenced to 1000 m   
            self.sigma = nc.variables['SIG1'][self.list_ctd,:].data  
        elif which=='potential_2': # potential density referenced to 2000 m  
            self.sigma = nc.variables['SIG2'][self.list_ctd,:].data  
        # - assumes all sigma have the same fill value - 
        self.sigma[self.sigma==nc.variables['SIGI']._FillValue] = np.nan 
        nc.close()
        return

    def get_ctd_bvf(self): 
        print(' ... get N2 from CTD file ... ') 
        nc = Dataset(self.file_ctd,'r') 
        self.zc_N2 = -nc.variables['DEPH'][self.list_ctd,:].data 
        self.zc_N2[self.zc_N2==-nc.variables['DEPH']._FillValue] = np.nan 
        self.N2 = nc.variables['BRV2'][self.list_ctd,:].data  
        self.N2[self.N2==nc.variables['BRV2']._FillValue] = np.nan 
        nc.close()
        return
    
    def get_ctd_oxygen(self,which='kg'): 
        print(' ... RREX '+self.year+': get CTD oxygen ... ') 
        nc = Dataset(self.file_ctd,'r') 
        self.zc_oxy = -nc.variables['DEPH'][:] 
        if which=='kg': # micromol/kg
            self.oxy = nc.variables['OXYK'][self.list_ctd,:] 
        elif which=='ml': # ml/l
            self.oxy = nc.variables['OXYL'][self.list_ctd,:] 
        nc.close() 
        return 
    
    def get_ctd_temperature(self,which='potential'): 
        print(' ... RREX '+self.year+': get CTD temperature ... ') 
        nc = Dataset(self.file_ctd,'r') 
        self.zc_temp = -nc.variables['DEPH'][self.list_ctd,:].data
        self.zc_temp[self.zc_temp==-nc.variables['DEPH']._FillValue] = np.nan 
        if which=='insitu': 
            self.temp = nc.variables['TEMP'][self.list_ctd,:].data 
        elif which=='potential': 
            self.temp = nc.variables['TPOT'][self.list_ctd,:].data
        self.temp[self.temp==nc.variables['TEMP']._FillValue] = np.nan 
        nc.close()
        return  
    
    def get_ctd_salinity(self): 
        print(' ... RREX '+self.year+': get CTD salinity ... ') 
        nc = Dataset(self.file_ctd,'r') 
        self.zc_salt = -nc.variables['DEPH'][self.list_ctd,:].data 
        self.zc_salt[self.zc_salt==-nc.variables['DEPH']._FillValue] = np.nan 
        self.salt = nc.variables['PSAL'][self.list_ctd,:] 
        self.salt[self.salt==nc.variables['PSAL']._FillValue] = np.nan 
        nc.close()
        return 

    def get_ladcp_velocity(self): 
        print(' ... RREX '+self.year+': get LADCP velocity ... ') 
        if self.year=='2015':
            mat = sio.loadmat(self.file_ladcp,squeeze_me=True,struct_as_record=False) 
            self.u_ladcp  = mat['Ul_inv'][:,self.list_ctd].T  
            self.v_ladcp  = mat['Vl_inv'][:,self.list_ctd].T  
            self.zc_ladcp = -mat['zl_inv'][:,self.list_ctd].T.astype(float) 
        elif self.year=='2017': 
            nc = Dataset(self.file_ladcp,'r')
            self.u_ladcp  = nc.variables['Ul_inv'][self.list_ctd,:] 
            self.v_ladcp  = nc.variables['Vl_inv'][self.list_ctd,:]   
            self.zc_ladcp = -nc.variables['zl_inv'][self.list_ctd,:].astype(float)    
            nc.close()
        # - get z-coordinate edges for velocity data - 
        self.dz_ladcp = abs(self.zc_ladcp[0,1]-self.zc_ladcp[0,0]) 
        self.ze_ladcp = np.concatenate((np.tile(self.zc_ladcp[:,0]+0.5*self.dz_ladcp,(1,1)).T, 
                                        0.5*(self.zc_ladcp[:,1:]+self.zc_ladcp[:,:-1]),
                                        np.tile(self.zc_ladcp[:,-1]-0.5*self.dz_ladcp,(1,1)).T),
                                        axis=1)
        self.nz_ladcp = self.zc_ladcp.shape[1] 
        return

    def regrid_ladcp(self,dz):
        print(' ... RREX '+self.year+': regrid LADCP velocity on the vertical ... ') 
        # - define new z coordinates -
        ze = np.arange(-5000,dz,dz) # edges  
        nz = ze.shape[0]-1 
        self.u_ladcp_new  = np.zeros((self.nctd,nz))   
        self.ze_ladcp_new = np.zeros((self.nctd,nz+1))   
        for i in range(self.nctd):  
            self.ze_ladcp_new[i,:] = ze 
            [self.u_ladcp_new[i,:],_,_]  = stats.binned_statistic(self.zc_ladcp[i,:],self.u_ladcp[i,:],
                                                            statistic=np.nanmean,bins=ze) 
        return      
    
    def interpolate_ladcp_2d(self,dx,dz,method_itp):
        print(' ... RREX '+self.year+': interpolate LADCP velocity on an x-z grid ... ') 
        print('                         NB: also makes use of bathymetry to fill gaps near the bottom ')
        # - define the 2-D grid - 
        xe_firstguess = np.arange(self.distcum[0],self.distcum[-1]+dx,dx) # edge
        xe = np.linspace(self.distcum[0],self.distcum[-1],xe_firstguess.shape[0]) # so that it has the right boundaries
        # /!\ consequently, dx slightly differs from the one prescribed ! 
        ze = np.arange(-5000,dz,dz)
        xc = 0.5*(xe[1:]+xe[:-1]) 
        zc = 0.5*(ze[1:]+ze[:-1]) 
        self.xc_itp,self.zc_itp = np.meshgrid(xc,zc) 
        self.xe_itp,self.ze_itp = np.meshgrid(xe,ze) 
        dist_tile = np.tile(self.distcum,(self.nz_ladcp,1)).T
        # - interpolate ladcp data - 
        self.u_ladcp_itp = itp.griddata((np.ravel(dist_tile[~np.isnan(self.u_ladcp)]),
                                         np.ravel(self.zc_ladcp[~np.isnan(self.u_ladcp)])),
                                         np.ravel(self.u_ladcp[~np.isnan(self.u_ladcp)]), 
                                         (self.xc_itp,self.zc_itp),method=method_itp) 
        return

    def get_thorpe(self,which='temp'):
        print(' ... RREX '+self.year+': get Thorpe overturns ... ')
        if which in ['temp','sig0','sig1','sig2']:
            nc = Dataset(self.file_thorpe+which+'.nc','r')
            self.L_thorpe = nc.variables['L_thorpe'][self.list_ctd,:] 
            self.N_thorpe = nc.variables['N_thorpe'][self.list_ctd,:] 
            self.zc_thorpe = nc.variables['z_thorpe'][self.list_ctd,:] # 'bin' centre 
            nc.close()
        elif which=='vmp': # i.e., Thorpe overturns from VMP data 
            nc = Dataset(self.file_thorpe_vmp,'r')
            self.L_thorpe  = nc.variables['L_thorpe'][self.list_vmp,:] 
            self.zc_thorpe = nc.variables['z_thorpe'][:] # 'bin' centre 
            nc.close()
        return
  

    def plot_fancy_dissipation(self,ax,i,dkm,eps_ref=-10,lw=0.3,ls=5): 
        ''' 
        CV 2019/05/13: arguments: [lw]      : linewidth for frame and plot
                                  [ls]      : label size 
                                  [eps_ref] : reference dissipation to center the bars  
                       NB: some plotting options are hard-coded (pad, ticks to be plotted, ...)
                       to avoid passing too many arguments     
        ''' 
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes 
        kmin = self.zc_eps[~np.isnan(self.zc_eps)].shape[0]-np.nanargmax(np.sort(self.eps[i,:]))-1
        zmin = self.hvmp[i] 
        axins = inset_axes(ax,width='100%',height='100%',
                   bbox_to_anchor=(self.distcum[i],zmin,dkm,abs(zmin)),
                   bbox_transform=ax.transData,borderpad=0)
        plt.barh(self.zc_eps,np.log10(self.eps[i,:])-eps_ref,self.dz_eps,
                 fill=False,linewidth=lw,zorder=5)
        axins.patch.set_facecolor('none')
        axins.spines['bottom'].set_linewidth(lw)
        axins.set_xticks([0,1,2,3,4]) # offset from eps_ref 
        if i==3: axins.set_xticklabels(['','',str(eps_ref)+' to '+str(eps_ref+4),'',''])
        else:    axins.set_xticklabels(())  
        axins.spines['left'].set_visible(False) #axins.axis('off') # removes the whole frame
        axins.spines['right'].set_visible(False)
        axins.spines['top'].set_visible(False)
        axins.tick_params(width=lw,length=3*lw,labelsize=ls,pad=1)
        try:
            plt.ylim(zmin,0)
        except: 
            plt.ylim(self.ze_eps[0],0) # to comply with Kunze's diffusivity 
        plt.xlim(0,4) # offset from eps_ref 
        axins.set_yticks(())
        return
    
    def plot_fancy_dissipation_keff(self,ax,i,dkm,eps_ref=-10,lw=0.3,ls=5):
        ''' 
        CV 2019/05/13: arguments: [lw]      : linewidth for frame and plot
                                  [ls]      : label size 
                                  [eps_ref] : reference dissipation to center the bars  
                       NB: some plotting options are hard-coded (pad, ticks to be plotted, ...)
                       to avoid passing too many arguments     
        '''
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        kmin = self.zc_eps[~np.isnan(self.zc_eps)].shape[0]-np.nanargmax(np.sort(self.eps_keff[i,:]))-1
        zmin = self.hvmp[i]
        axins = inset_axes(ax,width='100%',height='100%',
                   bbox_to_anchor=(self.distcum[i],zmin,dkm,abs(zmin)),
                   bbox_transform=ax.transData,borderpad=0)
        plt.barh(self.zc_eps,np.log10(self.eps_keff[i,:])-eps_ref,self.dz_eps,
                 fill=False,linewidth=lw,zorder=5)
        axins.patch.set_facecolor('none')
        axins.spines['bottom'].set_linewidth(lw)
        axins.set_xticks([0,1,2,3,4]) # offset from eps_ref 
        if i==3: axins.set_xticklabels(['','',str(eps_ref)+' to '+str(eps_ref+4),'',''])
        else:    axins.set_xticklabels(())
        axins.spines['left'].set_visible(False) #axins.axis('off') # removes the whole frame
        axins.spines['right'].set_visible(False)
        axins.spines['top'].set_visible(False)
        axins.tick_params(width=lw,length=3*lw,labelsize=ls,pad=1)
        try:
            plt.ylim(zmin,0)
        except:
            plt.ylim(self.ze_eps[0],0) # to comply with Kunze's diffusivity 
        plt.xlim(0,4) # offset from eps_ref 
        axins.set_yticks(())
        return



    def plot_fancy_thorpe(self,ax,i,dkm,lw=0.3,ls=5,color='magenta'): 
        ''' 
        CV 2019/08/07: arguments: [lw]      : linewidth for frame and plot
                                  [ls]      : label size 
                                  [color]   : color of the plot and grid  
                       NB: some plotting options are hard-coded (pad, ticks to be plotted, ...)
                       to avoid passing too many arguments     
        ''' 
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes 
        kmax = np.nanargmin(self.zc_thorpe[i,:])
        axins = inset_axes(ax,width='100%',height='100%',
                   bbox_to_anchor=(self.distcum[i],self.zc_thorpe[i,kmax],dkm,abs(self.zc_thorpe[i,kmax])),
                   bbox_transform=ax.transData,borderpad=0)
        plt.plot(self.thorpe[i,:],self.zc_thorpe[i,:],color=color,linewidth=lw,zorder=5)
        plt.plot([0,0],[self.zc_thorpe[i,kmax],0],color=color,linewidth=lw,zorder=5)
        axins.patch.set_facecolor('none')
        axins.spines['bottom'].set_linewidth(lw)
        if i==0:
            axins.spines['left'].set_visible(False)
            axins.spines['right'].set_visible(False)
            axins.spines['top'].set_visible(False)
            axins.spines['bottom'].set_color(color)
            axins.tick_params(width=lw,length=3*lw,labelsize=4,color=color,labelcolor=color,pad=2)
            axins.set_xticks([0,50]) 
        else:
            axins.axis('off') # removes the whole frame
        plt.ylim(self.zc_thorpe[i,kmax],0)
        plt.xlim(0,75) 
        axins.set_yticks(())
        return

    def get_aviso(self,lonmin=-60,lonmax=-10,latmin=45,latmax=65):
        print(' ... RREX '+self.year+': get AVISO fields ... ') 
        jul_beg = int(np.nanmin(self.julctd))   # first index of file to load 
        jul_end = int(np.nanmax(self.julctd))+1 # last  index of file to load
        imin, imax, jmin, jmax = 0,0,0,0
        # - get boundaries of domain to load - 
        nc  = Dataset(self.file_aviso,'r') 
        lon = nc.variables['longitude'][:]; lon[lon>180.]-=360.  
        lat = nc.variables['latitude'][:] 
        fillval = nc.variables['adt']._FillValue  
        nc.close()  
        imin = np.argmin(abs(lon-lonmin))-1 
        imax = np.argmin(abs(lon-lonmax))+1 
        jmin = np.argmin(abs(lat-latmin))-1 
        jmax = np.argmin(abs(lat-latmax))+1
        self.lonavi = lon[imin:imax] 
        self.latavi = lat[jmin:jmax]
        # - read fields - 
        adt    = np.zeros((jul_end-jul_beg+1,jmax-jmin,imax-imin))  
        ugos   = np.zeros((jul_end-jul_beg+1,jmax-jmin,imax-imin))  
        vgos   = np.zeros((jul_end-jul_beg+1,jmax-jmin,imax-imin))  
        taviso = np.zeros((jul_end-jul_beg+1,))  
        diff_days = (dt.datetime(1950,1,1) - dt.datetime(1970,1,1)).days # offset in CTD/AVISO compared to Python 
        timeline = [tm.gmtime((i+diff_days)*86400) for i in np.arange(jul_beg,jul_end+1,1)]
        for i in range(jul_end-jul_beg+1): 
            datestr  = '%.4i%.2i%.2i'%(timeline[i][0],timeline[i][1],timeline[i][2])  
            nc = Dataset(self.file_aviso[:-20]+datestr+self.file_aviso[-12:],'r') 
            adt[i,:,:]  = nc.variables['adt'][:,jmin:jmax,imin:imax].data  
            ugos[i,:,:] = nc.variables['ugos'][:,jmin:jmax,imin:imax].data  
            vgos[i,:,:] = nc.variables['vgos'][:,jmin:jmax,imin:imax].data  
            taviso[i]   = nc.variables['time'][:].data  
            nc.close() 
        # - now interpolate to CTD timeline - 
        interp_adt  = itp.interp1d(taviso,adt,axis=0)
        self.adt    = interp_adt(np.asarray(self.julctd)) 
        interp_ugos = itp.interp1d(taviso,ugos,axis=0)
        self.ugos   = interp_ugos(np.asarray(self.julctd)) 
        interp_vgos = itp.interp1d(taviso,vgos,axis=0)
        self.vgos   = interp_vgos(np.asarray(self.julctd)) 
        # - mask data - 
        self.adt[self.adt   == fillval] = 0 # not nan to enable interpolations      
        self.ugos[self.ugos == fillval] = 0      
        self.vgos[self.vgos == fillval] = 0      
        # - interpolate at CTD stations -
        self.ugos_ctd = np.zeros(self.nctd)  
        self.vgos_ctd = np.zeros(self.nctd)  
        for i in range(self.nctd): 
            spline = itp.RectBivariateSpline(self.lonavi,self.latavi,self.ugos[i,:,:].T,kx=1,ky=1)
            self.ugos_ctd[i] = spline.ev(self.lonctd[i],self.latctd[i]) 
            spline = itp.RectBivariateSpline(self.lonavi,self.latavi,self.vgos[i,:,:].T,kx=1,ky=1)
            self.vgos_ctd[i] = spline.ev(self.lonctd[i],self.latctd[i]) 
        return 
    

#####################################################################################
################################### OLD ROUTINES ####################################
#####################################################################################
    def get_dissipation_old(self,zbine): 
        # CV 2019/05/10: old version: uses the multiples files stored in data0 
        print(' ... RREX '+self.year+': get dissipation ... ')  
        nz   = zbine.shape[0]-1 
        eps  = np.zeros((self.npro,nz))
        loop = 0
        for i in range(self.npro):
            # /!\ +1 due to python index starting at 0 
            print('      --> '+self.file_vmp_ind[:-7]+'%.3i'%(self.list_vmp[i]+1)+'.mat') 
            mat  = sio.loadmat(self.file_vmp_ind[:-7]+'%.3i'%(self.list_vmp[i]+1)+'.mat',squeeze_me=True)
            pres = mat['Pmean'][:]
            epsm = mat['Emean'][:]
            z    = pres2depth(pres,self.latvmp[loop])
            # - suspicious profile(s) - 
            if self.list_vmp[i] in self.list_suspicious:
                print('   --> apply correction eps>1e-8 is set to nan ')  
                epsm[epsm>1e-8] = np.nan 
            # - bin in depth - 
            [eps[loop,:],_,_] = stats.binned_statistic(z,epsm[:],statistic=np.nanmean,bins=zbine)
            loop += 1
            self.eps = eps
        return  

    def get_nearby_ctd(self,radius=30.): 
        # CV 2019/05/10: now useless given the new strategy 
        print(' ... RREX '+self.year+': get indices of nearby CTD stations ... ') 
        nc = Dataset(self.file_ctd,'r')  
        latctd = nc.variables['LATITUDE'][:] 
        lonctd = nc.variables['LONGITUDE'][:] 
        nc.close()
        list_ctd = [] 
        for i in range(self.npro): 
            dd = dist_sphere(latctd,lonctd,self.latvmp[i],self.lonvmp[i])*1e-3 # km to m 
            ind = np.argwhere(dd<radius) 
            list_ctd.append([i[0] for i in np.argwhere(dd<radius)]) 
        list_ctd = [item for sublist in list_ctd for item in sublist] 
        list_ctd = list(set(list_ctd)) # get unique values 
        self.nctd     = len(list_ctd) 
        self.list_ctd = list_ctd 
        self.lonctd   = lonctd[list_ctd] 
        self.latctd   = latctd[list_ctd] 
        return 
