"""
NS : Choose one configuration outputs and compute for each tracer patch:
        (1) K_eff                  : The effective diapycnale diffusivity computed online and weighted by one tracer concentration
        (2) K_fit -named Kzh here- : The offline effective diapycnale diffusivity as defined in Holmes et al. 2019 (following Ledwell's 1D model 1991)
        (3) K_tracer               : The offline effective diapycnale diffusivity as defined in Ruan and Ferrari 2021 (following Taylor diffusivity 1922)
        (4) K_KPP -named AKt here- : The parameterized vertical diffusivity from KPP parameterization
"""

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
import gsw as gsw
import obsfit1d as fit
from croco_simulations_jonathan_hist import Croco

# ------------ parameters ------------ 
name_exp_jon      = ['rrexnum50','rrexnum50','rrexnum50',
                     'rrexnum100','rrexnum100','rrexnum100',
                     'rrexnum200','rrexnum200','rrexnum200',
                     'rrexnum200','rrexnum100','rrexnum100',
                     'rrexnum50','rrexnum100','rrexnum200',
                     'rrexnum100']
name_pathdata_jon = ['RREXNUM50_NOFILT_T','RREXNUM50_RSUP5_NOFILT_T','RREXNUM50_RSVWENO5_NOFILT_T',
                     'RREXNUM100_NOFILT_T','RREXNUM100_RSUP5_NOFILT_T','RREXNUM100_RSVWENO5_NOFILT_T',
                     'RREXNUM200_NOFILT_T','RREXNUM200_RSUP5_NOFILT_T','RREXNUM200_RSVWENO5_NOFILT_T',
                     'RREXNUMS200_RSUP5_NOFILT_T','RREXNUM100_C4_T','RREXNUM100_T',
                     'RREXNUM50_RSWENO5_NOFILT_T','RREXNUM100_RSWENO5_NOFILT_T','RREXNUM200_RSWENO5_NOFILT_T',
                     'RREXNUM100_UP3']
name_exp_grd_jon  = ['rrex50','rrex50','rrex50',
                     'rrex100-up3','rrex100-up5','rrex100-weno5',
                     'rrex200-up3','rrex200-up5','rrex200-weno5',
                     '','rrex100-up3','rrex100-up3',                     
                     'rrex50-weno5','rrex100-weno5','rrex200-weno5',
                     'rrex100-up3']
name_nc_jon       = ['rrexnum50-nofilt','rrexnum50-rsup5-nofilt','rrexnum50-rsvweno5-nofilt',
                     'rrexnum100-nofilt','rrexnum100-rsup5-nofilt','rrexnum100-rsvweno5-nofilt',
                     'rrexnum200-nofilt','rrexnum200-rsup5-nofilt','rrexnum200-rsvweno5-nofilt',
                     'rrexnums200-rsup5-nofilt','rrexnum100-c4','rrexnum100-filt',
                     'rrexnum50-rsweno5-nofilt','rrexnum100-rsweno5-nofilt','rrexnum200-rsweno5-nofilt',
                     'rrexnum100-up3-nofilt']
nbr_levels_jon    = ['50','50','50',
                     '100','100','100',
                     '200','200','200',
                     '200','100','100',
                     '50','100','200',
                     '100']
# --> select simulation
exp           = 15
name_exp      = name_exp_jon[exp]                  # name file netcdf
name_pathdata = name_pathdata_jon[exp]             # folder where are netcdf
name_exp_grd  = name_exp_grd_jon[exp]              # folder where grid data
name_nc       = name_nc_jon[exp]                   # name of output netcdf
nbr_levels    = nbr_levels_jon[exp]
# --> Select NetCDF to read
time        = [ '20','22','24','26','28','30','32','34','36','38','40',
                    '42','44','46','48','50','52','54','56','58','60',
                    '62','64','66','68','70','72','74','76','78','80',
                    '82','84','86','88','90','92','94','96','98']                    
ndfiles     = 2         # number of days per netcdf
nt = len(time)*ndfiles
dt = 3600*12            # time step between 2 outputs of CROCO
# - select tracers for analysis - 
tpas_list = ['tpas03','tpas05']
ntpas = len(tpas_list)
var_list  = ['zeta','temp','salt','AKt']
var_list += tpas_list 

# --> NetCDF output: contains diagnosis on diffusivities following several methods
file_diag_k = '/home/datawork-lops-rrex/nschifan/DIAGS/'+name_nc+'hist_diffusivities.nc'
gg = 9.81
rhoref   = 1027.4
# --> Create buoyancy bin
buoy_bin = 0.08
buoy_e   = np.arange(-49,-45,buoy_bin)*1e-3  # bin edges np.arange(-71,-68,buoy_bin)*1e-3 pour rhoref=1025
buoy_c   = 0.5*(buoy_e[1:]+buoy_e[:-1])      # bin centres 
nbin     = buoy_c.shape[0]

# - variables to be saved -
buoy_avg     = np.zeros((ntpas,nt))      # first order moment of tracer in buoyancy space  
buoy_var     = np.zeros((ntpas,nt))      # second order moment of tracer in buoyancy space (variance)    
N2_avg       = np.zeros((ntpas,nt))      # N2 averaged over the tracer
AKt_avg      = np.zeros((ntpas,nt))      # AKt averaged over the tracer  
Keff_avg     = np.zeros((ntpas,nt))      # Effective diapycnal diffusivity computed online on CROCO
tpas_bin     = np.zeros((ntpas,nt,nbin)) # tracer binned in buoyancy space
tpas_bin_ini = np.zeros((ntpas,nbin))    # tracer binned in buoyancy space
N2           = np.zeros((1002,802,int(nbr_levels)-1))
# variables as defined in Ruan and Ferrari 2021
Wtracer   = np.zeros((ntpas,nt)) 
Ktaylor   = np.zeros((ntpas,nt))
Komega    = np.zeros((ntpas,nt))
bp2       = np.zeros((ntpas,nt))
gradb2_avg= np.zeros((ntpas,nt))
# variables as defined in Holmes et al 2019
tpas_mod     = np.zeros((ntpas,nt,nbin)) # fit of tracer concentration following Ledwell's 1D  model (1991), from Holmes et al. 2019
K0        = np.zeros((ntpas,nt))
Kh        = np.zeros((ntpas,nt))
Kzh       = np.zeros((ntpas,nt))
nu_h      = np.zeros((ntpas,nt))

def tracer_avg(var,tpas,dvol):
    # dvol is the cell volume  
    #return np.nansum(var*tpas)/np.nansum(tpas)
    return np.nansum(var*tpas*dvol)/np.nansum(tpas*dvol)

print(' ............ time loop ...............................................  ')
tt = 0
for t_nc in range(len(time)):
    data = Croco(name_exp,nbr_levels,time[t_nc],name_exp_grd,name_pathdata)
    data.get_grid()
    dsurf   = 1./np.transpose(np.tile(data.pm*data.pn,(int(nbr_levels),1,1)),(1,2,0)) # horizontal surface area
    dsurf_w = 1./np.transpose(np.tile(data.pm*data.pn,(int(nbr_levels)-1,1,1)),(1,2,0)) # horizontal surface area at w-points
    for t in range(ndfiles):
        print('=====================  time index %.4i ====================='%t)
        print('    ---> read outputs ')
        data.get_outputs(t,var_list)
        AKt  = np.asfortranarray(data.var['AKt'][:,:,:])
        [z_r,z_w] = toolsF.zlevs(data.h,data.var['zeta'],data.hc,data.Cs_r,data.Cs_w)
        print(' ... compute potential density ... ')
        p       = gsw.p_from_z(z_r,np.nanmean(data.latr))
        SA      = gsw.SA_from_SP(data.var['salt'],p,np.nanmean(data.lonr),np.nanmean(data.latr))
        CT      = gsw.CT_from_pt(SA,data.var['temp'])
        for i in range(np.shape(z_r)[0]):
            for j in range(np.shape(z_r)[1]):
                [N2i,pmid] = gsw.Nsquared(SA[i,j,:],CT[i,j,:],p[i,j,:],data.latr[i,j])
                N2[i,j,:]  = N2i
        print(' ... N2 computed with gsw ... ')
        rho_po  = gsw.sigma1(SA,CT)+ 1000.    # potential density referenced at 1000 meters depth
        buoy    = -gg*(rho_po-rhoref)/rhoref
        dvol   = np.diff(z_w,axis=-1)*dsurf   # volume of grid cells on rho-coordinates 
        dvol_w = np.diff(z_r,axis=-1)*dsurf_w # volume of grid celles on w-coordinates on the vertical
        # --- compute grad(b)**2 ---
        dbdx   = tools.u2rho_3d(tools.diffxi_3d(buoy,data.pm,z_r,z_w,newz=None,mask=None))
        dbdy   = tools.v2rho_3d(tools.diffeta_3d(buoy,data.pn,z_r,z_w,newz=None,mask=None))
        dz_w   = np.diff(z_r,axis=-1)
        dbdz_w = np.diff(buoy,axis=-1)/dz_w
        dbdz   =  tools.vinterp(dbdz_w,z_r,z_w[:,:,1:-1],z_r)
        dbdz2  = dbdz**2
        gradb2 = dbdx**2 + dbdy**2 +dbdz**2
        # --> Loop on tracer 
        for i in range(ntpas):
            tpas = np.asfortranarray(data.var[tpas_list[i]][:,:,:])
            tpas[tpas<1e-6] = np.nan
            print(tpas.shape,dvol.shape)
            print('     --> bin variables in buoyancy space ')
            tpas_bin[i,tt,:] = stats.binned_statistic(buoy.ravel(),(tpas*dvol).ravel(),statistic=np.nansum,bins=buoy_e)[0]
            if t==0:
                tpas_bin_ini[i,:] = tpas_bin[i,tt,:]/np.nanmax(tpas_bin[i,tt,:])
            tpas_bin[i,tt,:] = tpas_bin[i,tt,:]*np.nansum(tpas_bin_ini[i,:])/np.nansum(tpas_bin[i,tt,:])
            print('     --> compute the moments of tracer and N2 ')
            buoy_avg[i,tt]  = tracer_avg(buoy,tpas,dvol)                        # called nu in Equation (24) in Holmes et al. 2019    
            buoy_var[i,tt]  = tracer_avg(buoy**2,tpas,dvol) - buoy_avg[i,tt]**2 # called sigma^2 in Equation (25) in Holmes et al. 2019
            N2_avg[i,tt]    = tracer_avg(N2,tools.w2rho(tpas),dvol_w)
            AKt_avg[i,tt]   = tracer_avg(tools.w2rho(AKt),tpas,dvol)            # AKt is the KKP parameterization on CROCO
            print('    --> Keff')
            # Load effective diapycnal diffusivity computed online and weighted it by the tracer concentration
            Keff             = data.get_diffusivity(t,'diffusivity',get_date=False) 
            print(Keff)
            Keff_avg[i,tt]   = tracer_avg(data.get_diffusivity(t,'diffusivity',get_date=False),tpas,dvol)
            print('     --> compute variables as in Ruan and Ferrari 2021')
            tpas_w       = 0.5*(tpas[:,:,:-1]+tpas[:,:,1:])                                         # at w-points with nsig-1 levels  
            omega        = np.diff(Keff[:,:,:-1]*dbdz_w,axis=-1)/np.diff(z_w[:,:,1:-1],axis=-1)     # at rho-points with nsig-2 levels
            omega_avg    = tracer_avg(omega,tpas[:,:,1:-1],dvol[:,:,1:-1])
            Wtracer[i,tt] = omega_avg / tracer_avg(dbdz_w,tpas_w,dvol_w)
            Ktaylor[i,tt] = tracer_avg(Keff[:,:,:-1]*dbdz_w**2,tpas_w,dvol_w) / tracer_avg(dbdz_w**2,tpas_w,dvol_w) 
            Komega[i,tt]  = 2*tracer_avg((omega-omega_avg)*(buoy[:,:,1:-1]-buoy_avg[i,t]),tpas[:,:,1:-1],dvol[:,:,1:-1])/tracer_avg(dbdz_w**2,tpas_w,dvol_w)
            bp2[i,tt] = tracer_avg((buoy - buoy_avg[i,tt])**2,tpas,dvol)
            gradb2_avg[i,tt] = tracer_avg(gradb2,tpas,dvol)
            print('    ')
        tt+=1

# --- compute Ktracer 
print(" ------ Ktracer following RUan and Ferrari 2021 ---------- ")
K_tracer=np.zeros((np.shape(bp2)[0],np.shape(bp2)[1]-1))
# --> Loop on tracer patchs
for itpas in range(ntpas):
    K_tracer[itpas,:] =  0.5*(bp2[itpas,1:]-bp2[itpas,:-1])/dt/(0.5 * (gradb2_avg[itpas,1:] + gradb2_avg[itpas,:-1]) )


print(" ------ Kzh following Holmes et al. 2019, from the code of Ryan Holmes (thank you again for sharing!) ---------- ")
# --> loop on tracer patchs
for itpas in range(len(tpas_list)):
    zF  = buoy_c/np.nanmean(N2_avg[itpas,:]) # simple change of coordinates          
    for t in range(1,ndfiles*(len(time))):
        print(' ---> time : ',t)
        zFt = zF - zF[np.argmax(tpas_bin[itpas,0,:])]
        res = fit.fit3par(zFt,tpas_bin[itpas,t,:],tpas_bin[itpas,0,:],(int(time[t//2])-int(time[0]))*dt)
        K0[itpas,t] = res.x[0]
        Kh[itpas,t] = res.x[1]
        tpas_mod[itpas,t,:] = fit.trMOD(res.x,zFt,tpas_bin[itpas,t,:],tpas_bin[itpas,0,:],(int(time[t//2])-int(time[0]))*dt)
    nu_h[itpas,:] = buoy_avg[itpas,:]/np.nanmean(N2_avg[itpas,:]) # cente of gravity in height space 
    nu_h[itpas,:] = nu_h[itpas,:]-nu_h[itpas,0]
    K0[itpas,0] = np.nan
    Kzh[itpas,:]=K0[itpas,:]+Kh[itpas,:]*nu_h[itpas,:]


# ----------- save parameters in netcdf file ------------ 
print(' ... save in netcdf file ... ')
nc = Dataset(file_diag_k,'w')
nc.rhoref      = rhoref
nc.createDimension('time',None)
nc.createDimension('nbin',nbin)
nc.createDimension('nbinp',nbin+1)
nc.createDimension('ntpas',ntpas)
nc.createVariable('time','f',('time'))
nc.createVariable('buoy_binc','f',('nbin'))
nc.createVariable('buoy_bine','f',('nbinp'))
nc.createVariable('buoy_avg','f',('ntpas','time'))
nc.createVariable('buoy_var','f',('ntpas','time'))
var = nc.createVariable('AKt_avg','f',('ntpas','time'))
var.long_name = 'AKt weighted by the tracer concentration'
var = nc.createVariable('Keff_avg','f',('ntpas','time'))
var.long_name = 'Keff weighted by the tracer concentration'
var = nc.createVariable('N2_avg','f',('ntpas','time'))
var.long_name = 'N2 weighted by the tracer concentration'
var = nc.createVariable('tpas_bin','f',('ntpas','time','nbin'))
var.long_name = 'tracer quantity in buoyancy space'
var = nc.createVariable('Wtracer','f',('ntpas','time'))
var = nc.createVariable('tpas_mod','f',('ntpas','time','nbin'))
var.long_name = 'modeled tracer quantity from 1D fit in buoyancy space'
var.long_name = 'diapycnal velocity of the centroid of the tracer'
var = nc.createVariable('Ktaylor','f',('ntpas','time'))
var.long_name = 'see Ruan and Ferrari JPO 2021'
var = nc.createVariable('Komega','f',('ntpas','time'))
var.long_name = 'see Ruan and Ferrari JPO 2021'
var = nc.createVariable('Ktr','f',('ntpas','time'))
var.long_name = 'see Ruan and Ferrari JPO 2021'
var = nc.createVariable('Kzh','f',('ntpas','time'))
var.long_name = 'see Holmes et al'
var = nc.createVariable('Kh','f',('ntpas','time'))
var.long_name = 'see Holmes et al'
var = nc.createVariable('K0','f',('ntpas','time'))
var.long_name = 'see Holmes et al'
var = nc.createVariable('bp2','f',('ntpas','time'))
var.long_name ='bp2'
var = nc.createVariable('gradb2_avg','f',('ntpas','time'))
var.long_name ='gradb2_avg'

nc.variables['Kzh'][:]   = Kzh
nc.variables['K0'][:]   = K0
nc.variables['Kh'][:]    = Kh
nc.variables['buoy_avg'][:]    = buoy_avg
nc.variables['buoy_var'][:]    = buoy_var
nc.variables['AKt_avg'][:]     = AKt_avg
nc.variables['Keff_avg'][:]    = Keff_avg
nc.variables['N2_avg'][:]      = N2_avg
nc.variables['tpas_bin'][:]    = tpas_bin
nc.variables['tpas_mod'][:]    = tpas_mod
nc.variables['Wtracer'][:]     = Wtracer
nc.variables['Ktaylor'][:]     = Ktaylor
nc.variables['Komega'][:]      = Komega
nc.variables['Ktr'][:]         = K_tracer
nc.variables['bp2'][:]         = bp2
nc.variables['gradb2_avg'][:]  = gradb2_avg
nc.close()


