'''
NS : Plot histograms of concentration (negative and positive) using RSUP3, RSUP5 and WENO5 combinations of scheme    
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
name_exp_jon      = ['rrexnum50','rrexnum50','rrexnum50']
name_pathdata_jon = ['RREXNUM50_NOFILT_T','RREXNUM50_RSUP5_NOFILT_T','RREXNUM50_RSVWENO5_NOFILT_T']
name_exp_grd_jon  = ['rrex50','rrex50','rrex50']
name_nc_jon       = ['rrexnum50-rsup3-nofilt','rrexnum50-rsup5-nofilt','rrexnum50-rsvweno5-nofilt']
name_title_jon    = ['a) exp50-rsup3','b) exp50-rsup5','c) exp50-weno5']
nbr_levels  = '50'

time        = [ '20','22','24','26','28','30','32','34','36','38','40',
                    '42','44','46','48','50','52','54','56','58','60',
                    '62','64','66','68','70','72','74','76','78','80']
ndfiles     = 2  # number of days per netcdf

# - select tracers for analysis - 
choose_tracer=['ridge','plain']
choice       = choose_tracer[1]
if choice=='plain':
    tpas        = 'tpas03'
    tpas_list   = ['tpas03']
    num_tpas    = [2]
    color_tracer = 'blue'
else:
    tpas        = 'tpas05'
    tpas_list   = ['tpas05']
    num_tpas    = [4] 
    color_tracer = 'red'

sig_min,sig_max = 27.9,27.55 # axis limits 
var_list        = tpas_list

# --- plot options --- 
fs      = 28      # fontsize 

# ------------ bin for concentration ------
minc      = -0.1
maxc      = 0.75
nce       = 0.017       # 0.0085
nbin      = 50          # 100
conc_e    = np.arange(minc,maxc,nce) # np.linspace(minc,maxc,num=nbin)
conc_c    = 0.5*(conc_e[1:]+conc_e[:-1])  # bin center  


# ------------ bin for neg concentration ------
mincn      = -0.1
maxcn      = 0
ncen       = 0.0085       # 0.0085
nbinn      = 12
conc_en    = np.arange(mincn,maxcn,ncen) # np.linspace(minc,maxc,num=nbin)
conc_cn    = 0.5*(conc_en[1:]+conc_en[:-1])  # bin center  


tpas1_neg = np.zeros((len(name_exp_jon),nbin-1))
tpas1_negn = np.zeros((len(name_exp_jon),nbinn-1))

neg1 = 0
neg2 = 0
maxg = 0
ming = 0


def make_plot(ax,figure,column,conc_c,tpas1_neg,minc,maxc,name_title_jon,conc_cn,tpas1_negn,fs,color_tracer):
    ax.bar(conc_c,tpas1_neg,width=0.01, fill=False,edgecolor=color_tracer) #, label='tracer 1')
    if column==0:
        plt.ylabel('Occurence',fontsize=fs)
    else:
        ax.set_yticks([])
    plt.xlabel('Tracer concentration',fontsize=fs)
    plt.yscale('log')
    plt.xlim(minc,maxc)
    plt.tick_params(labelsize=fs)
    ax.tick_params(labelsize=fs)
    plt.title(name_title_jon,fontsize=fs)
    # --- zoom negative
    # --->  box = [left, bottom, width, height]
    print('colmun :',column)
    if column==0:
        box = [0.22, 0.5, 0.1, 0.2]
    elif column==1:
        box = [0.5, 0.5, 0.1, 0.2]
    else:
        box = [0.77, 0.5, 0.1, 0.2]
    axes = figure.add_axes(box)
    plt.bar(conc_cn,tpas1_negn,width=0.01, fill=False,edgecolor=color_tracer)
    axes.set_yticks([10**3,10**5,10**7])
    axes.set_yticklabels([r'10$^{3}$',r'10$^{5}$',r'10$^{7}$'])
    axes.tick_params(labelsize=fs)
    plt.yscale('log')
    plt.ylim(0,3e7)  
    plt.xlim(minc,0)
    return 

# ------------ read data ------------ 
for exp in range(len(name_exp_jon)):
    name_exp      = name_exp_jon[exp]                  # name file netcdf
    name_pathdata = name_pathdata_jon[exp]             # folder where are netcdf
    name_exp_grd  = name_exp_grd_jon[exp]              # folder where grid data
    name_nc       = name_nc_jon[exp]                   # name of output netcdf
    tpas1_bin = np.zeros((ndfiles*len(time),nbin-1))
    tpas1_binn = np.zeros((ndfiles*len(time),nbinn-1))
    for t_nc in range(len(time)):
        data = Croco(name_exp,nbr_levels,time[t_nc],name_exp_grd,name_pathdata)
        data.get_grid()
        # ------------ make plot ------------ 
        print(' ... read avg file data + make plot ... ')
        for t in range(ndfiles):
            tt = (t+1)+ t_nc*ndfiles
            # ------------ read data ------------ 
            data.get_outputs(t,var_list)
            tpas1 = (data.var[tpas])
            eps   = 1e-6
            tpas1_r = np.ravel(tpas1)
            tpas1_bin[tt-1,:] = stats.binned_statistic(tpas1_r,tpas1_r,statistic='count',bins=conc_e)[0]
            if len(tpas1_r[tpas1_r<0])>0:
                tpas1_binn[tt-1,:] = stats.binned_statistic(tpas1_r[tpas1_r<0],tpas1_r[tpas1_r<0],statistic='count',bins=conc_en)[0]
            else:
                print('pas de valeur neg')
                tpas1_binn[tt-1,:] = np.zeros(np.shape(tpas1_binn[tt-1,:]))
            [minl1, maxl1] = [np.min(tpas1_r),np.max(tpas1_r)]
            minl = minl1 
            maxl = maxl1 
            if minl<ming :
                ming = minl
            if maxl>maxg:
                maxg = maxl
    tpas1_neg[exp,:] = np.nansum(tpas1_bin,axis=0)
    tpas1_negn[exp,:] = np.nansum(tpas1_binn,axis=0)

# find min and max for the plot
min1 = np.min(tpas1_neg)
max1 = np.max(tpas1_neg)

# ---- Make plot ----
column=0
figure = plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(1,3,)
for exp in range(len(name_exp_jon)):
    ax = plt.subplot(gs[0,column])  
    make_plot(ax,figure,column,conc_c,tpas1_neg[exp,:],minc,maxc,name_title_jon[exp],conc_cn,tpas1_negn[exp,:],fs,color_tracer)
    column+=1
plt.savefig('/home/datawork-lops-rrex/nschifan/Figures/NEG/neg_exp'+nbr_levels+'-rsup3-rsup5-weno5_Jon.pdf',bbox_inches='tight')
plt.close()




