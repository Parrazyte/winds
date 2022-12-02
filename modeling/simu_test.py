from xspec import *
from xspec_config_multisp import *
import time
import numpy as np
from tqdm import tqdm

#load XMM data first and then model_gauss as model

AllData.clear()
AllModels.clear()


sp=Spectrum('0670671301_pn_S003_Timing_auto_sp_src_grp_20_bgtested.ds')
mod=Xset.restore('mod_gauss.xcm')
mod_gauss=allmodel_data()

Pset()

nruns=10

delchis=np.zeros((nruns,20,4))

Xset.chatter=0

AllData.ignore('**-3.-10.-**')


with tqdm(total=nruns) as pbar:
    for i_run in range(nruns):
        for i in range(1,21):
            for i_e,e in enumerate([1,5,10,20]):
                if i_e<2:
                    continue
                fake=FakeitSettings(exposure=str(5000*i),fileName='sptest_'+('0' if e<10 else '')+str(e)+'eqw_'+('0' if i<2 else '')+str(i*5)+'k.ds')
                model_load(mod_gauss)
                AllModels(1).gaussian_7.norm.values=[AllModels(1).gaussian_7.norm.values[0]*e]+AllModels(1).gaussian_7.norm.values[1:]
                AllData.fakeit(settings=fake,applyStats=True)
                AllData.ignore('**-0.3 10.-**')
                
                # unfreeze()
                # AllModels(1)(1).frozen=True
                # AllModels(1)(11).frozen=True
                # AllModels(1)(12).frozen=True
                # AllModels(1)(13).frozen=True
                # AllModels(1)(14).frozen=True
                # Fit.perform()
                
                # Plot('ldata,ratio,delchi')
                chi_gauss=Fit.statistic
                delcomp('gaussian_7')
                # Fit.perform()
                
                # Plot('ldata,ratio,delchi')
                delchis[i_run][i-1][i_e]=Fit.statistic-chi_gauss
        
        pbar.update(1)
        
#%%
eqw_arr=[1,5,10,20]
col_arr=['red','blue','orange','green']
linestyle_arr=[':','-.','--']

plt.figure(figsize=(8,6))
plt.title('retrieved gaussian statistics for '+str(nruns)+' runs')
plt.xlabel('Exposure (ks)')
plt.ylabel(r' retrieved gaussian |$\Delta\chi^2$|')

plt.xlim(5,100)
exp=np.array(range(1,21))*5

# for i_run in range(nruns):
#     for i_e in range(4):
        
#         if i_e<2:
#             continue
        
#         plt.plot(exp,delchis[i_run].T[i_e],
#                  label=str(eqw_arr[i_e])+' eV' if i_run==0 else '',color=col_arr[i_e],alpha=0.1)
        
for i_e in range(4):
    
        
    if i_e<2:
        continue
        
    for i_ival,i_val in enumerate([0.1,0.5,0.9]):
        delchis_arr=np.copy(delchis).T[i_e]
        delchis_arr.sort()
        
        plt.plot(exp,delchis_arr.T[int(i_run*i_val)],label=str(100*i_val)+'% '+str(eqw_arr[i_e])+' eV',color=col_arr[i_e],alpha=1,
                  linestyle=linestyle_arr[i_ival])
    
plt.axhline(9.21,label=r'3$\sigma$ limit',color='black')

plt.legend()
        