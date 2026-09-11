import pandas as pd

from xspec_config_multisp import *
from fitting_tools import model_list
import os
import glob
from astropy.time import Time,TimeDelta
from general_tools import file_edit
from matplotlib.ticker import MultipleLocator

def sp_anal(obs_path,mod='powerlaw',baseload=False,scorpeon=True,overwrite=False):

    plt.ioff()

    outdir='s_a'

    os.system('mkdir -p '+outdir)

    if os.path.isfile("s_a/infos_fit.txt"):
        lines_infos=pd.read_csv('s_a/infos_fit.txt',sep='\t')

        if not overwrite:

            if obs_path in lines_infos['Observ_file'].values:
                print(obs_path+' already analyzed')
                return

    Plot.xLog=False
    AllData.clear()
    AllModels.clear()

    if baseload:
        Xset.restore(obs_path)
        obs=AllData(1).fileName

        AllModels.clear()

    else:
        AllData('1:1 '+obs_path)
        obs=obs_path

    AllData.ignore('**-0.3 10.-**')
    #file infos

    with fits.open(obs) as hdul:
        telescope=hdul[1].header['TELESCOP']
        if telescope=='NICER':
            start_obs_s = hdul[1].header['TSTART'] + \
                          (hdul[1].header['TIMEZERO'] if telescope == 'NICER' else 0)
            # saving for titles later
            mjd_ref = Time(hdul[1].header['MJDREFI'] + hdul[1].header['MJDREFF'], format='mjd')

            obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')

            obs_start=obs_start.isot

        elif telescope=='SWIFT':
            try:
                obs_start = hdul[0].header['DATE-OBS']
            except:
                try:
                    obs_start = hdul[1].header['DATE-OBS']
                except:
                    obs_start = Time(hdul[1].header['MJDSTART'], format='mjd').isot


    if telescope=='NICER' and scorpeon:
        xscorpeon.load('auto',frozen=True)

        if os.path.isfile('s_a/'+obs.split('.')[0]+'_baseload.xcm'):
            os.remove('s_a/'+obs.split('.')[0]+'_baseload.xcm')

        Xset.save('s_a/'+obs.split('.')[0]+'_baseload.xcm')

    if mod=='thcont':
        mod_list=['cont_diskbb', 'disk_thcomp', 'glob_TBfeo']
    elif mod=="powerlaw":
        mod_list=['powerlaw', 'glob_TBfeo']


    logfile_write, logfile = xLog_rw('s_a/'+obs.split('.')[0]+'_xlog.log')


    fitlines_strong=fitmod(mod_list,logfile,logfile_write)

    fitlines_strong.add_allcomps(split_fit=False)

    AllModels(1).TBfeo.nH.values=1.537
    AllModels(1).TBfeo.nH.frozen=True
    AllModels(1)(2).values=0.452
    AllModels(1)(3).values=2.33
    # AllModels(1)(2).frozen=False
    # AllModels(1)(3).frozen=False
    if mod=='thcont':
        AllModels(1).diskbb.Tin.values=[1.,0.01,0.4,0.4,2.,2.]

    Fit.perform()
    try:
        Fit.error('1-'+str(AllModels(1).nParameters))
    except:
        pass

    # if telescope=='NICER' and scorpeon:
    #     xscorpeon.load('auto',frozen=False,fit_SAA_norm=True)
    #
    # Fit.perform()
    # try:
    #     calc_error(logfile)
    # except:
    #     pass

    if os.path.isfile('s_a/'+obs.split('.')[0]+'_mod.xcm'):
        os.remove('s_a/'+obs.split('.')[0]+'_mod.xcm')

    Xset.save('s_a/'+obs.split('.')[0]+'_mod.xcm')

    #removing the calibation components and the interstellar absorption

    Plot_screen('ldata,ratio,delchi','s_a/'+obs.split('.')[0]+'_resid_screen.png')

    # saving the model str
    catch_model_str(logfile, savepath='s_a/'+obs.split('.')[0]+'_mod.txt')

    # if telescope=='NICER':
    #     delcomp('edge')
    #     delcomp('gaussian')
    delcomp('TBfeo')

    #computing the fluxes in different bands
    flux_arr=np.repeat(np.nan,9)
    for i in range(9):
        AllModels.calcFlux(str(float(i+1))+' '+str(float(i+2)))
        flux_arr[i]= AllData(1).flux[0]


    infos_store_path='s_a/'+'infos_fit.txt'
    line_str = obs + \
               '\t' +str(obs_start)+\
               '\t' + telescope+\
               '\t%.4e' %flux_arr[0]+ \
               '\t%.4e' %flux_arr[1] + \
               '\t%.4e' %flux_arr[2] + \
               '\t%.4e' %flux_arr[3] + \
               '\t%.4e' %flux_arr[4] + \
               '\t%.4e' %flux_arr[5] + \
               '\t%.4e' %flux_arr[6] + \
               '\t%.4e' %flux_arr[7] + \
               '\t%.4e' %flux_arr[8] + \
                '\n'

    line_store_header = 'Observ_file\tt_start\ttelescope\t'+\
                        'flux_1-2\tflux_2-3\tflux_3-4\tflux_4-5\tflux_5-6\tflux_6-7\tflux_7-8\tflux_8-9\tflux_9-10\n'

    file_edit(path=infos_store_path, line_id=obs, line_data=line_str, header=line_store_header)

def NICER_run_all_sp(sort=False,reverse=False,mod='thcont'):
    sp_list=glob.glob('**_grp_opt.pha')
    sp_list=[elem for elem in sp_list if 'MRG' not in elem and 'NF' not in elem]
    sp_list=np.array(sp_list)
    if sort:
        sp_list.sort()
    if reverse:
        sp_list=sp_list[::-1]

    for elem_sp in sp_list:
        sp_anal(elem_sp,mod=mod)

def BAT_plot_4U(path_BAT_lc):



    import matplotlib.dates as mdates


    fig_lc_BAT, ax_lc_BAT = plt.subplots(figsize=(14, 6))
    ax_lc_BAT.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_lc_BAT.xaxis.set_label('2025 observation dates')
    ax_lc_BAT.set_ylabel('BAT count rate (15-50 keV)')
    # ax_lc_BAT.spines['top'].set_color('forestgreen')
    # secax_BAT.xaxis.label.set_color('forestgreen')
    # secax_BAT.tick_params(axis='x', colors='forestgreen')
    # secax_BAT.xaxis.set_major_formatter(date_format)
    ax_lc_BAT.xaxis.set_minor_locator(MultipleLocator(1))
    # secax_BAT.xaxis.set_minor_locator(MultipleLocator(1))
    ax_lc_BAT.xaxis.set_tick_params(width=5)
    # secax_BAT.xaxis.set_tick_params(width=5)

    lc_BAT_infos = pd.read_csv(path_BAT_lc, delim_whitespace=True, skiprows=4, header=0)

    mdates_BAT=mdates.date2num(Time(lc_BAT_infos['TIME'].values,format='mjd').isot)

    # range_BAT_2007 = [54225, 54322]
    # range_BAT_2011 = [55530, 55652]
    # range_BAT_2016 = [57384, 57662]
    range_BAT_2021 = [59450, 59500]

    range_BAT_2025 = [60750, 60800]

    mask_BAT_2021=(lc_BAT_infos['TIME']>=range_BAT_2021[0]).values & (lc_BAT_infos['TIME']<=range_BAT_2021[1]).values
    mask_BAT_2025=(lc_BAT_infos['TIME']>=range_BAT_2025[0]).values & (lc_BAT_infos['TIME']<=range_BAT_2025[1]).values

    ax_lc_BAT.errorbar(mdates_BAT[mask_BAT_2021]-mdates_BAT[mask_BAT_2021][0]+mdates_BAT[mask_BAT_2025][0]+0.5-17,
                       lc_BAT_infos['RATE'][mask_BAT_2021],xerr=0.5,alpha=0.5,
                       yerr=lc_BAT_infos['ERROR'][mask_BAT_2021],
                  marker='x', ls='',color='orange',label='2022 evolution')
    ax_lc_BAT.errorbar(mdates_BAT[mask_BAT_2025]+0.5,
                       lc_BAT_infos['RATE'][mask_BAT_2025],xerr=0.5,alpha=0.5,
                       yerr=lc_BAT_infos['ERROR'][mask_BAT_2025],
                  marker='x', ls='',color='green',label='2025 evolution')

    ax_lc_BAT.axvline(mdates.date2num([Time(54301,format='mjd').isot])
                      -mdates_BAT[mask_BAT_2007][0]+mdates_BAT[mask_BAT_2025][0]+0.5,
                      color='blue',alpha=0.5,zorder=-1,lw=2,ls='--',
                      label='2006 state transition')

    ax_lc_BAT.axvline(mdates.date2num([Time(55612,format='mjd').isot])
                      -mdates_BAT[mask_BAT_2011][0]+mdates_BAT[mask_BAT_2025][0]+0.5,
                      color='purple',alpha=0.5,zorder=-1,lw=2,ls='--',
                      label='2011 state transition')

    ax_lc_BAT.axvline(mdates.date2num([Time(57470,format='mjd').isot])
                      -mdates_BAT[mask_BAT_2016][0]+mdates_BAT[mask_BAT_2025][0]+0.5,
                      color='red',alpha=0.5,zorder=-1,lw=2,ls='--',
                      label='2016 state transition')

    ax_lc_BAT.axvline(mdates.date2num([Time(59659,format='mjd').isot])
                      -mdates_BAT[mask_BAT_2022][0]+mdates_BAT[mask_BAT_2025][0]+0.5-17,
                      color='orange',alpha=0.5,zorder=-1,lw=2,ls='--',
                      label='2022 state transition')

    ax_lc_BAT.set_xlim(mdates_BAT[mask_BAT_2025][0],mdates.date2num(['2025-05-01']))

    ax_lc_BAT.set_ylim(0.,0.035)
    fig_lc_BAT.legend()


    # fig_HID,ax_HID=plt.subplots(1,figsize=(8,8))
    #
    # ax_HID.set_xlabel('')

    # for label in ax_lc_flux.get_xticklabels(which='major'):
    #     label.set(rotation=45, horizontalalignment='center')
    # for label in ax_lc_HR.get_xticklabels(which='major'):
    #     label.set(rotation=45, horizontalalignment='center')
    #
    # for label in secax_flux.get_xticklabels(which='major'):
    #     label.set(rotation=45, horizontalalignment='center')
    # for label in secax_HR.get_xticklabels(which='major'):
    #     label.set(rotation=45, horizontalalignment='center')


def standard_plots(path_22='/media/parrama/crucial_SSD/Observ/BHLMXB/NICER/IGRJ17091/visu/infos_fit_22_NICER_Swift.txt',
                   path_25='/media/parrama/crucial_SSD/Observ/BHLMXB/Swift/Sample/IGRJ17091/infos_fit_25_rise_glob.txt',
                   path_lc_22='/media/parrama/crucial_SSD/Observ/BHLMXB/NICER/IGRJ17091/2022/obsid/lcbatch/lc_a/infos_var.txt',
                   path_lc_25='/media/parrama/crucial_SSD/Observ/BHLMXB/Swift/Sample/IGRJ17091/2025-ToO/timing/infos_fit_timing.txt',
                   path_BAT_lc='/media/parrama/crucial_SSD/Observ/BHLMXB/Swift/Sample/IGRJ17091/IGRJ17091-3624_lc_BAT.txt'):

    from sources.rise_17091 import evol_plots
    evol_plots(path_22,path_25,path_lc_22,path_lc_25,path_BAT_lc)

def lc_anal(lc_path):

    outdir = 'lc_a'

    os.system('mkdir -p ' + outdir)


    var,varr_err=compute_RMS(lc_path)


    with fits.open(lc_path) as hdul:
        telescope=hdul[1].header['TELESCOP']
        if telescope=='NICER':
            start_obs_s = hdul[1].header['TSTART']
            # saving for titles later
            mjd_ref = Time(hdul[1].header['MJDREFI'] + hdul[1].header['MJDREFF'], format='mjd')

            obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')

            obs_start=obs_start.isot

        elif telescope=='SWIFT':
            try:
                obs_start = hdul[0].header['DATE-OBS']
            except:
                try:
                    obs_start = hdul[1].header['DATE-OBS']
                except:
                    obs_start = Time(hdul[1].header['MJDSTART'], format='mjd').isot

    infos_store_path='lc_a/'+'infos_var.txt'
    line_str = lc_path + \
               '\t' +str(obs_start)+\
               '\t' + telescope+\
               '\t%.4e' %var+ \
               '\t%.4e' %varr_err + \
                '\n'

    line_store_header = 'Observ_file\tt_start\ttelescope\t'+\
                        'frac_RMS\tfrac_RMS_error'+'\n'

    file_edit(path=infos_store_path, line_id=lc_path, line_data=line_str, header=line_store_header)

def NICER_run_all_lc(sort=True,reverse=False,band='0.3-10'):
    sp_list=glob.glob('**N_'+band+'_bin_1.0.lc')
    sp_list=[elem for elem in sp_list if 'MRG' not in elem and 'NF' not in elem and band in elem]
    sp_list=np.array(sp_list)
    if sort:
        sp_list.sort()
    if reverse:
        sp_list=sp_list[::-1]

    for elem_sp in sp_list:
        lc_anal(elem_sp)

def compute_RMS(lc_path):
    from stingray import Lightcurve, excess_variance

    lc_fits = fits.open(lc_path)
    lc = Lightcurve(lc_fits[1].data['TIME'], lc_fits[1].data['RATE'], err=lc_fits[1].data['ERROR'])

    return excess_variance(lc)

# from stingray import Powerspectrum
#
# plop=Powerspectrum(data=None)
# from stingray import fourier


def lc_EP():

    import matplotlib.dates as mdates

    from astropy.time import Time

    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/Propal/rise_4U')

    ep_csv=pd.read_csv('ep_37657_wxt.csv')

    flux_err_withul=np.array([ep_csv['flux'][i]/10 if ep_csv['upper_limit'][i] else ep_csv['flux_err'][i]\
                     for i in range(len(ep_csv))])


    dates_ep=mdates.date2num(Time(ep_csv['mjd'].values,format='mjd').isot)

    fig,ax=plt.subplots(figsize=(14,10))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H H'))

    ul_mask=ep_csv['upper_limit']

    #plotting upper limits
    plt.errorbar(ep_csv['mjd'][ul_mask],ep_csv['flux'][ul_mask],yerr=flux_err_withul[ul_mask],
                 uplims=ep_csv['upper_limit'][ul_mask],
                 ls='None',marker='.',color='grey',ecolor='grey',alpha=0.1)

    #plotting non upper limits
    mask2025=(ep_csv['mjd']<60820) & ~(ep_csv['upper_limit']) & (ep_csv['mjd']>60760)

    #from NuSTAR obs with transition on 13-04-2025
    tday25=60778.75

    #plotting upper limits
    plt.errorbar(ep_csv['mjd'][ul_mask]-tday25,ep_csv['flux'][ul_mask],yerr=flux_err_withul[ul_mask],
                 uplims=ep_csv['upper_limit'][ul_mask],
                 ls='None',marker='.',color='grey',ecolor='grey',alpha=0.2)

    #plotting the 2025 outburst normalized to the transition date
    plt.errorbar(ep_csv['mjd'][mask2025]-tday25,ep_csv['flux'][mask2025],yerr=flux_err_withul[mask2025],
                 uplims=ep_csv['upper_limit'][mask2025],
                 ls=':',marker='d',color='midnightblue',alpha=0.5,label='EP 2025 detections')

    mask2026=(ep_csv['mjd']>61270) & ~(ep_csv['upper_limit'])

    #plotting the 2026 outburst normalized to an equal flux value date offset (~equal points at 61287.19 and 60778.5)
    plt.errorbar(ep_csv['mjd'][mask2026]-tday25-(61287.19-60778.5),
                 ep_csv['flux'][mask2026],yerr=flux_err_withul[mask2026],
                 uplims=ep_csv['upper_limit'][mask2026],
                 ls=':',marker='o',color='darkred',alpha=1.0,label='EP 2026 (detections just started today) shifted to match EP 2025')

    plt.xlim(-2,5)
    plt.ylim(1.5e-11,1.5e-9)
    plt.yscale('log')
    plt.axvline(0,color='black',label='2025 start of transition (from NuSTAR seeing it happen in real time)')

    plt.axvspan(60779-tday25,60779-tday25+1.83,color='black',alpha=0.2,
                label='2021 start of transition\n(NICER/BAT vs 2025 Swift XRT/BAT, period of eq. HR_hard and/flux)')

    plt.axvline(Time.now().to_value('mjd')-tday25-(61287.19-60778.5),color="limegreen",alpha=1,
                                label='now: UTC '+Time.now().isot)
    plt.legend()

    plt.xlabel('time w.r.t. start of transition (days)')
    plt.ylabel('Einstein Probe flux 0.5-4keV flux (cgs)')
    plt.tight_layout()


'''
This source is one of the targets of XMM LP 094484. It was just detected as entering a new outburst as of less than 24 hours ago.
The flux increase has been remarkable, rising from non-detections in EP in September 01 to a flux level today that is extremely similar to the transition luminosities in the last 2 outbursts which covered the rise of this source (2021 and 2025). 
The figure at this URL https://drive.google.com/file/d/1LFHiX84nhu-2v-tQll5LX7F_P1odlSi8/view?usp=sharing summarizes our current monitoring of the source, normalized to its behavior during past outbursts. 

Our XMM-NuSTAR LP targets the state transition of a source like this, normally for a period of 10 days with 10 daily observations.

I have requested daily Swift observations of the source to start as soon as possible, but as we are expecting the transition to start imminently, I would like to schedule the first XMM-NuSTAR observation as soon as possible. 


'''

'''
NuSTAR SIMU

test=[FakeitSettings(response='nu91201307002A01_sr.rmf',arf='nu91201307002A01_sr.arf',exposure=1e4,
                              background='nu91201307002A01_bk.pha',
                              fileName='fake_NuSTARA_10ks.pha'),
    FakeitSettings(response='nu91201307002B01_sr.rmf',arf='nu91201307002B01_sr.arf',exposure=1e4,
                              background='nu91201307002B01_bk.pha',
                              fileName='fake_NuSTARB_10ks.pha')]
                  
#SPL normalized at 7e-9 cgs in absorbed 1-10keV luminosity            
Xset.restore('testmod.xcm')
set_ener('thcomp')
AllData.fakeit(settings=test)

'''

'''
First NuSTAR obs

ABSORBED
AllModels.calcFlux("3. 79.")
Spectrum Number: 1
Data Group Number: 1
 Model Flux   0.48686 photons (7.582e-09 ergs/cm^2/s) range (3.0000 - 79.000 keV)
Spectrum Number: 2
Data Group Number: 2
 Model Flux   0.48974 photons (7.6268e-09 ergs/cm^2/s) range (3.0000 - 79.000 keV)
AllModels.calcFlux("3. 10.")
Spectrum Number: 1
Data Group Number: 1
 Model Flux    0.3539 photons (3.1119e-09 ergs/cm^2/s) range (3.0000 - 10.000 keV)
Spectrum Number: 2
Data Group Number: 2
 Model Flux     0.356 photons (3.1303e-09 ergs/cm^2/s) range (3.0000 - 10.000 keV)
AllModels.calcFlux("6. 10.")
Spectrum Number: 1
Data Group Number: 1
 Model Flux   0.12269 photons (1.4972e-09 ergs/cm^2/s) range (6.0000 - 10.000 keV)
Spectrum Number: 2
Data Group Number: 2
 Model Flux   0.12342 photons (1.506e-09 ergs/cm^2/s) range (6.0000 - 10.000 keV)
AllModels.calcFlux("3. 6.")
Spectrum Number: 1
Data Group Number: 1
 Model Flux   0.23121 photons (1.6148e-09 ergs/cm^2/s) range (3.0000 - 6.0000 keV)
Spectrum Number: 2
Data Group Number: 2
 Model Flux   0.23258 photons (1.6243e-09 ergs/cm^2/s) range (3.0000 - 6.0000 keV)
AllModels.calcFlux("15. 50.")
Spectrum Number: 1
Data Group Number: 1
 Model Flux  0.068774 photons (2.724e-09 ergs/cm^2/s) range (15.000 - 50.000 keV)
Spectrum Number: 2
Data Group Number: 2
 Model Flux  0.069181 photons (2.7401e-09 ergs/cm^2/s) range (15.000 - 50.000 keV)
 


UNABSORBED FPMA

AllModels.calcFlux("0.3 10.")
Spectrum Number: 1
Data Group Number: 1
 Model Flux   0.41709 photons (3.3672e-09 ergs/cm^2/s) range (0.30000 - 10.000 keV)

AllModels.calcFlux("3. 10.")
Spectrum Number: 1
Data Group Number: 1
 Model Flux   0.50304 photons (4.118e-09 ergs/cm^2/s) range (3.0000 - 10.000 keV)

AllModels.calcFlux("3. 6.")
Spectrum Number: 1
Data Group Number: 1
 Model Flux   0.36525 photons (2.4396e-09 ergs/cm^2/s) range (3.0000 - 6.0000 keV)
 
 AllModels.calcFlux("6. 10.")
Spectrum Number: 1
Data Group Number: 1
 Model Flux   0.13779 photons (1.6785e-09 ergs/cm^2/s) range (6.0000 - 10.000 keV)

AllModels.calcFlux("15. 50.")
Spectrum Number: 1
Data Group Number: 1
 Model Flux  0.069382 photons (2.7428e-09 ergs/cm^2/s) range (15.000 - 50.000 keV)
 
  Edd ratio factor 15701344.992

15701344.992

FALSE		4U1630-47	81002318002	2026-09-06 07:44:00	NuSTAR	0.064658138677056	0.6880226266601082	1.1242826692900476



'''
