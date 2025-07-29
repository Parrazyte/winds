#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:59:26 2023

@author: parrama
"""

import numpy as np
from matplotlib.ticker import Locator
import time
import os
import astropy.units as u
import random
import glob
import warnings
from astropy.io import fits
from astroquery.simbad import Simbad
import io
import zipfile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time,TimeDelta
from matplotlib.widgets import Slider,Button

h_cgs = 6.624e-27
eV2erg = 1.6021773E-12
erg2eV = 1.0/eV2erg
Ryd2eV = 13.605693

compton_thick_thresh=1.5e24

# ! light speed in Km/s unit
c_Km = 2.99792e5
# ! light speed in cm/s unit
c_cgs = 2.99792e10
sigma_thomson_cgs = 6.6525e-25
PI = 3.14159265


def ang2kev(x):
    '''note : same thing on the other side due to the inverse

    also same thing for mAngtoeV'''

    return 12.398 / x

def absorb_to_L(logxi,NH_cm,R_rg,M_BH):
    return 10**(logxi)*NH_cm*R_rg*R_g(M_BH)*1e5

def make_zip(filebites_arr,filename_arr):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "a",
                         zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, data in zip(filename_arr,filebites_arr):
            zip_file.writestr(file_name, data.getvalue())

    return zip_buffer

def R_s(M_BH_sol):


    c_SI = 2.99792e8
    G_SI = 6.674e-11
    Msol_SI = 1.98892e30

    return 2*G_SI*M_BH_sol*Msol_SI/c_SI**2/1e3*u.km

def R_g(M_BH_sol):


    c_SI = 2.99792e8
    G_SI = 6.674e-11
    Msol_SI = 1.98892e30

    return G_SI*M_BH_sol*Msol_SI/c_SI**2/1e3*u.km

def dist_factor(d_kpc):
    return 4*np.pi*(d_kpc*1e3*3.086e18)**2*u.cm*u.cm

def edd_factor(d_kpc,m_sun):
    return dist_factor(d_kpc) / (1.26e38 * m_sun)

def norm_to_Rin(diskbb_norm,D_kpc=8,theta=45,phys=False,kappa=1.7):

    '''
    If phys is set to True, returns the physical radius instead using the correction of the appendix in
    https://articles.adsabs.harvard.edu/pdf/1998PASJ...50..667K

    '''

    r_in=(diskbb_norm/np.cos(theta*np.pi/180))**0.5*D_kpc/10 * u.km

    #from https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node165.html (see Eq 3. Careful kappa is missing in appendix)
    if phys:
        R_in_factor=kappa**2*(3/7)**(1/2)*(6/7)**3
        val_fin=r_in*R_in_factor
    else:
        val_fin=r_in
    return val_fin

def par_norm_to_Rin(par_number,model_number=1,D_kpc=8,theta=45,phys=False,kappa=1.7):

    from xspec import AllModels
    par_val=[AllModels(model_number)(par_number).values[0],AllModels(model_number)(par_number).error[0],
             AllModels(model_number)(par_number).error[1]]
    return norm_to_Rin(par_val,D_kpc,theta,phys,kappa)
# def ravel_ragged_test(array,mode=None):
#
#     '''ravels a 2/3d array/list even with ragged nested sequences'''
#
#     #leaving if the array is 0d
#
#     if type(array) not in [np.ndarray,list] or len(array)==0:
#         return array
#     #or 1d
#     if type(array[0]) not in [np.ndarray,list]:
#         return array
#
#     #testing if the array is 3d
#     if type(array[0][0]) in [np.ndarray,list]:
#         return np.array([array[i][j][k] for i in range(len(array)) for j in range(len(array[i])) for k in range(len(array[i][j]))],dtype=mode)
#     else:
#         return np.array([array[i][j] for i in range(len(array)) for j in range(len(array[i]))],dtype=mode)

def source_catal(spawn, dirpath, file, target_only=True, use_file_target=False):
    '''
    Tries to identify a Simbad object from either the directory structure or the source name in the file itself

    If use_file_target is set to True, does not cancel the process when Simbad
    doesn't recognize the source and uses the file target name instead

    '''

    # splitting the directories and searching every name in Simbad
    dir_list = dirpath.split('/')[1:]

    # removing a few problematic names
    crash_sources = ['M2', 'home', 'outputmos', 'BlackCAT', '']
    # as well as obsid type names that can cause crashes
    for elem_dir in dir_list:
        if len(elem_dir) == 10 and elem_dir.isdigit() or elem_dir in crash_sources:
            dir_list.remove(elem_dir)

    # Simbad.query_object gives a warning for a lot of folder names so we just skip them
    obj_list = None
    for elem_dir in dir_list:
        try:
            with warnings.catch_warnings():
                # warnings.filterwarnings('ignore','.*No known catalog could be found.*',)
                # warnings.filterwarnings('ignore','.*Identifier not found.*',)
                warnings.filterwarnings('ignore', category=UserWarning)
                elem_obj = Simbad.query_object(elem_dir)
                if type(elem_obj) != type(None):
                    obj_list = elem_obj
        except:
            breakpoint()
            print('\nProblem during the Simbad query. This is the current directory list:')
            print(dir_list)
            spawn.sendline('\ncd $currdir')
            return 'Problem during the Simbad query.'

    try:
        target_name = fits.open(os.path.join(dirpath,file))[0].header['OBJECT']
    except:
        target_name = fits.open(os.path.join(dirpath,file))[1].header['OBJECT']

    try:
        with warnings.catch_warnings():
            # warnings.filterwarnings('ignore','.*No known catalog could be found.*',)
            # warnings.filterwarnings('ignore','.*Identifier not found.*',)
            warnings.filterwarnings('ignore', category=UserWarning)
            file_query = Simbad.query_object(target_name)
    except:
        print('\nProblem during the Simbad query. This is the current obj name:')
        print(dir_list)
        spawn.sendline('\ncd $currdir')
        return 'Problem during the Simbad query.'

    if obj_list is None:
        print("\nSimbad didn't recognize any object name in the directories." +
              " Using the target of the observation instead...")
        obj_list = file_query

    if type(file_query) == type(None):
        print("\nSimbad didn't recognize the object name from the file header." +
              " Using the name of the directory...")
        target_query = ''
    else:
        target_query = file_query[0]['main_id']

    if type(obj_list) == type(file_query) and type(obj_list) == type(None):

        print("\nSimbad couldn't detect an object name.")
        if not use_file_target:
            print("\nSkipping this observation...")
            spawn.sendline('\ncd $currdir')

        return "Simbad couldn't detect an object name."

    # if we have at least one detections, it is assumed the "last" find is the name of the object
    obj_catal = obj_list[-1]

    print('\nValid name(s) detected. Object name assumed to be ' + obj_catal['main_id'])

    if obj_catal['main_id'] != target_query and target_only:
        print('\nTarget only mode activated and the source studied is not the main focus of the observation.' +
              '\nSkipping...')
        spawn.sendline('\ncd $currdir')
        return 'Target only mode activated and the source studied is not the main focus of the observation.'

    return obj_catal

def get_overlap(a, b,distance=False):
    #compute overlap between two intervals. if distance is set to true, returns the distance between the
    #intervals if they are disjoint
    if distance:
        return min(a[1], b[1]) - max(a[0], b[0])
    else:
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def shorten_epoch(file_ids_init):

    if file_ids_init=='auto':
        from xspec import AllData
        file_ids=[AllData(i+1).fileName.split('_sp')[0] for i in range(AllData.nGroups)]
    else:
        file_ids=file_ids_init

    # splitting obsids
    # obsids_list = [elem.split('-')[0] for elem in file_ids]

    #we split some weirder names, like bat files, to avoid splitting them incorrectly

    bat_survey_files=[elem for elem in file_ids if 'survey_point' in elem or \
                      (elem.startswith('BAT_') and elem.endswith('_mosaic.pha'))]

    obsids = np.unique([(elem.split('_heg')[0] if 'heg_' in elem else elem.split('-')[0]) for elem in file_ids if elem not in bat_survey_files])


    if len(obsids)==0:
        # returning the obsids directly if there's no gtis in the obsids
        obsids_ravel = ''.join(file_ids)
        if '-' not in obsids_ravel:
            return file_ids

    #only adding one element of the bat survey file because we will add them all at once
    obsids=obsids.tolist()+bat_survey_files[:1]

    # according the gtis in a shortened way
    epoch_str_list = []
    for elem_obsid in obsids:

        #for bat individual survey points
        if 'survey_point' in elem_obsid:

            #computing how much of the pointings can be compacted
            survey_ids = [elem.split('_survey_point_')[1].split('.')[0] for elem in bat_survey_files]

            if len(survey_ids)==1:
                epoch_str_list+=survey_ids
            else:
                survey_ids_char = np.array([[subelem for subelem in elem] for elem in survey_ids])
                survey_id_common_char=np.argwhere([len(np.unique(elem)) != 1 for elem in survey_ids_char.T]).T[0][0]

                #and compacting
                str_obsid=survey_ids[0][:survey_id_common_char]+'-'.join(elem[survey_id_common_char:] for elem in survey_ids)

                #recognizing BAT_analysis point, and adding them accordingly
                epoch_str_list+=[str_obsid]
            continue

        #for bat mosaics
        elif elem_obsid.startswith('BAT_') and elem_obsid.endswith('_mosaic.pha'):
            epoch_str_list += [elem_obsid.replace('.pha','').replace('_mosaic','')]
            continue

        str_gti_add = ''

        str_obsid = elem_obsid

        if '-' in ''.join([elem for elem in file_ids if elem.startswith(elem_obsid)]) and\
                len([elem for elem in file_ids if \
                                          elem.startswith(elem_obsid) and not 'heg_' in elem])>0:

            try:
                str_gti_add = '-' + '-'.join([elem.split('-')[1] for elem in file_ids if \
                                          elem.startswith(elem_obsid) and not 'heg_' in elem])
            except:
                breakpoint()
                pass

        str_gti_add += ''.join([elem.split(str_obsid)[-1].split('_grp_opt')[0] for elem in file_ids if \
                                  elem.startswith(elem_obsid) and 'heg_' in elem])

        epoch_str_list += [str_obsid + str_gti_add]

    return epoch_str_list


# not needed for now
def expand_epoch(shortened_epochs):

    '''
    Takes an array as argument so split the '_' joined short_id beforehand
    '''

    if type(shortened_epochs) in (str,int):
        shorten_list=[str(shortened_epochs)]
    else:
        shorten_list=shortened_epochs
    # splitting obsids
    file_ids = []

    for short_id in shorten_list:
        if short_id.count('-') <= 1:
            file_ids += [short_id]
        else:
            obsid = short_id.split('-')[0]
            gti_ids = short_id.split('-')[1:]

            file_ids += ['-'.join([obsid, elem_gti]) for elem_gti in gti_ids]

    return file_ids

def str_orbit(i_orbit):
    '''
    return regular str expression of orbit
    '''
    return ('%3.f' % (i_orbit + 1)).replace(' ', '0')


def plot_lc(lc_paths,binning='auto',directory='./',e_low='',e_high='',
                  lc_paths_HR_num=None,
                  interact=False,
                  interact_tstart=None,
                  instru='xrism',
                  show_date_evol=True,
                  save=False,suffix='',outdir=''):

    '''

    Wrapper to plot xrism lightcurves with or without interactivity

    if lc_path_HR is not None, assumes it is also an lc path with the same binning and properties and
    builds an HR from that as lc_path_HR/lc_path

    show_date_evol: if set to true, adds a secondary axis on top of the graph to show date formatters

    outdir: if set to None, saves the lightcure where lc_path is, otherwise in the outdir subdirectory
    '''

    if type(lc_paths)==str:
        lc_path_list=[lc_paths]
    else:
        lc_path_list=lc_paths

    if type(lc_paths_HR_num)==str:
        lc_path_HR_num_list=[lc_paths_HR_num]
    elif type(lc_paths_HR_num)==type(None):
        lc_path_HR_num_list=np.repeat(None,len(lc_paths))
    else:
        lc_path_HR_num_list=lc_paths_HR_num

    default_mpl_cycle=plt.rcParams['axes.prop_cycle'].by_key()['color']

    if save:
        plt.ioff()

    fig_lc, ax_lc = plt.subplots(1, figsize=(16, 8))

    time_zero_base=0

    for i_lc,(elem_lc,elem_lc_HR_num) in enumerate(zip(lc_path_list,lc_path_HR_num_list)):

        with fits.open(os.path.join(directory, elem_lc)) as fits_lc:
            data_lc_arr = fits_lc[1].data

            telescope = fits_lc[1].header['TELESCOP']
            instru = fits_lc[1].header['INSTRUME']

            time_zero = Time(fits_lc[1].header['MJDREFI'] + fits_lc[1].header['MJDREFF'], format='mjd')
            time_zero += TimeDelta(fits_lc[1].header['TIMEZERO'], format='sec')

            print(time_zero)

        if i_lc==0:
            time_zero_base=time_zero
        # and plotting it

        if binning=='auto':
            binning_use=data_lc_arr['TIME'][1]-data_lc_arr['TIME'][0]
        else:
            binning_use=str(binning)

        if elem_lc_HR_num is not None:
            with fits.open(os.path.join(directory, elem_lc_HR_num)) as fits_lc:
                data_lc_arr_num = fits_lc[1].data

            HR=data_lc_arr_num['RATE']/data_lc_arr['RATE']
            HR_err=((data_lc_arr_num['ERROR']/data_lc_arr_num['RATE'])**2+
                    (data_lc_arr['ERROR']/data_lc_arr['RATE'])**2)**(1/2)*HR

            plt.errorbar(data_lc_arr['TIME']+(time_zero-time_zero_base).value*86400,
                         HR,xerr=float(binning_use) / 2,yerr=HR_err,
                         ls='-', lw=1, color=(0.5, 0.5, 0.5,1/len(lc_path_list)),
                         ecolor='blue' if len(lc_path_list)==1 else default_mpl_cycle[i_lc],
                         label=elem_lc_HR_num+'/'+elem_lc if len(lc_path_list)>1 else '')

        else:

            plt.errorbar(data_lc_arr['TIME']+(time_zero-time_zero_base).value*86400, data_lc_arr['RATE'], xerr=float(binning_use) / 2,
                     yerr=data_lc_arr['ERROR'], ls='-', lw=1, color=(0.5, 0.5, 0.5, 1/len(lc_path_list)),
                         ecolor='blue' if len(lc_path_list)==1 else default_mpl_cycle[i_lc],
                         label=elem_lc if len(lc_path_list)>1 else '' )

    plt.legend()

    binning_str=str(binning_use)

    if instru=='xrism':
        if lc_paths_HR_num is not None:
            plt.suptitle(
            telescope + ' ' + instru + ' Hardness Ratio for observation ' + lc_path_list[0].split('_lc')[0].split('_pixel')[0] +
            (' with pixel ' + lc_path_list[0].split('_pixel_')[-1].split('_')[0] if instru == 'RESOLVE' else
             ' with region ' + lc_path_list[0].split('_cl_')[-1].split('_lc')[0]) +
            ' in [' + str(e_high) +']/[' + str(e_low) + '] keV with ' + binning_str + ' s binning')
        else:
            plt.suptitle(
            telescope + ' ' + instru + ' lightcurve for observation ' + lc_path_list[0].split('_lc')[0].split('_pixel')[0] +
            (' with pixel ' + lc_path_list[0].split('_pixel_')[-1].split('_')[0] if instru == 'RESOLVE' else
             ' with region ' + lc_path_list[0].split('_cl_')[-1].split('_lc')[0]) +
            ' in [' + str(e_low) + '-' + str(e_high) + '] keV with ' + binning_str + ' s binning')
    else:

        if lc_paths_HR_num is not None:
            plt.suptitle(
            telescope + ' ' + instru + ' Hardness Ratio for observation(s) ' + lc_path_list[0].split('_lc')[0].split('_pixel')[0] +
        ' in [' + str(e_high) +']/[' + str(e_low) + '] keV with ' + binning_str + ' s binning')
        else:
            plt.suptitle(
            telescope + ' ' + instru + ' lightcurve for observation(s) ' + lc_path_list[0].split('_lc')[0].split('_pixel')[0] +
            ' in [' + str(e_low) + '-' + str(e_high) + '] keV with ' + binning_str + ' s binning')

    plt.xlabel('Time (s) after ' + time_zero.isot)

    if show_date_evol:

        def func_time_to_date(x):

            val_out= mdates.date2num([(time_zero+TimeDelta(x,format='sec')).isot])[0]

            return val_out
        def func_date_to_time(x):

            if type(x) in (np.ndarray,list):
                x_use=x

                val_out=np.copy(x_use)
                for i,elem in enumerate(x_use):
                    if type(elem) in (np.ndarray,list):
                        for j,elem in enumerate(x_use[i]):
                            val_out[i][j] = (Time(str(mdates.num2date(elem)).split('+')[0]) - time_zero).to_value('sec')
                    else:
                        val_out[i]=(Time(str(mdates.num2date(elem)).split('+')[0])-time_zero).to_value('sec')

            else:
                val_out=(Time(str(mdates.num2date([x])).split('+')[0])-time_zero).to_value('sec')

            return val_out

        secax_lc = ax_lc.secondary_xaxis('top',functions=(func_time_to_date,func_date_to_time))

        #adding the formatter
        x_bounds_sec=ax_lc.get_xlim()[1]-ax_lc.get_xlim()[0]

        if x_bounds_sec< 10*86400:
            date_format = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
        elif x_bounds_sec < 365*86400:
            date_format = mdates.DateFormatter('%Y-%m-%d')
        else:
            date_format = mdates.DateFormatter('%Y-%m')

        secax_lc.xaxis.set_major_formatter(date_format)


    if lc_paths_HR_num is not None:
        plt.ylabel('Hardness Ratio')

    else:
        plt.ylabel('RATE (counts/s)')

    plt.tight_layout()

    if interact:

        plt.ion()

        plt.subplots_adjust(bottom=0.2)

        ax_slider_start = fig_lc.add_axes([0.2, 0.1, 0.65, 0.03])

        ax_slider_end = fig_lc.add_axes([0.2, 0.05, 0.65, 0.03])


        #note: we add one binning unit to allow to go all the way and take the last bin
        slider_start = Slider(ax_slider_start, label='gti start (s)',
                      valmin=data_lc_arr['TIME'][0]+float(binning_use)/2 if interact_tstart is None else interact_tstart,
                      valmax=data_lc_arr['TIME'][-1]+float(binning_use), valstep=float(binning_use))

        slider_end = Slider(ax_slider_end, label='gti end (s)',
                      valmin=data_lc_arr['TIME'][0]+float(binning_use)/2,
                      valmax=data_lc_arr['TIME'][-1]+float(binning_use)/2, valstep=float(binning_use))
        def slider_update(val):

            for elem_child in ax_lc.get_children():
                if elem_child._label == 'current gti':
                    elem_child.remove()

            ax_lc.axvspan(slider_start.val, slider_end.val,0, 1, alpha=0.3, color='green', label='current gti')

            fig_lc.legend()

        slider_start.on_changed(slider_update)
        slider_end.on_changed(slider_update)

        ax_button_finish = fig_lc.add_axes([0.9, 0.1, 0.08, 0.05])

        ax_button = fig_lc.add_axes([0.9, 0.025, 0.08, 0.04])


        but = Button(ax=ax_button, label='Save GTI')

        but_stop = Button(ax=ax_button_finish, label='Save GTI &    \n end computation')

        def func_button(val):
            plt.close()
            print(slider_start.val)
            print(slider_end.val)

        class finish_button:
            def __init__(self):
                self.finish=False

            def press(self,event):
                self.finish=True
                plt.close()
                print(slider_start.val)
                print(slider_end.val)

        plt.show()

        but.on_clicked(func_button)

        button_end=finish_button()
        but_stop.on_clicked(button_end.press)

        #will block the code as long as the button isn't pressed
        plt.show(block=True)

        if not save:
            return slider_start.val,slider_end.val,button_end.finish

    if save:
        lc_path_extension=lc_path_list[0][lc_path_list[0].rfind('.'):]

        os.system('mkdir -p '+os.path.join(directory, outdir))

        fig_lc.savefig(os.path.join(directory, outdir, lc_path_list[0].split('/')[-1].replace(lc_path_extension,
                                                        ('_'+ suffix if suffix!='' else '')+'_screen.png')))
        plt.close()
        plt.ion()

        if interact:
            return slider_start.val,slider_end.val,button_end.finish

def rescale_flex(ax,xlims,ylims,margin,std_x=None,std_y=None):

    '''

    std_x and std_y are bounds to use as default if the xlim/ylim values are smaller

    variant for negative symlog (aka with 0 as max value) not implemented
    '''


    
    if ax.get_xscale()=='linear':
        
        xrange = xlims[1] - xlims[0]

        del_xrange = xrange * margin
        
        if std_x is not None:
            ax.set_xlim((min(xlims[0]- del_xrange,std_x[0]),max(xlims[1]+ del_xrange,std_x[1])))
        else:
            ax.set_xlim((xlims[0]-del_xrange, xlims[1]+del_xrange))
            
    elif ax.get_xscale() in ['log','symlog']:

        xrange = xlims[1] / (ax.get_xticks()[1] if ax.get_xscale() == 'symlog' else xlims[0])

        del_xrange = np.log10(xrange) * margin
        
        if std_x is not None:
            ax.set_xlim((min(xlims[0]*10**(-del_xrange),std_x[0]),max(xlims[1]*10**(del_xrange),std_x[1])))
        else:
            ax.set_xlim((xlims[0] * 10 ** (-del_xrange)), xlims[1] * 10 ** (del_xrange))

    if ax.get_yscale() == 'linear':

        yrange = ylims[1] - ylims[0]

        del_yrange = yrange * margin

        if std_y is not None:
            ax.set_ylim((min(ylims[0] - del_yrange, std_y[0]), max(ylims[1] + del_yrange, std_y[1])))
        else:
            ax.set_ylim((ylims[0] - del_yrange, ylims[1] + del_yrange))

    elif ax.get_yscale() in ['log', 'symlog']:

        yrange = ylims[1] / (ax.get_yticks()[1] if ax.get_yscale() == 'symlog' else ylims[0])

        del_yrange = np.log10(yrange) * margin

        if std_y is not None:
            ax.set_ylim((min(ylims[0] * 10 ** (-del_yrange), std_y[0]), max(ylims[1] * 10 ** (del_yrange), std_y[1])))
        else:
            ax.set_ylim((ylims[0] * 10 ** (-del_yrange)), ylims[1] * 10 ** (del_yrange))

def ravel_ragged(array,mode=None,ragtuples=True):

    '''
    ravels a multi dimensional ragged nested sequence
    only considers tuples if ragtuples is set to True
    '''

    list_elem=[]

    for elem in array:
        if type(elem) in [np.ndarray,list] + ([tuple] if ragtuples else []):
            if len(elem)!=0:
                list_elem+=ravel_ragged(elem,mode=None,ragtuples=ragtuples).tolist()
            else:
                continue
        else:
            list_elem+=[elem]

    #necessary to avoid converting tuples into arrays if ragtuples is not set to True
    if np.all(type(elem)==tuple for elem in list_elem) and not ragtuples:
        return_arr=np.array([None]*len(list_elem))
        for i_elem in range(len(list_elem)):
            return_arr[i_elem]=list_elem[i_elem]
        return return_arr

    return np.array(list_elem,dtype=mode)

def interval_extract(list):

    '''
    From a list of numbers, outputs a list of the integer intervals contained inside it
    '''

    if len(list)==0:
        return list

    list = sorted(set(list))
    range_start = previous_number = list[0]

    for number in list[1:]:
        if number == previous_number + 1:
            previous_number = number
        else:
            yield [range_start, previous_number]
            range_start = previous_number = number
    yield [range_start, previous_number]

def file_edit(path,line_id,line_data,header):
    
    '''
    Edits (or create) the file given in the path and replaces/add the line(s) where the line_id str/LIST is with
    the line-content str/LIST.
    
    line_id should be included in line_content.
    
    Header is the first line (or lines if it's a list/array) of the file, with usually different informations.
    '''
    
    lines=[]
    if type(line_id)==str or type(line_id)==np.str_:
        line_identifier=[line_id]
    else:
        line_identifier=line_id
        
    if type(line_data)==str or type(line_data)==np.str_:
        line_content=[line_data]
    else:
        line_content=line_data

    '''
    Complicated mess to avoid several edits to a single file with i.e. parallel processing
    '''

    path_dir = '/'.join(path.split('/')[:-1])
    file_temp = path.split('/')[-1]
    file_temp_prefix = file_temp[:file_temp.rfind('.')]

    import random
    random_n = random.random()

    path_temp = os.path.join(path_dir, file_temp_prefix + '_temp_' + str(random_n) + file_temp[file_temp.rfind('.'):])

    # creating the temp_file
    with open(path_temp, 'w+') as f:
        pass

    # waiting 1 second (so that in case of parallel procs, all the temp files are created
    time.sleep(1)

    curr_files_temp = [elem for elem in glob.glob(os.path.join(path_dir, '**')) if
                       file_temp_prefix + '_temp' in elem.split('/')[-1]]

    curr_files_temp.sort()

    iter_concu=0

    while len(curr_files_temp) > 1:

        # waiting if this is not the first of the temp file, otherwise going forward
        if path_temp == curr_files_temp[0]:
            break

        time.sleep(0.1)
        curr_files_temp = [elem for elem in glob.glob(os.path.join(path_dir, '**')) if
                           file_temp_prefix + '_temp' in elem.split('/')[-1]]

        curr_files_temp.sort()

        iter_concu+=1

        assert iter_concu<30,'Issue with '+path+' overwriting. Check for dead temps'
    #only reading the file after the wait to get the last version

    if os.path.isfile(path):
        with open(path) as file:
            lines=file.readlines()
            
            #loop for all the lines to add
            for single_identifier,single_content in zip(line_identifier,line_content):
                line_exists=False
                if not single_content.endswith('\n'):
                    single_content+='\n'
                    
                #loop for all the lines in the file
                for l,single_line in enumerate(lines):
                    if single_line.startswith(single_identifier):
                        
                        lines[l]=single_content
                        line_exists=True
                if line_exists==False:
                    lines+=[single_content]
                
    else:
        #adding everything
        lines=line_content

    #here we use 0 as an identifier for a non line
    
    if type(header) is np.ndarray:
        header_eff=header.tolist()
    elif type(header) is list:
        header_eff=header
    else:
        header_eff=[header]


    with open(path_temp,'w+') as file:
        #adding the header lines
        
        if lines[:len(header_eff)]==header_eff:
            file.writelines(lines)
        else:
            file.writelines(header_eff+lines)

    if os.path.isfile(path):
        os.remove(path)
    os.system('mv '+path_temp+' '+path)

def print_log(elem,logfile_io,silent=False):

    '''
    prints and logs at once
    '''

    if not silent:
        print(elem)

    if logfile_io is not None:
        logfile_io.write(str(elem)+('\n' if not str(elem).endswith('\n') else ''))

class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh, nints=10):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        dmlower = majorlocs[1] - majorlocs[0]    # major tick difference at lower end
        dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) \
                                   or (dmlower == self.linthresh and majorlocs[0] < 0)):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) \
                                    or (dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                          '%s type.' % type(self))