#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:33:19 2022

@author: parrama
"""

#general imports
import warnings


import numpy as np
import pandas as pd


#matplotlib imports

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates

import dill

from astropy.time import Time
from astroquery.simbad import Simbad


#correlation values and uncertainties with MC distribution from the uncertainties
from custom_pymccorrelation import pymccorrelation

#Note : as of the writing of this code, the standard pymccorrelation doesn't accept differing +/- uncertainties, so I tweaked their 
#'perturb values' function

# import time

'''Custom'''

from pathlib import Path
import sys


#adding the top directory to the path to avoid issues when importing fitting_tools
sys.path.append(str(Path(__file__).resolve().parent.parent))

#custom script with some lines and fit utilities and variables
from fitting_tools import lines_std,lines_std_names,ravel_ragged



telescope_colors={'XMM':'red',
                  'Chandra':'blue',
                  'NICER':'green',
                  'Suzaku':'magenta',
                  'Swift':'orange'
    }

#inclination, mass and other values

#note: in order to make sure we can use our overlap function, we add a very slight uncertainty to inclinations without uncertainties, 
#to make sur the overlap result with the inclination constraints is not 0
incl_dic={
        '1H1659-487':[57.5,20.5,20.5],
        #grs with both ortographs to swap colors sometimes
        'GRS1915+105':[60,5,5],
        'GRS 1915+105':[60,5,5],
        'MAXIJ1820+070':[74,7,7],
        'H1743-322':[75,3,3],
        '4U1630-472':[67.5,7.5,7.5],
        '4U1630-47':[67.5,7.5,7.5],
        'SwiftJ1658.2-4242':[64,3,2],
        'IGRJ17091-3624':[70,0,0.001],
        'MAXIJ1535-571':[45,45,0],
        '1E1740.7-2942':[50,0,40],
        'GROJ1655-40':[69,2,2],
        'SwiftJ1753.5-0127':[55,7,2],
         #mixed both reflection measurements here
        'EXO1846-031':[56.5,16.5,16.5],
        '4U1957+115':[13,0,0.001],
        'MAXIJ1348-630':[28,3,3],
        'XTEJ1752-223':[49,49,0],
        'MAXIJ1659-152':[70,10,10],
        'GS1354-64':[70,0,0.001],
        'XTEJ1650-500':[47,0,43],
        #for 17451 we only have eclipses constraints
        'IGRJ17451-3022':[70,0,20],
        'SwiftJ1357.2-0933':[80,0,10],
        'XTEJ1652-453':[32,32,0],
        'MAXIJ1803-298':[70,0,20],
        'SAXJ1819.3-2525':[72,4,4],
        'GS2023+338':[67,1,3],
        'XTEJ1550-564':[75,4,4],
        'MAXIJ1305-704':[72,8,5],
        'AT2019wey':[30,30,0],
        '4U1543-475':[21,2,2],
        'GRS1739-278':[33,0,0.001],
        'XTEJ1118+480':[72,2,2],
        'GRS1716-249':[50,10,10]
        
         }

dippers_list=['4U1543-475',
              '4U1630-47',
              'GROJ1655-40',
              'H1743-322',
              'GRS1915+105',
              'GRS 1915+105',
              'IGRJ17091-3624',
              'IGRJ17451-3022',
              'MAXIJ1305-704',
              'MAXIJ1659-152',
              'MAXIJ1803-298',
              'MAXIJ1820+070',
              'SwiftJ1357.2-0933',
              'SwiftJ1658.2-4242',
              'XTEJ1817-330']

#custom distande dictionnary for measurements which are not up to date in blackcat/watchdog
dist_dic={
    'MAXIJ1535-571':[4.1,0.6,0.5],
    'GRS 1915+105':[8.6,1.6,2],
    'GRS1915+105':[8.6,1.6,2],
    'MAXIJ1348-630':[3.39,0.385,0.382],
    'H1743-322':[8.5,0.8,0.8],
    'SwiftJ1357.2-0933':[8,0,0],
    }

#note : only dynamical measurements for BHs
mass_dic={
    '1H1659-487':[5.9,3.6,3.6],
    'GRS 1915+105':[12.4,1.8,2],
    'GRS1915+105':[12,2,2],
    'MAXIJ1820+070':[6.9,1.2,1.2],
    'GROJ1655-40':[5.4,0.3,0.3],
    'SAXJ1819.3-2525':[6.4,0.6,0.6],
    'GS2023+338':[9,0.6,0.2],
    'XTEJ1550-564':[11.7,3.9,3.9],
    'MAXIJ1305-704':[8.9,1.,1.6],
    '4U1543-475':[8.4,1,1],
    'XTEJ1118+480':[7.55,0.65,0.65],
    #NS:
    'XTEJ1701-462':1.4
    }

sources_det_dic=['GRS1915+105','GRS 1915+105','GROJ1655-40','H1743-322','4U1630-47','IGRJ17451-3022']

with open('/app/winds/visualisation/visual_line_vars.pkl','rb') as dump_file:
    dump_dict=dill.load(dump_file)
    
dict_lc_rxte=dump_dict['dict_lc_rxte']


#current number of informations in abslines_infos
n_infos=7

info_str=['equivalent widths','blueshifts','energies','line flux']

info_hid_str=['Hardness Ratio','Flux','Time']

axis_str=['Line equivalent width (eV)','Line blueshift (km/s)','Line energy (keV)',r'line flux (erg/s/cm$^{-2}$)',None,r'Line width (keV)']

axis_hid_str=['Hardness Ratio ([6-10]/[3-6] keV bands)',r'Luminosity in the [3-10] keV band in (L/L$_{Edd}$) units']

#indexes for ratio choices in lines between specific ions/complexes
ratio_choices={
                '25':[3,0],
                '26':[4,1],
                'Ka':[1,0],
                'Kb':[4,3],
              }

#theoretical ratio values
ratio_vals={'Ka':0.545
    
            }

ratio_choices_str={'25': lines_std[lines_std_names[6]]+' over '+lines_std[lines_std_names[3]],
                   '26': lines_std[lines_std_names[7]]+' over '+lines_std[lines_std_names[4]],
                   'Ka': lines_std[lines_std_names[4]]+' over '+lines_std[lines_std_names[4]],
                   'Kb': lines_std[lines_std_names[7]]+' over '+lines_std[lines_std_names[6]]}

#minimum nH value to consider for the colormaps
min_nh=5e-1


def silent_Simbad_query(source_name):

    '''just here to avoid seing the Simbad warning for when no source is found everytime'''
    
    #converting strings into lists to avoid slicing unique names since we're using the query for lists
    if type(source_name) in [str,np.str_]:
        source_id=[source_name]
    else:
        source_id=source_name
        
    with warnings.catch_warnings():
        # warnings.filterwarnings('ignore','.*No known catalog could be found.*',)
        # warnings.filterwarnings('ignore','.*Identifier not found.*',)
        warnings.filterwarnings('ignore',category=UserWarning)
        result=Simbad.query_objects(source_id)
        
    return result
    
def fetch_maxi_lightcurve(ctl_maxi_df,ctl_maxi_simbad,name):
    
    '''
    Attempt to identify a MAXI source corresponding to the name given through Simbad identification
    If the attempt is successful, loads a dataframe contaiting the 1 day MAXI lightcurve for this object, from the MAXI website
    '''

    simbad_query=silent_Simbad_query([name[0].split('_')[0]])
    
    if simbad_query is None:
        return None
    
    if simbad_query['MAIN_ID'] not in ctl_maxi_simbad['MAIN_ID']:
        return None
    
    #we fetch the script id instead of the direct column number because Simbad erases the columns with no match 
    #(-1 because the id starts at 1)
    source_id=ctl_maxi_simbad['SCRIPT_NUMBER_ID'][ctl_maxi_simbad['MAIN_ID']==simbad_query['MAIN_ID']][0]-1
    
    source_lc=pd.read_csv(ctl_maxi_df['standard'][source_id],names=['MJDcenter','2-20keV[ph/s/cm2]','err_2-20',
                                              '2-4keV','err_2-4','4-10keV','err_4-10','10-20keV','err_10-20'],sep=' ')

    return source_lc

def fetch_rxte_lightcurve(name,dict_rxte=dict_lc_rxte):
    
    '''
    Attempts to fetch a lightcurve for the given namen in the lightcurve rxte dictionnary
    
    the dictionnary has dataframes of all the main band 1-day averaged lightcurves found in the RXTE archive for the sample,
    with simbad ids as keys
    '''
    
    simbad_query=silent_Simbad_query([name[0].split('_')[0]])
    
    
    if simbad_query is None:
        return None

    if simbad_query[0]['MAIN_ID'] not in dict_lc_rxte.keys():
        return None
    
    return dict_lc_rxte[simbad_query[0]['MAIN_ID']]

def plot_lightcurve(dict_linevis,ctl_maxi_df,ctl_maxi_simbad,name,dict_rxte=dict_lc_rxte,mode='full',display_hid_interval=True,superpose_ew=False):

    '''
    plots various MAXI based lightcurve for sources in the Sample if a match is found in the MAXI source list
    
    full : full lightcurve in 2-20 keV
    HR : HR in 4-10/2-4 bands
    '''
    
    slider_date=dict_linevis['slider_date']
    zoom_lc=dict_linevis['zoom_lc']
    mask_obj=dict_linevis['mask_obj']
    #for the two variables below we fetch the first element because they are 1 size arrays containing the info for the only source displayed
    date_list=dict_linevis['date_list'][mask_obj][0]
    instru_list=dict_linevis['instru_list'][mask_obj][0]
    
    #for the EW superposition
    abslines_plot_restrict=dict_linevis['abslines_plot_restrict']
    mask_lines=dict_linevis['mask_lines']
    conf_thresh=dict_linevis['slider_sign']
    
    maxi_lc_df=fetch_maxi_lightcurve(ctl_maxi_df,ctl_maxi_simbad,name)
    
    rxte_lc_df=fetch_rxte_lightcurve(name, dict_lc_rxte)
    
    if maxi_lc_df is None and rxte_lc_df is None:
        return None
    
    fig_lc,ax_lc=plt.subplots(figsize=(12,4))
    
    #main axis definitions
    ax_lc.set_xlabel('Time')
    if mode=='full':
        ax_lc.set_title(name[0]+' broad band lightcurve')
        
        maxi_y_str='MAXI '+maxi_lc_df.columns[1] if maxi_lc_df is not None else ''
        rxte_y_str='RXTE '+rxte_lc_df.columns[1]+'/25' if rxte_lc_df is not None else ''
        ax_lc.set_ylabel(maxi_y_str+('/' if maxi_lc_df is not None and rxte_lc_df is not None else '')+rxte_y_str)
    elif mode=='HR':
        ax_lc.set_title(name[0]+' HR ratio evolution')
        ax_lc.set_ylabel('MAXI counts HR in [4-10]/[2-4] bands')
        
        maxi_y_str='MAXI counts HR in [4-10]/[2-4] keV' if maxi_lc_df is not None else ''
        rxte_y_str='RXTE band C/(B+A) [5-12]/[1.5-5] keV' if rxte_lc_df is not None else ''
        ax_lc.set_ylabel(maxi_y_str+\
                         ("/" if maxi_lc_df is not None and rxte_lc_df is not None else '')+\
                         rxte_y_str,fontsize=8 if maxi_lc_df is not None and rxte_lc_df is not None else None)
        
        
    '''MAXI'''
    
    #empty list initialisation to append easily the dates list afterwards no matter 
    num_maxi_dates=[]
    num_rxte_dates=[]
    
    if rxte_lc_df is not None:
            
        #creating a variable for the dates
        num_rxte_dates=mdates.date2num(Time(rxte_lc_df[rxte_lc_df.columns[0]],format='mjd').datetime)
        
        if mode=='full':    
            ax_lc.set_yscale('log')
            
            #plotting the full lightcurve
            #note: the conversion factor is here to convert(ish) the rxte counts into maxi counts
            ax_lc.errorbar(num_rxte_dates,rxte_lc_df[rxte_lc_df.columns[1]]/25,xerr=0.5,yerr=rxte_lc_df[rxte_lc_df.columns[2]]/25,
                        linestyle='',color='sienna',marker='',elinewidth=0.5,label='RXTE standard counts')
            ax_lc.set_ylim(0.1,ax_lc.get_ylim()[1])
    
        if mode=='HR':
            
            '''
            for RXTE, 
            band A is [1.5-3] keV
            band B is [3-5] keV
            band C is [5-12] keV
            
            for the HR, we do C/A+B
            '''
            
            #computing the HR evolution and uncertainties
            
            rxte_hr_denom=rxte_lc_df[rxte_lc_df.columns[3]]+rxte_lc_df[rxte_lc_df.columns[5]]
            rxte_hr=rxte_lc_df[rxte_lc_df.columns[7]]/rxte_hr_denom
            
            rxte_hr_err=(rxte_lc_df[rxte_lc_df.columns[8]]/rxte_lc_df[rxte_lc_df.columns[7]]+(rxte_lc_df[rxte_lc_df.columns[4]]+rxte_lc_df[rxte_lc_df.columns[6]])/rxte_hr_denom)*rxte_hr
            
            ax_lc.set_yscale('log')
            
            ax_lc.set_ylim(0.05,5)
            
            #plotting the full lightcurve
            rxte_hr_errbar=ax_lc.errorbar(num_rxte_dates,rxte_hr,xerr=0.5,yerr=rxte_hr_err,
                        linestyle='',color='sienna',marker='',elinewidth=0.5,label='RXTE HR')
    
            #adapting the transparency to hide the noisy elements
            rxte_hr_alpha_val=1/(20*abs(rxte_hr_err/rxte_hr)**2)
            rxte_hr_alpha=[min(1,elem) for elem in rxte_hr_alpha_val]
            
            #replacing indiviudally the alpha values for each point but the line
            for elem_children in rxte_hr_errbar.get_children()[1:]:
    
                elem_children.set_alpha(rxte_hr_alpha)
                
    if maxi_lc_df is not None:
            
        #creating a variable for the dates
        num_maxi_dates=mdates.date2num(Time(maxi_lc_df[maxi_lc_df.columns[0]],format='mjd').datetime)
        
        if mode=='full':    
            ax_lc.set_yscale('log')
            
            #plotting the full lightcurve
            ax_lc.errorbar(num_maxi_dates,maxi_lc_df[maxi_lc_df.columns[1]],xerr=0.5,yerr=maxi_lc_df[maxi_lc_df.columns[2]],
                        linestyle='',color='black',marker='',elinewidth=0.5,label='MAXI standard counts')
            ax_lc.set_ylim(0.1,ax_lc.get_ylim()[1])
    
        if mode=='HR':
            #computing the HR evolution and uncertainties
            maxi_hr=maxi_lc_df[maxi_lc_df.columns[5]]/maxi_lc_df[maxi_lc_df.columns[3]]
            
            maxi_hr_err=(maxi_lc_df[maxi_lc_df.columns[6]]/maxi_lc_df[maxi_lc_df.columns[5]]+maxi_lc_df[maxi_lc_df.columns[4]]/maxi_lc_df[maxi_lc_df.columns[3]])*maxi_hr
            ax_lc.set_yscale('log')
            
            ax_lc.set_ylim(0.05,5)
            
            #plotting the full lightcurve
            maxi_hr_errbar=ax_lc.errorbar(num_maxi_dates,maxi_hr,xerr=0.5,yerr=maxi_hr_err,
                        linestyle='',color='black',marker='',elinewidth=0.5,label='MAXI HR')
    
            #adapting the transparency to hide the noisy elements
            maxi_hr_alpha_val=1/(20*abs(maxi_hr_err)*2)/abs(maxi_hr)
            maxi_hr_alpha=[min(1,elem) for elem in maxi_hr_alpha_val]
            
            #replacing indiviudally the alpha values for each point but the line
            for elem_children in maxi_hr_errbar.get_children()[1:]:
    
                elem_children.set_alpha(maxi_hr_alpha)

        # ax_lc.set_ylim(0.1,ax_lc.get_ylim()[1])
        
    #displaying observations with other instruments
    
    label_tel_list=[]
        
    if superpose_ew:
        #creating a second y axis with common x axis
        ax_lc_ew=ax_lc.twinx()
        ax_lc_ew.set_yscale('log')
        ax_lc_ew.set_ylabel('EW (eV)')
        ax_lc_ew.set_ylim(5,100)
        
        #plotting the detection and upper limits following what we do for the scatter graphs
    
        date_list_repeat=np.array([date_list for repeater in range(sum(mask_lines))])

        instru_list_repeat=np.array([instru_list for repeater in range(sum(mask_lines))])
        
        #these boolean arrays distinguish non detections (i.e. 0/nan significance) and statistically significant detections from the others            
        bool_sign=ravel_ragged(abslines_plot_restrict[4][0]).astype(float)
        
        #standard detection mask (we don't need the intime here since the graph bounds will be cut if needed
        bool_det=(bool_sign!=0.) & (~np.isnan(bool_sign))
        
        #mask used for upper limits only
        bool_nondetsign=((bool_sign<=conf_thresh) | (np.isnan(bool_sign)))
        
        #restricted significant mask, to be used in conjunction with bool_det
        bool_sign=bool_sign[bool_det]>=conf_thresh

        x_data_det=mdates.date2num(ravel_ragged(date_list_repeat))[bool_det][bool_sign]
        y_data_det=ravel_ragged(abslines_plot_restrict[0][0])[bool_det][bool_sign]
        
        y_error_det=np.array([ravel_ragged(abslines_plot_restrict[0][1])[bool_det][bool_sign],
                              ravel_ragged(abslines_plot_restrict[0][2])[bool_det][bool_sign]]).T
        
        x_data_ul=mdates.date2num(ravel_ragged(date_list_repeat))[bool_nondetsign]
        y_data_ul=ravel_ragged(abslines_plot_restrict[5][0])[bool_nondetsign]
                    
        color_det=[telescope_colors[elem] for elem in ravel_ragged(instru_list_repeat)[bool_det][bool_sign]]
        
        color_ul=[telescope_colors[elem] for elem in ravel_ragged(instru_list_repeat)[bool_nondetsign]]
        
        #zipping the errorbars to allow different colors
        for x_data,y_data,y_err,color in zip(x_data_det,y_data_det,y_error_det,color_det):
            
            ax_lc_ew.errorbar(x_data,y_data,xerr=0.5,yerr=np.array([y_err]).T,color=color,marker='d',markersize=2,elinewidth=1)
            

        for x_data,y_data,color in zip(x_data_ul,y_data_ul,color_ul):
         
            ax_lc_ew.errorbar(x_data,y_data,xerr=0.5,yerr=0.05*y_data,marker='.',color=color,uplims=True,markersize=2,elinewidth=1,capsize=2)

                
    for i_obs,date_obs in enumerate(date_list):
        
        num_date_obs=mdates.date2num(Time(date_obs).datetime)
        
        #we add a condition for the label to only plot each instrument once
        ax_lc.axvline(x=num_date_obs,ymin=0,ymax=1,color=telescope_colors[instru_list[i_obs]],
                        label=instru_list[i_obs]+' exposure' if instru_list[i_obs] not in label_tel_list else '',ls=':',lw=0.5)

        if instru_list[i_obs] not in label_tel_list:
            label_tel_list+=[instru_list[i_obs]]
        
    #resizing the x axis and highlighting depending on wether we are zooming on a restricted time interval or not
    
    tot_dates_list=[]
    tot_dates_list+=[] if maxi_lc_df is None else num_maxi_dates.tolist()
    tot_dates_list+=[] if rxte_lc_df is None else num_rxte_dates.tolist()
    
    if zoom_lc:
        ax_lc.set_xlim(max(mdates.date2num(slider_date[0]),min(tot_dates_list)),
                                             min(mdates.date2num(slider_date[1]),max(tot_dates_list)))
        time_range=min(mdates.date2num(slider_date[1]),max(tot_dates_list))-max(mdates.date2num(slider_date[0]),min(tot_dates_list))
        
    else:
        ax_lc.set_xlim(min(tot_dates_list),max(tot_dates_list))
        time_range=max(tot_dates_list)-min(tot_dates_list)
        
        #highlighting the time interval in the main HID
        
        if display_hid_interval:
            plt.axvspan(mdates.date2num(slider_date[0]),mdates.date2num(slider_date[1]),0,1,color='grey',alpha=0.3,label='HID interval')
        
    #creating an appropriate date axis
    #manually readjusting for small durations because the AutoDateLocator doesn't work well
    if time_range<150:
        date_format=mdates.DateFormatter('%Y-%m-%d')
    else:
        date_format=mdates.DateFormatter('%Y-%m')
    # else:
    #     date_format=mdates.AutoDateFormatter(mdates.AutoDateLocator())

    ax_lc.xaxis.set_major_formatter(date_format)
                    
    for label in ax_lc.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')
            
    ax_lc.legend()
    
    return fig_lc

def distrib_graph(data_perinfo,info,dict_linevis,data_ener=None,conf_thresh=0.99,indiv=False,save=False,close=False,streamlit=False,bigger_text=False):
    
    '''
    repartition diagram from all the observations in the current pool (i.e. all objects/obs/lines).
    
    Use the 'info' keyword to graph flux,eqwidth, bshift or ener
    
    Use the 'indiv' keyword to plot for all lines simultaneously or 6 plots for the 6 single lines
    
    Non detections are filtered via 0 values in significance
    
    Detections above and below the given threshold are highlighted
    
    we ravel the last 2 dimensions with a custom function since for multi object plots the sequences can be ragged and the custom .ravel function
    doesn't work
    '''
    n_infos=dict_linevis['n_infos']
    range_absline=dict_linevis['range_absline']
    mask_lines=dict_linevis['mask_lines']
    bins_bshift=dict_linevis['bins_bshift']
    bins_ener=dict_linevis['bins_ener']
    if streamlit:
        display_nonsign=dict_linevis['display_nonsign']
    else:
        display_nonsign=True
        
    save_dir=dict_linevis['save_dir']
    save_str_prefix=dict_linevis['save_str_prefix']
    args_cam=dict_linevis['args_cam']
    args_line_search_e=dict_linevis['args_line_search_e']
    args_line_search_norm=dict_linevis['args_line_search_norm']

    ind_info=np.argwhere([elem in info for elem in ['eqw','bshift','ener','lineflux']])[0][0]
    
    data_plot=data_perinfo[ind_info] if ind_info!=2 else data_ener
    
    if 'ratio' in info:
        ratio_mode=True
    else:
        ratio_mode=False
    
    if indiv:
        graph_range=range_absline
    else:
        #using a list index for the global graphs allows us to keep the same line structure
        #however we need to restrict it to the currently plotted lines in streamlit mode
        if streamlit:
            graph_range=[range(len([elem for elem in mask_lines if elem]))]
        else:
            graph_range=[range_absline]

    #computing the range of the eqw bins from the global graph to get the same scale for all individual graphs)

    bins_eqw=np.geomspace(1,max(100,(max(ravel_ragged(data_perinfo[0][0]))//5+1)*5),20)
    
    bins_eqwratio=np.geomspace(0.5,5,20)
    
    if n_infos>=5:
        if len(ravel_ragged(data_perinfo[3][0])[ravel_ragged(data_perinfo[4][0])>0])>0 and len(ravel_ragged(data_perinfo[3][0]).nonzero()[0])!=0:
            
            bins_flux=np.geomspace(max(1e-16,min(1e-13,(min(ravel_ragged(data_perinfo[3][0])[ravel_ragged(data_perinfo[4][0])>0])*0.9))),
                                   max(1e-10,(max(ravel_ragged(data_perinfo[3][0])[ravel_ragged(data_perinfo[4][0])>0])*1.1)),20)
        else:
            bins_flux=np.geomspace(1e-13,1e-10,20)
    else:
        bins_flux=None

    #sorting the bins in an array depending on the info asked
    bins_info=[bins_eqw,bins_bshift,bins_ener,bins_flux]
    
    bins_ratio=[bins_eqwratio]
    
    #and fetching the current one    
    hist_bins=bins_info[ind_info] if not ratio_mode else bins_ratio[ind_info]
    
    #fetching the global boolean array for the graph size definition
    bool_det_glob=ravel_ragged(data_perinfo[4][0]).astype(float)

    #the "and" keyword doesn't work for arrays so we use & instead (which requires parenthesis)
    bool_det_glob=(bool_det_glob!=0.) & (~np.isnan(bool_det_glob))
    
    #computing the maximal height of the graphs (in order to keep the same y axis scaling on all the individual lines graphs)

    max_height=max((np.histogram(ravel_ragged(data_plot[0])[bool_det_glob],bins=bins_info[ind_info]))[0])
       
    for i in graph_range:
    
        if ratio_mode:
            
            #we fetch the index list corresponding to the info string at the end of the info provided
            ratio_indexes_x=ratio_choices[info[-2:]]
            
        fig_hist,ax_hist=plt.subplots(1,1,figsize=(6,6) if bigger_text else (10,8))
        
        if not bigger_text:
            if indiv:
                fig_hist.suptitle('Repartition of the '+info_str[ind_info]+' of the '+lines_std[lines_std_names[3+i]]+' absorption line')
            else:
                if ratio_mode:
                    fig_hist.suptitle('Repartition of the '+ratio_choices_str[info[-2:]]+' ratio in currently selected sources')
                else:
                    fig_hist.suptitle('Repartition of the '+info_str[ind_info]+
                                  (' in currently selected lines and sources' if streamlit else ' absorption lines'))
            
        ax_hist.set_xlabel(axis_str[ind_info])
        
        #using a log x axis if the 
        ax_hist.set_ylabel(r'Detection')
        ax_hist.set_ylim([0,max_height+0.25])
        
        if ind_info in [0,3]:
            ax_hist.set_xscale('log')
        #these boolean arrays distinguish non detections (i.e. 0/nan significance) and statistically significant detections from the others

        bool_sign=ravel_ragged(data_perinfo[4][0][i]).astype(float)
        bool_det=(bool_sign!=0.) & (~np.isnan(bool_sign))
        bool_sign=bool_sign[bool_det]>=conf_thresh
        
        #this time data_plot is an array
        if ratio_mode:
            
            #we need to create restricted bool sign and bool det here, with only the index of each line
            sign_ratio_arr=np.array([ravel_ragged(data_perinfo[4][0][ratio_indexes_x[i]]).astype(float) for i in [0,1]])
            
            #here we create different mask variables to keep the standard bool_det/sign for the other axis if needed
            bool_det_ratio=(sign_ratio_arr[0]!=0.) & (~np.isnan(sign_ratio_arr[0])) & (sign_ratio_arr[1]!=0.) & (~np.isnan(sign_ratio_arr[1]))
            
            #same problem with bitwise comparison here
            bool_sign_ratio=bool_det_ratio[sign_ratio_arr.T.min(1)>=conf_thresh]
            
            
            #before using them to create the data ratio
            hist_data=np.array(
                  [ravel_ragged(data_plot[0][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio]/\
                   ravel_ragged(data_plot[0][ratio_indexes_x[1]])[bool_det_ratio][bool_sign_ratio],
                   ravel_ragged(data_plot[0][ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio]/\
                   ravel_ragged(data_plot[0][ratio_indexes_x[1]])[bool_det_ratio][~bool_sign_ratio]],dtype=object)
                    
        else:
            hist_data=[ravel_ragged(data_plot[0][i])[bool_det][bool_sign],ravel_ragged(data_plot[0][i])[bool_det][~bool_sign]]
        
        hist_cols=['blue','grey']
        hist_labels=['detections above '+str(conf_thresh*100)+'% treshold','detections below '+str(conf_thresh*100)+'% treshold']
        
        if display_nonsign:
            ax_hist.hist(hist_data,stacked=True,color=hist_cols,label=hist_labels,bins=hist_bins)
        else:
            ax_hist.hist(hist_data[0],color=hist_cols[0],label=hist_labels[0],bins=hist_bins)
            
        #forcing only integer ticks on the y axis
        ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))

        if not bigger_text:
            plt.legend()
        plt.tight_layout()
        if save:
            if indiv:
                suffix_str='_'+lines_std_names[3+i]
            else:
                suffix_str='_all'
                
            plt.savefig(save_dir+'/graphs/distrib/'+save_str_prefix+'autofit_distrib_'+info+suffix_str+'_cam_'+args_cam+'_'+\
                            args_line_search_e.replace(' ','_')+'_'+args_line_search_norm.replace(' ','_')+'.png')
        if close:
            plt.close(fig_hist)
        
    #returning the graph for streamlit display
    if streamlit:
        return fig_hist
    
def correl_graph(data_perinfo,infos,data_ener,dict_linevis,mode='intrinsic',mode_vals=None,conf_thresh=0.99,indiv=False,save=False,close=False,
                 streamlit=False,compute_correl=False,bigger_text=False,show_linked=True,show_ul_eqw=False):
    
    '''
    Scatter plots for specific line parameters from all the observations in the current pool (i.e. all objects/obs/lines).
    
    infos:
        Intrinsic:
            -eqwidth
            -bshift
            -ener
            -eqwratio+25/26/Ka/Kb : ratio between the two 25/26/Ka/Kb lines. Always uses the highest line (in energy) as numerator
            -width
            -nH
            
        observ:
            -HR
            -flux
            -Time

        source:
            -inclin
    Modes:
        -Intrinsic    Use the 'info' keyword to plot infos vs one another
        -observ          Use the 'info' keyword to plot infos vs HR or flux (observ info always on y axis)
        -source       Use the 'info' keyword to plot eqwidth, bshift, ener or flux vs the source inclinations
        -eqwratio   Use the 'info' keyword to specify which the lines for which the eqw will be compared
        
    In HID mode, requires the mode_vals variable to be set to the flux values
    in inclin mode, idem with the inclination values
    
    Color modes (not an argument, ruled by the dicionnary variable color_scatter):
        -None
        -Instrument
        -Sources
        -Time
        -HR
        -width
        
    (Note : displaying non significant results overrides the coloring)
    
    Use the 'indiv' keyword to plot for all lines simultaneously or 6 plots for the 6 single lines
    the streamlit mode uses the indiv=False since the lines are already selected in the data provided to the function
    
    Non detections are filtered via 0 values in significance
    
    Detections above and below the given threshold and linked values are highlighted if the checkbox to display non significant detectoins is set, else 
    only significant detections are displayed
    
    The person and spearman coefficients are computed from significant detections only, with MC propagation of their uncertainties through a custom library
    (https://github.com/privong/pymccorrelation)
    As such, the distribution of the errors are assumed to be gaussian 
    (which is not the case, but using the distributions themselves would be a nightmare)
    
    we ravel the last 2 dimensions with a custom function since for multi object plots the sequences can be ragged and the custom .ravel function
    doesn't work

    Note : there's an explicit call to the FeKg26 being the last of the tested abs line (and assumption that Kb25 and Kb26 are just before), 
    so the uncertainty shift part will have to be changed if this is modified
    '''
    
    range_absline=dict_linevis['range_absline']
    mask_lines=dict_linevis['mask_lines']

    if streamlit:
        display_nonsign=dict_linevis['display_nonsign']
        scale_log_hr=dict_linevis['scale_log_hr']
        scale_log_eqw=dict_linevis['scale_log_eqw']
        plot_trend=dict_linevis['plot_trend']
        lock_lims_det=dict_linevis['lock_lims_det']
        
    else:
        display_nonsign=True
        scale_log_hr=True
        scale_log_eqw=True
        plot_trend=False
        lock_lims_det=False
        
    mask_obj=dict_linevis['mask_obj']
    abslines_ener=dict_linevis['abslines_ener']
    abslines_plot=dict_linevis['abslines_plot']

    
    save_dir=dict_linevis['save_dir']
    save_str_prefix=dict_linevis['save_str_prefix']
    args_cam=dict_linevis['args_cam']
    args_line_search_e=dict_linevis['args_line_search_e']
    args_line_search_norm=dict_linevis['args_line_search_norm']
    
    if streamlit:
        slider_date=dict_linevis['slider_date']
        instru_list=dict_linevis['instru_list'][mask_obj]
    
    if 'color_scatter' in dict_linevis:    
        color_scatter=dict_linevis['color_scatter']
        obj_disp_list=dict_linevis['obj_list'][mask_obj]
        date_list=dict_linevis['date_list'][mask_obj]
        hid_plot=dict_linevis['hid_plot_restrict']
        width_plot_restrict=dict_linevis['width_plot_restrict']
        nh_plot_restrict=dict_linevis['nh_plot_restrict']
    
    else:
        color_scatter='None'
        date_list=None
        hid_plot=None
        width_plot_restrict=None
        nh_plot_restrict=None
    
    observ_list=dict_linevis['observ_list'][mask_obj]
    

    
    n_obj=len(observ_list)
    
    infos_split=infos.split('_')
    alpha_ul=0.3
    
    x_error=None
    y_error=None
    
    #using time changes the way things are plotted, uncertainties are computed and significance are used
    if 'time' in infos_split[0]:
        time_mode=True
        infos_split=infos_split[::-1]
    else:
        time_mode=False
    
    #using ratios changes the way things are plotted, uncertainties are computed and significance are used
    if 'ratio' in infos:
        ratio_mode=True
        
        if 'ratio' in infos_split[0]:
            #fetching the line indexes corresponding to the names
            ratio_indexes_x=ratio_choices[infos_split[0][-2:]]
        
            #keeping the same architecture in time mode even if the ratio is plotted on the y axis for simplicity
        elif 'ratio' in infos_split[1]:
            #fetching the line indexes corresponding to the names
            ratio_indexes_x=ratio_choices[infos_split[1][-2:]]
    else:
        ratio_mode=False
        
    #not showing upper limit for the width plot as there's no point currently
    if 'width' in infos:
        show_ul_eqw=False
        
    #failsafe to prevent wrong colorings for intrinsic plots
    if (ratio_mode or mode=='eqwratio') and color_scatter=='width':
        color_scatter='None'
        
    data_list=[data_perinfo[0],data_perinfo[1],data_ener,data_perinfo[3],date_list,width_plot_restrict]
        
    #infos and data definition depending on the mode
    if mode=='intrinsic':

        ind_infos=[np.argwhere([elem in infos_split[i] for elem in ['eqw','bshift','ener','lineflux','time','width']])[0][0] for i in [0,1]]

        data_plot=[data_list[ind] for ind in ind_infos]
        
    elif mode=='observ':
        #the 'None' is here to have the correct index for the width element
        ind_infos=[np.argwhere([elem in infos_split[0] for elem in ['eqw','bshift','ener','lineflux','None','width']])[0][0],
                    np.argwhere(np.array(['HR','flux','time'])==infos_split[1])[0][0]]

        data_plot=[data_list[ind_infos[0]], date_list if ind_infos[1]==2 else mode_vals[ind_infos[1]]]
        
        if time_mode:
            data_plot=data_plot[::-1]
        
    elif mode=='source':
        ind_infos=[np.argwhere([elem in infos_split[0] for elem in ['eqw','bshift','ener','lineflux','None','width']])[0][0],-1]
        data_plot=[data_list[ind_infos[0]], mode_vals]
        
    elif mode=='eqwratio':
        ind_infos=[np.argwhere([elem in infos_split[i] for elem in lines_std_names[3:]])[0][0] for i in [0,1]]
        data_plot=[data_perinfo[0],data_perinfo[0]]
    
    if indiv:
        graph_range=range_absline
    else:
        #using a list index for the global graphs allows us to keep the same line structure
        #however we need to restrict it to the currently plotted lines in streamlit mode
        if streamlit:
            graph_range=[range(len([elem for elem in mask_lines if elem]))]
        else:
            graph_range=[range_absline]

        
    for i in graph_range:
        
        figsize_val=(6,6)
        fig_scat,ax_scat=plt.subplots(1,1,figsize=figsize_val if bigger_text else (11,10.5))
        
        if not bigger_text:
            if indiv:
                fig_scat.suptitle(info_str[ind_infos[0]]+' - '+(info_str[ind_infos[1]] if mode=='intrinsic' else\
                                  info_hid_str[ind_infos[1]])+' scatter for the '+lines_std[lines_std_names[3+i]]+
                                  ' absorption line')
            else:
                fig_scat.suptitle(((infos_split[0]+' - '+infos_split[1]+' equivalent widths') if mode=='eqwratio' else\
                    ((info_str[ind_infos[0]] if not ratio_mode else infos_split[0])+' - '+(info_str[ind_infos[1]] if mode=='intrinsic' else\
                              (info_hid_str[ind_infos[1]]) if mode=='observ' else 'inclination')))\
                              +(' for currently selected '+('lines and ' if not ratio_mode else '')+'sources' if streamlit else ' for all absorption lines'))
            
        if mode!='eqwratio':

            ax_scat.set_xlabel('Time' if time_mode else (axis_str[ind_infos[0]]+' '+(infos_split[0][-2:]+' ratio')) if ratio_mode else\
                               axis_str[ind_infos[0]])

        else:
            ax_scat.set_xlabel(infos_split[0]+' '+axis_str[ind_infos[0]])
            ax_scat.set_ylabel(infos_split[1]+' '+axis_str[ind_infos[0]])
            
        #putting a logarithmic y scale if it shows equivalent widths or one of the hid parameters
        if mode!='source' and ((mode=='observ' and scale_log_hr and not time_mode) or ((ind_infos[0 if time_mode else 1] in [0,3] or mode=='eqwratio') and scale_log_eqw)):
                            
            ax_scat.set_yscale('log')

        #logarithmic scale by default for eqw ratios
        if ratio_mode:
            if time_mode:
                ax_scat.set_yscale('log')
            else:
                ax_scat.set_xscale('log')
        
        #putting a logarithmic x scale if it shows equivalent widths
        if ind_infos[0] in [0,3] and scale_log_eqw and not time_mode:
            ax_scat.set_xscale('log')

        
        if mode=='intrinsic':
            ax_scat.set_ylabel(axis_str[ind_infos[1]])
        elif mode=='observ':
            ax_scat.set_ylabel(axis_str[ind_infos[0]] if (time_mode and not ratio_mode) else\
                               axis_str[ind_infos[0]]+' '+(infos_split[0][-2:]+' ratio') if time_mode else\
                               axis_hid_str[ind_infos[1]])
        elif mode=='source':
            ax_scat.set_ylabel('Source inclination (??)')
            ax_scat.set_ylim((0,90))

        #creating a time variable for time mode to be used later
        
        if time_mode:
            
            #creating an appropriate date axis
            #manually readjusting for small durations because the AutoDateLocator doesn't work well
            time_range=min(mdates.date2num(slider_date[1]),max(mdates.date2num(ravel_ragged(date_list))))-\
                       max(mdates.date2num(slider_date[0]),min(mdates.date2num(ravel_ragged(date_list))))
            
            if time_range<150:
                date_format=mdates.DateFormatter('%Y-%m-%d')
            elif time_range<1825:
                date_format=mdates.DateFormatter('%Y-%m')
            else:
                date_format=mdates.AutoDateFormatter(mdates.AutoDateLocator())
            ax_scat.xaxis.set_major_formatter(date_format)
                            
            plt.xticks(rotation=70)

        date_list_repeat=np.array([date_list for repeater in (i if type(i)==range else [i])]) if not ratio_mode else date_list
        
        if streamlit:
            mask_intime=(Time(ravel_ragged(date_list_repeat))>=slider_date[0]) & (Time(ravel_ragged(date_list_repeat))<=slider_date[1])
        
            mask_intime_norepeat=(Time(ravel_ragged(date_list))>=slider_date[0]) & (Time(ravel_ragged(date_list))<=slider_date[1])
        else:
            mask_intime=True
        
            mask_intime_norepeat=True
            
        #the boolean masks for detections and significance are more complex when using ratios instead of the standard data since 
        #two lines need to be checked
        if mode=='eqwratio' or ratio_mode:
                                
            #we can use the data constructs used for the eqw ratio mode to create the ratios in ratio_mode
            #we just need to get common indexing variable
            ind_ratio=ind_infos if mode=='eqwratio' else ratio_indexes_x
                
            #in eqw ratio mode we need to make sure than both lines are defined for each point so we must combine the mask of both lines
            bool_sign_x=ravel_ragged(data_perinfo[4][0][ind_ratio[0]]).astype(float)
            bool_sign_y=ravel_ragged(data_perinfo[4][0][ind_ratio[1]]).astype(float)
            
            bool_det_ratio=(bool_sign_x!=0.) & (~np.isnan(bool_sign_x)) & (bool_sign_y!=0.) & (~np.isnan(bool_sign_y)) & mask_intime_norepeat
            
            #for whatever reason we can't use the bitwise comparison so we compute the minimum significance of the two lines before testing for a single arr
            bool_sign_ratio=np.array([bool_sign_x[bool_det_ratio],bool_sign_y[bool_det_ratio]]).T.min(1)>=conf_thresh
            
            #masks for upper limits (needs at least one axis to have detection and significance)
            bool_nondetsign_x=np.array(((bool_sign_x<=conf_thresh).tolist() or (np.isnan(bool_sign_x).tolist()))) &\
                              (np.array((bool_sign_y>=conf_thresh).tolist()) & np.array((~np.isnan(bool_sign_y)).tolist())) & mask_intime_norepeat
            bool_nondetsign_y=np.array(((bool_sign_y<=conf_thresh).tolist() or (np.isnan(bool_sign_y).tolist()))) &\
                              (np.array((bool_sign_x>=conf_thresh).tolist()) & np.array((~np.isnan(bool_sign_x)).tolist())) & mask_intime_norepeat
                        
            #the boool sign and det are only used for the ratio in ratio_mode, but are global in eqwratio mode
            if mode=='eqwratio':
                
                #converting the standard variables to the ratio ones
                bool_det=bool_det_ratio
                bool_sign=bool_sign_ratio
                
                #note: we don't care about the 'i' index of the graph range here since this graph is never made in indiv mode/not all lines mode
                
                #here we can build both axis variables and error variables identically
                x_data,y_data=[np.array([ravel_ragged(data_plot[0][0][ind_infos[i]])[bool_det][bool_sign],
                                         ravel_ragged(data_plot[0][0][ind_infos[i]])[bool_det][~bool_sign]],dtype=object) for i in [0,1]]
                
                #same thing for the uncertainties                
                x_error,y_error=[np.array([[ravel_ragged(data_plot[i][l][ind_infos[i]])[bool_det][bool_sign] for l in [1,2]],                                         
                                           [ravel_ragged(data_plot[i][l][ind_infos[i]])[bool_det][~bool_sign] for l in [1,2]]],
                                          dtype=object) for i in [0,1]]
                
            #here we assume this is ratio_mode
            else:  
            #this time data_plot is an array
            #here, in the ratio X is the numerator so the values are obtained by dividing X/Y
            
                #here we can directly create the data ratio
                if time_mode:

                    x_data=np.array([mdates.date2num(ravel_ragged(date_list_repeat))[bool_det_ratio][bool_sign_ratio],mdates.date2num(ravel_ragged(date_list_repeat))[bool_det_ratio][~bool_sign_ratio]],
                                    dtype=object)

                        
                    #in this case the eqw ratio is on the Y axis
                    y_data=np.array(
                          [ravel_ragged(data_plot[1][0][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio]/\
                           ravel_ragged(data_plot[1][0][ratio_indexes_x[1]])[bool_det_ratio][bool_sign_ratio],
                           ravel_ragged(data_plot[1][0][ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio]/\
                           ravel_ragged(data_plot[1][0][ratio_indexes_x[1]])[bool_det_ratio][~bool_sign_ratio]],dtype=object)            
                else:
                    x_data=np.array(
                          [ravel_ragged(data_plot[0][0][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio]/\
                           ravel_ragged(data_plot[0][0][ratio_indexes_x[1]])[bool_det_ratio][bool_sign_ratio],
                           ravel_ragged(data_plot[0][0][ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio]/\
                           ravel_ragged(data_plot[0][0][ratio_indexes_x[1]])[bool_det_ratio][~bool_sign_ratio]],dtype=object) 
        else:
            
            #these boolean arrays distinguish non detections (i.e. 0/nan significance) and statistically significant detections from the others            
            bool_sign=ravel_ragged(data_perinfo[4][0][i]).astype(float)
            
            #standard detection mask
            bool_det=(bool_sign!=0.) & (~np.isnan(bool_sign)) & (mask_intime)
            
            #mask used for upper limits only
            bool_nondetsign=((bool_sign<=conf_thresh) | (np.isnan(bool_sign))) & (mask_intime)
            
            #restricted significant mask, to be used in conjunction with bool_det
            bool_sign=bool_sign[bool_det]>=conf_thresh
            
            if time_mode:

                x_data=np.array([mdates.date2num(ravel_ragged(date_list_repeat))[bool_det][bool_sign],mdates.date2num(ravel_ragged(date_list_repeat))[bool_det][~bool_sign]],
                                dtype=object)
                y_data=np.array([ravel_ragged(data_plot[1][0][i])[bool_det][bool_sign],ravel_ragged(data_plot[1][0][i])[bool_det][~bool_sign]],
                        dtype=object)

            else:

                x_data=np.array([ravel_ragged(data_plot[0][0][i])[bool_det][bool_sign],ravel_ragged(data_plot[0][0][i])[bool_det][~bool_sign]],
                        dtype=object)
                    
        #applying the same thing to the y axis if ratios are also plotted there
        if type(ind_infos[1]) not in [int,np.int64] and 'ratio' in ind_infos[1]:
                #note:not needed for now
                pass
            # #we fetch the index list corresponding to the info string at the end of the info provided
            # ratio_indexes_y=ratio_choices[ind_infos[1][-2:]]
            
            # #before using them to create the data ratio (here we don't bother using the non significant detections)
            # #we don't bother masking the infinites here because they are masked natively when plotting
            # y_data=np.array(
            #       [ravel_ragged(data_plot[0][0][ratio_indexes_y[0]])[bool_det[ratio_indexes_y[0]]][bool_sign[ratio_indexes_y[0]]]/\
            #        ravel_ragged(data_plot[0][0][ratio_indexes_y[1]])[bool_det[ratio_indexes_y[1]]][bool_sign[ratio_indexes_y[1]]],
            #        ravel_ragged(data_plot[0][0][ratio_indexes_y[0]])[bool_det[ratio_indexes_y[0]]][~bool_sign[ratio_indexes_y[0]]]/\
            #        ravel_ragged(data_plot[0][0][ratio_indexes_y[1]])[bool_det[ratio_indexes_y[1]]][~bool_sign[ratio_indexes_y[1]]]],dtype=object)
        else:
              
            if mode=='intrinsic':

                y_data=np.array([ravel_ragged(data_plot[1][0][i])[bool_det][bool_sign],ravel_ragged(data_plot[1][0][i])[bool_det][~bool_sign]],
                            dtype=object)

            elif mode=='observ' and not time_mode:
                
                #since the hid data is the same no matter the line, we need to repeat it for the number of lines used when plotting the global graph
                y_data_repeat=np.array([data_plot[1][0] for repeater in (i if type(i)==range else [i])])
                                        
                #only then can the linked mask be applied correctly (doesn't change anything in individual mode)
                if ratio_mode:
                    #here we just select one of all the lines in the repeat and apply the ratio_mode mask onto it
                    y_data=np.array([ravel_ragged(y_data_repeat[ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio],
                                     ravel_ragged(y_data_repeat[ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio]],dtype=object)
                else:
                    y_data=np.array([ravel_ragged(y_data_repeat)[bool_det][bool_sign],ravel_ragged(y_data_repeat)[bool_det][~bool_sign]],
                                    dtype=object)
                    
            elif mode=='source':
                y_data_repeat=np.array([data_plot[1][i_obj][0] for repeater in (i if type(i)==range else [i])\
                                        for i_obj in range(n_obj) for i_obs in range(len(data_plot[0][0][repeater][i_obj]))\
                                    ]).ravel()
                    
    
                y_data=np.array([y_data_repeat[bool_det][bool_sign],y_data_repeat[bool_det][~bool_sign]],dtype=object)
                        
        #### upper limit computation
        
        if show_ul_eqw:
            
            if mode=='eqwratio' or ratio_mode:

                #we use the same double definition here
                #in ratio_mode, x is the numerator so the ul_x case amounts to an upper limit
                x_data_ul_x=np.array(ravel_ragged(data_perinfo[5][0][ind_ratio[0]])[bool_nondetsign_x],
                                dtype=object)                
                y_data_ul_x=np.array(ravel_ragged(data_perinfo[0][0][ind_ratio[1]])[bool_nondetsign_x],
                                dtype=object)
                
                #here in ratio mode ul_y amounts to a lower limit
                x_data_ul_y=np.array(ravel_ragged(data_perinfo[0][0][ind_ratio[0]])[bool_nondetsign_y],
                                dtype=object)
                y_data_ul_y=np.array(ravel_ragged(data_perinfo[5][0][ind_ratio[1]])[bool_nondetsign_y],
                                dtype=object)
                
                #same way of defining the errors
                y_error_ul_x=np.array([ravel_ragged(data_perinfo[0][l][ind_ratio[1]])[bool_nondetsign_x] for l in [1,2]],dtype=object)
                x_error_ul_y=np.array([ravel_ragged(data_perinfo[0][l][ind_ratio[0]])[bool_nondetsign_y] for l in [1,2]],dtype=object)
                
                #computing two (upper and lower) limits for the ratios
                
                #the upper limit corresponds to points for which the numerator line is not detected, so it's the ul_y 
                #similarly, the lower limit is for ul_x
 
                #this is only in ratio_mode
                if ratio_mode:
                    if time_mode:
                        #switching x/y and ll/ul in time mode since the ratio is now on the y axis
                        
                        y_data_ll=x_data_ul_y/y_data_ul_y
                        y_data_ul=x_data_ul_x/y_data_ul_x
                        
                        x_data_ll=np.array(mdates.date2num(ravel_ragged(date_list_repeat))[bool_nondetsign_y],dtype=object)
                        x_data_ul=np.array(mdates.date2num(ravel_ragged(date_list_repeat))[bool_nondetsign_x],dtype=object)
                    else:
                        x_data_ll=y_data_ul_y/x_data_ul_y
                        y_data_ll=np.array(ravel_ragged(y_data_repeat[ind_ratio[1]])[bool_nondetsign_y],dtype=object)
                        
                        x_data_ul=y_data_ul_x/x_data_ul_x
                        y_data_ul=np.array(ravel_ragged(y_data_repeat[ind_ratio[0]])[bool_nondetsign_x],dtype=object)
                        
            else:
                
                if time_mode:

                        y_data_ul=np.array(ravel_ragged(data_perinfo[5][0][i])[bool_nondetsign],
                                    dtype=object)
                        x_data_ul=np.array(mdates.date2num(ravel_ragged(date_list_repeat))[bool_nondetsign],dtype=object)

                else:
                    
                    x_data_ul=np.array(ravel_ragged(data_perinfo[5][0][i])[bool_nondetsign],
                                dtype=object)
                    

                    #we can directly create the y_data_ul from y_data_repeat no matter which one it comes from
                    y_data_ul=np.array(ravel_ragged(y_data_repeat)[bool_nondetsign],dtype=object)

                        
        #in the errors we want to distinguish the parameters which were linked so we start compute the masks of the 'linked' values
        
        #we don't really care about the linked state in eqwratio mode
        if mode=='eqwratio':
            linked_mask=np.array([np.repeat(False,len(x_data[0])),np.repeat(False,len(x_data[1]))],dtype=object)
        else:
            if ratio_mode:
                
                linked_mask=np.array([np.array([type(elem)==str for elem in ravel_ragged(data_perinfo[1][1][ratio_indexes_x[0]])\
                                    [bool_det_ratio][bool_sign_ratio]] and\
                          [type(elem)==str for elem in ravel_ragged(data_perinfo[1][1][ratio_indexes_x[1]])\
                                    [bool_det_ratio][bool_sign_ratio]]).astype(object),
                      np.array([type(elem)==str for elem in ravel_ragged(data_perinfo[1][1][ratio_indexes_x[0]])\
                                [bool_det_ratio][~bool_sign_ratio]] and\
                          [type(elem)==str for elem in ravel_ragged(data_perinfo[1][1][ratio_indexes_x[1]])\
                                    [bool_det_ratio][~bool_sign_ratio]]).astype(object)],
                     dtype=object)
            else:         
                linked_mask=np.array([np.array([type(elem)==str for elem in ravel_ragged(data_perinfo[1][1][i])[bool_det][bool_sign].astype(object)]),
                                  np.array([type(elem)==str for elem in ravel_ragged(data_perinfo[1][1][i])[bool_det][~bool_sign].astype(object)])],
                                 dtype=object)
            
        #reconverting the array to bools and replacing it by true if it is empty to avoid conflicts when there is not data to plot
        #note : this doesn't work for arrays of length 1 so we add a reconversion when we use it just in case
        
        #resticting to cases where there's more than one resulting element
        if type(linked_mask[0])!=bool:
            linked_mask[0]=linked_mask[0].astype(bool)
        else:
            linked_mask[0]=np.array([linked_mask[0]])
        if type(linked_mask[1])!=bool:
            linked_mask[1]=linked_mask[1].astype(bool)
        else:
            linked_mask[1]=np.array([linked_mask[1]])
            

        #defining the jumps in indexes for linked values in the all line arrays, which is the number of lines shifted times the number of obs in 
        #all current objects
        n_obs=len(ravel_ragged(data_perinfo[0][0][0]))
        jump_Kb=3*n_obs
        jump_Kg=4*n_obs

        #unused for now        
        # #defining the number of lines to skip when not all lines are used
        # n_jump_Kg=len([elem for elem in mask_lines[:-1] if elem is True])
        # n_jump_Kb26=len([elem for elem in mask_lines[:-2] if elem is True])
        # n_jump_Kb25=len([elem for elem in mask_lines[:-3] if elem is True])
        
        #custom function so that the array indexation is not many lines long 
        def linked_uncer(ind,ind_incer,ind_info):
            
            '''
            For all lines at once it is easier to use absolute index positions after developping the array
            We do the index changes before bool_det and bool_sign else we could lose the regularity in the shifts
            linked_ind is only applied in abslines_plot (and not the restricted array) because else the information of the line the current line
            is linked to can be missing if the other line is masked
            '''
            
            linked_line_ind_restrict=ind//n_obs
            
            #here we retrieve the index of the original line and translate it into a position for the non masked array            
            linked_line_ind=np.argwhere(mask_lines)[linked_line_ind_restrict][0]

            linked_line_pos=ind%n_obs+n_obs*linked_line_ind
            
            '''
            Since the line shifts can be either positive or negative depending on if the first complexes are linked to the others or vice-versa, 
            we directly fetch all of the uncertainty values for each linked complex and pick the associated value
            We also make sure each line is linked properly by restricting the lines we check to the ones with the same base values
            '''
            
            ###TODO: update this for future linedets
            # #first we isolate the values for the same exposure
            # curr_exp_values=ravel_ragged(abslines_plot[ind_info][0].transpose(1,0,2)[mask_obj].transpose(1,0,2))[(ind//6)*6:(ind//6+1)*6]
            
            #then we identify 
            
            if linked_line_ind>2:
                val_line_pos=linked_line_pos-jump_Kg if linked_line_ind>4 else linked_line_pos-jump_Kb
            else:
                #should not happen
                breakpoint()
                
            if ind_info==2:
                #testing if there are 5 dimensions which would mean that we need to transpose with specific axes due to the array being regular
                if np.ndim(abslines_plot)==5:
                     return ravel_ragged(abslines_ener[ind_incer].transpose(1,0,2)[mask_obj].transpose(1,0,2))[val_line_pos]
                else:
                    return ravel_ragged(abslines_ener[ind_incer].T[mask_obj].T)[val_line_pos]
            else:
                if np.ndim(abslines_plot)==5:
                    return ravel_ragged(abslines_plot[ind_info][ind_incer].transpose(1,0,2)[mask_obj].transpose(1,0,2))[val_line_pos]
                else:
                    return ravel_ragged(abslines_plot[ind_info][ind_incer].T[mask_obj].T)[val_line_pos]
            
        #### error computation
        #(already done earlier in specific cases)
        
        #in ratio mode the errors are always for the eqw and thus are simply computed from composing the uncertainties of each eqw
        #then coming back to the ratio
        if ratio_mode:
            if time_mode:
                
                '''
                reminder of the logic since this is quite confusing
                the uncertainty is computed through the least squares because it's the correct formula for these kind of computation
                (correctly considers the skewness of the bivariate distribution)
                here, we can use ratio_indiexes_x[0] and ratio_indexes_x[1] independantly as long was we divide the uncertainty 
                ([i] which will either be [1] or [2]) by the main value [0]. However, as the uncertainties are asymetric, each one must be paired with the opposite one to maintain consistency 
                (aka the + uncertaintiy of the numerator and the - incertainty of the denominator, when composed,
                 give the + uncertainty of the quotient)
                on top of that, the adressing is important as the minus uncertainty of the quotient needs to be first, 
                so the -num/+det must be first
                so for the first index in [1,2] which is i=1, we must have i for the num and 2 for the det
                the num is the one with ratio_indexes[0]
                '''

                y_error=np.array([[((ravel_ragged(data_plot[1][i][ratio_indexes_x[0]])[bool_det_ratio][elem_sign_mask]/\
                                     ravel_ragged(data_plot[1][0][ratio_indexes_x[0]])[bool_det_ratio][elem_sign_mask])**2+
                                   (ravel_ragged(data_plot[1][2 if i==1 else 1][ratio_indexes_x[1]])[bool_det_ratio][elem_sign_mask]/\
                                    ravel_ragged(data_plot[1][0][ratio_indexes_x[1]])[bool_det_ratio][elem_sign_mask])**2)**(1/2)*
                                   y_data[i_sign] for i in [1,2]] for i_sign,elem_sign_mask in enumerate([bool_sign_ratio,~bool_sign_ratio])],dtype=object)
                
            else:
                x_error=np.array([[((ravel_ragged(data_plot[0][i][ratio_indexes_x[0]])[bool_det_ratio][elem_sign_mask]/\
                                     ravel_ragged(data_plot[0][0][ratio_indexes_x[0]])[bool_det_ratio][elem_sign_mask])**2+
                                   (ravel_ragged(data_plot[0][2 if i==1 else 1][ratio_indexes_x[1]])[bool_det_ratio][elem_sign_mask]/\
                                    ravel_ragged(data_plot[0][0][ratio_indexes_x[1]])[bool_det_ratio][elem_sign_mask])**2)**(1/2)*
                                   x_data[i_sign] for i in [1,2]] for i_sign,elem_sign_mask in enumerate([bool_sign_ratio,~bool_sign_ratio])],dtype=object)
                    
        #defining the errors and shifting the linked values accordingly for blueshifts
        #here we do not need the linked_ind shenanigans because this is not called during streamlit (and so all the lines are there)
        
        #THIS ALSO NEEDS TO CHANGE
        if indiv:
            x_error=np.array([[elem if type(elem)!=str else (ravel_ragged(data_plot[0][j][i-3])[ind_val] if i<5 else\
                      ravel_ragged(data_plot[0][1][i-4])[ind_val]) for ind_val,elem in enumerate(ravel_ragged(data_plot[0][j][i]))] for j in [1,2]])
                
            if mode=='intrinsic':
                y_error=np.array([[elem if type(elem)!=str else (ravel_ragged(data_plot[1][j][i-3])[ind_val] if i<5 else\
                      ravel_ragged(data_plot[0][1][i-4])[ind_val]) for ind_val,elem in enumerate(ravel_ragged(data_plot[1][j][i]))] for j in [1,2]])
            
        elif not ratio_mode and mode!='eqwratio':
            if time_mode:
                #swapping the error to the y axis
                y_error=np.array([[elem if type(elem)!=str else linked_uncer(ind_val,j,ind_infos[1])\
                                                                for ind_val,elem in enumerate(ravel_ragged(data_plot[1][j][i]))] for j in [1,2]])
            else:
                x_error=np.array([[elem if type(elem)!=str else linked_uncer(ind_val,j,ind_infos[0])\
                                                            for ind_val,elem in enumerate(ravel_ragged(data_plot[0][j][i]))] for j in [1,2]])
                    
            if mode=='intrinsic':
                y_error=np.array([[elem if type(elem)!=str else linked_uncer(ind_val,j,ind_infos[0])\
                                   for ind_val,elem in enumerate(ravel_ragged(data_plot[1][j][i]))] for j in [1,2]])
                
        '''
        in observ and source mode, we can compute the errors in the same formula in indiv mode or not since the values are the same for each line
        '''
        #note : all the cases of use of time mode are already created before
        if mode=='observ' and not time_mode:

            #l is error index here
            y_err_repeat=np.array([[data_plot[1][l] for repeater in (i if type(i)==range else [i])] for l in [1,2]])
                                    
            #only then can the linked mask be applied correctly (doesn't change anything in individual mode)
            if ratio_mode:

                #here we just select one of all the lines in the repeat and apply the ratio_mode mask onto it
                y_error=np.array([[ravel_ragged(y_err_repeat[l][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio] for l in [0,1]],
                                [ravel_ragged(y_err_repeat[l][ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio] for l in [0,1]]],dtype=object)
            else:
                y_error=np.array([[ravel_ragged(y_err_repeat[l])[bool_det][bool_sign] for l in [0,1]],
                                [ravel_ragged(y_err_repeat[l])[bool_det][~bool_sign] for l in [0,1]]],dtype=object)
            
        if mode=='source':
            y_err_repeat=np.array([[data_plot[1][i_obj][l] for j in (i if type(i)==range else [i])\
                                    for i_obj in range(n_obj) for i_obs in range(len(data_plot[0][0][j][i_obj]))]\
                                    for l in [1,2]])
            

            y_error=np.array([[ravel_ragged(y_err_repeat[l])[bool_det][bool_sign] for l in [0,1]],
                              [ravel_ragged(y_err_repeat[l])[bool_det][~bool_sign] for l in [0,1]]],
                            dtype=object)

        #maybe missing something for time mode here
        ###TODO
        if show_ul_eqw and mode!='eqwratio' and not time_mode:
            if ratio_mode:
                #creation of y errors from both masks for upper and lower limits
                y_error_ul=np.array([ravel_ragged(y_err_repeat[l][ind_ratio[0]])[bool_nondetsign_x] for l in [0,1]],dtype=object)
                y_error_ll=np.array([ravel_ragged(y_err_repeat[l][ind_ratio[1]])[bool_nondetsign_y] for l in [0,1]],dtype=object)
            else:
                #creating y errors from the upper limit mask
                y_error_ul=np.array([ravel_ragged(y_err_repeat[l])[bool_nondetsign] for l in [0,1]],dtype=object)
                
        #applying the det and sign masks
        if mode!='eqwratio' and not ratio_mode:

            if time_mode:
                y_error=[np.array([y_error[0][bool_det][bool_sign],y_error[1][bool_det][bool_sign]]),
                          np.array([y_error[0][bool_det][~bool_sign],y_error[1][bool_det][~bool_sign]])]
            else:
                x_error=[np.array([x_error[0][bool_det][bool_sign],x_error[1][bool_det][bool_sign]]),
                      np.array([x_error[0][bool_det][~bool_sign],x_error[1][bool_det][~bool_sign]])]
            
        
        if mode=='intrinsic':
            y_error=[np.array([y_error[0][bool_det][bool_sign],y_error[1][bool_det][bool_sign]]).astype(float),
                      np.array([y_error[0][bool_det][~bool_sign],y_error[1][bool_det][~bool_sign]]).astype(float)]
        
        #no need to do anything for y error in hid mode since it's already set to None
        
        data_cols=['blue','grey']
        
        data_link_cols=['dodgerblue','silver'] if show_linked else data_cols
        
        data_labels=np.array(['detections above '+str(conf_thresh*100)+'% treshold','detections below '+str(conf_thresh*100)+'% treshold'])

        #### plots
                
        #plotting the unlinked and linked results with a different set of colors to highlight the differences
        #note : we don't use markers here since we cannot map their color

        errbar_list=['' if len(x_data[s])==0 else ax_scat.errorbar(x_data[s].astype(float)[~(linked_mask[s].astype(bool))],
                                                      y_data[s].astype(float)[~(linked_mask[s].astype(bool))],
                          xerr=None if x_error is None else [elem[~(linked_mask[s].astype(bool))] for elem in x_error[s]],
                          yerr=[elem[~(linked_mask[s].astype(bool))] for elem in y_error[s]],linewidth=1,
                          c=data_cols[s],label=data_labels[s] if color_scatter=='None' else '',linestyle='',
                          marker='D' if color_scatter=='None' else None,alpha=1,zorder=1)\
         for s in ([0,1] if display_nonsign else [0])]

           
        errbar_list_linked=['' if len(x_data[s])==0 else ax_scat.errorbar(x_data[s].astype(float)[linked_mask[s].astype(bool)],
                                                        y_data[s].astype(float)[linked_mask[s].astype(bool)],
                          xerr=None if x_error is None else [elem[linked_mask[s].astype(bool)] for elem in x_error[s]],
                          yerr=[elem[linked_mask[s].astype(bool)] for elem in y_error[s]],linewidth=1,
                          color=data_link_cols[s],label='linked '+data_labels[s]  if color_scatter=='None' else '',linestyle='',
                          marker='D' if color_scatter=='None' else None,alpha=0.6 if show_linked else 1.0,zorder=1000)\
         for s in ([0,1] if display_nonsign else [0])]        
            
        #locking the graph if asked to to avoid things going haywire with upper limits
        if lock_lims_det:
            ax_scat.set_xlim(ax_scat.get_xlim())
            ax_scat.set_ylim(ax_scat.get_ylim())
                
        #### coloring
        
        #for the colors, we use the same logic than to create y_data with observation/source level parameters here
        
        #default colors for when color_scatter is set to None, will be overwriten
        color_arr=['blue']
        color_arr_ul='black'
        color_arr_ul_x=color_arr_ul
        color_arr_ul_y=color_arr_ul
            
        if color_scatter=='Instrument':

            #there's no need to repeat in eqwratio since the masks are computed for a single line                
            if mode=='eqwratio' or ratio_mode:
                color_data_repeat=instru_list
            else:
                color_data_repeat=np.array([instru_list for repeater in (i if type(i)==range else [i])])
            
            if ratio_mode:
                color_data=np.array([ravel_ragged(color_data_repeat)[bool_det_ratio][bool_sign_ratio],
                                 ravel_ragged(color_data_repeat)[bool_det_ratio][~bool_sign_ratio]],dtype=object)
                                                          
            else:

                color_data=np.array([ravel_ragged(color_data_repeat)[bool_det][bool_sign],ravel_ragged(color_data_repeat)[bool_det][~bool_sign]],
                                dtype=object)
                
                if mode!='eqwratio':
                    #same thing for the upper limits
                    color_data_ul=ravel_ragged(color_data_repeat)[bool_nondetsign]

            if mode=='eqwratio' or ratio_mode:
                #same thing for the upper limits in x and y
                color_data_ul_x=ravel_ragged(color_data_repeat)[bool_nondetsign_x]
                color_data_ul_y=ravel_ragged(color_data_repeat)[bool_nondetsign_y]

            #computing the actual color array for the detections
            color_arr=np.array([np.array([telescope_colors[elem] for elem in color_data[s]]) for s in [0,1]],dtype=object)
            
            #and for the upper limits if needed
            if show_ul_eqw:
                                    
                if ratio_mode or mode=='eqwratio':
                    color_arr_ul_x=np.array([telescope_colors[elem] for elem in color_data_ul_x])
        
                    color_arr_ul_y=np.array([telescope_colors[elem] for elem in color_data_ul_y])
                    
                else:
                    color_arr_ul=np.array([telescope_colors[elem] for elem in color_data_ul])

            #here we can keep a simple labeling
            label_dict=telescope_colors
            
        elif color_scatter in ['Time','HR','width','nH']:
            
            color_var_arr=date_list if color_scatter=='Time' else hid_plot[0][0] if color_scatter=='HR'\
                else width_plot_restrict[0] if color_scatter=='width' else nh_plot_restrict[0]
                            
            #there's no need to repeat in eqwratio since the masks are computed for a single line                
            if color_scatter=='width':
                color_data_repeat=color_var_arr
            else:
                if mode=='eqwratio' or ratio_mode:
                    color_data_repeat=color_var_arr
                else:
                    color_data_repeat=np.array([color_var_arr for repeater in (i if type(i)==range else [i])])
            
            if ratio_mode:
                color_data=np.array([ravel_ragged(color_data_repeat)[bool_det_ratio][bool_sign_ratio],
                                 ravel_ragged(color_data_repeat)[bool_det_ratio][~bool_sign_ratio]],dtype=object)
                                                          
            else:

                color_data=np.array([ravel_ragged(color_data_repeat)[bool_det][bool_sign],ravel_ragged(color_data_repeat)[bool_det][~bool_sign]],
                                dtype=object)

                if mode!='eqwratio':
                    #same thing for the upper limits
                    color_data_ul=ravel_ragged(color_data_repeat)[bool_nondetsign]

            if mode=='eqwratio' or ratio_mode:
                
                #same thing for the upper limits in x and y
                color_data_ul_x=ravel_ragged(color_data_repeat)[bool_nondetsign_x]
                color_data_ul_y=ravel_ragged(color_data_repeat)[bool_nondetsign_y]
                
            #here we compute a conversion of the dates to numerical values in the case of a Time colormap
            if color_scatter=='Time':
                c_arr=np.array([mdates.date2num(color_data[s]) for s in [0,1]],dtype=object)
            
                if ratio_mode or mode=='eqwratio':
                    c_arr_ul_x=mdates.date2num(color_data_ul_x)
                    c_arr_ul_y=mdates.date2num(color_data_ul_y)
                    
                else:
                    c_arr_ul=mdates.date2num(color_data_ul)
            else:
                c_arr=color_data
                
                if ratio_mode or mode=='eqwratio':
                    c_arr_ul_x=color_data_ul_x
                    c_arr_ul_y=color_data_ul_y
                else:
                    c_arr_ul=color_data_ul

            c_arr_tot=c_arr.tolist()
            
            # #adding the upper limits to the normalisation if necessary
            if show_ul_eqw and ('eqw' in infos or mode=='eqwratio'):
                if mode=='eqwratio' or ratio_mode:
                    
                    c_arr_tot+=[c_arr_ul_x,c_arr_ul_y]
                else:
                    c_arr_tot+=[c_arr_ul]    
                        
            #differing norms for Time and HR:
            if color_scatter in ['HR','nH']:
                
                c_norm=colors.LogNorm(vmin=min(ravel_ragged(c_arr_tot)),
                                      vmax=max(ravel_ragged(c_arr_tot)))
            else:

                c_norm=mpl.colors.Normalize(vmin=min(ravel_ragged(c_arr_tot)),
                                            vmax=max(ravel_ragged(c_arr_tot)))

                    
            color_cmap=mpl.cm.plasma
            colors_func_date=mpl.cm.ScalarMappable(norm=c_norm,cmap=color_cmap)
            
            #computing the actual color array for the detections
            color_arr=np.array([[colors_func_date.to_rgba(elem) for elem in c_arr[s]] for s in ([0,1] if display_nonsign else [0]) ])
            
            #and for the upper limits
            if ratio_mode or mode=='eqwratio':
                
                #the axes swap in timemode requires swapping the indexes to fetch the uncertainty locations
                color_arr_ul_x=np.array([colors_func_date.to_rgba(elem) for elem in c_arr_ul_x])
                color_arr_ul_y=np.array([colors_func_date.to_rgba(elem) for elem in c_arr_ul_y])
            else:
                color_arr_ul=np.array([colors_func_date.to_rgba(elem) for elem in c_arr_ul]) 
            
        elif color_scatter=='Source':
            
            #there's no need to repeat in eqwratio since the masks are computed for a single line                
            if mode=='eqwratio' or ratio_mode:
                color_data_repeat=np.array([obj_disp_list[i_obj] for i_obj in range(n_obj) for i_obs in range(len(data_perinfo[0][0][0][i_obj]))])
                
            else:
                color_data_repeat=np.array([obj_disp_list[i_obj] for repeater in (i if type(i)==range else [i])\
                                        for i_obj in range(n_obj) for i_obs in range(len(data_perinfo[0][0][repeater][i_obj]))\
                                    ])
            
            if ratio_mode:
                
                color_data=np.array([ravel_ragged(color_data_repeat)[bool_det_ratio][bool_sign_ratio],
                                 ravel_ragged(color_data_repeat)[bool_det_ratio][~bool_sign_ratio]],dtype=object)

            else:

                color_data=np.array([ravel_ragged(color_data_repeat)[bool_det][bool_sign],ravel_ragged(color_data_repeat)[bool_det][~bool_sign]],
                            dtype=object)
                
                if mode!='eqwratio':
                    #same thing for the upper limits
                    color_data_ul=ravel_ragged(color_data_repeat)[bool_nondetsign]

            if mode=='eqwratio' or ratio_mode:
                #same thing for the upper limits in x and y
                color_data_ul_x=ravel_ragged(color_data_repeat)[bool_nondetsign_x]
                color_data_ul_y=ravel_ragged(color_data_repeat)[bool_nondetsign_y]
                
            color_data_tot=[color_data[s] for s in ([0,1] if display_nonsign else [0])]
            #global array for unique extraction
            if show_ul_eqw and ('eqw' in infos or mode=='eqwratio'):
                
                if not ratio_mode and mode!='eqwratio':
                    #same thing for the upper limits
                    color_data_tot+=[color_data_ul]
                else:
                    #same thing for the upper limits in x and y
                    color_data_tot+=[color_data_ul_x,color_data_ul_y]
                    
                
            #we extract the number of objects with detection from the array of sources
            disp_objects=np.unique(ravel_ragged(color_data_tot))
            
            #and compute a color mapping accordingly
            norm_colors_obj=mpl.colors.Normalize(vmin=0,vmax=len(disp_objects))
            colors_func_obj=mpl.cm.ScalarMappable(norm=norm_colors_obj,cmap=mpl.cm.hsv_r)
        
            color_arr=np.array([colors_func_obj.to_rgba(np.argwhere(disp_objects==elem)[0][0]) for s in ([0,1] if display_nonsign else [0]) for elem in color_data[s]])
                
            label_dict={disp_objects[i]:colors_func_obj.to_rgba(i) for i in range(len(disp_objects))}
            
            if not display_nonsign:
                color_arr=np.array([color_arr,'None'],dtype=object)

            #same for the upper limits if needed
            if show_ul_eqw and ('eqw' in infos or mode=='eqwratio'):
                                    
                if ratio_mode or mode=='eqwratio':
                    color_arr_ul_x=np.array([colors_func_obj.to_rgba(np.argwhere(disp_objects==elem)[0][0]) for elem in color_data_ul_x])
        
                    color_arr_ul_y=np.array([colors_func_obj.to_rgba(np.argwhere(disp_objects==elem)[0][0]) for elem in color_data_ul_y])
                    
                else:

                    color_arr_ul=np.array([colors_func_obj.to_rgba(np.argwhere(disp_objects==elem)[0][0]) for elem in color_data_ul])

                        
        def plot_ul_err(lims_bool,x_data,y_data,x_err,y_err,col,label=''):
            
            #this construction allows us to maintain a string color name when not using color_scatter

            for i_err,(x_data_single,y_data_single,x_err_single,y_err_single,col_single) in\
                enumerate(zip(x_data,y_data,x_err,y_err,col if color_scatter!='None' else np.repeat(col,len(x_data)))):
                
                ax_scat.errorbar(x_data_single,y_data_single,xerr=np.array([x_err_single]).T if x_err_single is not None else x_err_single,
                                 yerr=np.array([y_err_single]).T if y_err_single is not None else y_err_single,
                                 xuplims=lims_bool[0]==1,xlolims=lims_bool[0]==-1,uplims=lims_bool[1]==1,lolims=lims_bool[1]==-1,
                                 color=col_single,linestyle='',marker='.' if color_scatter=='None' else None,
                                 label=label if i_err==0 else '',alpha=alpha_ul if color_scatter=='None' else 1)
            
        ####plotting upper limits
        if show_ul_eqw and ('eqw' in infos or mode=='eqwratio'):
            
            if mode=='eqwratio':
                #xuplims here
                plot_ul_err([1,0],x_data_ul_x,y_data_ul_x,x_data_ul_x*0.05,y_error_ul_x.T,color_arr_ul_x)
                
                #uplims here
                plot_ul_err([0,1],x_data_ul_y,y_data_ul_y,x_error_ul_y.T,y_data_ul_y*0.05,color_arr_ul_y)
                
            else:

                # else:
                #     ax_scat.set_xlim(mdates.date2num(['2012-01-01']),mdates.date2num(['2012-10-01']))
                
                if time_mode:
                    #uplims here, the upper limits display has the same construction no matter if in ratio mode or not
                    plot_ul_err([0,1],x_data_ul,y_data_ul,[None]*len(x_data_ul),y_data_ul*0.05,color_arr_ul)
                
                else:
                    
                    #xuplims here, the upper limits display has the same construction no matter if in ratio mode or not
                    plot_ul_err([1,0],x_data_ul,y_data_ul,x_data_ul*0.05,y_error_ul.T,color_arr_ul)
                
                #adding the lower limits in ratio mode
                if ratio_mode:
                    
                    if time_mode:
                          #lolims here
                          plot_ul_err([0,-1],x_data_ll,y_data_ll,[None]*len(x_data_ll),y_data_ll*0.05,color_arr_ul_y)
                    else:
                        #lolims here
                        plot_ul_err([-1,0],x_data_ll,y_data_ll,x_data_ll*0.05,y_error_ll.T,color_arr_ul_y)
                        
        if color_scatter!='None':
            
            for s,elem_errbar in enumerate(errbar_list):
                #replacing indiviudally the colors for each point but the line
                for elem_children in elem_errbar.get_children()[1:]:

                    elem_children.set_colors(color_arr[s][~(linked_mask[s].astype(bool))])
                    
            for s,elem_errbar_linked in enumerate(errbar_list_linked):
                #replacing indiviudally the colors for each point but the line
                for elem_children in elem_errbar_linked.get_children()[1:]:
                    
                    elem_children.set_colors(color_arr[s][(linked_mask[s].astype(bool))])
                
            if color_scatter in ['Time','HR','width','nH']:
                
                #adding space for the colorbar
                ax_cb=fig_scat.add_axes([0.99, 0.123, 0.02, 0.84 if not compute_correl else 0.73])
    
                #scatter plot on top of the errorbars to be able to map the marker colors and create the colormap
                
                #no labels needed here
                scat_list=[ax_scat.scatter(x_data[s].astype(float)[~(linked_mask[s].astype(bool))],
                                           y_data[s].astype(float)[~(linked_mask[s].astype(bool))],
                c=c_arr[s][~(linked_mask[s].astype(bool))],cmap=color_cmap,norm=c_norm,marker='D',alpha=1,zorder=1)\
                           for s in ([0,1] if display_nonsign else [0])]

                scat_list_linked=[ax_scat.scatter(x_data[s].astype(float)[(linked_mask[s].astype(bool))],
                                                  y_data[s].astype(float)[(linked_mask[s].astype(bool))],
                c=c_arr[s][(linked_mask[s].astype(bool))],cmap=color_cmap,norm=c_norm,marker='D',alpha=1,zorder=1)\
                                  for s in ([0,1] if display_nonsign else [0])]
                        
                #forcing a common normalisation for all the scatters
                scat_list_tot=scat_list+scat_list_linked
                        
                for elem_scat in scat_list_tot:

                    elem_scat.set_clim(min(ravel_ragged(c_arr_tot)),max(ravel_ragged(c_arr_tot)))
                                                
                if color_scatter=='Time':
                    #creating the colormap
                    
                    #manually readjusting for small durations because the AutoDateLocator doesn't work well
                    time_range=max(mdates.date2num(ravel_ragged(date_list)))-min(mdates.date2num(ravel_ragged(date_list)))
                    
                    if time_range<150:
                        date_format=mdates.DateFormatter('%Y-%m-%d')
                    elif time_range<1825:
                        date_format=mdates.DateFormatter('%Y-%m')
                    else:
                        date_format=mdates.AutoDateFormatter(mdates.AutoDateLocator())   
                        
                    plt.colorbar(scat_list[0],cax=ax_cb,ticks=mdates.AutoDateLocator(),format=date_format)  
                
                elif color_scatter in ['HR','width','nH']:

                    #creating the colormap (we add a bottom extension for nH to indicate the values cut)
                    plt.colorbar(scat_list[0],cax=ax_cb,label=r'nH ($10^{22}$ cm$^{-2}$)' if color_scatter=='nH' else color_scatter,extend='min' if color_scatter=='nH' else None)  


            else:
                
                #scatter plot on top of the errorbars to be able to map the marker colors
                #The loop allows to create scatter with colors according to the labels
                for s in ([0,1] if display_nonsign else [0]):
                    for i_col,color_label in enumerate(label_dict.keys()):
        
                        #creating a mask for the points of the right color
                        if color_scatter=='Instrument':
                            color_mask=[elem==label_dict[color_label] for elem in color_arr[s][~(linked_mask[s].astype(bool))]]
                            color_mask_linked=[elem==label_dict[color_label] for elem in color_arr[s][(linked_mask[s].astype(bool))]]


                            #checking if there is at least one upper limit:
                            #(a bit convoluted but we cannot concatenate 0 len arrays so we add a placeholder that'll never get recognized instead)
                            no_ul_displayed=np.sum([elem==label_dict[color_label] for elem in\
                                                (np.concatenate((color_arr_ul_x if len(color_arr_ul_x)>0 and color_arr_ul_x!='black' else ['temp'],\
                                                                 color_arr_ul_y if len(color_arr_ul_y)>0 and color_arr_ul_y!='black' else ['temp']))\
                                                 if (ratio_mode or mode=='eqwratio') else color_arr_ul)])==0
                                
                        #same idea but here since the color is an RGB tuple we need to convert the element before the comparison
                        elif color_scatter=='Source':
                            color_mask=[tuple(elem) ==label_dict[color_label] for elem in color_arr[s][~(linked_mask[s].astype(bool))]]
                            color_mask_linked=[tuple(elem) ==label_dict[color_label] for elem in color_arr[s][(linked_mask[s].astype(bool))]]
        
                            #checking if there is at least one upper limit:
                            #(a bit convoluted but we cannot concatenate 0 len arrays so we add a placeholder that'll never get recognized instead)
                            no_ul_displayed=np.sum([tuple(elem)==label_dict[color_label] for elem in\
                                                (np.concatenate((color_arr_ul_x if len(color_arr_ul_x)>0 and color_arr_ul_x!='black' else ['temp'],\
                                                                 color_arr_ul_y if len(color_arr_ul_y)>0 and color_arr_ul_y!='black' else ['temp']))\
                                                 if (ratio_mode or mode=='eqwratio') else color_arr_ul)])==0
                                
                        #not displaying color/labels that are not actually in the plot
                        if np.sum(color_mask)==0 and np.sum(color_mask_linked)==0 and (not show_ul_eqw or no_ul_displayed):
                            continue
                        
                        ax_scat.scatter(x_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask],
                                                              y_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask],
                                  color=label_dict[color_label],label=color_label,marker='D',alpha=1,zorder=1)
                        
                        #No label for the second plot to avoid repetitions
                        ax_scat.scatter(x_data[s].astype(float)[(linked_mask[s].astype(bool))][color_mask_linked],
                                                              y_data[s].astype(float)[(linked_mask[s].astype(bool))][color_mask_linked],
                                  color=label_dict[color_label],
                        label='',marker='D',alpha=1,zorder=1)
            
                ax_scat.legend()
            
        # ax_scat.set_xlim(mdates.date2num(['2012-01-01']),mdates.date2num(['2012-10-01']))
                        
        #computing the statistical coefficients for the significant portion of the sample (we don't care about the links here)
        if compute_correl and mode!='source' and len(x_data[0])>1 and not time_mode:
            
            #we cannot transpose the whole arrays since in eqwratio/HID mode they are most likely ragged due to having both
            #the significant and unsignificant data, so we create regular versions manually
            x_error_sign_T=np.array([elem for elem in x_error[0]]).T
            y_error_sign_T=np.array([elem for elem in y_error[0]]).T
            

            
            r_pearson=np.array(pymccorrelation(x_data[0].astype(float),y_data[0].astype(float),dx=x_error_sign_T/1.65,
                                      dy= y_error_sign_T/1.65,
                                      Nperturb=1000,coeff='pearsonr',percentiles=(50,5,95)))

                
            #switching back to uncertainties from quantile values
            r_pearson=np.array([[r_pearson[ind_c][0],r_pearson[ind_c][0]-r_pearson[ind_c][1],r_pearson[ind_c][2]-r_pearson[ind_c][0]]\
                                for ind_c in [0,1]])
    
            r_spearman=np.array(pymccorrelation(x_data[0].astype(float),y_data[0].astype(float),dx=x_error_sign_T/1.65,
                                      dy= y_error_sign_T/1.65,
                                      Nperturb=1000,coeff='spearmanr',percentiles=(50,5,95)))
        
            #switching back to uncertainties from quantile values
            r_spearman=np.array([[r_spearman[ind_c][0],r_spearman[ind_c][0]-r_spearman[ind_c][1],r_spearman[ind_c][2]-r_spearman[ind_c][0]]\
                                for ind_c in [0,1]])
            
            ax_scat.set_title(r'Pearson: '+
                              '$r='+str(round(r_pearson[0][0],3))+'_{-'+str(round(r_pearson[0][1],3))+'}^{+'+str(round(r_pearson[0][2],3))+'}$ '+
                              '$p='+str(round(r_pearson[1][0],3))+'_{-'+str(round(r_pearson[1][1],3))+'}^{+'+str(round(r_pearson[1][2],3))+'}$\n'+
                              'Spearman: '+
                              '$r='+str(round(r_spearman[0][0],3))+'_{-'+str(round(r_spearman[0][1],3))+'}^{+'+str(round(r_spearman[0][2],3))+
                              '}$ '+
                              '$p='+str(round(r_spearman[1][0],3))+'_{-'+str(round(r_spearman[1][1],3))+'}^{+'+str(round(r_spearman[1][2],3))+
                              '}$ ')
            
        #adjusting the axis sizes for eqwratio mode to get the same scale
        if mode=='eqwratio':
            ax_scat.set_ylim(max(min(ax_scat.get_xlim()[0],ax_scat.get_ylim()[0]),0),max(ax_scat.get_xlim()[1],ax_scat.get_ylim()[1]))
            ax_scat.set_xlim(max(min(ax_scat.get_xlim()[0],ax_scat.get_ylim()[0]),0),max(ax_scat.get_xlim()[1],ax_scat.get_ylim()[1]))
            ax_scat.plot(ax_scat.get_xlim(),ax_scat.get_ylim(),ls='--',color='grey')
                
        #plotting the trend lines for both the significant sample and the whole sample
        if len(x_data[0])>1 and mode!='source' and plot_trend:
            try:
                fit_sign=np.polyfit(x_data[0].astype(float),y_data[0].astype(float),1)
                poly_sign=np.poly1d(fit_sign)
    
                ax_scat.plot(np.sort(x_data[0].astype(float)),poly_sign(np.sort(x_data[0].astype(float))),color=data_cols[0],linestyle='--',
                              label='linear trend line for detections above '+str(conf_thresh*100)+'%')
            except:
                pass
                
                
        if len(np.concatenate(x_data,axis=0))>1 and len(x_data[1])>0 and mode!='source' and plot_trend and display_nonsign:
            
            try:
                fit_all=np.polyfit(np.concatenate(x_data,axis=0).astype(float),np.concatenate(y_data,axis=0).astype(float),1)
                poly_all=np.poly1d(fit_all)
                ax_scat.plot(np.sort(np.concatenate(x_data,axis=0).astype(float)),poly_all(np.sort(np.concatenate(x_data,axis=0).astype(float))),
                              color=data_cols[1],linestyle='--',label='linear trend line for all detections')
            except:
                pass
        
        if show_linked:
            plt.legend()
        plt.tight_layout()
        

        
        if save:
            if indiv:
                suffix_str='_'+lines_std_names[3+i]
            else:
                suffix_str='_all'
                
            plt.savefig(save_dir+'/graphs/'+mode+'/'+save_str_prefix+'autofit_correl_'+infos+suffix_str+'_cam_'+args_cam+'_'+\
                            args_line_search_e.replace(' ','_')+'_'+args_line_search_norm.replace(' ','_')+'.png')
        if close:
            plt.close(fig_scat)
            
    #returning the graph for streamlit display
    if streamlit:
        return fig_scat