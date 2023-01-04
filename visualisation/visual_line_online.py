#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 00:11:45 2022

@author: parrama
"""

#general imports
import io
import numpy as np
import pandas as pd
from copy import deepcopy
import streamlit as st

#matplotlib imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

#custom imports
from astropy.time import Time
# import time
import dill

from pathlib import Path
import sys

#adding the top directory to the path to avoid issues when importing fitting_tools
sys.path.append(str(Path(__file__).resolve().parent)+'/spectral_analysis/')

#custom script with some lines and fit utilities and variables
from fitting_tools import lines_std,lines_std_names,ravel_ragged,range_absline

#visualisation functions
from visual_line_online_tools import dump_dict,distrib_graph,correl_graph,incl_dic,\
    n_infos, plot_lightcurve, telescope_colors, sources_det_dic, dippers_list
    
'''Astro'''




def getoverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

# import mpld3

# import streamlit.components.v1 as components

'''
in the abslines_infos_perline form, the order is:
    -each habsorption line
    -the number of sources
    -the number of obs for each source
    -the info (5 rows, eqw/bshift/delchi/sign)
    -it's uncertainty (3 rows, main value/neg uncert/pos uncert,useless for the delchi and sign)
'''
    
instru_list=dump_dict['instru_list']
telescope_list=dump_dict['telescope_list']
choice_telescope=dump_dict['choice_telescope']
dist_obj_list=dump_dict['dist_obj_list']
mass_obj_list=dump_dict['mass_obj_list']
line_search_e=dump_dict['line_search_e']
multi_obj=dump_dict['multi_obj']
observ_list=dump_dict['observ_list']
bad_flags=dump_dict['bad_flags']
obj_list=dump_dict['obj_list']
date_list=dump_dict['date_list']
abslines_plot=dump_dict['abslines_plot']
hid_plot=dump_dict['hid_plot']
flux_plot=dump_dict['flux_plot']
nh_plot=dump_dict['nh_plot']
incl_plot=dump_dict['incl_plot']
abslines_infos_perobj=dump_dict['abslines_infos_perobj']
flux_list=dump_dict['flux_list']
abslines_ener=dump_dict['abslines_ener']
width_plot=dump_dict['width_plot']
dict_linevis=dump_dict['dict_linevis']
catal_maxi_df=dump_dict['catal_maxi_df']
catal_maxi_simbad=dump_dict['catal_maxi_simbad']

'''Page creation'''
#### Streamlit page creation

line_display_str=np.array([r'FeXXV Ka (6.70 keV)',r'FeXXVI  Ka (6.97 keV)','NiXXVII Ka (7.80 keV)',
                      'FeXXV Kb (7.89 keV)','FeXXVI Kb (8.25 keV)','FeXXVI Kg (8.70 keV)'])

if multi_obj:
    radio_single=st.sidebar.radio('Display options:',('All Objects','Multiple Objects','Single Object'))
    
    restrict_sources_detection=st.sidebar.checkbox('Restrict to sources with detection')
    ####Done manually as of now, should be changed
    
    if radio_single=='Single Object':
        display_single=True
    else:
        display_single=False
    
    if radio_single=='Multiple Objects':
        display_multi=True
    else:
        display_multi=False
        
    if display_multi:
        with st.sidebar.expander('Source'):
            choice_source=st.multiselect('',options=[elem for elem in obj_list if elem in sources_det_dic] if restrict_sources_detection else obj_list,default=[elem for elem in obj_list if elem in sources_det_dic] if restrict_sources_detection else obj_list)     
        
    if display_single:
        #switching to array to keep the same syntax later on
        choice_source=[st.sidebar.selectbox('Source',obj_list)]

with st.sidebar.expander('Absorption lines restriction'):
    selectbox_abstype=st.multiselect('',
                    options=line_display_str,default=line_display_str)

#creating the line mask from that
mask_lines=np.array([elem in selectbox_abstype for elem in line_display_str])

with st.sidebar.expander('inclination'):
    slider_inclin=st.slider('Inclination restriction (°)',min_value=0.,max_value=90.,step=0.5,value=[0.,90.])
    
    include_noinclin=st.checkbox('Include Sources with no inclination information',value=True)
    
    incl_inside=st.checkbox('Only include sources with uncertainties compatible with the current limits',value=False)
    
    radio_dipper=st.radio('Dipping sources restriction',('Off','Restrict to dippers','Restrict to non-dippers'))
    
    
#not used currently
# radio_dispmode=st.sidebar.radio('HID display',('Autofit line detections','blind search peak detections'))
# if radio_dispmode=='Autofit line detections':
#     display_final=True
# else:
#     display_final=False

#     radio_info_cmap=st.sidebar.radio('Color map options:',('Source','Peak delchi'))
#     slider_ener=st.sidebar.slider('Peak energy range',min_value=line_search_e[0],max_value=line_search_e[1],
#                                   step=line_search_e[2],value=[6.,9.])
#     display_abslines=st.sidebar.checkbox('Display absorption Lines')
#     display_emlines=st.sidebar.checkbox('Display emission Lines')
    
restrict_time=st.sidebar.checkbox('Restrict time interval',value=True)
        
display_source_table=st.sidebar.checkbox('Display source parameters table',value=False)

st.sidebar.header('HID options')

#not used right now
# else:
#     #full true mask
#     mask_lines=np.array(line_display_str!='')
            
slider_sign=st.sidebar.slider('Detection significance treshold',min_value=0.9,max_value=1.,step=1e-3,value=0.997,format="%.3f")

display_nonsign=st.sidebar.checkbox('Show detections below significance threshold',value=False)

if display_nonsign:
    restrict_threshold=st.sidebar.checkbox('Prioritize showing maximal values of significant detections',value=True)
else:
    restrict_threshold=True
        
radio_info_cmap=st.sidebar.radio('HID colormap',('Source','Blueshift','Delchi','EQW ratio','Inclination','Time','Instrument','nH'))

if radio_info_cmap!='Source':
    display_edgesource=st.sidebar.checkbox('Color code sources in marker edges',value=False)
else:
    display_edgesource=False

if radio_info_cmap=='Inclination':
    #order change here to have more logic in the variables
    cmap_incl_str=['Main value','Lower limit','Upper limit']
    cmap_incl_type_str=st.sidebar.radio('Inclination information to use:',cmap_incl_str)

    cmap_incl_type=[i for i in range(3) if cmap_incl_str[i]==cmap_incl_type_str][0]
else:
    cmap_inclination_limit=False
    cmap_incl_type=1
    
if radio_info_cmap=='EQW ratio':
    with st.sidebar.expander('Lines selection for EQW ratio:'):
        selectbox_ratioeqw=st.sidebar.multiselect('',options=line_display_str[mask_lines],default=line_display_str[mask_lines][:2])
        
else:
    selectbox_ratioeqw=''
    
    checkbox_zoom=st.sidebar.checkbox('Zoom around the displayed elements',value=False)
            
display_nondet=st.sidebar.checkbox('Display non detection',value=True)

if display_nondet:
    with st.sidebar.expander('Upper limits'):
        display_upper=st.checkbox('Display upper limits',value=True)    
        if display_upper:
                selectbox_upperlines=st.multiselect('Lines selection for upper limit display:',
                                                            options=line_display_str[mask_lines],default=line_display_str[mask_lines][:2])
                mask_lines_ul=np.array([elem in selectbox_upperlines for elem in line_display_str])
else:
    display_upper=False
    
if display_single:
    display_evol_single=st.sidebar.checkbox('Highlight time evolution in the HID',value=True)
               
save_format=st.sidebar.radio('Graph format:',('pdf','svg','png'))

        
with st.sidebar.expander('Visualisation'):
    
    display_obj_zerodet=st.checkbox('Color sources with no detection',value=True)
    
    display_hid_error=st.checkbox('Display errorbar for HID position',value=True)
    
    display_central_abs=st.checkbox('Display centers for absorption detections',value=False)

    alpha_abs=st.checkbox('Plot with transparency',value=False)
    
    global_colors=st.checkbox('Normalise colors/colormaps over the entire sample',value=False)
        
    paper_look=st.checkbox('Paper look',value=True)
    
    bigger_text=st.checkbox('Bigger text size',value=True)
    
    square_mode=st.checkbox('Square mode',value=True)

    show_linked=st.checkbox('Distinguish linked detections',value=False)
    
if alpha_abs:
    alpha_abs=0.5
else:
    alpha_abs=1
    
with st.sidebar.expander('Lightcurves'):
    
    plot_lc_monit=st.checkbox('Plot monitoring lightcurve',value=False)
    plot_hr_monit=st.checkbox('Plot monitoring HR',value=False)
    monit_highlight_hid=st.checkbox('Highlight HID coverage',value=False)
    
    if plot_lc_monit or plot_hr_monit:
        zoom_lc=st.checkbox('Zoom on the restricted time period in the lightcurve',value=False)
    else:
        zoom_lc=False
        
    fig_lc_monit=None
    fig_hr_monit=None
    
    plot_maxi_ew=st.checkbox('Superpose measured EW',value=False)
    
    ####ADD LC DOWNLOAD
    # def save_lc():
        
    #     '''
    #     Saves the current maxi_graph in a svg (i.e. with clickable points) format.
    #     '''
    #     if display_single:
    #         fig_lc_monit.savefig(save_dir+'/'+choice_source[0]+'_lc_'+str(round(time.time()))+'.'+save_format,bbox_inches='tight')
    #         fig_hr_monit.savefig(save_dir+'/'+choice_source[0]+'_hr_'+str(round(time.time()))+'.'+save_format,bbox_inches='tight')
            
    # st.button('Save current MAXI curves',on_click=save_lc,key='save_lc_key')
    
compute_only_withdet=st.sidebar.checkbox('Skip parameter analysis when no detection remain with the current constraints',value=False)

mpl.rcParams.update({'font.size': 10})

if not square_mode:
    fig_hid,ax_hid=plt.subplots(1,1,figsize=(8,5) if bigger_text else (12,6))
else:
    fig_hid,ax_hid=plt.subplots(1,1,figsize=(8,6))
ax_hid.clear()


'''HID GRAPH'''

#log x scale for an easier comparison with Ponti diagrams
ax_hid.set_xscale('log')
ax_hid.set_xlabel('Hardness Ratio ([6-10]/[3-6] keV bands)')
ax_hid.set_ylabel(r'Luminosity in the [3-10] keV band in (L/L$_{Edd}$) units')
ax_hid.set_yscale('log')

'''Dichotomy'''

#fetching the line indexes when plotting eqw ratio as colormap
eqw_ratio_ids=np.argwhere([elem in selectbox_ratioeqw for elem in line_display_str]).T[0]

#string of the colormap legend for the informations
radio_info_label=['Blueshift', r'$\Delta\chi^2$', 'Equivalent width ratio']

#masking for restriction to single objects
if display_single or display_multi:
    mask_obj_single=np.array([elem in choice_source for elem in obj_list])
else:
    mask_obj_single=np.repeat(True,len(obj_list))
    
#masking the objects depending on inclination
mask_inclin=[include_noinclin if elem not in incl_dic else getoverlap((incl_dic[elem][0]-incl_dic[elem][1],incl_dic[elem][0]+incl_dic[elem][2]),slider_inclin)>0 for elem in obj_list]

#masking objects whose inclination bond go beyond the inclination restrictions if asked to
if incl_inside:
    mask_inclin=[False if elem not in incl_dic else round(getoverlap((incl_dic[elem][0]-incl_dic[elem][1],incl_dic[elem][0]+incl_dic[elem][2]),                slider_inclin),3)==incl_dic[elem][1]+incl_dic[elem][2] for elem in obj_list]
    
#masking dippers/non dipper if asked to
mask_dippers=np.array([elem in dippers_list for elem in obj_list])

if radio_dipper=='Restrict to dippers':
    mask_inclin=(mask_inclin) & (mask_dippers)
elif radio_dipper=='Restrict to non-dippers':
    mask_inclin=(mask_inclin) & ~(mask_dippers)
#double mask taking into account both single/multiple display mode and the inclination    

mask_obj_base=(mask_obj_single) & (mask_inclin)
    
####Dates restriction
if restrict_time:
    slider_date=st.slider('Dates restriction',min_value=Time(min(ravel_ragged(date_list[mask_obj_base]))).datetime,
                          max_value=Time(max(ravel_ragged(date_list[mask_obj_base]))).datetime,
                          value=[Time(min(ravel_ragged(date_list[mask_obj_base]))).datetime,
                                 Time(max(ravel_ragged(date_list[mask_obj_base]))).datetime])
else:
    slider_date=[Time(min(ravel_ragged(date_list[mask_obj_base]))).datetime,
                                 Time(max(ravel_ragged(date_list[mask_obj_base]))).datetime]
    
#creating a mask according to the sources with observations in the current date restriction
mask_obj_intime=np.array([((np.array([Time(subelem) for subelem in elem])>=Time(slider_date[0])) &\
                  (np.array([Time(subelem) for subelem in elem])<=Time(slider_date[1]))).any() for elem in date_list])

#restricting mask_obj_base with the new base
mask_obj_base=mask_obj_base & mask_obj_intime

#checking which sources have no detection in the current combination
global_displayed_sign=abslines_plot[4][0][mask_lines].T

#creating a mask from the ones with at least one detection 
#(or at least one significant detections if we don't consider non significant detections)
if display_nonsign:
    mask_obj_withdet=np.array([(np.array([subelem for subelem in elem])>0).any() for elem in global_displayed_sign])
else:
    mask_obj_withdet=np.array([(np.array([subelem for subelem in elem])>slider_sign).any() for elem in global_displayed_sign])
    
if not display_obj_zerodet:
    mask_obj=mask_obj_base & mask_obj_withdet
else:
    mask_obj=mask_obj_base
    
hid_plot_restrict=hid_plot.T[mask_obj].T
incl_plot_restrict=incl_plot[mask_obj]

#creating variables with values instead of uncertainties for the inclination and nh colormaps

incl_cmap=np.array([incl_plot.T[0],incl_plot.T[0]-incl_plot.T[1],incl_plot.T[0]+incl_plot.T[2]]).T
incl_cmap_base=incl_cmap[mask_obj_base]
incl_cmap_restrict=incl_cmap[mask_obj]

nh_plot_restrict=deepcopy(nh_plot)
nh_plot_restrict=nh_plot_restrict.T[mask_obj].T

#defining the dataset that will be used in the plots for the colormap limits
if radio_info_cmap in ['Blueshift','Delchi']:
    radio_cmap_i=1 if radio_info_cmap=='Blueshift' else 2
else:
    radio_cmap_i=0
    
#computing the extremal values of the whole sample/plotted sample to get coherent colormap normalisations, and creating the range of object colors
if global_colors:
    global_plotted_sign=abslines_plot[4][0].ravel()
    global_plotted_data=abslines_plot[radio_cmap_i][0].ravel()
    #objects colormap
    norm_colors_obj=mpl.colors.Normalize(vmin=0,vmax=len(abslines_infos_perobj)+1)
    colors_obj=mpl.cm.ScalarMappable(norm=norm_colors_obj,cmap=mpl.cm.hsv)

else:
    global_plotted_sign=abslines_plot[4][0][mask_lines].T[mask_obj].ravel()
    global_plotted_data=abslines_plot[radio_cmap_i][0][mask_lines].T[mask_obj].ravel()
    
    #the date is an observation-level parameter so it needs to be repeated to have the same dimension as the other global variables
    global_plotted_datetime=np.array([elem for elem in date_list[mask_obj] for i in range(sum(mask_lines))],dtype='object')

    #objects colormap
    norm_colors_obj=mpl.colors.Normalize(vmin=0,vmax=len(abslines_infos_perobj[mask_obj])+1)
    colors_obj=mpl.cm.ScalarMappable(norm=norm_colors_obj,cmap=mpl.cm.hsv)
    
# else:
#     global_plotted_instr=

#adapting the plotted data in regular array for each object in order to help
#global masks to take off elements we don't want in the comparison

global_mask_intime=(Time(ravel_ragged(global_plotted_datetime))>=Time(slider_date[0])) &\
    (Time(ravel_ragged(global_plotted_datetime))<=Time(slider_date[1]))

global_mask_intime_norepeat=(Time(ravel_ragged(date_list[mask_obj]))>=Time(slider_date[0])) &\
    (Time(ravel_ragged(date_list[mask_obj]))<=Time(slider_date[1]))

#global_nondet_mask=(np.array([subelem for elem in global_plotted_sign for subelem in elem])<=slider_sign) & (global_mask_intime)

global_det_mask=(np.array([subelem for elem in global_plotted_sign for subelem in elem])>0) & (global_mask_intime)

global_sign_mask=(np.array([subelem for elem in global_plotted_sign for subelem in elem])>slider_sign) & (global_mask_intime)

global_det_data=np.array([subelem for elem in global_plotted_data for subelem in elem])[global_det_mask]

#this second array is here to restrict the colorbar scalings to take into account significant detections only
global_sign_data=np.array([subelem for elem in global_plotted_data for subelem in elem])[global_sign_mask]

#same for the color-coded infos
cmap_info=mpl.cm.plasma_r if radio_info_cmap not in ['Time','nH'] else mpl.cm.plasma

#normalisation of the colormap
if radio_cmap_i==1 or radio_info_cmap=='EQW ratio':
    gamma_colors=1 if radio_cmap_i==1 else 0.5
    cmap_norm_info=colors.PowerNorm(gamma=gamma_colors)
    
elif radio_info_cmap not in ['Inclination','Time']:
    cmap_norm_info=colors.LogNorm()
else:
    #keeping a linear norm for the inclination
    cmap_norm_info=colors.Normalize()
    
#building the powerlaw ticks for

#putting the axis limits at standard bounds or the points if the points extend further
flux_list_ravel=np.array([subelem for elem in flux_list for subelem in elem])
bounds_x=[min(flux_list_ravel.T[2][0]/flux_list_ravel.T[1][0]),max(flux_list_ravel.T[2][0]/flux_list_ravel.T[1][0])]
bounds_y=[min(flux_list_ravel.T[4][0]),max(flux_list_ravel.T[4][0])]

if checkbox_zoom:
    ax_hid.margins(0.05)   
else:
    ax_hid.set_xlim((min(bounds_x[0]*0.9,0.1),max(bounds_x[1]*1.1,2)))
    ax_hid.set_ylim((min(bounds_y[0]*0.9,1e-5),max(bounds_y[1]*1.1,1)))

#creating space for the colorbar
if radio_info_cmap not in ['Source','Instrument']:
    ax_cb=plt.axes([0.92, 0.105, 0.02, 0.775])

#markers
marker_abs='o'
marker_nondet='d'
marker_ul='v'
marker_ul_top='^'

alpha_ul=0.3

label_obj_plotted=np.repeat(False
                            ,len(abslines_infos_perobj[mask_obj]))
is_colored_scat=False

#creating the plotted colors variable#defining the mask for detections and non detection
plotted_colors_var=[]

#### detections HID


#loop on the objects for detections (restricted or not depending on if the mode is detection only)
for i_obj,abslines_obj in enumerate(abslines_infos_perobj[mask_obj]):
    
    #defining the index of the object in the entire array if asked to, in order to avoid changing colors
    if global_colors:
        i_obj_glob=np.argwhere(obj_list==obj_list[mask_obj][i_obj])[0][0]
    else:
        i_obj_glob=i_obj
        
    '''
    The shape of each abslines_obj is (uncert,info,line,obs)
    '''
    
    #defining the hid positions of each point
    x_hid=flux_list[mask_obj][i_obj].T[2][0]/flux_list[mask_obj][i_obj].T[1][0]
    y_hid=flux_list[mask_obj][i_obj].T[4][0]
            
    #defining the masks and shapes of the markers for the rest

    #defining the mask for the time interval restriction
    datelist_obj=Time(np.array([date_list[mask_obj][i_obj] for i in range(sum(mask_lines))]).astype(str))
    mask_intime=(datelist_obj>=Time(slider_date[0])) & (datelist_obj<=Time(slider_date[1]))
    
    #defining the mask for detections and non detection        
    mask_det=(abslines_obj[0][4][mask_lines]>0.) & (mask_intime)
    
    #defining the mask for significant detections
    mask_sign=(abslines_obj[0][4][mask_lines]>slider_sign) & (mask_intime)
        
    #these ones will only be used if the restrict values chexbox is checked

    obj_val_cmap_sign=np.array(
        [np.nan if len(abslines_obj[0][radio_cmap_i][mask_lines].T[i_obs][mask_sign.T[i_obs]])==0 else\
                                   (max(abslines_obj[0][radio_cmap_i][mask_lines].T[i_obs][mask_sign.T[i_obs]])\
                                    if radio_info_cmap!='EQW ratio' else\
                                        0 if abslines_obj[0][radio_cmap_i][eqw_ratio_ids[0]].T[i_obs]==0 else \
                                        abslines_obj[0][radio_cmap_i][eqw_ratio_ids[1]].T[i_obs]/\
                                        abslines_obj[0][radio_cmap_i][eqw_ratio_ids[0]].T[i_obs])\
                                   for i_obs in range(len(abslines_obj[0][radio_cmap_i][mask_lines].T))])

        
    #the size is always tied to the EQW
    obj_size_sign=np.array([np.nan if len(abslines_obj[0][0][mask_lines].T[i_obs][mask_sign.T[i_obs]])==0 else\
                               max(abslines_obj[0][0][mask_lines].T[i_obs][mask_sign.T[i_obs]])\
                                   for i_obs in range(len(abslines_obj[0][0][mask_lines].T))])
        
    #and we can create the plot mask from it (should be the same wether we take obj_size_sign or the size)
    if radio_info_cmap!='EQW ratio':
        obj_val_mask_sign=~np.isnan(obj_size_sign)
    else:
        obj_val_mask_sign=np.array((~np.isnan(obj_size_sign)).tolist() and (obj_val_cmap_sign!=0).tolist())
    
    #creating a display order which is the reverse of the eqw size order to make sure we do not hide part the detections
    obj_order_sign=obj_size_sign[obj_val_mask_sign].argsort()[::-1]
            
    #same thing for all detections
    obj_val_cmap=np.array([np.nan if len(abslines_obj[0][radio_cmap_i][mask_lines].T[i_obs][mask_det.T[i_obs]])==0 else\
                               max(abslines_obj[0][radio_cmap_i][mask_lines].T[i_obs][mask_det.T[i_obs]])\
                                   for i_obs in range(len(abslines_obj[0][radio_cmap_i][mask_lines].T))])
        
    obj_size=np.array([np.nan if len(abslines_obj[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]])==0 else\
                               max(abslines_obj[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]])\
                                   for i_obs in range(len(abslines_obj[0][0][mask_lines].T))])
             
    obj_val_mask=~np.isnan(obj_size)

    #creating a display order which is the reverse of the eqw size order to make sure we show as many detections as possible
    obj_order=obj_size[obj_val_mask].argsort()[::-1]
    
    # not used for now                
    # else:
    #     #in single line mode we can directly fetch the single lines values and the mask for the specific line
    #     obj_val_cmap_sign=abslines_obj[0][radio_cmap_i][mask_lines][mask_sign[mask_lines]].astype(float)
        
    #     #and the mask
    #     obj_val_mask_sign=mask_sign[mask_lines]
        
    #     #in single line mode we can directly fetch the single lines values and the mask for the specific line
    #     obj_val_cmap=abslines_obj[0][radio_cmap_i][mask_lines][mask_det[mask_lines]].astype(float)
        
    #     obj_val_mask=mask_det[mask_lines]
        
    #this mask is used to plot 'unsignificant only' detection points
    obj_val_mask_nonsign=(obj_val_mask) & (~obj_val_mask_sign)

    #plotting everything
    
    #we put the color mapped scatter into a list to clim all of them at once at the end
    if i_obj==0:
        scat_col=[]
                
    #plotting the detection centers if asked for
    
    if len(x_hid[obj_val_mask])>0 and display_central_abs:
        ax_hid.scatter(x_hid[obj_val_mask],y_hid[obj_val_mask],marker=marker_abs,color=colors_obj.to_rgba(i_obj_glob)\
                       if radio_info_cmap=='Source' else 'grey',label='',zorder=1000,edgecolor='black')

        
    #### detection scatters
    #plotting statistically significant absorptions before values
    
    if radio_info_cmap=='Instrument':
        color_instru=[telescope_colors[elem] for elem in instru_list[mask_obj][i_obj][obj_val_mask_sign][obj_order_sign]]
        
        if display_nonsign:
            color_instru_nonsign=[telescope_colors[elem] for elem in instru_list[mask_obj][i_obj][obj_val_mask_nonsign]]
                    
    #note: there's no need to reorder for source level informations (ex: inclination) since the values are the same for all the points                    
    c_scat=None if radio_info_cmap=='Source' else\
        mdates.date2num(date_list[mask_obj][i_obj][obj_val_mask_sign][obj_order_sign]) if radio_info_cmap=='Time' else\
        np.repeat(incl_cmap_restrict[i_obj][cmap_incl_type],len(x_hid[obj_val_mask_sign])) if radio_info_cmap=='Inclination' else\
        color_instru if radio_info_cmap=='Instrument' else\
        nh_plot_restrict[0][i_obj][obj_val_mask_sign][obj_order_sign] if radio_info_cmap=='nH' else\
        obj_val_cmap_sign[obj_val_mask_sign][obj_order_sign]
            
    
        ###TODO : test the dates here with just IGRJ17451 to solve color problem

    #adding a failsafe to avoid problems when nothing is displayed
    if c_scat is not None and len(c_scat)==0:
        c_scat=None
        
    if restrict_threshold:
        
        #displaying "significant only" cmaps/sizes
        scat_col+=[ax_hid.scatter(x_hid[obj_val_mask_sign][obj_order_sign],y_hid[obj_val_mask_sign][obj_order_sign],
                                  marker=marker_abs,color=colors_obj.to_rgba(i_obj_glob) if radio_info_cmap=='Source' else None,
        c=c_scat,s=5*obj_size_sign[obj_val_mask_sign][obj_order_sign]**1.15,
                       edgecolor='black' if not display_edgesource else colors_obj.to_rgba(i_obj_glob),
                       linewidth=1+int(display_edgesource)/2,
                       norm=cmap_norm_info,
                           label=obj_list[mask_obj][i_obj] if not label_obj_plotted[i_obj] and\
                               (radio_info_cmap=='Source' or display_edgesource) and len(x_hid[obj_val_mask_sign])>0 else '',
                           cmap=cmap_info,alpha=alpha_abs)]
            
        if (radio_info_cmap=='Source' or display_edgesource) and len(x_hid[obj_val_mask_sign])>0:
            label_obj_plotted[i_obj]=True

    #plotting the maximum value and hatch coding depending on if there's a significant abs line in the obs
    else:
                
        #displaying "all" cmaps/sizes but only where's at least one significant detection (so we don't hatch)
        scat_col+=[ax_hid.scatter(x_hid[obj_val_mask_sign][obj_order_sign],y_hid[obj_val_mask_sign][obj_order_sign],
                                  marker=marker_abs,color=colors_obj.to_rgba(i_obj_glob) if radio_info_cmap=='Source' else None,
        c=c_scat,s=5*obj_size[obj_val_mask_sign][obj_order_sign]**1.15,
                       edgecolor='black' if not display_edgesource else colors_obj.to_rgba(i_obj_glob),
                       linewidth=1+int(display_edgesource),
                       norm=cmap_norm_info,
                           label=obj_list[mask_obj][i_obj] if not label_obj_plotted[i_obj] and\
                               (radio_info_cmap=='Source' or display_edgesource) and len(x_hid[obj_val_mask_sign])>0 else '',
                           cmap=cmap_info,alpha=alpha_abs)]
        
        if (radio_info_cmap=='Source' or display_edgesource) and len(x_hid[obj_val_mask_sign])>0:
            label_obj_plotted[i_obj]=True
            
    #adding the plotted colors into a list to create the ticks from it at the end
    plotted_colors_var+=[elem for elem in (incl_cmap_restrict.T[cmap_incl_type] if radio_info_cmap=='Inclination' else \
                                           (obj_val_cmap_sign[obj_val_mask_sign][obj_order_sign] if restrict_threshold\
                         else obj_val_cmap[obj_val_mask_sign][obj_order_sign]).tolist()) if not np.isnan(elem)]
    
    if display_nonsign:
        
        c_scat_nonsign=None if radio_info_cmap=='Source' else\
            mdates.date2num(date_list[mask_obj][i_obj][obj_val_mask_nonsign]) if radio_info_cmap=='Time' else\
            np.repeat(incl_cmap_restrict[i_obj][cmap_incl_type],len(x_hid[obj_val_mask_nonsign])) if radio_info_cmap=='Inclination' else\
            nh_plot_restrict[0][i_obj][obj_val_mask_nonsign] if radio_info_cmap=='nH' else\
                obj_val_cmap[obj_val_mask_nonsign]
                       
        #adding a failsafe to avoid problems when nothing is displayed
        if c_scat is not None and len(c_scat)==0:
            c_scat=None
            
        #and "unsignificant only" in any case is hatched. Edgecolor sets the color of the hatch
        scat_col+=[ax_hid.scatter(x_hid[obj_val_mask_nonsign],y_hid[obj_val_mask_nonsign],marker=marker_abs,
                       color=colors_obj.to_rgba(i_obj_glob) if radio_info_cmap=='Source' else None,
        c=c_scat_nonsign,s=5*obj_size[obj_val_mask_nonsign]**1.15,hatch='///',
                       edgecolor='grey' if not display_edgesource else colors_obj.to_rgba(i_obj_glob),
                       linewidth=1+int(display_edgesource),
                       norm=cmap_norm_info,
                       label=obj_list[mask_obj][i_obj] if not label_obj_plotted[i_obj] and\
                               (radio_info_cmap=='Source' or display_edgesource) else '',
                       cmap=cmap_info,
                       alpha=alpha_abs)]
        if (radio_info_cmap=='Source' or display_edgesource) and len(x_hid[obj_val_mask_nonsign])>0:
            label_obj_plotted[i_obj]=True
    
        plotted_colors_var+=[elem for elem in (incl_cmap_restrict.T[cmap_incl_type] if radio_info_cmap=='Inclination' else obj_val_cmap[obj_val_mask_nonsign].tolist()) if not np.isnan(elem)]
        
    
    #resizing all the colors and plotting the colorbar, only done at the last iteration
    if radio_info_cmap not in ['Source','Instrument'] and i_obj==len(abslines_infos_perobj[mask_obj])-1 and len(plotted_colors_var)>0:
        
        is_colored_scat=False
        
        for elem_scatter in scat_col:
            
            #standard limits for the inclination and Time
            if radio_info_cmap=='Inclination':
                elem_scatter.set_clim(vmin=0,vmax=90)
            elif radio_info_cmap=='Time':

                elem_scatter.set_clim(
                    vmin=max(min(mdates.date2num(ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])),mdates.date2num(slider_date[0])),
                    vmax=min(max(mdates.date2num(ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])),mdates.date2num(slider_date[1])))          
                
            elif radio_info_cmap=='nH':
                elem_scatter.set_clim(vmin=min(ravel_ragged(nh_plot_restrict[0])[global_mask_intime_norepeat]),
                                      vmax=max(ravel_ragged(nh_plot_restrict[0])[global_mask_intime_norepeat]))
                
            else:
                
                #dynamical limits for the rest
                if global_colors and radio_info_cmap not in ('EQW ratio','Inclination','Time','nH'):
                    if display_nonsign:
                        elem_scatter.set_clim(vmin=min(global_det_data),vmax=max(global_det_data))
                    else:
                        elem_scatter.set_clim(vmin=min(global_sign_data),vmax=max(global_sign_data))
                else:
                    elem_scatter.set_clim(vmin=min(plotted_colors_var),vmax=max(plotted_colors_var))

            if len(elem_scatter.get_sizes())>0:
                is_colored_scat=True
                
                #keeping the scatter to create the colorbar from it
                elem_scatter_forcol=elem_scatter
            # ax_cb.set_axis_off()
            
        #defining the ticks from the currently plotted objects
        
        if radio_cmap_i==1 or radio_info_cmap=='EQW ratio':

            cmap_min_sign=1 if min(plotted_colors_var)==0 else min(plotted_colors_var)/abs(min(plotted_colors_var))

            #round numbers for the blueshift                
            if radio_info_cmap=='Blueshift':
                bshift_step=500
                
                #the -1 and +2 are here to ensure we see the extremal ticks
                cmap_norm_ticks=np.arange((cmap_min_sign//bshift_step-1)*bshift_step,((max(plotted_colors_var)//bshift_step)+3)*bshift_step,
                                          1000)
                elem_scatter.set_clim(vmin=min(cmap_norm_ticks),vmax=max(cmap_norm_ticks))
                
            else:
                cmap_norm_ticks=np.linspace(cmap_min_sign*abs(min(plotted_colors_var))**(gamma_colors),
                                        max(plotted_colors_var)**(gamma_colors),7,endpoint=True)
            
            #adjusting to round numbers
            
            if radio_info_cmap=='EQW ratio':

                cmap_norm_ticks=np.concatenate((cmap_norm_ticks,np.array([1])))

                cmap_norm_ticks.sort()
                
            if radio_cmap_i==1 and min(plotted_colors_var)<0:
                # cmap_norm_ticks=np.concatenate((cmap_norm_ticks,np.array([0])))
                # cmap_norm_ticks.sort()
                pass
            
            if radio_info_cmap!='Blueshift':
                #maintaining the sign with the square norm
                cmap_norm_ticks=cmap_norm_ticks**(1/gamma_colors)           

                cmap_norm_ticks=np.concatenate((np.array([min(plotted_colors_var)]),cmap_norm_ticks))

                cmap_norm_ticks.sort()

            
        else:
            cmap_norm_ticks=None
        
        #only creating the colorbar if there is information to display
        if is_colored_scat or radio_info_cmap in ['Inclination','Time','nH'] :
            
            if radio_info_cmap=='Time':
                
                #manually readjusting for small durations because the AutoDateLocator doesn't work well
                time_range=min(max(mdates.date2num(ravel_ragged(date_list[mask_obj]))),mdates.date2num(slider_date[1]))-\
                        max(min(mdates.date2num(ravel_ragged(date_list[mask_obj]))),mdates.date2num(slider_date[0]))
                
                if time_range<150:
                    date_format=mdates.DateFormatter('%Y-%m-%d')
                elif time_range<1825:
                    date_format=mdates.DateFormatter('%Y-%m')
                else:
                    date_format=mdates.AutoDateFormatter(mdates.AutoDateLocator())
                
                cb=plt.colorbar(elem_scatter_forcol,cax=ax_cb,ticks=mdates.AutoDateLocator(),format=date_format)    
            else:
                cb=plt.colorbar(elem_scatter_forcol,cax=ax_cb,extend='min' if radio_info_cmap=='nH' else None)
                cb.set_ticks(cmap_norm_ticks)
            
        # cb.ax.minorticks_off()
        
            if radio_cmap_i==1:
                cb_add_str=' (km/s)'
            else:
                cb_add_str=''
                
            if radio_info_cmap=='Inclination':
                cb.set_label(cmap_incl_type_str+' of the source inclination (°)',labelpad=10)
            elif radio_info_cmap=='Time':
                cb.set_label('Observation date',labelpad=30)
            elif radio_info_cmap=='nH':
                cb.set_label(r'nH ($10^{22}$ cm$^{-1}$)',labelpad=10)
            else:
                if restrict_threshold:
                    cb.set_label(('maximal ' if radio_info_cmap!='EQW ratio' else '')+radio_info_label[radio_cmap_i-1].lower()+
                                 ' in significant detections\n for each observation'+cb_add_str,labelpad=10)
                else:
                    cb.set_label(('maximal ' if radio_info_cmap!='EQW ratio' else '')+radio_info_label[radio_cmap_i-1].lower()+
                                 ' in all detections\n for each observation'+cb_add_str,labelpad=10)
                 
                    
label_obj_plotted=np.repeat(False,len(abslines_infos_perobj[mask_obj]))

#### non detections HID

#loop for non detection, separated to be able to restrict the color range in case of non detection
for i_obj_base,abslines_obj_base in enumerate(abslines_infos_perobj[mask_obj_base]):
    
    #skipping everything if we don't plot nondetections
    if not display_nondet:
        continue
    
    #defining the index of the object in the entire array if asked to, in order to avoid changing colors
    if global_colors:
        i_obj_glob=np.argwhere(obj_list==obj_list[mask_obj_base][i_obj_base])[0][0]
    else:
        i_obj_glob=i_obj_base
    
    '''
    The shape of each abslines_obj is (uncert,info,line,obs)
    '''
        
    #we use non-detection-masked arrays for non detection to plot them even while restricting the colors to a part of the sample 
    x_hid_base=flux_list[mask_obj_base][i_obj_base].T[2][0]/flux_list[mask_obj_base][i_obj_base].T[1][0]
    y_hid_base=flux_list[mask_obj_base][i_obj_base].T[4][0]
    
    x_hid_incert=hid_plot.T[mask_obj_base][i_obj_base].T[0]
    y_hid_incert=hid_plot.T[mask_obj_base][i_obj_base].T[1]

        
    #reconstructing standard arrays
    x_hid_incert=np.array([[subelem for subelem in elem] for elem in x_hid_incert])
    y_hid_incert=np.array([[subelem for subelem in elem] for elem in y_hid_incert])
    #defining the masks and shapes of the markers for the rest

    #defining the non detection as strictly non detection or everything below the significance threshold
    if display_nonsign:
        mask_det=abslines_obj_base[0][4][mask_lines]>0.
    
    else:
        mask_det=abslines_obj_base[0][4][mask_lines]>slider_sign

    #defining the mask for the time interval restriction
    datelist_obj=Time(np.array([date_list[mask_obj_base][i_obj_base]\
                                for i in range(sum(mask_lines_ul if display_upper else mask_lines))]).astype(str))
    mask_intime=(datelist_obj>=Time(slider_date[0])) & (datelist_obj<=Time(slider_date[1]))
    
    mask_intime_norepeat=(Time(date_list[mask_obj_base][i_obj_base].astype(str))>=Time(slider_date[0])) & (Time(date_list[mask_obj_base][i_obj_base].astype(str))<=Time(slider_date[1]))
    
    #defining the mask
    prev_mask_nondet=np.isnan(np.array([np.nan if len(abslines_obj_base[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]])==0 else\
                               max(abslines_obj_base[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]])\
                                   for i_obs in range(len(abslines_obj_base[0][0][mask_lines].T))]))
        
    mask_nondet=(np.isnan(np.array([np.nan if len(abslines_obj_base[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]])==0 else\
                               max(abslines_obj_base[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]])\
                                   for i_obs in range(len(abslines_obj_base[0][0][mask_lines].T))]))) & (mask_intime_norepeat)

    if len(x_hid_base[mask_nondet])>0:
        #note: due to problems with colormapping of the edgecolors we directly compute the color of the edges with a normalisation
        norm_cmap_incl = mpl.colors.Normalize(0,90)
        
        norm_cmap_time = mpl.colors.Normalize(min(mdates.date2num(ravel_ragged(date_list[mask_obj_base])[global_mask_intime_norepeat])),
                                              max(mdates.date2num(ravel_ragged(date_list[mask_obj_base])[global_mask_intime_norepeat])))
        if display_upper:
            
            #we define the upper limit range of points independantly to be able to have a different set of lines used for detection and
            #upper limits if necessary
            
            mask_det_ul=(abslines_obj_base[0][4][mask_lines_ul]>0.) & (mask_intime)
            mask_det_ul=(abslines_obj_base[0][4][mask_lines_ul]>slider_sign) & (mask_intime)

            mask_nondet_ul=np.isnan(np.array(\
                                        [np.nan if len(abslines_obj_base[0][0][mask_lines_ul].T[i_obs][mask_det_ul.T[i_obs]])==0 else\
                                       max(abslines_obj_base[0][0][mask_lines_ul].T[i_obs][mask_det_ul.T[i_obs]])\
                                           for i_obs in range(len(abslines_obj_base[0][0][mask_lines].T))])) & (mask_intime_norepeat)
            
            #defining the sizes of upper limits
            obj_size_ul=np.array([np.nan if len(abslines_obj_base[0][0][mask_lines_ul].T[i_obs][mask_det_ul.T[i_obs]])!=0 else\
                                       max(abslines_obj_base[0][5][mask_lines_ul].T[i_obs][~mask_det_ul.T[i_obs]])\
                                           for i_obs in range(len(abslines_obj_base[0][0][mask_lines_ul].T))])
            
            #creating a display order which is the reverse of the eqw size order to make sure we do not hide part the ul
            #not needed now that the UL are not filled colorwise
            # obj_order_sign_ul=obj_size_ul[mask_nondet_ul].argsort()[::-1]
            
            #there is no need to use different markers unless we display source per color, so we limit the different triangle to this case
            marker_ul_curr=marker_ul_top if i_obj_base%2!=0 and radio_info_cmap=='Source' else marker_ul
                            
            if radio_info_cmap=='Instrument':
                color_data=[telescope_colors[elem] for elem in instru_list[mask_obj_base][i_obj_base][mask_nondet_ul]]
                
                edgec_scat=[colors.to_rgba(elem) for elem in color_data]
            else:
                
                edgec_scat=colors_obj.to_rgba(i_obj_glob) if radio_info_cmap=='Source' and display_obj_zerodet else\
                            cmap_info(norm_cmap_incl(incl_cmap_base[i_obj_base][cmap_incl_type]))\
                            if (not np.isnan(incl_cmap_base[i_obj_base][cmap_incl_type]) and radio_info_cmap=='Inclination') else\
                    cmap_info(norm_cmap_time(mdates.date2num(date_list[mask_obj_base][i_obj_base][mask_nondet_ul])))\
                            if radio_info_cmap=='Time' else\
                        cmap_info(cmap_norm_info(nh_plot.T[mask_obj_base].T[0][i_obj_base][mask_nondet_ul])) if radio_info_cmap=='nH' else\
                            'grey'
            #adding a failsafe to avoid problems when nothing is displayed
            if len(edgec_scat)==0:
                edgec_scat=None
            
            elem_scatter_nondet=ax_hid.scatter(
                x_hid_base[mask_nondet_ul],y_hid_base[mask_nondet_ul],marker=marker_ul_curr,
                           color='none',edgecolor=edgec_scat,s=5*obj_size_ul[mask_nondet_ul]**1.15,
                           label='' if not display_obj_zerodet else (obj_list[mask_obj][i_obj_base] if not label_obj_plotted[i_obj_base] and\
                               (radio_info_cmap=='Source' or display_edgesource) else ''),zorder=500,alpha=1.0,
                               cmap=cmap_info if radio_info_cmap in ['Inclination','Time'] else None)
        else:

            if radio_info_cmap=='Instrument':
                color_data=[telescope_colors[elem] for elem in instru_list[mask_obj_base][i_obj_base][mask_nondet]]
                
                c_scat_nondet=[colors.to_rgba(elem) for elem in color_data]
            else:                    
                c_scat_nondet=colors_obj.to_rgba(i_obj_glob) if radio_info_cmap=='Source' and display_obj_zerodet else\
                            cmap_info(norm_cmap_incl(incl_cmap_base[i_obj_base][cmap_incl_type]))\
                            if (not np.isnan(incl_cmap_base[i_obj_base][cmap_incl_type]) and radio_info_cmap=='Inclination') else\
                    mdates.date2num(date_list[mask_obj_base][i_obj_base][mask_nondet])\
                            if radio_info_cmap=='Time' else\
                            nh_plot.T[mask_obj_base].T[0][i_obj_base][mask_nondet] if radio_info_cmap=='nH' else\
                            'grey'

            elem_scatter_nondet=ax_hid.scatter(x_hid_base[mask_nondet],y_hid_base[mask_nondet],marker=marker_nondet,
                           c=c_scat_nondet,cmap=cmap_info,norm=cmap_norm_info,
                           label='' if not display_obj_zerodet else (obj_list[mask_obj][i_obj_base] if not label_obj_plotted[i_obj_base] and\
                               (radio_info_cmap=='Source' or display_edgesource) else ''),zorder=1000,edgecolor='black',alpha=1.)
        
            if display_hid_error:
                
                #in order to get the same clim as with the standard scatter plots, we manually readjust the rgba values of the colors before plotting
                #the errorbar "empty" and changing its color manually (because as of now matplotlib doesn't like multiple color inputs for errbars)
                if radio_info_cmap in ['Inclincation','Time']:
                    if radio_info_cmap=='Inclination':
                        cmap_norm_info.vmin=0
                        cmap_norm_info.vmax=90
                    elif radio_info_cmap=='Time':
                        cmap_norm_info.vmin=max(min(mdates.date2num(ravel_ragged(date_list[mask_obj_base])[global_mask_intime_norepeat])),
                                                mdates.date2num(slider_date[0]))
                        cmap_norm_info.vmax=min(max(mdates.date2num(ravel_ragged(date_list[mask_obj_base])[global_mask_intime_norepeat])),
                                                mdates.date2num(slider_date[1]))
                        
                    elif radio_info_cmap=='nH':
                        cmap_norm_info.vmin=min(ravel_ragged(nh_plot.T[mask_obj_base].T[0])[global_mask_intime_norepeat])
                        cmap_norm_info.vmax=max(ravel_ragged(nh_plot.T[mask_obj_base].T[0])[global_mask_intime_norepeat])
                        
                    colors_func=mpl.cm.ScalarMappable(norm=cmap_norm_info,cmap=cmap_info)
                    
                    c_scat_nondet_rgba_clim=colors_func.to_rgba(c_scat_nondet)

                elem_err_nondet=ax_hid.errorbar(x_hid_incert[0][mask_nondet],y_hid_incert[0][mask_nondet],xerr=x_hid_incert[1:].T[mask_nondet].T,yerr=y_hid_incert[1:].T[mask_nondet].T,marker='None',linestyle='None',linewidth=0.5,
                               c=c_scat_nondet if radio_info_cmap not in ['Inclination','Time','nH'] else None,label='',zorder=1000,alpha=1.)
                
                if radio_info_cmap in ['Inclincation','Time','nH']:
                    for elem_children in elem_err_nondet.get_children()[1:]:

                        elem_children.set_colors(c_scat_nondet_rgba_clim)
                
        if radio_info_cmap=='Source' and display_obj_zerodet:
            label_obj_plotted[i_obj_base]=True
        
        
        if radio_info_cmap in ['Inclination','Time','nH']:
            
            if radio_info_cmap=='Inclination':
                elem_scatter_nondet.set_clim(vmin=0,vmax=90)
                
                # if display_hid_error:
                #     elem_err_nondet.set_clim(vmin=0,vmax=90)
                
            elif radio_info_cmap=='Time':
                elem_scatter_nondet.set_clim(
                    vmin=max(min(mdates.date2num(ravel_ragged(date_list[mask_obj_base])[global_mask_intime_norepeat])),mdates.date2num(slider_date[0])),
                    vmax=min(max(mdates.date2num(ravel_ragged(date_list[mask_obj_base])[global_mask_intime_norepeat])),mdates.date2num(slider_date[1])))       
                # if display_hid_error:
                #     elem_err_nondet.set_clim(
                #     vmin=max(min(mdates.date2num(ravel_ragged(date_list[mask_obj_base][global_mask_intime_norepeat]))),mdates.date2num(slider_date[0])),
                #     vmax=min(max(mdates.date2num(ravel_ragged(date_list[mask_obj_base][global_mask_intime_norepeat]))),mdates.date2num(slider_date[1])))
                
            elif radio_info_cmap=='nH':
                elem_scatter_nondet.set_clim(vmin=min(ravel_ragged(nh_plot.T[mask_obj_base].T[0])[global_mask_intime_norepeat]),
                                             vmax=max(ravel_ragged(nh_plot.T[mask_obj_base].T[0])[global_mask_intime_norepeat]))
                
            if len(elem_scatter_nondet.get_sizes())>0:
                is_colored_scat_nondet=True
            
            #creating the colorbar at the end if it hasn't been created with the detections
            if i_obj_base==len(abslines_infos_perobj[mask_obj_base])-1 and not is_colored_scat and is_colored_scat_nondet:

                #creating an empty scatter with a 'c' value to serve as base for the colorbar
                elem_scatter_empty=ax_hid.scatter(x_hid_base[mask_nondet][False],y_hid_base[mask_nondet][False],marker=None,
                               c=cmap_info(norm_cmap_time(mdates.date2num(date_list[mask_obj_base][i_obj_base][mask_nondet])))[False],
                               label='',zorder=1000,edgecolor=None,cmap=cmap_info,alpha=1.)
        
                if radio_info_cmap=='Inclination':
                    
                    elem_scatter_empty.set_clim(vmin=0,vmax=90)

                    cb=plt.colorbar(elem_scatter_empty,cax=ax_cb)
                        
                    cb.set_label(cmap_incl_type_str+' of the source inclination (°)',labelpad=10)
                elif radio_info_cmap=='Time':

                    elem_scatter_empty.set_clim(
                    vmin=max(min(mdates.date2num(ravel_ragged(date_list[mask_obj_base]))),mdates.date2num(slider_date[0])),
                    vmax=min(max(mdates.date2num(ravel_ragged(date_list[mask_obj_base]))),mdates.date2num(slider_date[1])))
                                           
                    #manually readjusting for small durations because the AutoDateLocator doesn't work well
                    time_range=min(max(mdates.date2num(ravel_ragged(date_list[mask_obj_base]))),mdates.date2num(slider_date[1]))-\
                            max(min(mdates.date2num(ravel_ragged(date_list[mask_obj_base]))),mdates.date2num(slider_date[0]))
                    
                    if time_range<150:
                        date_format=mdates.DateFormatter('%Y-%m-%d')
                    elif time_range<1825:
                        date_format=mdates.DateFormatter('%Y-%m')
                    else:
                        date_format=mdates.AutoDateFormatter(mdates.AutoDateLocator())
                    
                    cb=plt.colorbar(elem_scatter_empty,cax=ax_cb,ticks=mdates.AutoDateLocator(),format=date_format)    

                    cb.set_label('Observation date',labelpad=10)
                    
                elif radio_info_cmap=='nH':
                    elem_scatter_empty.set_clim(vmin=min(ravel_ragged(nh_plot.T[mask_obj_base].T[0])[global_mask_intime_norepeat]),
                                             vmax=max(ravel_ragged(nh_plot.T[mask_obj_base].T[0])[global_mask_intime_norepeat]))
                    cb=plt.colorbar(elem_scatter_empty,cax=ax_cb,extend='min')
                    
                    cb.set_label(r'nH ($10^{22}$ cm$^{-1}$)')
                        
#### Displaying arrow evolution if needed and if there are points
if display_single and display_evol_single and sum(global_mask_intime_norepeat)>1:
    
    #odering the points depending on the observation date
    date_order=datelist_obj[0][mask_intime[0]].argsort()
    
    #plotting the main line between all points
    ax_hid.plot(x_hid_base[mask_intime[0]][date_order],y_hid_base[mask_intime[0]][date_order],color='grey',linewidth=0.5)
    
    #computing the position of the arrows to superpose to the lines
    xarr_start=x_hid_base[mask_intime[0]][date_order][range(len(x_hid_base[mask_intime[0]][date_order])-1)]
    xarr_end=x_hid_base[mask_intime[0]][date_order][range(1,len(x_hid_base[mask_intime[0]][date_order]))]
    yarr_start=y_hid_base[mask_intime[0]][date_order][range(len(y_hid_base[mask_intime[0]][date_order])-1)]
    yarr_end=y_hid_base[mask_intime[0]][date_order][range(1,len(y_hid_base[mask_intime[0]][date_order]))]
    xpos=(xarr_start+xarr_end)/2
    ypos=(yarr_start+yarr_end)/2
    xdir=xarr_end-xarr_start
    ydir=yarr_end-yarr_start
    
    for X,Y,dX,dY in zip(xpos,ypos,xdir,ydir):
        ax_hid.annotate("",xytext=(X,Y),xy=(X+0.001*dX,Y+0.001*dY),arrowprops=dict(arrowstyle='->',color='grey'),size=10)
        
    
if radio_info_cmap=='Source' or display_edgesource:
    if paper_look:
        old_legend_size=mpl.rcParams['legend.fontsize']
        mpl.rcParams['legend.fontsize']=5.5 if sum(mask_obj)>30 and radio_info_cmap=='Source' else 7
        hid_legend=fig_hid.legend(loc='upper right',ncol=1,bbox_to_anchor=(1.11,0.895) if bigger_text and radio_info_cmap=='Source' \
                                  and display_obj_zerodet else (0.9,0.88))
        mpl.rcParams['legend.fontsize']=old_legend_size
    else:
        hid_legend=fig_hid.legend(loc='upper left',ncol=2,bbox_to_anchor=(0.01,0.99))
            
    #maintaining a constant marker size in the legend (but only for markers)
    for elem_legend in hid_legend.legendHandles:
        if type(elem_legend)==mpl.collections.PathCollection:
            if len(elem_legend._sizes)!=0:
                for i in range(len(elem_legend._sizes)):
                    elem_legend._sizes[i]=30 if (sum(mask_obj)>30 and radio_info_cmap=='Source') else 50

#### legends

hid_det_examples=[
    ((Line2D([0],[0],marker=marker_ul,color='white',markersize=50**(1/2),alpha=alpha_ul,linestyle='None',markeredgecolor='black',markeredgewidth=2),
      Line2D([0],[0],marker=marker_ul_top,color='white',markersize=50**(1/2),alpha=alpha_ul,linestyle='None',markeredgecolor='black',markeredgewidth=2))\
     if radio_info_cmap=='Source' else Line2D([0],[0],marker=marker_ul,color='white',markersize=50**(1/2),alpha=alpha_ul,linestyle='None',markeredgecolor='black',markeredgewidth=2))\
     if display_upper else
    (Line2D([0],[0],marker=marker_nondet,color='white',markersize=50**(1/2),linestyle='None',markeredgecolor='black',markeredgewidth=2)),
    (Line2D([0],[0],marker=marker_abs,color='white',markersize=50**(1/2),linestyle='None',markeredgecolor='black',markeredgewidth=2))]

if display_nonsign:
    hid_det_examples+=[
    (Line2D([0],[0],marker=marker_abs,color='white',markersize=50**(1/2),linestyle='None',markeredgecolor='grey',markeredgewidth=2))]
    
mpl.rcParams['legend.fontsize']=7

#marker legend
fig_hid.legend(handles=hid_det_examples,loc='center left',labels=['upper limit' if display_upper else 'non detection ','absorption line detection\n above '+(r'3$\sigma$' if slider_sign==0.997 else str(slider_sign*100)+'%')+' significance','absorption line detection below '+str(slider_sign*100)+' significance.'],title='Markers',
            bbox_to_anchor=(0.125,0.815) if bigger_text and square_mode else (0.125,0.82),handler_map = {tuple:mpl.legend_handler.HandlerTuple(None)},
            handlelength=2,handleheight=2.,columnspacing=1.)

#note: upper left anchor (0.125,0.815)
#note : upper right anchor (0.690,0.815)
#note: 0.420 0.815

#size legend

if display_upper:
    #displaying the 
    if radio_info_cmap=='Source':
        hid_size_examples=[(Line2D([0],[0],marker=marker_abs,color='black',markersize=(5*5**1.15)**(1/2),linestyle='None'),
                Line2D([0],[0],marker=marker_ul,color='None',markeredgecolor='grey',markersize=(5*5**1.15)**(1/2),linestyle='None',zorder=500),
                Line2D([0],[0],marker=marker_ul_top,color='None',markeredgecolor='grey',markersize=(5*5**1.15)**(1/2),linestyle='None',zorder=500)),
                        (Line2D([0],[0],marker=marker_abs,color='black',markersize=(5*20**1.15)**(1/2),linestyle='None'),
                Line2D([0],[0],marker=marker_ul,color='None',markeredgecolor='grey',markersize=(5*20**1.15)**(1/2),linestyle='None',zorder=500),
                Line2D([0],[0],marker=marker_ul_top,color='None',markeredgecolor='grey',markersize=(5*20**1.15)**(1/2),linestyle='None',zorder=500)),
                        (Line2D([0],[0],marker=marker_abs,color='black',markersize=(5*50**1.15)**(1/2),linestyle='None'),
                Line2D([0],[0],marker=marker_ul,color='None',markeredgecolor='grey',markersize=(5*50**1.15)**(1/2),linestyle='None',zorder=500),
                Line2D([0],[0],marker=marker_ul_top,color='None',markeredgecolor='grey',markersize=(5*50**1.15)**(1/2),linestyle='None',zorder=500))]
    else:
        hid_size_examples=[(Line2D([0],[0],marker=marker_abs,color='black',markersize=(5*5**1.15)**(1/2),linestyle='None'),
                Line2D([0],[0],marker=marker_ul,color='None',markeredgecolor='grey',markersize=(5*5**1.15)**(1/2),linestyle='None',zorder=500)),
                        (Line2D([0],[0],marker=marker_abs,color='black',markersize=(5*20**1.15)**(1/2),linestyle='None'),
                Line2D([0],[0],marker=marker_ul,color='None',markeredgecolor='grey',markersize=(5*20**1.15)**(1/2),linestyle='None',zorder=500)),
                        (Line2D([0],[0],marker=marker_abs,color='black',markersize=(5*50**1.15)**(1/2),linestyle='None'),
                Line2D([0],[0],marker=marker_ul,color='None',markeredgecolor='grey',markersize=(5*50**1.15)**(1/2),linestyle='None',zorder=500))]
else:
    hid_size_examples=[(Line2D([0],[0],marker=marker_abs,color='black',markersize=(5*5**1.15)**(1/2),linestyle='None')),
                        (Line2D([0],[0],marker=marker_abs,color='black',markersize=(5*20**1.15)**(1/2),linestyle='None')),
                        (Line2D([0],[0],marker=marker_abs,color='black',markersize=(5*50**1.15)**(1/2),linestyle='None'))]

eqw_legend=fig_hid.legend(handles=hid_size_examples,loc='center left',labels=['5 eV','20 eV','50 eV'],
                          title='Equivalent widths',
            bbox_to_anchor=(0.125,0.218) if bigger_text and square_mode else (0.125,0.218),handleheight=4, handlelength=4,facecolor='None')

if radio_info_cmap=='Instrument':
    instru_examples=np.array([Line2D([0],[0],marker=marker_abs,color='red',markeredgecolor='black',markersize=(5*5**1.15)**(1/2),linestyle='None'),
                     Line2D([0],[0],marker=marker_abs,color='blue',markeredgecolor='black',markersize=(5*5**1.15)**(1/2),linestyle='None'),
                     Line2D([0],[0],marker=marker_abs,color='green',markeredgecolor='black',markersize=(5*5**1.15)**(1/2),linestyle='None'),
                     Line2D([0],[0],marker=marker_abs,color='magenta',markeredgecolor='black',markersize=(5*5**1.15)**(1/2),linestyle='None'),
                     Line2D([0],[0],marker=marker_abs,color='orange',markeredgecolor='black',markersize=(5*5**1.15)**(1/2),linestyle='None')])
                
    instru_ind=[np.argwhere(np.array(telescope_list)==elem)[0][0] for elem in np.array(choice_telescope)]
    
    instru_legend=fig_hid.legend(handles=instru_examples[instru_ind].tolist(),loc='upper right',labels=choice_telescope,
                              title=radio_info_cmap,
                bbox_to_anchor=(0.900,0.88) if bigger_text and square_mode else (0.825,0.918),handleheight=1, handlelength=4,facecolor='None')

#note: 0.9 0.53
#destacked version
# fig_hid.legend(handles=hid_size_examples,loc='center left',labels=['5 eV','20 eV','50 eV'],title='Equivalent widths',
#             bbox_to_anchor=(0.125,0.235) if bigger_text and square_mode else (0.125,0.235),handler_map = {tuple:mpl.legend_handler.HandlerTuple(None)},handlelength=8,handleheight=5,columnspacing=5.)

# fig_hid_html = mpld3.fig_to_html(fig_hid)
# components.html(fig_hid_html, height=1000)

st.pyplot(fig_hid)

# @st.cache
# def convert_image(fig,s_format):

#     fig_save = io.BytesIO()

#     fig.savefig(fig_save, format=s_format,bbox_inches='tight')
    
#     return fig_save

# save_fig_hid= convert_image(fig_hid,s_format=save_format)

# st.download_button(
#      label="Download data as CSV",
#      data=save_fig_hid,
#      file_name='large_df.csv',
#      mime='text/csv',
#  )


'''
SOURCE TABLE
'''

if display_source_table:
    
    df_source_arr=np.array([obj_list,dist_obj_list,mass_obj_list,incl_plot.T[0],incl_plot.T[1],incl_plot.T[2]]).astype(str).T
    
    df_source=pd.DataFrame(df_source_arr,columns=['source','distance (kpc)','mass (M_sun)','inclination (°)','incl err -','incl err +'])
    
    st.dataframe(df_source)
    
    
#### Transposing the tables into plot arrays

flag_noabsline=False

#bin values for all the histograms below
#for the blueshift and energies the range is locked so we can use a global binning for all the diagrams
bins_bshift=np.linspace(-5e3,1e4,num=25,endpoint=True)
bins_ener=np.arange(6.,9.,2*line_search_e[2])

#creating restricted ploting arrays witht the current streamlit object and lines selections

abslines_plot_restrict=deepcopy(abslines_plot)
for i_info in range(len(abslines_plot_restrict)):
    for i_incer in range(len(abslines_plot_restrict[i_info])):
        abslines_plot_restrict[i_info][i_incer]=abslines_plot_restrict[i_info][i_incer][mask_lines].T[mask_obj].T
        
abslines_ener_restrict=deepcopy(abslines_ener)
for i_incer in range(len(abslines_ener_restrict)):
    abslines_ener_restrict[i_incer]=abslines_ener_restrict[i_incer][mask_lines].T[mask_obj].T

width_plot_restrict=deepcopy(width_plot)
width_plot_restrict=np.transpose(np.transpose(width_plot_restrict,(1,0,2))[mask_lines].T[mask_obj],(1,2,0))

dict_linevis['mask_obj']=mask_obj
dict_linevis['zoom_lc']=zoom_lc
dict_linevis['slider_date']=slider_date
dict_linevis['date_list']=date_list
dict_linevis['instru_list']=instru_list
dict_linevis['abslines_plot_restrict']=abslines_plot_restrict
dict_linevis['slider_sign']=slider_sign

dict_linevis['mask_lines']=mask_lines
dict_linevis['bins_bshift']=bins_bshift
dict_linevis['bins_ener']=bins_ener
dict_linevis['display_nonsign']=display_nonsign
dict_linevis['save_dir']=None
dict_linevis['save_str_prefix']= None

'''''''''''''''''''''
 ####MAXI LIGHTCURVE
'''''''''''''''''''''
if plot_lc_monit:
    
    fig_lc_monit=plot_lightcurve(dict_linevis,catal_maxi_df,catal_maxi_simbad,choice_source,display_hid_interval=monit_highlight_hid,
                                     superpose_ew=plot_maxi_ew)

    #wrapper to avoid streamlit trying to plot a None when resetting while loading
    if fig_lc_monit is not None:
        st.pyplot(fig_lc_monit)

if plot_hr_monit:
    
    fig_hr_monit=plot_lightcurve(dict_linevis,catal_maxi_df,catal_maxi_simbad,choice_source,mode='HR',display_hid_interval=monit_highlight_hid,
                                     superpose_ew=plot_maxi_ew)
    # fig_maxi_lc_html = mpld3.fig_to_html(fig_maxi_lc)
    # components.html(fig_maxi_lc_html,height=500,width=1000)

    #wrapper to avoid streamlit trying to plot a None when resetting while loading
    if plot_hr_monit is not None:
        st.pyplot(fig_hr_monit)


if (plot_lc_monit and fig_lc_monit is None) or (plot_hr_monit and fig_hr_monit is None) :
    st.text('No match in MAXI source list found.')
        
'''''''''''''''''''''
   ####DIAGRAMS
'''''''''''''''''''''

if compute_only_withdet:
    
    if sum(global_mask_intime_norepeat)==0 or sum(global_sign_mask)==0:
        if sum(global_mask_intime_norepeat)==0:
            st.text('No point left in selected dates interval. Cannot compute parameter analysis.')
        elif sum(global_sign_mask)==0:
            st.text('There are no detections for current object/date selection. Cannot compute parameter analysis.')
        st.stop()

'''
AUTOFIT LINES
'''

'''Distributions'''
    
    
def streamlit_distrib():
    distrib_eqw=distrib_graph(abslines_plot_restrict,'eqw',dict_linevis,conf_thresh=slider_sign,streamlit=True,bigger_text=bigger_text)
    distrib_bshift=distrib_graph(abslines_plot_restrict,'bshift',dict_linevis,conf_thresh=slider_sign,streamlit=True,bigger_text=bigger_text)
    distrib_ener=distrib_graph(abslines_plot_restrict,'ener',dict_linevis,abslines_ener_restrict,conf_thresh=slider_sign,streamlit=True,
                               bigger_text=bigger_text)
    if use_eqwratio:
        distrib_eqwratio=distrib_graph(abslines_plot_restrict,'eqwratio'+eqwratio_type,dict_linevis,conf_thresh=slider_sign,streamlit=True,
                                       bigger_text=bigger_text)
    if n_infos>=5 and use_lineflux:
        distrib_lineflux=distrib_graph(abslines_plot_restrict,'lineflux',dict_linevis,conf_thresh=slider_sign,streamlit=True,bigger_text=bigger_text)
        
    with st.expander('Distribution graphs'):
        
        col_list={'eqw':None,'bshift':None,'ener':None}
        
        if use_eqwratio:
            col_list['eqwratio']=None
        if n_infos>=5 and use_lineflux:
            col_list['lineflux']=None
            
        st_cols=st.columns(len(col_list))
        
        for i_col,col_name in enumerate(list(col_list.keys())):
            col_list[col_name]=st_cols[i_col]
            
        with col_list['eqw']:
            st.pyplot(distrib_eqw)

        with col_list['bshift']:
            st.pyplot(distrib_bshift)
        with col_list['ener']:
            st.pyplot(distrib_ener)
                
        if use_eqwratio:
            with col_list['eqwratio']:
                st.pyplot(distrib_eqwratio)
        
        if use_lineflux and n_infos>=5:
            with col_list['lineflux']:
                st.pyplot(distrib_lineflux)
    
'''1-1 Correlations'''

'''Intrinsic line parameters'''

def streamlit_scat(mode):
    
    scat_eqw=[]
    scat_bshift=[]
    scat_ener=[]
    scat_width=[]
    
    if mode=='eqwratio':
        scat_eqw=[correl_graph(abslines_plot_restrict,eqwratio_comp[0]+'_'+eqwratio_comp[1],abslines_ener_restrict,dict_linevis,mode='eqwratio',
                               conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                               show_ul_eqw=show_scatter_ul)]
        if use_time and use_eqwratio:
            #not actually bshift but we keep the same column names
            scat_bshift=[correl_graph(abslines_plot_restrict,'time_eqwratio'+eqwratio_type,abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,
                                        mode='observ',conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            
    if mode=='intrinsic':
        scat_eqw=[correl_graph(abslines_plot_restrict,'bshift_eqw',abslines_ener_restrict,dict_linevis,conf_thresh=slider_sign,streamlit=True,
                                compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        scat_bshift=[correl_graph(abslines_plot_restrict,'ener_eqw',abslines_ener_restrict,dict_linevis,conf_thresh=slider_sign,streamlit=True,
                               compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        scat_ener=[correl_graph(abslines_plot_restrict,'ener_bshift',abslines_ener_restrict,dict_linevis,conf_thresh=slider_sign,streamlit=True,
                                 compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        if n_infos>=5 and use_lineflux:
            scat_lineflux=[correl_graph(abslines_plot_restrict,'lineflux_bshift',abslines_ener_restrict,dict_linevis,conf_thresh=slider_sign,
                                        streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
            
        if use_time:
            scat_eqw+=[correl_graph(abslines_plot_restrict,'time_eqw',abslines_ener_restrict,dict_linevis,mode_vals=None,mode='observ',
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            scat_bshift+=[correl_graph(abslines_plot_restrict,'time_bshift',abslines_ener_restrict,dict_linevis,mode_vals=None,mode='observ',
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            scat_ener+=[correl_graph(abslines_plot_restrict,'time_ener',abslines_ener_restrict,dict_linevis,mode_vals=None,mode='observ',
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            scat_width+=[correl_graph(abslines_plot_restrict,'time_width',abslines_ener_restrict,dict_linevis,mode_vals=None,mode='observ',
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            
        if use_width:
            scat_eqw+=[correl_graph(abslines_plot_restrict,'eqw_width',abslines_ener_restrict,dict_linevis,mode_vals=None,
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            scat_bshift+=[correl_graph(abslines_plot_restrict,'bshift_width',abslines_ener_restrict,dict_linevis,mode_vals=None,
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            scat_ener+=[correl_graph(abslines_plot_restrict,'ener_width',abslines_ener_restrict,dict_linevis,mode_vals=None,
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
    elif mode=='observ':
        scat_eqw=\
        [correl_graph(abslines_plot_restrict,'eqw_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',conf_thresh=slider_sign,
                      streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,show_ul_eqw=show_scatter_ul),
         correl_graph(abslines_plot_restrict,'eqw_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',conf_thresh=slider_sign,
                      streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
        
        scat_bshift=\
            [correl_graph(abslines_plot_restrict,'bshift_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked),
             correl_graph(abslines_plot_restrict,'bshift_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
            
        scat_ener=\
        [correl_graph(abslines_plot_restrict,'ener_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                      conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked),
         correl_graph(abslines_plot_restrict,'ener_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                      conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
    
        scat_width=\
        [correl_graph(abslines_plot_restrict,'width_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                      conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked),
         correl_graph(abslines_plot_restrict,'width_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                      conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        
        if use_eqwratio:
         scat_eqwratio=\
             [correl_graph(abslines_plot_restrict,'eqwratio'+eqwratio_type+'_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                      conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                      show_ul_eqw=show_scatter_ul),
              correl_graph(abslines_plot_restrict,'eqwratio'+eqwratio_type+'_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                      conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                      show_ul_eqw=show_scatter_ul)]
        
        if n_infos>=5 and use_lineflux:
            scat_lineflux=\
            [correl_graph(abslines_plot_restrict,'lineflux_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked),
             correl_graph(abslines_plot_restrict,'lineflux_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        
    elif mode=='source':
        scat_eqw+=[correl_graph(abslines_plot_restrict,'eqw_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,mode='source',
                                conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                                show_ul_eqw=show_scatter_ul)]
        
        scat_bshift+=[correl_graph(abslines_plot_restrict,'bshift_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,mode='source',
                               conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        
        scat_ener+=[correl_graph(abslines_plot_restrict,'ener_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,mode='source',
                                 conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        scat_width+=[correl_graph(abslines_plot_restrict,'width_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,mode='source',
                                 conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        if n_infos>=5:
            scat_lineflux=[correl_graph(abslines_plot_restrict,'lineflux_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,
                                        mode='source',conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked)]
    
    with st.expander('Correlation graphs for '+mode+' parameters'):

        col_list={'eqw':None,'bshift':None,'ener':None}
        
        if use_width:
            col_list['width']=None
        
        #defining columns for each data type
        if use_eqwratio and mode=='observ':
            col_list['eqwratio']=None
        if n_infos>=5 and use_lineflux:
            col_list['lineflux']=None
        st_cols=st.columns(len(col_list))
        
        for i_col,col_name in enumerate(list(col_list.keys())):
            col_list[col_name]=st_cols[i_col]
        
        with col_list['eqw']:
            [st.pyplot(elem) for elem in scat_eqw]
                        
        if mode!='eqwratio' or use_time:

            with col_list['bshift']:
                [st.pyplot(elem) for elem in scat_bshift]

        if mode!='eqwratio':
            with col_list['ener']:
                [st.pyplot(elem) for elem in scat_ener]
            if use_eqwratio and mode=='observ':
                with col_list['eqwratio']:
                    [st.pyplot(elem) for elem in scat_eqwratio]
                    
        if use_lineflux and n_infos>=5:
            with col_list['lineflux']:
                [st.pyplot(elem) for elem in scat_lineflux]
                
        if use_width:
            with col_list['width']:
                [st.pyplot(elem) for elem in scat_width]
            
mpl.rcParams.update({'font.size': 14})

#### Streamlit Parameter analysis

with st.sidebar.expander('Parameter analysis'):
    
    st.header('Data selection')
    display_distrib=st.checkbox('Plot distributions')
    display_scat_intr=st.checkbox('Correlations between intrinsic parameters',value=False)
    display_scat_hid=st.checkbox('Correlations with observation parameters',value=False)
    display_scat_inclin=st.checkbox('Correlations with source parameters',value=False)
    display_scat_eqwcomp=st.checkbox('Plot eqw vs eqw correlations',value=False)
    if display_scat_eqwcomp:
        eqwratio_comp=st.multiselect('Lines to compare', [elem for elem in lines_std_names[3:] if 'abs' in elem],default=lines_std[3:5])
    use_eqwratio=st.checkbox('Add eqw ratios as an intrinsic parameter')
    if use_eqwratio:
        eqwratio_type=st.selectbox('Ratio to use',['FeKa26/FeKa25 (Ka)','FeKb26/FeKb25 (Kb)','FeKb25/FeKa25 (25)','FeKb26/FeKa26 (26)'])\
            .split(' ')[1].replace('(','').replace(')','')
    use_width=st.checkbox('Plot width evolution of intrinsic parameters',value=False)
    use_time=st.checkbox('Plot time evolution of intrinsic parameters',value=False)
    use_lineflux=st.checkbox('Add lineflux as an intrinsic parameter')
    compute_correl=st.checkbox('Compute Pearson/Spearman for the scatter plots',value=True)
    
    st.header('Visualisation')
    radio_color_scatter=st.radio('Scatter plot color options:',('None','Instrument','Source','Time','HR','width','nH'))
    scale_log_eqw=st.checkbox('Use a log scale for the equivalent width and line fluxes in scatters')
    scale_log_hr=st.checkbox('Use a log scale for the HID parameters',value=True)
    plot_trend=st.checkbox('Display linear trend lines in the scatter plots',value=False)
    
    st.header('Upper limits')
    show_scatter_ul=st.checkbox('Display upper limits in eqw plots',value=True if display_single else False)
    lock_lims_det=not(st.checkbox('Include upper limits in graph bounds computations',value=True))

#storing arguments to reduce the number of arguments in the scatter plot functions    
dict_linevis['scale_log_hr']=scale_log_hr
dict_linevis['scale_log_eqw']=scale_log_eqw

dict_linevis['abslines_ener']=abslines_ener
dict_linevis['abslines_plot']=abslines_plot

dict_linevis['lock_lims_det']=lock_lims_det
dict_linevis['plot_trend']=plot_trend

dict_linevis['color_scatter']=radio_color_scatter
dict_linevis['observ_list']=observ_list
dict_linevis['hid_plot_restrict']=hid_plot_restrict
dict_linevis['width_plot_restrict']=width_plot_restrict
dict_linevis['nh_plot_restrict']=nh_plot_restrict

if display_distrib:
    streamlit_distrib()
    
if display_scat_intr:
    streamlit_scat('intrinsic')

if display_scat_eqwcomp:
    streamlit_scat('eqwratio')
    
if display_scat_hid:
    streamlit_scat('observ')

if display_scat_inclin:
    streamlit_scat('source')
    
# display_general_infos=st.sidebar.checkbox('Display general informations (only with all lines)',value=False)

#number of individual line detections

#Not used right now
display_general_infos=False

if display_general_infos:
    
    #for all lines
    n_sources_restrict=mask_obj.sum()
    n_sources_withdet=(mask_obj & mask_obj_withdet).sum()
    n_sources_nodet=(mask_obj & ~mask_obj_withdet).sum()
    
    #number of individual detections for all lines
    n_detections=(ravel_ragged(global_displayed_sign[mask_obj])>0).sum()
    n_detections_sign=(ravel_ragged(global_displayed_sign[mask_obj])>slider_sign).sum()
    
    #sources with detections for individual lines
    n_sources_withdet_perline=[]
    #number of inidividual detections for individual lines
    n_detections_perline=[]
    for i in range_absline:
        n_sources_withdet_perline+=\
            [np.array([(global_displayed_sign[mask_obj].T[i][j]>0).any() for j in range(n_sources_restrict)]).sum()]
        n_detections_perline+=[[(ravel_ragged(global_displayed_sign[mask_obj].T[i])>0).sum(),
                               (ravel_ragged(global_displayed_sign[mask_obj].T[i])>slider_sign).sum()]]
        
    n_detections_perline=np.array(n_detections_perline).T
    
    st.write('number of sources in the current sample:'+str(n_sources_restrict))
    st.write('number of sources with detections in the current sample:'+str(n_sources_withdet))
    st.write('number of sources with no detection:'+str(n_sources_nodet))
    st.write('number of individual detections:'+str(n_detections))
    st.write('number of individual significant detections:'+str(n_detections_sign))
    for i in range_absline:
        st.write('number of sources with '+str(lines_std_names[i+3])+' detections:'+str(n_sources_withdet_perline[i]))
        st.write('number of individual'+str(lines_std_names[i+3])+' detections:'+str(n_detections_perline[0][i]))
        st.write('number of individual significant'+str(lines_std_names[i+3])+' detections:'+str(n_detections_perline[1][i]))
