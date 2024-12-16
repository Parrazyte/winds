import os,sys
import glob

import argparse
import warnings


import numpy as np
import pandas as pd
from decimal import Decimal

import streamlit as st
#matplotlib imports

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go

sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/general/')

from general_tools import interval_extract

from scipy.interpolate import CubicSpline
from scipy.ndimage import label, find_objects

import dill

#don't know where the bug comes from tbf
try:
    st.set_page_config(layout='wide')
except:
    pass

'''
Important notes:
for now we don't consider the luminosity uncertainty (assumed to be negligible against the one for the EW
'''

range_nH = np.arange(21, 25.01, 0.5)
range_v_turb = np.array([0, 500, 1000, 2500, 5000])

online='parrama' not in os.getcwd()
if online:
    project_dir='/mount/src/winds/'
else:
    project_dir='/home/parrama/Documents/Work/PhD/Scripts/Python/'

update_dump=False

if online:
    dump_path = '/mount/src/winds/observations/visualisation/ion_visu/dump_ions.pkl'

    if not os.path.isfile(dump_path):
        print(dump_path)
        st.error('Dump file not found for this configuration')
else:
    dump_path = '/home/parrama/Documents/Work/PhD/docs/Observations/4U/photo_Stefano/dump_ions.pkl'

    update_dump = st.sidebar.button('Update dump')

if update_dump or not os.path.isfile(dump_path):

    base_photo_dir = '/home/parrama/Documents/Work/PhD/docs/Observations/4U/photo_Stefano'

    explore_photo_dir = '/home/parrama/Documents/Work/PhD/docs/Observations/4U/photo_Stefano/more_nH/4U'

    flux_3_10_csv = pd.read_csv(base_photo_dir + '/ionization_observ.txt')

    ew_25_csv = pd.read_csv(base_photo_dir + '/ionization_FeKa25.txt')

    ew_26_csv = pd.read_csv(base_photo_dir + '/ionization_FeKa26.txt')

    # assuming 8kpc of distance here
    olum_3_10 = 7.66e45 * flux_3_10_csv['Flux_3-10']

    h = 1.054571817e-34
    eV = 1.0218e-19
    nu2eV = eV / h

    SEDs = {elem.split('/')[-1].replace('SED_', '').replace('.txt', ''): np.loadtxt(elem) for elem in
            glob.glob(os.path.join(base_photo_dir, 'SED_**.txt'))}

    COG_base = {elem.split('/')[-1].replace('SED_', '').replace('_ew.dat', ''): np.loadtxt(elem) for elem in
                glob.glob(os.path.join(base_photo_dir, 'SED_**_ew.dat'))}

    lion = {}

    COG_invert = {}

    for elem_SED in list(SEDs.keys()):

        # the specific olum value for this SED
        olum_indiv = olum_3_10[flux_3_10_csv['SED'] == elem_SED]

        # isolating the region of nu within the different energy ranges
        nu_3_10_mask = (SEDs[elem_SED].T[0] >= 3e3 * nu2eV) & (SEDs[elem_SED].T[0] <= 10e3 * nu2eV)
        nu_ion_mask = (SEDs[elem_SED].T[0] >= 13.6 * nu2eV) & (SEDs[elem_SED].T[0] <= 13.6e3 * nu2eV)

        # masking the SEDs according to the different bands
        indiv_SED_masked_3_10 = SEDs[elem_SED][nu_3_10_mask]
        indiv_SED_masked_ion = SEDs[elem_SED][nu_ion_mask]

        # integrating to get the different luminosities
        l_3_10_SED = np.trapz(indiv_SED_masked_3_10.T[2], indiv_SED_masked_3_10.T[0])
        l_ion_SED = np.trapz(indiv_SED_masked_ion.T[2], indiv_SED_masked_ion.T[0]) / l_3_10_SED * olum_indiv

        lion[elem_SED] = float(l_ion_SED.iloc[0])

        # this one will also have the modified inverted y axis
        COG_invert_indiv = np.array([[None] * len(range_v_turb)] * len(range_nH))

        for i_nh, nH in enumerate(range_nH):
            for i_v_turb, v_turb in enumerate(range_v_turb):
                curr_COG = np.loadtxt(
                    os.path.join(explore_photo_dir, 'SED_' + elem_SED + '_ew_%.1f' % nH + '_' + str(v_turb) + '.dat'))

                curr_nRtwo = float(np.log10(l_ion_SED).iloc[0]) - curr_COG.T[0]

                # order is nr², 26, 25
                COG_invert_indiv[i_nh][i_v_turb] = np.array([curr_nRtwo, curr_COG.T[1], curr_COG.T[2]])

        COG_invert[elem_SED] = COG_invert_indiv

    with st.spinner(text='Updating dump file...' if update_dump else \
            'Creating dump file...'):

        dump_arr =np.array([flux_3_10_csv,
                            ew_25_csv,
                            ew_26_csv,
                            olum_3_10,
                            SEDs,
                            COG_base,
                            COG_invert],dtype=object)
        with open(dump_path, 'wb+') as dump_file:
            dill.dump(dump_arr, file=dump_file)

with open(dump_path, 'rb') as dump_file:
    dump_arr = dill.load(dump_file)

flux_3_10_csv=dump_arr[0]
ew_25_csv=dump_arr[1]
ew_26_csv=dump_arr[2]
olum_3_10=dump_arr[3]
SEDs=dump_arr[4]
COG_base=dump_arr[5]
COG_invert=dump_arr[6]

slider_nH=st.select_slider('nH value',range_nH,value=23.0)
slider_v_turb=st.select_slider(r'$v_{turb}$ value',range_v_turb,value=1000)

fig_2D,ax_2D = plt.subplots(1,3, figsize=(10,8),sharey=True)
fig_2D.subplots_adjust(wspace=0)

plt.suptitle(r'Curve of Growths for log(nH/cm²)='+str(slider_nH)+' and $v_{turb}$='+str(slider_v_turb)+' km/s',
             position=[0.5,0.05])

base_cmap=plt.rcParams['axes.prop_cycle'].by_key()['color']

ax_2D[0].set_ylim(33,38)

radio_single = st.sidebar.radio('Display options:', ('Multiple Objects', 'Single Object'))

if radio_single == 'Single Object':
    list_SEDs_disp=[st.sidebar.selectbox('SED to display',options=list(SEDs.keys()))]


if radio_single == 'Multiple Objects':

    list_SEDs_disp=st.sidebar.multiselect(label='SEDs to display',options=list(SEDs.keys()),default=list(SEDs.keys()))

interpolate=st.sidebar.toggle(label='interpolate',value=True)

highlight_EW_vert=st.sidebar.toggle('highlight EW range')

highlight_valid_range=st.sidebar.toggle('highlight valid nR² range',value=True)


if interpolate:
    COG_invert_interp={}

    for elem_SED in list(COG_invert.keys()):

        arr_shape=np.shape(COG_invert[elem_SED])

        interp_arr=np.repeat(None,arr_shape[0]*arr_shape[1])\
                          .reshape(arr_shape)

        for i_nh in np.arange(arr_shape[0]):
            for i_turb in np.arange(arr_shape[1]):
                interp_x=np.arange(33,40,0.001)

                #need to invert to keep increasing X axis
                interp_f_25=CubicSpline(COG_invert[elem_SED][i_nh][i_turb][0][::-1],COG_invert[elem_SED][i_nh][i_turb][2][::-1])
                interp_f_26=CubicSpline(COG_invert[elem_SED][i_nh][i_turb][0][::-1],COG_invert[elem_SED][i_nh][i_turb][1][::-1])

                interp_arr[i_nh][i_turb]=np.array([interp_x,interp_f_26(interp_x),interp_f_25(interp_x)])

        COG_invert_interp[elem_SED]=interp_arr

    COG_invert_use=COG_invert_interp
else:
    COG_invert_use=COG_invert

for i_SED,elem_SED in enumerate(list(SEDs.keys())):

    if elem_SED not in list_SEDs_disp:
        continue

    COG_invert_SED=COG_invert_use[elem_SED]

    COG_invert_indiv=COG_invert_SED[np.where(range_nH==slider_nH)[0][0]][np.where(slider_v_turb==range_v_turb)[0][0]]

    #plotting the COGs

    #restricting the display to the part between 34 and 37 to avoid issues due to interpolation
    mask_noissue=(COG_invert_indiv[0]>=34.) & (COG_invert_indiv[0]<=37.)
    ax_2D[0].plot((COG_invert_indiv[1]/COG_invert_indiv[2])[mask_noissue],
                  COG_invert_indiv[0][mask_noissue],
                  label=elem_SED,color=base_cmap[i_SED])

    ax_2D[1].plot(COG_invert_indiv[2],COG_invert_indiv[0],color=base_cmap[i_SED])
    ax_2D[2].plot(COG_invert_indiv[1],COG_invert_indiv[0],color=base_cmap[i_SED])

    try:
        ew_25_vals=ew_25_csv[flux_3_10_csv['SED'] == elem_SED]
    except:
        breakpoint()

    ew_25_vals_arr=np.array(ew_25_vals)[0][1:]

    ew_26_vals=ew_26_csv[flux_3_10_csv['SED'] == elem_SED]
    ew_26_vals_arr=np.array(ew_26_vals)[0][1:]

    if ew_25_vals_arr[0]!=0:
        ew_ratio_vals=np.array([ew_26_vals_arr[0]/ew_25_vals_arr[0],
                                ew_26_vals_arr[0]/ew_25_vals_arr[0]-(ew_26_vals_arr[0] - ew_26_vals_arr[1]) / (ew_25_vals_arr[0] + ew_25_vals_arr[1]),
                                (ew_26_vals_arr[0] + ew_26_vals_arr[1]) / (ew_25_vals_arr[0] - ew_25_vals_arr[1])\
                                -ew_26_vals_arr[0]/ew_25_vals_arr[0]])
    else:
        ew_ratio_vals=np.repeat(0,3)

    if highlight_EW_vert:
        #filling according to the EW value
        ax_2D[1].axvspan(ew_25_vals_arr[0]-ew_25_vals_arr[1],ew_25_vals_arr[0]+ew_25_vals_arr[2],
                               color=base_cmap[i_SED],alpha=0.3)
        ax_2D[2].axvspan(ew_26_vals_arr[0]-ew_26_vals_arr[1],ew_26_vals_arr[0]+ew_26_vals_arr[2],
                               color=base_cmap[i_SED],alpha=0.3)

        ax_2D[0].axvspan(ew_ratio_vals[0]-ew_ratio_vals[1],ew_ratio_vals[0]+ew_ratio_vals[2],
                               color=base_cmap[i_SED],alpha=0.3)

    ew_25_mask=np.argwhere((COG_invert_indiv[2] >ew_25_vals_arr[0]-ew_25_vals_arr[1]) &
                           (COG_invert_indiv[2] <ew_25_vals_arr[0]+ew_25_vals_arr[2])).T[0]

    ew_26_mask=np.argwhere((COG_invert_indiv[1] >ew_26_vals_arr[0]-ew_26_vals_arr[1]) &
                           (COG_invert_indiv[1] <ew_26_vals_arr[0]+ew_26_vals_arr[2])).T[0]

    if ew_25_vals_arr[-1]!=0.:
        ew_25_mask_ul=np.argwhere(COG_invert_indiv[2] <ew_25_vals_arr[-1]).T[0]
    else:
        ew_25_mask_ul=[]

    if ew_26_vals_arr[-1]!=0.:
        ew_26_mask_ul=np.argwhere(COG_invert_indiv[2] <ew_26_vals_arr[-1]).T[0]
    else:
        ew_26_mask_ul=[]

    if ew_25_vals_arr[-1]!=0 or ew_26_vals_arr[-1]!=0:
        ew_ratio_mask=[]
    else:
        ew_ratio_mask=np.argwhere((COG_invert_indiv[1]/COG_invert_indiv[2] >\
                                   (ew_26_vals_arr[0]-ew_26_vals_arr[1])/(ew_25_vals_arr[0]+ew_25_vals_arr[1])) &
                                  (COG_invert_indiv[1] / COG_invert_indiv[2] < \
                                   (ew_26_vals_arr[0]+ew_26_vals_arr[1]) / (ew_25_vals_arr[0] - ew_25_vals_arr[1]))).T[0]

    if np.any(ew_25_mask_ul) or np.any(ew_26_mask_ul):

        if np.any(ew_25_mask_ul) and np.any(ew_26_mask_ul):
            ew_ratio_mask_l = (ew_25_mask_ul) | (ew_26_mask_ul)
        elif np.any(ew_25_mask_ul):
            ew_ratio_mask_l=ew_25_mask_ul
        else:
            ew_ratio_mask_l=ew_26_mask_ul
    else:
        ew_ratio_mask_l = []

    if len(ew_25_mask)!=0:
        ew_25_intervals=list(interval_extract(ew_25_mask))

        #25 outline plot
        [ax_2D[1].plot(COG_invert_indiv[2][ew_25_intervals[i_inter][0]:ew_25_intervals[i_inter][1]+1],
                  COG_invert_indiv[0][ew_25_intervals[i_inter][0]:ew_25_intervals[i_inter][1]+1],
                  color=base_cmap[i_SED],lw=5,alpha=0.7,zorder=100)\
         for i_inter in range(len(ew_25_intervals))]

    if len(ew_25_mask_ul)!=0:

        ew_25_intervals_ul=list(interval_extract(ew_25_mask_ul))

        #25 outline plot
        [ax_2D[1].plot(COG_invert_indiv[2][ew_25_intervals_ul[i_inter][0]:ew_25_intervals_ul[i_inter][1]+1],
                  COG_invert_indiv[0][ew_25_intervals_ul[i_inter][0]:ew_25_intervals_ul[i_inter][1]+1],
                  color=base_cmap[i_SED],lw=5,alpha=0.7,zorder=100,ls=':')\
         for i_inter in range(len(ew_25_intervals_ul))]

    # if elem_SED=="diagonal_lower_low_highE_flux":
    #     breakpoint()

    ax_2D[1].set_xlim(0,70)


    if len(ew_26_mask)!=0:

        try:
            ew_26_intervals=list(interval_extract(ew_26_mask))
        except:
            breakpoint()

        #26 outline plot
        [ax_2D[2].plot(COG_invert_indiv[1][ew_26_intervals[i_inter][0]:ew_26_intervals[i_inter][1]+1],
                  COG_invert_indiv[0][ew_26_intervals[i_inter][0]:ew_26_intervals[i_inter][1]+1],
                  color=base_cmap[i_SED],lw=5,alpha=0.7,zorder=100)\
         for i_inter in range(len(ew_26_intervals))]

    if len(ew_26_mask_ul)!=0:

        ew_26_intervals_ul=list(interval_extract(ew_26_mask_ul))

        #25 outline plot
        [ax_2D[1].plot(COG_invert_indiv[1][ew_26_intervals_ul[i_inter][0]:ew_26_intervals_ul[i_inter][1]+1],
                  COG_invert_indiv[0][ew_26_intervals_ul[i_inter][0]:ew_26_intervals_ul[i_inter][1]+1],
                  color=base_cmap[i_SED],lw=5,alpha=0.7,zorder=100,ls=':')\
         for i_inter in range(len(ew_26_intervals_ul))]
        
    ax_2D[2].set_xlim(0,70)


    if len(ew_ratio_mask)!=0:

        try:
            ew_ratio_intervals=list(interval_extract(ew_ratio_mask))
        except:
            breakpoint()

        # breakpoint()


        #preventing showing the ratios that are due to interpolation issues
        for i_inter in range(len(ew_ratio_intervals)):

            if (COG_invert_indiv[0][ew_ratio_intervals[i_inter][0]:ew_ratio_intervals[i_inter][1]+1] >=37.).all() or \
                (COG_invert_indiv[0][ew_ratio_intervals[i_inter][0]:ew_ratio_intervals[i_inter][1] + 1] <= 34.).all():
                continue

            #ratio outline plot
            ax_2D[0].plot((COG_invert_indiv[1]/COG_invert_indiv[2])\
                [ew_ratio_intervals[i_inter][0]:ew_ratio_intervals[i_inter][1]+1],
                      COG_invert_indiv[0][ew_ratio_intervals[i_inter][0]:ew_ratio_intervals[i_inter][1]+1],
                      color=base_cmap[i_SED],lw=5,alpha=0.7,zorder=100)

    if len(ew_ratio_mask_l)!=0:

        ew_ratio_intervals_l=list(interval_extract(ew_ratio_mask_l))

        # preventing showing the ratios that are due to interpolation issues
        for i_inter in range(len(ew_ratio_intervals_l)):

            # if "upper_high_highE" in elem_SED:
            #     breakpoint()
            if (COG_invert_indiv[0][ew_ratio_intervals_l[i_inter][0]:ew_ratio_intervals_l[i_inter][1] + 1] >= 37.).all() or \
                    (COG_invert_indiv[0][
                     ew_ratio_intervals_l[i_inter][0]:ew_ratio_intervals_l[i_inter][1] + 1] <= 34.).all():
                continue

            mask_bounds=(COG_invert_indiv[0][ew_ratio_intervals_l[i_inter][0]:ew_ratio_intervals_l[i_inter][1] + 1] <= 37.)\
                        & (COG_invert_indiv[0][
                     ew_ratio_intervals_l[i_inter][0]:ew_ratio_intervals_l[i_inter][1] + 1] >= 34.)

            # ratio outline plot
            ax_2D[0].plot((COG_invert_indiv[1] / COG_invert_indiv[2]) \
                              [ew_ratio_intervals_l[i_inter][0]:ew_ratio_intervals_l[i_inter][1] + 1][mask_bounds],
                          COG_invert_indiv[0][ew_ratio_intervals_l[i_inter][0]:ew_ratio_intervals_l[i_inter][1] + 1][mask_bounds],
                          color=base_cmap[i_SED], lw=5, alpha=0.7, zorder=100,ls=':')

    #displaying the valid range of parameters by combining all the restrictions on EW and EW ratio
    if highlight_valid_range:

        valid_range_par=[elem for elem in np.arange(len(COG_invert_indiv[0])) if
                                                    (elem in ew_25_mask or elem in ew_25_mask_ul)\
                                                    and (elem in ew_26_mask or elem in ew_26_mask_ul)
                                                    and (elem in ew_ratio_mask or elem in ew_ratio_mask_l)]

        if len(valid_range_par)!=0:
            valid_range_intervals = list(interval_extract(valid_range_par))
            for i_inter in range(len(valid_range_intervals)):
                ax_2D[0].axhspan(COG_invert_indiv[0][valid_range_intervals[i_inter][0]],
                                 COG_invert_indiv[0][valid_range_intervals[i_inter][1]],
                             color=base_cmap[i_SED], alpha=0.3)
                ax_2D[1].axhspan(COG_invert_indiv[0][valid_range_intervals[i_inter][0]],
                                 COG_invert_indiv[0][valid_range_intervals[i_inter][1]],
                             color=base_cmap[i_SED], alpha=0.3)
                ax_2D[2].axhspan(COG_invert_indiv[0][valid_range_intervals[i_inter][0]],
                                 COG_invert_indiv[0][valid_range_intervals[i_inter][1]],
                             color=base_cmap[i_SED], alpha=0.3)
    ax_2D[0].set_xscale('log')
    ax_2D[0].set_xlim(0.1,10.)
    ax_2D[0].set_ylabel(r'log$_{10}$(nR²)')


#plotting a horizontal background grid to help visualisation
[ax_2D[0].plot([1e-2,1e2],[nr_val,nr_val],color='grey',lw=0.5,alpha=0.3) for nr_val in np.arange(33.1,38,0.1)]
[ax_2D[1].plot([0,80],[nr_val,nr_val],color='grey',lw=0.5,alpha=0.3) for nr_val in np.arange(33.1,38,0.1)]
[ax_2D[2].plot([0,80],[nr_val,nr_val],color='grey',lw=0.5,alpha=0.3) for nr_val in np.arange(33.1,38,0.1)]

fig_2D.legend(ncol=3,bbox_to_anchor=(0.91, 1.))


tab_2D,tab_3D=st.tabs(['2D Curve of Growths','nR² evolution'])

with tab_2D:

    st.pyplot(fig_2D)

#computing the valid range of parameters for all the different obs and all parameter combinations
valid_par_range_dict={}
valid_volumes_dict={}

for elem_SED in list(SEDs.keys()):

    valid_par_arr=np.repeat(-1.,2*len(range_nH)*len(range_v_turb)).reshape(2,len(range_nH),len(range_v_turb))
    valid_par_arr_3D=np.repeat(None,2*len(range_nH)*len(range_v_turb)).reshape(2,len(range_nH),len(range_v_turb))

    for i_nh in range(len(range_nH)):
        for i_v_turb in range(len(range_v_turb)):


            COG_invert_SED = COG_invert_use[elem_SED]

            COG_invert_indiv = COG_invert_SED[i_nh][i_v_turb]

            ew_25_vals = ew_25_csv[flux_3_10_csv['SED'] == elem_SED]
            ew_25_vals_arr = np.array(ew_25_vals)[0][1:]

            ew_26_vals = ew_26_csv[flux_3_10_csv['SED'] == elem_SED]
            ew_26_vals_arr = np.array(ew_26_vals)[0][1:]

            if ew_25_vals_arr[0] != 0:
                ew_ratio_vals = np.array([ew_26_vals_arr[0] / ew_25_vals_arr[0],
                                          ew_26_vals_arr[0] / ew_25_vals_arr[0] - (
                                                      ew_26_vals_arr[0] - ew_26_vals_arr[1]) / (
                                                      ew_25_vals_arr[0] + ew_25_vals_arr[1]),
                                          (ew_26_vals_arr[0] + ew_26_vals_arr[1]) / (
                                                      ew_25_vals_arr[0] - ew_25_vals_arr[1]) \
                                          - ew_26_vals_arr[0] / ew_25_vals_arr[0]])
            else:
                ew_ratio_vals = np.repeat(0, 3)

            ew_25_mask = np.argwhere((COG_invert_indiv[2] > ew_25_vals_arr[0] - ew_25_vals_arr[1]) &
                                     (COG_invert_indiv[2] < ew_25_vals_arr[0] + ew_25_vals_arr[2])).T[0]

            ew_26_mask = np.argwhere((COG_invert_indiv[1] > ew_26_vals_arr[0] - ew_26_vals_arr[1]) &
                                     (COG_invert_indiv[1] < ew_26_vals_arr[0] + ew_26_vals_arr[2])).T[0]

            if ew_25_vals_arr[-1] != 0.:
                ew_25_mask_ul = np.argwhere(COG_invert_indiv[2] < ew_25_vals_arr[-1]).T[0]
            else:
                ew_25_mask_ul = []

            if ew_26_vals_arr[-1] != 0.:
                ew_26_mask_ul = np.argwhere(COG_invert_indiv[2] < ew_26_vals_arr[-1]).T[0]
            else:
                ew_26_mask_ul = []

            if ew_25_vals_arr[-1] != 0 or ew_26_vals_arr[-1] != 0:
                ew_ratio_mask = []
            else:
                ew_ratio_mask = np.argwhere((COG_invert_indiv[1] / COG_invert_indiv[2] > \
                                             (ew_26_vals_arr[0] - ew_26_vals_arr[1]) / (
                                                         ew_25_vals_arr[0] + ew_25_vals_arr[1])) &
                                            (COG_invert_indiv[1] / COG_invert_indiv[2] < \
                                             (ew_26_vals_arr[0] + ew_26_vals_arr[1]) / (
                                                         ew_25_vals_arr[0] - ew_25_vals_arr[1]))).T[0]

            if np.any(ew_25_mask_ul) or np.any(ew_26_mask_ul):

                if np.any(ew_25_mask_ul) and np.any(ew_26_mask_ul):
                    ew_ratio_mask_l = (ew_25_mask_ul) | (ew_26_mask_ul)
                elif np.any(ew_25_mask_ul):
                    ew_ratio_mask_l = ew_25_mask_ul
                else:
                    ew_ratio_mask_l = ew_26_mask_ul
            else:
                ew_ratio_mask_l = []


            valid_range_par = [elem for elem in np.arange(len(COG_invert_indiv[0])) if
                               (elem in ew_25_mask or elem in ew_25_mask_ul) \
                               and (elem in ew_26_mask or elem in ew_26_mask_ul)
                               and (elem in ew_ratio_mask or elem in ew_ratio_mask_l)]

            if len(valid_range_par)!=0:
                valid_par_arr[0][i_nh][i_v_turb]=COG_invert_indiv[0][valid_range_par[0]]
                valid_par_arr[1][i_nh][i_v_turb] = COG_invert_indiv[0][valid_range_par[-1]]

                valid_par_arr_3D[0][i_nh][i_v_turb]=np.array([range_nH[i_nh],range_v_turb[i_v_turb],
                                                              COG_invert_indiv[0][valid_range_par[0]]]).astype(object)
                valid_par_arr_3D[1][i_nh][i_v_turb]=np.array([range_nH[i_nh],range_v_turb[i_v_turb],
                                                              COG_invert_indiv[0][valid_range_par[-1]]]).astype(object)
            else:
                valid_par_arr_3D[0][i_nh][i_v_turb]=np.repeat(-1,3).astype(object)
                valid_par_arr_3D[1][i_nh][i_v_turb]=np.repeat(-1,3).astype(object)

                # breakpoint()

    valid_par_range_dict[elem_SED]=valid_par_arr
    #identifying individual volumes in the final array (chatgpt made this)

    # Label connected components
    structure = np.ones((3, 3, 3))  # Define connectivity (26-connected here)
    labeled_array, num_features = label(valid_par_arr > 0, structure=structure)

    # Find slices for each volume
    slices = find_objects(labeled_array)
    volume_list=[]
    volume_3D_list=[]
    for i, sl in enumerate(slices):
        volume_list += [valid_par_arr[sl] * (labeled_array[sl] == (i + 1))]

        try:
            #protected by a fourth dimension that is kept as an object to avoid issues with the additional dimension
            high_d_volume = valid_par_arr_3D[sl] * (labeled_array[sl] == (i + 1))
        except:
            breakpoint()

        #once the array is created, we can put it back to floats to transpose it
        for i in range(len(high_d_volume)):
            for j in range(len(high_d_volume[i])):
                for k in range(len(high_d_volume[i][j])):
                    high_d_volume[i][j][k]=high_d_volume[i][j][k].astype(float)
        high_d_volume=np.array([[[subsubelem for subsubelem in subelem] for subelem in elem] for elem in high_d_volume])

        #and now we can set it up into a x y z array type
        high_d_volume_clean= np.array([elem.ravel() for elem in high_d_volume.transpose(3,0,1,2)])

        #removing the zeros that sometimes get there if the definition of the volume needs additional space
        #in the 3D range
        high_d_volume_clean=np.array([elem for elem in high_d_volume_clean.T if not np.all(elem==0.)])

        volume_3D_list+=[high_d_volume_clean]

    valid_volumes_dict[elem_SED]=volume_3D_list

breakpoint()
#now just gotta test how to plot the volumes
# with tab_3D:
#
#
#     elem_SED:go.volume()}
#     fig = go.Figure(data=[
#         go.Surface(z=z1),
#         go.Surface(z=z2, showscale=False, opacity=0.9),
#         go.Surface(z=z3, showscale=False, opacity=0.9)
#
#     ])
#
#     st.plotly_chart(fig)




# breakpoint()

#color the intervals of nR² where each obs may exist

#make a 3D graph of the minimum/all nR² distance between observations?
#ou alors 3D surface plot ?
