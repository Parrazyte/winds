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

sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/general/')

from general_tools import interval_extract

import dill


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

        for i_nH, nH in enumerate(range_nH):
            for i_v_turb, v_turb in enumerate(range_v_turb):
                curr_COG = np.loadtxt(
                    os.path.join(explore_photo_dir, 'SED_' + elem_SED + '_ew_%.1f' % nH + '_' + str(v_turb) + '.dat'))

                curr_nRtwo = float(np.log10(l_ion_SED).iloc[0]) - curr_COG.T[0]

                # order is nr², 26, 25
                COG_invert_indiv[i_nH][i_v_turb] = np.array([curr_nRtwo, curr_COG.T[1], curr_COG.T[2]])

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

slider_nH=st.select_slider('nH value',range_nH)
slider_v_turb=st.select_slider(r'$v_{turb}$ value',range_v_turb)

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
for i_SED,elem_SED in enumerate(list(SEDs.keys())):

    if elem_SED not in list_SEDs_disp:
        continue
    COG_invert_SED=COG_invert[elem_SED]

    COG_invert_indiv=COG_invert_SED[np.where(range_nH==slider_nH)[0][0]][np.where(slider_v_turb==range_v_turb)[0][0]]

    #plotting the COGs
    ax_2D[0].plot(COG_invert_indiv[1]/COG_invert_indiv[2],COG_invert_indiv[0],
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

    if ew_25_vals_arr[-1]!=0 or ew_26_vals_arr[-1]!=0:
        ew_ratio_mask=[]
    else:
        ew_ratio_mask=np.argwhere((COG_invert_indiv[1]/COG_invert_indiv[2] >\
                                   (ew_26_vals_arr[0]-ew_26_vals_arr[1])/(ew_25_vals_arr[0]+ew_25_vals_arr[1])) &
                                  (COG_invert_indiv[1] / COG_invert_indiv[2] < \
                                   (ew_26_vals_arr[0]+ew_26_vals_arr[1]) / (ew_25_vals_arr[0] - ew_25_vals_arr[1]))).T[0]

    if len(ew_25_mask)!=0:
        ew_25_intervals=list(interval_extract(ew_25_mask))

        #25 outline plot
        [ax_2D[1].plot(COG_invert_indiv[2][ew_25_intervals[i_inter][0]:ew_25_intervals[i_inter][1]+1],
                  COG_invert_indiv[0][ew_25_intervals[i_inter][0]:ew_25_intervals[i_inter][1]+1],
                  color=base_cmap[i_SED],lw=5,
                  marker='s' if ew_25_intervals[i_inter][0]==ew_25_intervals[i_inter][1] else '',alpha=0.7)\
         for i_inter in range(len(ew_25_intervals))]

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
                  color=base_cmap[i_SED],lw=5,
                  marker='s' if ew_26_intervals[i_inter][0]==ew_26_intervals[i_inter][1] else '',alpha=0.7)\
         for i_inter in range(len(ew_26_intervals))]

    ax_2D[2].set_xlim(0,70)


    if len(ew_ratio_mask)!=0:

        try:
            ew_ratio_intervals=list(interval_extract(ew_ratio_mask))
        except:
            breakpoint()

        # breakpoint()


        #ratio outline plot
        [ax_2D[0].plot((COG_invert_indiv[1]/COG_invert_indiv[2])\
            [ew_ratio_intervals[i_inter][0]:ew_ratio_intervals[i_inter][1]+1],
                  COG_invert_indiv[0][ew_ratio_intervals[i_inter][0]:ew_ratio_intervals[i_inter][1]+1],
                  color=base_cmap[i_SED],lw=5,marker='s' if\
                ew_ratio_intervals[i_inter][0]==ew_ratio_intervals[i_inter][1] else '',alpha=0.7)\
         for i_inter in range(len(ew_ratio_intervals))]

    ax_2D[0].set_xscale('log')
    ax_2D[0].set_xlim(0.1,10.)
    ax_2D[0].set_ylabel(r'log$_{10}$(nR²)')

fig_2D.legend(ncol=3,bbox_to_anchor=(0.91, 1.))

st.pyplot(fig_2D)

# breakpoint()

#color the intervals of nR² where each obs may exist

#make a 3D graph of the minimum/all nR² distance between observations?
#ou alors 3D surface plot ?
