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

online='parrama' not in os.getcwd()
if online:
    project_dir='/mount/src/winds/'
else:
    project_dir='/home/parrama/Documents/Work/PhD/Scripts/Python/'

sys.path.append(project_dir+'/general/')

from general_tools import interval_extract

from scipy.interpolate import CubicSpline
from scipy.ndimage import label, find_objects
from scipy.spatial import Delaunay

import plotly.graph_objects as go
from webcolors import name_to_rgb
from scipy.spatial import ConvexHull

import dill

#don't know where the bug comes from tbf
try:
    st.set_page_config(layout='wide')
except:
    pass


#Important notes:
#for now we don't consider the luminosity uncertainty (assumed to be negligible against the one for the EW

#previous version
# range_nH = np.arange(21, 25.01, 0.5)
# range_v_turb = np.array([0, 500, 1000, 2500, 5000])

#better sampling (with rounding to avoid precision errors that will display in the values
range_nH=np.arange(21.5,25.01,0.05).round(3)
range_v_turb=np.arange(0.5,4.01,0.05).round(3)
interp_x = np.arange(33, 38, 0.01)

update_dump=False

if online:
    dump_path = project_dir+'observations/visualisation/ion_visu/dump_ions.pkl'

    if not os.path.isfile(dump_path):
        print(dump_path)
        st.error('Dump file not found for this configuration')
else:
    dump_path = '/home/parrama/Documents/Work/PhD/docs/Observations/4U/photo_Stefano/dump_ions.pkl'

    update_dump = st.sidebar.button('Update dump')

if not online:
    update_online = st.sidebar.button('Update online version')
else:
    update_online = False

if update_online:

    # updating script
    path_online = __file__.replace('.py', '_online.py')
    os.system('cp ' + __file__ + ' ' + path_online)

    # updating dumps to one level above the script
    os.system('cp '+dump_path+' '+ path_online[:path_online.rfind('/')] + '/ion_visu/')


if update_dump or not os.path.isfile(dump_path):

    base_photo_dir = '/home/parrama/Documents/Work/PhD/docs/Observations/4U/photo_Stefano'

    explore_photo_dir = '/home/parrama/Documents/Work/PhD/docs/Observations/4U/photo_Stefano/better_sampling/4U'

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

    loading_bar = st.progress(0, text='loading dumps')

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
                    os.path.join(explore_photo_dir, 'SED_' + elem_SED + '_ew_%.2f' % nH + '_%.2f'  % v_turb + '.dat'))

                curr_nRtwo = float(np.log10(l_ion_SED).iloc[0]) - curr_COG.T[0]

                # order is nr², 26, 25
                COG_invert_indiv[i_nh][i_v_turb] = np.array([curr_nRtwo, curr_COG.T[1], curr_COG.T[2]])

                loading_bar.progress(1 / (len(SEDs)*len(range_nH)*len(range_v_turb)))


        COG_invert[elem_SED] = COG_invert_indiv

    loading_bar.empty()

    #interpolating makes both the sampling and our lives easier, which also means we don' need to store the nR² range
    COG_invert_interp = {}

    for elem_SED in list(COG_invert.keys()):

        arr_shape = np.shape(COG_invert[elem_SED])

        interp_arr = np.repeat(None, arr_shape[0] * arr_shape[1]) \
            .reshape(arr_shape)

        for i_nh in np.arange(arr_shape[0]):
            for i_turb in np.arange(arr_shape[1]):

                #lets try a lower resolution
                # interp_x = np.arange(33, 40, 0.001)

                # need to invert to keep increasing X axis
                interp_f_25 = CubicSpline(COG_invert[elem_SED][i_nh][i_turb][0][::-1],
                                          COG_invert[elem_SED][i_nh][i_turb][2][::-1])
                interp_f_26 = CubicSpline(COG_invert[elem_SED][i_nh][i_turb][0][::-1],
                                          COG_invert[elem_SED][i_nh][i_turb][1][::-1])

                interp_arr[i_nh][i_turb] = np.array([interp_f_26(interp_x), interp_f_25(interp_x)]).astype(np.float16)

        COG_invert_interp[elem_SED] = np.array([[subelem for subelem in elem] for elem in interp_arr])

    COG_invert_use = COG_invert_interp

    # computing the valid range of parameters for all the different obs and all parameter combinations
    valid_par_range_dict = {}
    valid_volumes_dict = {}

    def find_volumes():

        volume_search_bar = st.progress(0, text='Decomposing volumes')

        for elem_SED in list(SEDs.keys()):

            valid_par_arr = np.repeat(-1., 2 * len(range_nH) * len(range_v_turb)).reshape(2, len(range_nH),
                                                                                          len(range_v_turb))
            valid_par_arr_3D = np.repeat(None, 2 * len(range_nH) * len(range_v_turb)).reshape(2, len(range_nH),
                                                                                              len(range_v_turb))

            param_space_bar = st.progress(0, text='Finding parameter space')

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

                    ew_25_mask = np.argwhere((COG_invert_indiv[1] > ew_25_vals_arr[0] - ew_25_vals_arr[1]) &
                                             (COG_invert_indiv[1] < ew_25_vals_arr[0] + ew_25_vals_arr[2])).T[0]

                    ew_26_mask = np.argwhere((COG_invert_indiv[0] > ew_26_vals_arr[0] - ew_26_vals_arr[1]) &
                                             (COG_invert_indiv[0] < ew_26_vals_arr[0] + ew_26_vals_arr[2])).T[0]

                    if ew_25_vals_arr[-1] != 0.:
                        ew_25_mask_ul = np.argwhere(COG_invert_indiv[1] < ew_25_vals_arr[-1]).T[0]
                    else:
                        ew_25_mask_ul = []

                    if ew_26_vals_arr[-1] != 0.:
                        ew_26_mask_ul = np.argwhere(COG_invert_indiv[1] < ew_26_vals_arr[-1]).T[0]
                    else:
                        ew_26_mask_ul = []

                    if ew_25_vals_arr[-1] != 0 or ew_26_vals_arr[-1] != 0:
                        ew_ratio_mask = []
                    else:
                        ew_ratio_mask = np.argwhere((COG_invert_indiv[0] / COG_invert_indiv[1] > \
                                                     (ew_26_vals_arr[0] - ew_26_vals_arr[1]) / (
                                                             ew_25_vals_arr[0] + ew_25_vals_arr[1])) &
                                                    (COG_invert_indiv[0] / COG_invert_indiv[1] < \
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

                    valid_range_par = [elem for elem in np.arange(len(interp_x)) if
                                       (elem in ew_25_mask or elem in ew_25_mask_ul) \
                                       and (elem in ew_26_mask or elem in ew_26_mask_ul)
                                       and (elem in ew_ratio_mask or elem in ew_ratio_mask_l)]

                    if len(valid_range_par) != 0:
                        valid_par_arr[0][i_nh][i_v_turb] = interp_x[valid_range_par[0]]
                        valid_par_arr[1][i_nh][i_v_turb] = interp_x[valid_range_par[-1]]

                        valid_par_arr_3D[0][i_nh][i_v_turb] = np.array([range_nH[i_nh], 10 ** (range_v_turb[i_v_turb]),
                                                                        interp_x[
                                                                            valid_range_par[0]]]).astype(object)
                        valid_par_arr_3D[1][i_nh][i_v_turb] = np.array([range_nH[i_nh], 10 ** (range_v_turb[i_v_turb]),
                                                                        interp_x[
                                                                            valid_range_par[-1]]]).astype(object)
                    else:
                        valid_par_arr_3D[0][i_nh][i_v_turb] = np.repeat(-1, 3).astype(object)
                        valid_par_arr_3D[1][i_nh][i_v_turb] = np.repeat(-1, 3).astype(object)

                    param_space_bar.progress(1 / (len(range_nH) * len(range_v_turb)))
                    print((i_nh, i_v_turb))

            valid_par_range_dict[elem_SED] = valid_par_arr
            # identifying individual volumes in the final array (chatgpt made this)

            # Label connected components
            structure = np.ones((3, 3, 3))  # Define connectivity (26-connected here)
            labeled_array, num_features = label(valid_par_arr > 0, structure=structure)

            # Find slices for each volume
            slices = find_objects(labeled_array)
            volume_list = []
            volume_3D_list = []
            for i, sl in enumerate(slices):
                volume_list += [valid_par_arr[sl] * (labeled_array[sl] == (i + 1))]

                # protected by a fourth dimension that is kept as an object to avoid issues with the additional dimension
                high_d_volume = valid_par_arr_3D[sl] * (labeled_array[sl] == (i + 1))

                # once the array is created, we can put it back to floats to transpose it
                for i in range(len(high_d_volume)):
                    for j in range(len(high_d_volume[i])):
                        for k in range(len(high_d_volume[i][j])):
                            high_d_volume[i][j][k] = high_d_volume[i][j][k].astype(float)

                high_d_volume = np.array(
                    [[[subsubelem for subsubelem in subelem] for subelem in elem] for elem in high_d_volume])

                # #flattening the array to get only one dimension
                high_d_volume = np.array(
                    [elem.flatten().reshape(len(elem.flatten()) // 3, 3) for elem in high_d_volume])
                #
                # #removing the empty points
                high_d_volume = np.array(
                    [[subelem for subelem in elem if not np.all(subelem == 0.)] for elem in high_d_volume])

                # # #and now we can set it up into a x y z array type
                # high_d_volume_clean= np.array([elem.ravel() for elem in high_d_volume.transpose(3,0,1,2)])
                #
                # #removing the zeros that sometimes get there if the definition of the volume needs additional space
                # #in the 3D range
                # high_d_volume_clean=np.array([elem for elem in high_d_volume_clean.T if not np.all(elem==0.)]).T

                volume_3D_list += [high_d_volume]

                volume_search_bar.progress(1 / (len(SEDs) * len(slices)), text='Decomposing volumes')
                print('One more slice')

            valid_volumes_dict[elem_SED] = volume_3D_list

            volume_search_bar.progress(1 / len(SEDs), text='Decomposing volumes')

        return valid_volumes_dict

    valid_volumes_dict = find_volumes()

    with st.spinner(text='Updating dump file...' if update_dump else \
            'Creating dump file...'):

        dump_arr =np.array([flux_3_10_csv,
                            ew_25_csv,
                            ew_26_csv,
                            olum_3_10,
                            SEDs,
                            COG_base,
                            COG_invert_use,
                            valid_par_range_dict,
                            valid_volumes_dict],dtype=object)

        with open(dump_path, 'wb+') as dump_file:
            dill.dump(dump_arr, file=dump_file)

@st.cache_data
def load_dumps(arg=1):
    with open(dump_path, 'rb') as dump_file:
        dump_arr = dill.load(dump_file)

        dump_file.close()

    return dump_arr

dump_arr=load_dumps(arg=1)


flux_3_10_csv=dump_arr[0]
ew_25_csv=dump_arr[1]
ew_26_csv=dump_arr[2]
olum_3_10=dump_arr[3]
SEDs=dump_arr[4]
COG_base=dump_arr[5]
COG_invert_use=dump_arr[6]
valid_par_range_dict=dump_arr[7]
valid_volumes_dict=dump_arr[8]


# # test=np.array([[subelem for subelem in elem] for elem in COG_invert_use['outlier_diagonal_lower_floor']])
# test=valid_par_range_dict['outlier_diagonal_lower_floor']
# breakpoint()

tab_2D,tab_3D=st.tabs(['2D Curve of Growths','nR² evolution'])

with tab_2D:
    slider_nH=st.select_slider('nH value',range_nH,value=23.0)
    slider_v_turb=st.select_slider(r'$v_{turb}$ value',range_v_turb,value=3.00)

fig_2D,ax_2D = plt.subplots(1,3, figsize=(10,8),sharey=True)
fig_2D.subplots_adjust(wspace=0)

plt.suptitle(r'Curve of Growths for log(nH/cm²)='+str(slider_nH)+' and $v_{turb}$='+str(slider_v_turb)+' km/s',
             position=[0.5,0.05])

cmap_choice=st.sidebar.radio('Color map choice',['Trying to match Stefano', 'matplotlib default'])

init_cmap_vals={'outlier_diagonal_middle':'red',
                'outlier_diagonal_lower_floor':'maroon',
                'diagonal_lower_low_highE_flux':'orange',
                'diagonal_upper_low_highE_flux':'powderblue',
                'diagonal_upper_mid_highE_flux':'turquoise',
                'diagonal_upper_high_highE_flux':'pink',
                'SPL_whereline':'forestgreen'}

if cmap_choice=="matplotlib default":
    base_cmap=plt.rcParams['axes.prop_cycle'].by_key()['color']
elif cmap_choice=='Trying to match Stefano':
    base_cmap=[init_cmap_vals[elem_SED] for elem_SED in list(SEDs.keys())]


ax_2D[0].set_ylim(33,38)

radio_single = st.sidebar.radio('Display options:', ('Multiple Observations', 'Single Observation'))

if radio_single == 'Single Observation':
    list_SEDs_disp=[st.sidebar.selectbox('SED to display',options=list(SEDs.keys()))]

elif radio_single == 'Multiple Observations':

    list_SEDs_disp=st.sidebar.multiselect(label='SEDs to display',options=list(SEDs.keys()),default=list(SEDs.keys()))

list_SEDs_surface=st.sidebar.multiselect(label='SEDs to draw 3D surfaces for',
                                        options=list_SEDs_disp,default=list_SEDs_disp)


with tab_3D:
    if len(list_SEDs_disp)>1:
        st.info('Undersampling the volumes to avoid lags. For the full volumes, select only one SED.')

plot_points=st.sidebar.toggle(label='overlay points',value=False)


with st.sidebar.expander('Curve of growth visualisation options:'):
    highlight_EW_vert=st.toggle('highlight EW range')

    highlight_valid_range=st.toggle('highlight valid nR² range',value=True)

for i_SED,elem_SED in enumerate(list(SEDs.keys())):

    if elem_SED not in list_SEDs_disp:
        continue

    COG_invert_SED=COG_invert_use[elem_SED]

    COG_invert_indiv=COG_invert_SED[np.where(range_nH==slider_nH)[0][0]][np.where(slider_v_turb==range_v_turb)[0][0]]

    #plotting the COGs

    #restricting the display to the part between 34 and 37 to avoid issues due to interpolation
    mask_noissue=(interp_x>=34.) & (interp_x<=37.)

    ax_2D[0].plot((COG_invert_indiv[0]/COG_invert_indiv[1])[mask_noissue],
                  interp_x[mask_noissue],
                  label=elem_SED,color=base_cmap[i_SED])

    ax_2D[1].plot(COG_invert_indiv[1],interp_x,color=base_cmap[i_SED])
    ax_2D[2].plot(COG_invert_indiv[0],interp_x,color=base_cmap[i_SED])

    ew_25_vals=ew_25_csv[flux_3_10_csv['SED'] == elem_SED]
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

    ew_25_mask=np.argwhere((COG_invert_indiv[1] >ew_25_vals_arr[0]-ew_25_vals_arr[1]) &
                           (COG_invert_indiv[1] <ew_25_vals_arr[0]+ew_25_vals_arr[2])).T[0]

    ew_26_mask=np.argwhere((COG_invert_indiv[0] >ew_26_vals_arr[0]-ew_26_vals_arr[1]) &
                           (COG_invert_indiv[0] <ew_26_vals_arr[0]+ew_26_vals_arr[2])).T[0]

    if ew_25_vals_arr[-1]!=0.:
        ew_25_mask_ul=np.argwhere(COG_invert_indiv[1] <ew_25_vals_arr[-1]).T[0]
    else:
        ew_25_mask_ul=[]

    if ew_26_vals_arr[-1]!=0.:
        ew_26_mask_ul=np.argwhere(COG_invert_indiv[1] <ew_26_vals_arr[-1]).T[0]
    else:
        ew_26_mask_ul=[]

    if ew_25_vals_arr[-1]!=0 or ew_26_vals_arr[-1]!=0:
        ew_ratio_mask=[]
    else:
        ew_ratio_mask=np.argwhere((COG_invert_indiv[0]/COG_invert_indiv[1] >\
                                   (ew_26_vals_arr[0]-ew_26_vals_arr[1])/(ew_25_vals_arr[0]+ew_25_vals_arr[1])) &
                                  (COG_invert_indiv[0] / COG_invert_indiv[1] < \
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
        temp=[ax_2D[1].plot(COG_invert_indiv[1][ew_25_intervals[i_inter][0]:ew_25_intervals[i_inter][1]+1],
                  interp_x[ew_25_intervals[i_inter][0]:ew_25_intervals[i_inter][1]+1],
                  color=base_cmap[i_SED],lw=5,alpha=0.7,zorder=100)\
         for i_inter in range(len(ew_25_intervals))]

    if len(ew_25_mask_ul)!=0:

        ew_25_intervals_ul=list(interval_extract(ew_25_mask_ul))

        #25 outline plot
        temp=[ax_2D[1].plot(COG_invert_indiv[1][ew_25_intervals_ul[i_inter][0]:ew_25_intervals_ul[i_inter][1]+1],
                  interp_x[ew_25_intervals_ul[i_inter][0]:ew_25_intervals_ul[i_inter][1]+1],
                  color=base_cmap[i_SED],lw=5,alpha=0.7,zorder=100,ls=':')\
         for i_inter in range(len(ew_25_intervals_ul))]

    ax_2D[1].set_xlim(0,70)


    if len(ew_26_mask)!=0:

        ew_26_intervals=list(interval_extract(ew_26_mask))

        #26 outline plot
        temp=[ax_2D[2].plot(COG_invert_indiv[0][ew_26_intervals[i_inter][0]:ew_26_intervals[i_inter][1]+1],
                  interp_x[ew_26_intervals[i_inter][0]:ew_26_intervals[i_inter][1]+1],
                  color=base_cmap[i_SED],lw=5,alpha=0.7,zorder=100)\
         for i_inter in range(len(ew_26_intervals))]

    if len(ew_26_mask_ul)!=0:

        ew_26_intervals_ul=list(interval_extract(ew_26_mask_ul))

        #25 outline plot
        temp=[ax_2D[1].plot(COG_invert_indiv[0][ew_26_intervals_ul[i_inter][0]:ew_26_intervals_ul[i_inter][1]+1],
                  interp_x[ew_26_intervals_ul[i_inter][0]:ew_26_intervals_ul[i_inter][1]+1],
                  color=base_cmap[i_SED],lw=5,alpha=0.7,zorder=100,ls=':')\
         for i_inter in range(len(ew_26_intervals_ul))]
        
    ax_2D[2].set_xlim(0,70)


    if len(ew_ratio_mask)!=0:

        ew_ratio_intervals=list(interval_extract(ew_ratio_mask))

        #preventing showing the ratios that are due to interpolation issues
        for i_inter in range(len(ew_ratio_intervals)):

            if (interp_x[ew_ratio_intervals[i_inter][0]:ew_ratio_intervals[i_inter][1]+1] >=37.).all() or \
                (interp_x[ew_ratio_intervals[i_inter][0]:ew_ratio_intervals[i_inter][1] + 1] <= 34.).all():
                continue

            #ratio outline plot
            temp=ax_2D[0].plot((COG_invert_indiv[0]/COG_invert_indiv[1])\
                [ew_ratio_intervals[i_inter][0]:ew_ratio_intervals[i_inter][1]+1],
                      interp_x[ew_ratio_intervals[i_inter][0]:ew_ratio_intervals[i_inter][1]+1],
                      color=base_cmap[i_SED],lw=5,alpha=0.7,zorder=100)

    if len(ew_ratio_mask_l)!=0:

        ew_ratio_intervals_l=list(interval_extract(ew_ratio_mask_l))

        # preventing showing the ratios that are due to interpolation issues
        for i_inter in range(len(ew_ratio_intervals_l)):

            if (interp_x[ew_ratio_intervals_l[i_inter][0]:ew_ratio_intervals_l[i_inter][1] + 1] >= 37.).all() or \
                    (interp_x[
                     ew_ratio_intervals_l[i_inter][0]:ew_ratio_intervals_l[i_inter][1] + 1] <= 34.).all():
                continue

            mask_bounds=(interp_x[ew_ratio_intervals_l[i_inter][0]:ew_ratio_intervals_l[i_inter][1] + 1] <= 37.)\
                        & (interp_x[
                     ew_ratio_intervals_l[i_inter][0]:ew_ratio_intervals_l[i_inter][1] + 1] >= 34.)

            # ratio outline plot
            temp=ax_2D[0].plot((COG_invert_indiv[0] / COG_invert_indiv[1]) \
                              [ew_ratio_intervals_l[i_inter][0]:ew_ratio_intervals_l[i_inter][1] + 1][mask_bounds],
                          interp_x[ew_ratio_intervals_l[i_inter][0]:ew_ratio_intervals_l[i_inter][1] + 1][mask_bounds],
                          color=base_cmap[i_SED], lw=5, alpha=0.7, zorder=100,ls=':')

    #displaying the valid range of parameters by combining all the restrictions on EW and EW ratio
    if highlight_valid_range:

        valid_range_par=[elem for elem in np.arange(len(interp_x)) if
                                                    (elem in ew_25_mask or elem in ew_25_mask_ul)\
                                                    and (elem in ew_26_mask or elem in ew_26_mask_ul)
                                                    and (elem in ew_ratio_mask or elem in ew_ratio_mask_l)]

        if len(valid_range_par)!=0:
            valid_range_intervals = list(interval_extract(valid_range_par))
            for i_inter in range(len(valid_range_intervals)):
                ax_2D[0].axhspan(interp_x[valid_range_intervals[i_inter][0]],
                                 interp_x[valid_range_intervals[i_inter][1]],
                             color=base_cmap[i_SED], alpha=0.3)
                ax_2D[1].axhspan(interp_x[valid_range_intervals[i_inter][0]],
                                 interp_x[valid_range_intervals[i_inter][1]],
                             color=base_cmap[i_SED], alpha=0.3)
                ax_2D[2].axhspan(interp_x[valid_range_intervals[i_inter][0]],
                                 interp_x[valid_range_intervals[i_inter][1]],
                             color=base_cmap[i_SED], alpha=0.3)
    ax_2D[0].set_xscale('log')
    ax_2D[0].set_xlim(0.1,10.)
    ax_2D[0].set_ylabel(r'log$_{10}$(nR²)')


#plotting a horizontal background grid to help visualisation
temp=[ax_2D[0].plot([1e-2,1e2],[nr_val,nr_val],color='grey',lw=0.5,alpha=0.3) for nr_val in np.arange(33.1,38,0.1)]
temp=[ax_2D[1].plot([0,80],[nr_val,nr_val],color='grey',lw=0.5,alpha=0.3) for nr_val in np.arange(33.1,38,0.1)]
temp=[ax_2D[2].plot([0,80],[nr_val,nr_val],color='grey',lw=0.5,alpha=0.3) for nr_val in np.arange(33.1,38,0.1)]

fig_2D.legend(ncol=3,bbox_to_anchor=(0.91, 1.))



with tab_2D:

    st.pyplot(fig_2D)

def plot_3d_surface(planes, color='lightblue', volume_number=1, plot_points=False,
                    legendgroup='',i_SED=-1,draw_surface=True,full_planes=None,under_sampling_v_turb=1,under_sampling_nh=1,
                    single_mode=False):
    """
    Plot a 3D structure, distinguishing between volumes, planes, and lines.

    Args:
        planes (numpy.ndarray): A 2D array of shape (n_planes, n_points, 3), where each plane is a set of 3D points.
        color (str): The color to use for the surface or points (default is 'lightblue').
        volume_number (int): The name of the volume to display on hover (default is 1').
        show_volume_name (bool): Whether to display the `volume_name` on hover for each point (default is True).

        plot_points (bool): whether to plots the points from which the volume/plane is made

        legend_group: the SED name to only have one legend element per SED

        i_SED: used with high sampling to determine the alphashape coefficient manually for each SED

        draw_surface (bool): decides if the plot is only of the points or also of the surfaces between the points

        full_volumes: used when undersampling the planes to still draw of the points if desired, mostly for checking
                      purposes

        under_sampling_v_turb and _nH: used for checking purposes

        single_mode: to know whether the volumes should be partly transparent

    Returns:
        a list of  A Plotly graph objects with the appropriate 3D object type.
    """

    volume_str='volume ' + str(volume_number)

    def make_shape_triangles_planes(points, color='blue', volume_str='', legendgroup='',check_overflow='base',
                                    plane_type='delauney',line_mode=False):
        """
        This one was adapted must work specifically to not overplot too many triangles
        Tailored for planes

        #note: the line_mode option wasn't really doing it so I'm keeping it but it's not on
        """
        x = points[:, 0]  # x is always 23
        y = points[:, 1]
        z = points[:, 2]

        if len(np.unique(x)) == 1:
            points_2d = points[:, 1:3]
        elif len(np.unique(y)) == 1:
            points_2d = points[:, [0, 2]]
        else:
            points_2d = points[:, 0:2]

        # if check_overflow:
        # Perform Delaunay triangulation

        mesh_list = []

        if plane_type=='nearest':

            triangle_list=[]
            v_turb_range_plane=np.unique(points.T[1])

            #doing this on the first axis
            for i_v_turb,v_turb in enumerate(v_turb_range_plane[:-1]):

                #finding the two points with the current v_turb value
                points_curr_v_turb=points[points.T[1]==v_turb_range_plane[i_v_turb]]

                points_next_v_turb=points[points.T[1]==v_turb_range_plane[i_v_turb+1]]

                #making two triangles out of these
                points_fill=np.array([points_curr_v_turb.tolist()+points_next_v_turb.tolist()])

                # breakpoint()

                #adding the lower and upper triangle
                tri_fill_lower=np.array(points_curr_v_turb.tolist()+[points_next_v_turb[points_next_v_turb.T[2].argmin()].tolist()])
                tri_fill_higher=np.array(points_next_v_turb.tolist()+[points_curr_v_turb[points_next_v_turb.T[2].argmax()].tolist()])

                triangle_list+=[tri_fill_lower]
                triangle_list+=[tri_fill_higher]
                # tri_fill_lower=tri_fill_lower[::-1]
                # tri_fill_higher=tri_fill_higher[::-1]

                if line_mode:
                    for points_couple in (points_curr_v_turb,[points_curr_v_turb[0].tolist()]+[points_next_v_turb[0].tolist()],
                                          points_next_v_turb,[points_curr_v_turb[1].tolist()]+[points_next_v_turb[1].tolist()]):

                        mesh_list += [
                            go.Scatter3d(x=np.array(points_couple).T[0],
                                         y=np.array(points_couple).T[1],
                                         z=np.array(points_couple).T[2],
                                      line_color=color,mode='lines',
                                      name=volume_str, legendgroup=legendgroup, legendgrouptitle={'text': legendgroup},
                                      showlegend=False)]

                    # mesh_list += [
                    #     go.Scatter3d(x=tri_fill_higher.T[0].tolist(),
                    #                  y=tri_fill_higher.T[1].tolist(),
                    #                  z=tri_fill_higher.T[2].tolist(),
                    #               line_color=color,mode='lines',
                    #               name=volume_str, legendgroup=legendgroup, legendgrouptitle={'text': legendgroup},
                    #               showlegend=False)]
                else:
                    mesh_list += [
                        go.Mesh3d(x=tri_fill_lower.T[0], y=tri_fill_lower.T[1], z=tri_fill_lower.T[2],
                                  color=color, i=[0], j=[1], k=[2], opacity=0.5 if single_mode else 0.4, showscale=False,
                                  name=volume_str, legendgroup=legendgroup, legendgrouptitle={'text': legendgroup},
                                  showlegend=False)]


                    mesh_list += [
                        go.Mesh3d(x=tri_fill_higher.T[0], y=tri_fill_higher.T[1], z=tri_fill_higher.T[2],
                                  color=color, i=[0], j=[1], k=[2], opacity=0.5 if single_mode else 0.4, showscale=False,
                                  name=volume_str, legendgroup=legendgroup, legendgrouptitle={'text': legendgroup},
                                  showlegend=False)]

                #and on the second axis

        elif plane_type=='delauney':

            triangulation = Delaunay(points_2d)

            # Add triangles to the plot

            #for line drawings
            couple_list=[]

            for simplex in triangulation.simplices:
                x_tris = [x[simplex[0]], x[simplex[1]], x[simplex[2]]]
                y_tris = [y[simplex[0]], y[simplex[1]], y[simplex[2]]]
                z_tris = [z[simplex[0]], z[simplex[1]], z[simplex[2]]]

                if check_overflow=='base':
                    # skipping the triangles with 3 different x or y axis coordinates since these will go beyond the plane we want
                    # to draw
                    if len(np.unique(y_tris)) > 2 or len(np.unique(x_tris)) > 2:
                        continue

                elif check_overflow.isdigit():

                    args_y=np.array([np.argwhere(range_v_turb == np.log10(elem))[0][0] for elem in y_tris])
                    dist_x=abs(args_y.max()-args_y.min())
                    args_x=np.array([np.argwhere(range_nH==elem)[0][0] for elem in x_tris])
                    dist_y=abs(args_x.max()-args_x.min())

                    if np.sqrt(dist_y**2+dist_x**2)>int(check_overflow):
                        continue

                if line_mode:

                    # skipping the triangles with 3 different x or y axis coordinates since these will go beyond the plane we want
                    # to draw
                    if len(np.unique(y_tris)) > 2 or len(np.unique(x_tris)) > 2:
                        continue

                    for couple in [[0,1],[1,2],[2,0]]:
                        x_couple=x[simplex[couple]]
                        y_couple=y[simplex[couple]]
                        z_couple=z[simplex[couple]]

                        #skipping diagonals and already drawn segments
                        #note: imperfect: should instead loop through all possible triangles in this case
                        if not (len(np.unique(x_couple))==1 or len(np.unique(y_couple))==1) or simplex[couple] in\
                            np.array(couple_list):
                            continue

                        mesh_list += [
                            go.Scatter3d(x=x_couple,
                                         y=y_couple,
                                         z=z_couple,
                                         line_color='rgba('+','.join(np.array(name_to_rgb(color)).astype(str).tolist()+['0.2'])+')',
                                         mode='lines',
                                         name=volume_str, legendgroup=legendgroup,
                                         legendgrouptitle={'text': legendgroup},
                                         showlegend=False)]

                        # breakpoint()

                        couple_list+=[simplex[couple]]

                    #breakpoint()
                    # mesh_list += [
                    # go.Scatter3d(x=x_tris+[None],
                    #              y=y_tris+[None],
                    #              z=z_tris+[None],
                    #              line_color=color,mode='lines',
                    #              name=volume_str, legendgroup=legendgroup, legendgrouptitle={'text': legendgroup},
                    #              showlegend=False)]

                else:
                    mesh_list += [
                        go.Mesh3d(x=x_tris, y=y_tris, z=z_tris, color=color, i=[0], j=[1], k=[2], opacity=0.5 if single_mode else 0.4,
                                  showscale=False,
                                  name=volume_str, legendgroup=legendgroup, legendgrouptitle={'text': legendgroup},
                                  showlegend=False)]

        return mesh_list

    def make_shape_triangles_volumes(points, color='blue', volume_str='', legendgroup=''):
        """
        This one was adapted from chatgpt but modified because must work specifically to not overplot too many triangles
        Tailored for planes
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        if len(np.unique(x)) == 1:
            points_2d = points[:, 1:3]
        elif len(np.unique(y)) == 1:
            points_2d = points[:, [0, 2]]
        elif  len(np.unique(z)) == 1:
            points_2d = points[:, 0:2]
        else:
            points_2d=points
        # Perform Delaunay triangulation
        triangulation = Delaunay(points_2d)

        # Add triangles to the plot
        mesh_list = []

        #computing the list of good enough triangles
        for simplex in triangulation.simplices:
            x_tris = [x[simplex[0]], x[simplex[1]], x[simplex[2]]]
            y_tris = [y[simplex[0]], y[simplex[1]], y[simplex[2]]]
            z_tris = [z[simplex[0]], z[simplex[1]], z[simplex[2]]]

            # skipping the triangles with 3 different coordinates for each point since these will go beyond the plane we want
            # to draw
            if len(np.unique(y_tris)) > 2 or len(np.unique(x_tris)) > 2 or len(np.unique(z_tris)) > 2:
                continue

            # skipping the triangles with non-adjacent points:
            ids_x_grid=np.unique(np.array([np.argwhere(range_nH==elem)[0][0] for elem in x_tris]))
            if abs(ids_x_grid[1]-ids_x_grid[0])>1:
                continue

            # breakpoint()
            ids_y_grid=np.unique(np.array([np.argwhere(range_v_turb==np.log10(elem))[0][0] for elem in y_tris]))
            if abs(ids_y_grid[1]-ids_y_grid[0])>1:
                continue
                
            # ids_z_grid=np.unique(np.array([np.argwhere(range_nH==elem)[0][0] for elem in x_tris]))
            # if abs(ids_z_grid[1]-ids_z_grid[0])>0:
            #     continue

            mesh_list += [
                go.Mesh3d(x=x_tris, y=y_tris, z=z_tris, color=color, i=[0], j=[1], k=[2], opacity=0.5 if single_mode else 0.4, showscale=False,
                          name=volume_str, legendgroup=legendgroup, legendgrouptitle={'text': legendgroup},
                          showlegend=False)]

        return mesh_list

    points = np.vstack(planes)
    rank = np.linalg.matrix_rank(points[:, :3])  # Determine rank based on X, Y, Z coordinates

    if full_planes is not None:
        points_for_points=np.vstack(full_planes)
    else:
        points_for_points=points

    #adding a failsafe for the cases where this doesn't work
    if rank==3:
        rank=rank-sum([len(np.unique(elem))==1 for elem in points.T])

    if rank==2:
        rank=rank-max(sum([len(np.unique(elem))==1 for elem in points.T])-1,0)

    shapes=[]
    if rank == 3:

        lower_p=planes[0]
        higher_p=planes[1]

        if i_SED in [0,1]:
            alpha_higher=8
            alpha_lower=6
        else:
            alpha_higher=6
            alpha_lower=6

        alpha_higher=str(alpha_higher+(0 if under_sampling_nh==1 and under_sampling_v_turb==1 else 3))
        alpha_lower=str(alpha_lower+(0 if under_sampling_nh==1 and under_sampling_v_turb==1 else 3))

        if draw_surface:
            #making the base "horizontal" planes
            shapes+=make_shape_triangles_planes(lower_p,color=color,volume_str=volume_str,legendgroup=legendgroup,
                                                check_overflow=alpha_lower)
            shapes+=make_shape_triangles_planes(higher_p,color=color,volume_str=volume_str,legendgroup=legendgroup,
                                                check_overflow=alpha_higher)

        #finding the edges of the "horizontal" planes by fetching the extremal values for each v_turb
        plane_highest=[]
        plane_lowest=[]
        for i_v_turb,elem_v_turb in enumerate(range_v_turb):

            #skipping the elements that have been removed when undersampling
            if i_v_turb not in [0,len(range_v_turb)-1]+np.arange(len(range_v_turb))[np.arange(len(range_v_turb))\
                                                                                    %under_sampling_v_turb==0].tolist():
                continue

            elem_lower_p_v_turb=lower_p[np.log10(planes[0].T[1]) == elem_v_turb]

            if len(elem_lower_p_v_turb)==0:
                if under_sampling_nh>1:
                    continue
                else:
                    raise ValueError('Error: there are separations in the planes.')

            elem_lower_p_highest=elem_lower_p_v_turb[elem_lower_p_v_turb.T[2].argmax()]
            elem_lower_p_lowest=elem_lower_p_v_turb[elem_lower_p_v_turb.T[2].argmin()]

            elem_higher_p_v_turb=higher_p[np.log10(planes[0].T[1]) == elem_v_turb]

            if len(elem_lower_p_v_turb)==0:
                if under_sampling_nh>1:
                    continue
                else:
                    raise ValueError('Error: there are separations in the planes.')

            elem_higher_p_highest=elem_higher_p_v_turb[elem_higher_p_v_turb.T[2].argmax()]
            elem_higher_p_lowest=elem_higher_p_v_turb[elem_higher_p_v_turb.T[2].argmin()]

            plane_highest+=[elem_higher_p_highest,elem_lower_p_highest]
            plane_lowest+=[elem_higher_p_lowest,elem_lower_p_lowest]

        plane_highest=np.array([elem for elem in plane_highest])
        plane_lowest=np.array([elem for elem in plane_lowest])

        #and closing with two "vertical" planes

        if draw_surface:

            shapes+=make_shape_triangles_planes(plane_highest,color=color,volume_str=volume_str,legendgroup=legendgroup,
                                                plane_type='nearest')
            shapes+=make_shape_triangles_planes(plane_lowest,color=color,volume_str=volume_str,legendgroup=legendgroup,
                                                plane_type='nearest')

        if plot_points:
            shapes += [go.Scatter3d(x=points_for_points.T[0], y=points_for_points.T[1], z=points_for_points.T[2],
                                    mode='markers',
                                       marker=dict(size=2, color=color,opacity=0.4 if single_mode else 1.),
                                    name=volume_str, legendgroup=legendgroup, legendgrouptitle={'text': legendgroup},
                                    showlegend=True)]

    elif rank == 2:

        #note: shouldn't need to undersample here

        shapes+=make_shape_triangles_planes(points,color=color,volume_str=volume_str,legendgroup=legendgroup)

        if plot_points:
            shapes += [go.Scatter3d(x=points_for_points.T[0], y=points_for_points.T[1], z=points_for_points.T[2],
                                    mode='markers',
                                       marker=dict(size=2, color=color),
                                    name=volume_str,legendgroup=legendgroup,legendgrouptitle={'text': legendgroup},
                                    showlegend=True)]


    elif rank == 1:

        #note: shouldn't need to undersample here

        # 1D Line
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        shapes += [go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            line=dict(color=color, width=5),
            marker=dict(size=5, color=color, opacity=0.5),
            name=volume_str,legendgroup=legendgroup,legendgrouptitle={'text': legendgroup})]
    else:

        #note: shouldn't need to undersample here

        # Degenerate case
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        shapes += [go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color=color, opacity=0.5),
            name=volume_str,legendgroup=legendgroup,legendgrouptitle={'text': legendgroup})]

    return shapes



@st.cache_data
def make_3D_figure(SEDs_disp,SEDs_surface,plot_points=False,under_sampling_v_turb='var',under_sampling_nh='var'):

    '''
    Under sampling gives how much we divide in one axis to reduce the number of vertices
    '''

    # getting all the surfaces into an array
    mult_d_surfaces = []

    single_mode=len(SEDs_disp)==1

    n_SEDs=len(SEDs_disp)

    for i_SED,elem_SED in enumerate(list(SEDs.keys())):

        if under_sampling_v_turb == 'var':

            if i_SED in [2,4]:
                under_sampling_v_turb=2 if single_mode else 5+n_SEDs//3
            else:
                under_sampling_v_turb=1 if single_mode else 4+n_SEDs//3

        if under_sampling_nh == 'var':
            if i_SED in [2,4]:
                under_sampling_nh=2 if single_mode else 5+n_SEDs//3
            else:
                under_sampling_nh=1+n_SEDs//6

        if elem_SED not in SEDs_disp:
            continue

        valid_volumes=valid_volumes_dict[elem_SED]

        for i_vol in range(len(valid_volumes)):

            pos_v_turb_lower=np.array([np.argwhere(np.log10(elem) == range_v_turb)[0][0]
                                                for elem in valid_volumes[i_vol][0].T[1]])
            pos_v_turb_higher=np.array([np.argwhere(np.log10(elem) == range_v_turb)[0][0]
                                                for elem in valid_volumes[i_vol][1].T[1]])

            #combining undersampling and adding the first and last v_turbs in any case to avoid missing out the edes of the surface
            mask_under_sampling_v_turb_lower=(pos_v_turb_lower % under_sampling_v_turb == 0) | (pos_v_turb_lower==max(pos_v_turb_lower)) \
                                                                                | (pos_v_turb_lower==min(pos_v_turb_lower))

            mask_under_sampling_v_turb_higher=(pos_v_turb_higher % under_sampling_v_turb == 0) | (pos_v_turb_higher==max(pos_v_turb_higher)) \
                                                                                | (pos_v_turb_higher==min(pos_v_turb_higher))

            pos_nh_lower=np.array([np.argwhere(elem == range_nH)[0][0]
                                                for elem in valid_volumes[i_vol][0].T[0]])
            pos_nh_higher=np.array([np.argwhere(elem == range_nH)[0][0]
                                                for elem in valid_volumes[i_vol][1].T[0]])

            #combining undersampling and adding the first and last nhs in any case to avoid missing out the edes of the surface
            mask_under_sampling_nh_lower=(pos_nh_lower % under_sampling_nh == 0) | (pos_nh_lower==max(pos_nh_lower)) \
                                                                                | (pos_nh_lower==min(pos_nh_lower))

            mask_under_sampling_nh_higher=(pos_nh_higher % under_sampling_nh == 0) | (pos_nh_higher==max(pos_nh_higher)) \
                                                                                | (pos_nh_higher==min(pos_nh_higher))
            
            mask_under_sampling_lower=mask_under_sampling_v_turb_lower & mask_under_sampling_nh_lower
            mask_under_sampling_higher=mask_under_sampling_v_turb_higher & mask_under_sampling_nh_higher


            valid_volumes_under_sampled=[valid_volumes[i_vol][0][mask_under_sampling_lower],
                                         valid_volumes[i_vol][1][mask_under_sampling_higher]]

            mult_d_surfaces+=plot_3d_surface(valid_volumes_under_sampled,color=base_cmap[i_SED],volume_number=i_vol+1,
                                             legendgroup=elem_SED,i_SED=i_SED,draw_surface=elem_SED in SEDs_surface,
                                             full_planes=valid_volumes[i_vol],
                                             under_sampling_v_turb=under_sampling_v_turb,
                                             under_sampling_nh=under_sampling_nh,plot_points=plot_points,
                                             single_mode=single_mode)

    fig=go.Figure(data=mult_d_surfaces)


    # Layout settings with custom axis names
    fig.update_layout(
        scene=dict(
            xaxis_title='nH',  # Custom X-axis label
            yaxis_title='v_turb',  # Custom Y-axis label
            zaxis_title='log10(nR²)',  # Custom Z-axis label
            aspectmode="cube",
            yaxis=dict(type='log')
        ),
        height=1000
    )

    # fig_3D.show()
    # fig_3D.show()

    #
    with tab_3D:
        st.plotly_chart(fig,use_container_width=True,theme=None)

make_3D_figure(list_SEDs_disp,list_SEDs_surface,plot_points=plot_points)




