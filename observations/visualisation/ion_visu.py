import os,sys
import glob

import argparse
import warnings

import time
import numpy as np
import pandas as pd
from decimal import Decimal

import streamlit as st
#matplotlib imports

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plotly.graph_objects as go

online='parrama' not in os.getcwd()
if online:
    project_dir='/mount/src/winds/'
else:
    project_dir='/home/parrama/Documents/Work/PhD/Scripts/Python/'

sys.path.append(project_dir+'/general/')

from general_tools import interval_extract,get_overlap
from bipolar import hotcold

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
# range_nh = np.arange(21, 25.01, 0.5)
# range_v_turb = np.array([0, 500, 1000, 2500, 5000])

#better sampling (with rounding to avoid precision errors that will display in the values
step_nh=0.05
range_nh=np.arange(21.5,25.01,step_nh).round(3)

step_v_turb=0.05
range_v_turb=np.arange(0.5,4.01,step_v_turb).round(3)
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
        COG_invert_indiv = np.array([[None] * len(range_v_turb)] * len(range_nh))

        for i_nh, nH in enumerate(range_nh):
            for i_v_turb, v_turb in enumerate(range_v_turb):
                curr_COG = np.loadtxt(
                    os.path.join(explore_photo_dir, 'SED_' + elem_SED + '_ew_%.2f' % nH + '_%.2f'  % v_turb + '.dat'))

                curr_nRtwo = float(np.log10(l_ion_SED).iloc[0]) - curr_COG.T[0]

                # order is nr², 26, 25
                COG_invert_indiv[i_nh][i_v_turb] = np.array([curr_nRtwo, curr_COG.T[1], curr_COG.T[2]])

                loading_bar.progress(1 / (len(SEDs)*len(range_nh)*len(range_v_turb)))


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

            valid_par_arr = np.repeat(-1., 2 * len(range_nh) * len(range_v_turb)).reshape(2, len(range_nh),
                                                                                          len(range_v_turb))
            valid_par_arr_3D = np.repeat(None, 2 * len(range_nh) * len(range_v_turb)).reshape(2, len(range_nh),
                                                                                              len(range_v_turb))

            param_space_bar = st.progress(0, text='Finding parameter space')

            for i_nh in range(len(range_nh)):
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

                        valid_par_arr_3D[0][i_nh][i_v_turb] = np.array([range_nh[i_nh], 10 ** (range_v_turb[i_v_turb]),
                                                                        interp_x[
                                                                            valid_range_par[0]]]).astype(object)
                        valid_par_arr_3D[1][i_nh][i_v_turb] = np.array([range_nh[i_nh], 10 ** (range_v_turb[i_v_turb]),
                                                                        interp_x[
                                                                            valid_range_par[-1]]]).astype(object)
                    else:
                        valid_par_arr_3D[0][i_nh][i_v_turb] = np.repeat(-1, 3).astype(object)
                        valid_par_arr_3D[1][i_nh][i_v_turb] = np.repeat(-1, 3).astype(object)

                    param_space_bar.progress(1 / (len(range_nh) * len(range_v_turb)))
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

tab_2D,tab_3D,tab_delta=st.tabs(['2D Curve of Growths','3D nR² evolution','2D projections of distance between SEDs'])

with tab_2D:
    slider_nH=st.select_slider('nH value',range_nh,value=23.0)
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

base_nolag=np.array(list(SEDs.keys()))[[0,1,3,5,6]]

plot_3D=st.sidebar.toggle('Plot nR² evolution in 3D',value=False)

if not plot_3D:
    with tab_3D:
        st.info('To start plotting the 3D evolution, toggle the option in the sidebar.')

list_SEDs_surface=st.sidebar.multiselect(label='SEDs to draw 3D surfaces for',
                                        options=list_SEDs_disp,default=[elem for elem in base_nolag if elem in list_SEDs_disp])


with tab_3D:
    if plot_3D and len(list_SEDs_disp)>1:
        st.info('The volumes are undersampled according to the number of observations drawn.'
                'To see the full volumes, select only one SED.')

plot_points=st.sidebar.toggle(label='overlay points',value=False)

control_camera=st.sidebar.toggle(label='manual camera control',value=False)

setup_camera=st.sidebar.toggle(label='setup camera',value=False)


if setup_camera and not online:
    val_up_x=st.sidebar.number_input('camera up x',format="%.3f")
    val_up_y=st.sidebar.number_input('camera up y',format="%.3f")
    val_up_z=st.sidebar.number_input('camera up z',format="%.3f")

    val_center_x=st.sidebar.number_input('camera center x',format="%.3f")
    val_center_y=st.sidebar.number_input('camera center y',format="%.3f")
    val_center_z=st.sidebar.number_input('camera center z',format="%.3f")
    
    val_eye_x=st.sidebar.number_input('camera eye x',format="%.3f")
    val_eye_y=st.sidebar.number_input('camera eye y',format="%.3f")
    val_eye_z=st.sidebar.number_input('camera eye z',format="%.3f")

    camera = dict(
        up=dict(x=val_up_x,y=val_up_y,z=val_up_z),
        center=dict(x=val_center_x,y=val_center_y,z=val_center_z),
        eye=dict(x=val_eye_x,y=val_eye_y,z=val_eye_z)
    )

    #view 2:
    # 0 0 1 0.07 -0.108 0.009 1.614 -0.005 -0.855

    #view 3:
    #0 0 1 0.013 0.057 -0.101 -0.303 1.303 1.132

elif control_camera and not online:

    slider_up_x = st.sidebar.slider(label='camera X angle vector',min_value=-1., max_value=1., step=0.01,value=0.)
    slider_up_y = st.sidebar.slider(label='camera Y angle vector',min_value=-1., max_value=1., step=0.01,value=0.)
    slider_up_z = st.sidebar.slider(label='camera Z angle vector',min_value=-1., max_value=1., step=0.01,value=1.)
    
    slider_eye_x = st.sidebar.slider(label='camera X pos',min_value=-3., max_value=3., step=0.1,value=1.)
    slider_eye_y = st.sidebar.slider(label='camera Y pos',min_value=-3., max_value=3., step=0.1,value=1.)
    slider_eye_z = st.sidebar.slider(label='camera Z pos',min_value=-3., max_value=3., step=0.1,value=1.)

    #good views: 1.6 1.6 0.3 (facing the planes)

    camera = dict(
        up=dict(x=slider_up_x, y=slider_up_y, z=slider_up_z),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=slider_eye_x, y=slider_eye_y, z=slider_eye_z)
    )
else:
    camera=None


with st.sidebar.expander('2D projections'):
    plot_distance_SEDs=st.toggle('Plot distance between observations',value=False)

    if plot_distance_SEDs:
        plot_distance_dim=st.radio('projection',('nR²','v_turb','NH'),index=0)
    else:
        plot_distance_dim=''

    combine_plot_distance=st.toggle('Combine distance plots',value=False)

    plot_distance_names = st.toggle('Write SED names in the distance plot titles', value=False)

    interpolate_nr2=st.toggle('Interpolate NR² values on a coarser grid when it is not the projected axe',value=True)

if not plot_distance_SEDs:
    with tab_delta:
        st.info('To start plotting these elements, toggle the option in the sidebar.')



with st.sidebar.expander('Curve of growth visualisation options:'):
    highlight_EW_vert=st.toggle('highlight EW range')

    highlight_valid_range=st.toggle('highlight valid nR² range',value=True)

for i_SED,elem_SED in enumerate(list(SEDs.keys())):

    if elem_SED not in list_SEDs_disp:
        continue

    COG_invert_SED=COG_invert_use[elem_SED]

    COG_invert_indiv=COG_invert_SED[np.where(range_nh==slider_nH)[0][0]][np.where(slider_v_turb==range_v_turb)[0][0]]

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

#distance plots


@st.cache_data
def plot_distance(_ax,SED_1,SED_2,mode='nR²',write_names=True,interpolate_nr2=False):

    '''
    2D projections of the 3D graphs for couples of 2 SEDs.
    Highlight the valid 2D parameter space of each individualSED in hashes,
    when both spaces are compatible, highlights the distance in the projection dimension (mode) between the regions

    interpolate_nr2:
        if set to True, replaces the 0.01 grid with a 0.05

    if ax is not None, builds the figure in the given ax, does not create a colorbar, and returns the
    min and max value of the imshow and the imshow for climming
    '''

    assert len(valid_volumes_dict[SED_1])==1,'Error: only implemented for single volumes'
    assert len(valid_volumes_dict[SED_2])==1,'Error: only implemented for single volumes'

    # here we consider that the nr² values are sampled over a 0.01 grid between 33 and 36.5
    # we add the rounding to avoid comparison issues
    range_nr2 = np.arange(33, 36.51, 0.05 if interpolate_nr2 and mode!='nR²' else 0.01).round(3)


    plane_lower_SED_1_init,plane_higher_SED_1_init=valid_volumes_dict[SED_1][0]
    plane_lower_SED_2_init,plane_higher_SED_2_init=valid_volumes_dict[SED_2][0]

    if interpolate_nr2 and mode !='nR²':
        plane_lower_SED_1=plane_lower_SED_1_init.copy()
        plane_lower_SED_1.T[2]=(plane_lower_SED_1.T[2]*20).round()/20

        plane_higher_SED_1=plane_higher_SED_1_init.copy()
        plane_higher_SED_1.T[2]=(plane_higher_SED_1.T[2]*20).round()/20
        
        plane_lower_SED_2=plane_lower_SED_2_init.copy()
        plane_lower_SED_2.T[2]=(plane_lower_SED_2.T[2]*20).round()/20

        plane_higher_SED_2=plane_higher_SED_2_init.copy()
        plane_higher_SED_2.T[2]=(plane_higher_SED_2.T[2]*20).round()/20
    else:
        plane_lower_SED_1 = plane_lower_SED_1_init.copy()

        plane_higher_SED_1 = plane_higher_SED_1_init.copy()

        plane_lower_SED_2 = plane_lower_SED_2_init.copy()

        plane_higher_SED_2 = plane_higher_SED_2_init.copy()

    #creating the plot
    if _ax is None:
        fig_dist,ax_dist= plt.subplots(1,1, figsize=(6,6),)
    else:
        ax_dist=_ax

    ax_dist.set_facecolor('grey')
    cmap_bipolar=hotcold(neutral=1)

    if mode=='nR²':

        range_x=range_v_turb
        range_y=range_nh

        #this index is for computing the values from the plane_lower/higher_SED arrays
        index_x=1
        index_y=0
        index_z=2

        if _ax is None:
            ax_dist.set_xlabel(r'log$_{10}$(v$_{turb}$)')
            ax_dist.set_ylabel(r'log$_{10}$(NH)')

    if mode=='v_turb':

        range_x=range_nr2
        range_y=range_nh

        #this index is for computing the values from the plane_lower/higher_SED arrays
        index_x=2
        index_y=0
        index_z=1

        if _ax is None:
            ax_dist.set_xlabel(r'log$_{10}$(nR$^2$)')
            ax_dist.set_ylabel(r'log$_{10}$(NH)')

    if mode=='NH':

        range_x=range_nr2
        range_y=range_v_turb

        #this index is for computing the values from the plane_lower/higher_SED arrays
        index_x=2
        index_y=1
        index_z=0

        if _ax is None:
            ax_dist.set_xlabel(r'log$_{10}$(nR$^2$)')
            ax_dist.set_ylabel(r'log$_{10}$(v_turb)')

    distance_arr = np.repeat(np.nan, len(range_x) * len(range_y)).reshape((len(range_x), len(range_y)))

    surface_SED_1 = np.repeat(np.nan, len(range_x) * len(range_y)).reshape((len(range_x), len(range_y)))

    surface_SED_2 = np.repeat(np.nan, len(range_x) * len(range_y)).reshape((len(range_x), len(range_y)))


    #computing the distance array
    for i_x,elem_x in enumerate(range_x):
        for i_y,elem_y in enumerate(range_y):

            #creating the x and y valid arrays separately for clarity
            #we only apply the log10 for v_turb (index_x==1)
            mask_x_valid_SED_1_lower=((np.log10(plane_lower_SED_1.T[index_x])==elem_x) if index_x==1 else \
                              ((plane_lower_SED_1.T[index_x])==elem_x))

            mask_y_valid_SED_1_lower=((np.log10(plane_lower_SED_1.T[index_y])==elem_y) if index_y==1 else \
                              ((plane_lower_SED_1.T[index_y])==elem_y))

            mask_x_valid_SED_2_lower = ((np.log10(plane_lower_SED_2.T[index_x]) == elem_x) if index_x == 1 else \
                                      ((plane_lower_SED_2.T[index_x]) == elem_x))

            mask_y_valid_SED_2_lower = ((np.log10(plane_lower_SED_2.T[index_y]) == elem_y) if index_y == 1 else \
                                      ((plane_lower_SED_2.T[index_y]) == elem_y))


            mask_valid_SED_1_lower=mask_x_valid_SED_1_lower & mask_y_valid_SED_1_lower
            mask_valid_SED_2_lower=mask_x_valid_SED_2_lower & mask_y_valid_SED_2_lower
            
            mask_x_valid_SED_1_higher=((np.log10(plane_higher_SED_1.T[index_x])==elem_x) if index_x==1 else \
                              ((plane_higher_SED_1.T[index_x])==elem_x))

            mask_y_valid_SED_1_higher=((np.log10(plane_higher_SED_1.T[index_y])==elem_y) if index_y==1 else \
                              ((plane_higher_SED_1.T[index_y])==elem_y))

            mask_x_valid_SED_2_higher = ((np.log10(plane_higher_SED_2.T[index_x]) == elem_x) if index_x == 1 else \
                                      ((plane_higher_SED_2.T[index_x]) == elem_x))

            mask_y_valid_SED_2_higher = ((np.log10(plane_higher_SED_2.T[index_y]) == elem_y) if index_y == 1 else \
                                      ((plane_higher_SED_2.T[index_y]) == elem_y))


            mask_valid_SED_1_higher=mask_x_valid_SED_1_higher & mask_y_valid_SED_1_higher
            mask_valid_SED_2_higher=mask_x_valid_SED_2_higher & mask_y_valid_SED_2_higher
            

            # x_square=[[range_v_turb[i_v_turb],range_v_turb[i_v_turb+1]],range_v_turb[i_v_turb],range_v_turb[i_v_turb+1]]
            #
            # y_square=[[range_nh[i_nh],range_nh[i_nh]],range_nh[i_nh+1],range_nh[i_nh+1]]

            if mask_valid_SED_1_lower.any() or mask_valid_SED_1_higher.any():
                surface_SED_1[i_x][i_y]=1
                # ax_dist.pcolor(x_square,y_square,[[1]],cmap='viridis',alpha=1,hash='//',vmin=0,vmax=1)

            if mask_valid_SED_2_lower.any() or mask_valid_SED_2_higher.any():
                surface_SED_2[i_x][i_y]=1
                # ax_dist.pcolor(x_square,y_square,1,[[1]],cmap='viridis',alpha=1,hash="\\",vmin=0,vmax=1)

            if not (((sum(mask_valid_SED_1_lower)+sum(mask_valid_SED_1_higher))>0) and \
                    ((sum(mask_valid_SED_2_lower)+sum(mask_valid_SED_2_higher))>0)):

                continue

            if mode =='nR²' and (sum(mask_valid_SED_1_lower)>1 or sum(mask_valid_SED_1_higher)>1\
                            or (sum(mask_valid_SED_2_lower)>1 or sum(mask_valid_SED_2_higher)>1)):
                print('This should not happen')
                breakpoint()

            if mode=='nR²':

                if not (mask_valid_SED_1_lower==mask_valid_SED_1_higher).all() or \
                   not (mask_valid_SED_2_lower == mask_valid_SED_2_higher).all():
                    print('This should not happen')
                    breakpoint()

                #in this mode, each NH-v_turb couple only has one range of projection coordinate so computing the
                #distance between that of the two SEDs is easy
                projec_range_SED_1=np.array([plane_lower_SED_1[mask_valid_SED_1_lower][0][2],
                                             plane_higher_SED_1[mask_valid_SED_1_higher][0][2]])
                projec_range_SED_2=np.array([plane_lower_SED_2[mask_valid_SED_2_lower][0][2],
                                             plane_higher_SED_2[mask_valid_SED_2_higher][0][2]])

                #note that this value is positive if the intervals are compatible and negative otherwise
                distance_vals=get_overlap(projec_range_SED_1,projec_range_SED_2,distance=True)

                if distance_vals>=0:
                    distance_arr[i_x][i_y] = 0.
                else:
                    distance_sign=(projec_range_SED_1[0]-projec_range_SED_2[0])/abs(projec_range_SED_1[0]-projec_range_SED_2[0])

                    distance_arr[i_x][i_y] = distance_sign*abs(distance_vals)
            else:
                #otherwise, it is necessary to compute all the continuous intervals of the projection for each SED
                #then find the smallest distance between each of them
                if mode=='v_turb':

                    if sum(mask_valid_SED_1_lower)>0:
                        projec_ranges_SED_1_lower=np.argwhere([elem==range_v_turb for elem in
                                                 np.log10(plane_lower_SED_1[mask_valid_SED_1_lower].T[1])]).T[1]
                    else:
                        projec_ranges_SED_1_lower=np.array([])

                    if sum(mask_valid_SED_1_higher)>0:
                        projec_ranges_SED_1_higher=np.argwhere([elem==range_v_turb for elem in
                                                 np.log10(plane_lower_SED_1[mask_valid_SED_1_higher].T[1])]).T[1]
                    else:
                        projec_ranges_SED_1_higher=np.array([])


                    if sum(mask_valid_SED_2_lower)>0:
                        projec_ranges_SED_2_lower=np.argwhere([elem==range_v_turb for elem in
                                                 np.log10(plane_lower_SED_2[mask_valid_SED_2_lower].T[1])]).T[1]
                    else:
                        projec_ranges_SED_2_lower=np.array([])

                    if sum(mask_valid_SED_2_higher)>0:
                        projec_ranges_SED_2_higher=np.argwhere([elem==range_v_turb for elem in
                                                 np.log10(plane_lower_SED_2[mask_valid_SED_2_higher].T[1])]).T[1]
                    else:
                        projec_ranges_SED_2_higher=np.array([])


                if mode=='NH':

                    if sum(mask_valid_SED_1_lower) > 0:
                        projec_ranges_SED_1_lower = np.argwhere([round(elem,3) == range_nh.round(3) for elem in
                                                                 plane_lower_SED_1[mask_valid_SED_1_lower].T[0]]).T[1]
                    else:
                        projec_ranges_SED_1_lower = np.array([])

                    if sum(mask_valid_SED_1_higher) > 0:
                        projec_ranges_SED_1_higher = np.argwhere([round(elem,3) == range_nh.round(3) for elem in
                                                                  plane_lower_SED_1[mask_valid_SED_1_higher].T[0]]).T[1]
                    else:
                        projec_ranges_SED_1_higher = np.array([])

                    if sum(mask_valid_SED_2_lower) > 0:
                        projec_ranges_SED_2_lower = np.argwhere([round(elem,3) == range_nh.round(3) for elem in
                                                                 plane_lower_SED_2[mask_valid_SED_2_lower].T[0]]).T[1]
                    else:
                        projec_ranges_SED_2_lower = np.array([])

                    if sum(mask_valid_SED_2_higher) > 0:
                        projec_ranges_SED_2_higher = np.argwhere([round(elem,3) == range_nh.round(3) for elem in
                                                                  plane_lower_SED_2[mask_valid_SED_2_higher].T[0]]).T[1]
                    else:
                        projec_ranges_SED_2_higher = np.array([])


                project_intervals_SED_1=list(interval_extract(np.unique(projec_ranges_SED_1_lower.tolist()+
                                                              projec_ranges_SED_1_higher.tolist())))

                project_intervals_SED_2 = list(interval_extract(np.unique(projec_ranges_SED_2_lower.tolist() +
                                                                          projec_ranges_SED_2_higher.tolist())))

                distance_couples = []
                for elem_inter_1 in project_intervals_SED_1:
                    for elem_inter_2 in project_intervals_SED_2:
                        distance_vals = get_overlap(elem_inter_1, elem_inter_2, distance=True)

                        if distance_vals >= 0:
                            distance_couples += [0.]
                        else:
                            distance_sign = (elem_inter_1[0] - elem_inter_2[0]) / abs(
                                elem_inter_1[0] - elem_inter_2[0])

                            distance_couples += [distance_sign * abs(distance_vals)]

                absmin_distance_id = np.argmin(abs(np.array(distance_couples)))
                distance_arr[i_x][i_y] = distance_couples[absmin_distance_id] * (step_v_turb if mode=='v_turb' else step_nh)

    #note: lw=0 is used to remove the borders. The weird implementation of the hatch coloring is necessary
    #because pcolor doesn't have enough arguments
    #this axis is always with either v_turb or nh so the delta is always 1/2*0.05
    delta_hash_y=0.025

    #this one can also be nR² when not in nR² mode
    delta_hash_x=0.025 if mode == 'nR²' else 0.005


    # try:
    hash_x = np.repeat(range_x - delta_hash_x, len(range_y)).reshape(len(range_x), len(range_y))
    hash_y = np.repeat(range_y - delta_hash_y, len(range_x)).reshape(len(range_y), len(range_x)).T

    # breakpoint()
    hash_1=ax_dist.pcolor(hash_x,
           hash_y,
           surface_SED_1[:-1, :-1],
           cmap='viridis',alpha=1.,hatch='///',lw=0,vmin=0,vmax=1,zorder=10)
    # except:
    #     breakpoint()

    # except:
    #     breakpoint()

    hash_1.set_facecolor('none')
    hash_1.set_edgecolor('lightgrey')

    hash_2=ax_dist.pcolor(hash_x,
           hash_y,
           surface_SED_2[:-1, :-1],
           cmap='viridis',alpha=1.,hatch='\\\\\\',lw=0,vmin=0,vmax=1,zorder=10)

    # hash_2=ax_dist.pcolor(np.repeat(range_x-delta_hash_x,len(range_y)).reshape(len(range_x),len(range_y)),
    #                np.repeat(range_y-delta_hash_y,len(range_x)).reshape(len(range_x),len(range_y)).T,
    #                surface_SED_2[:-1, :-1],
    #                cmap='viridis',alpha=1.,hatch='\\\\\\',lw=0,vmin=0,vmax=1,zorder=10)

    hash_2.set_facecolor('none')
    hash_2.set_edgecolor('lightgrey')

    cm_ticks = (np.linspace(np.nanmin(distance_arr), 0, 6, endpoint=True).tolist() if np.nanmin(distance_arr)<=0 else [-0.0001])+ \
               (np.linspace(0, np.nanmax(distance_arr), 6, endpoint=True).tolist() if np.nanmin(distance_arr) >= 0 else [0.0001])

    cm_norm = colors.TwoSlopeNorm(vcenter=0,
                                  vmin=np.nanmin(distance_arr) if np.nanmin(distance_arr)<0 else
                                  (-0.0001 if np.nanmax(distance_arr)>0 else -1),
                                  vmax=np.nanmax(distance_arr) if np.nanmax(distance_arr)>0 else
                                  (0.0001 if np.nanmin(distance_arr)<0 else 1))

    img=ax_dist.pcolormesh(range_x,range_y,distance_arr.T,cmap=cmap_bipolar,norm=cm_norm,zorder=100)

    if _ax is None:
        cb=fig_dist.colorbar(img,location='bottom', orientation='horizontal',spacing='proportional',
                         pad=0.16 if write_names else 0.16,
                         ticks=cm_ticks)
        #important to rescale the colorbar properly, otherwise both sides will always be 50/50
        # ('proportional' in the cb settings isn't really working for twoslopenorm)
        cb.ax.set_xscale('linear')

        if mode == 'nR²':
            cb.ax.set_title(r'$\Delta$lognR$^{2}$')
        if mode == 'v_turb':
            cb.ax.set_title(r'$\Delta$log$_{10}v_{turb}$')

        if mode == 'NH':
            cb.ax.set_title(r'$\Delta$log$_{10}NH$')

        if write_names:
            if mode=='nR²':
                plt.suptitle(r'log$_{10}$nR²$_{' + SED_1.replace('_','\_') + '}$' +
                             ' - log$_{10}$nR²$_{' + SED_2.replace('_','\_') + '}$')
            if mode=='v_turb':
                plt.suptitle(r'log$_{10}v_{turb,' + SED_1.replace('_','\_') + '}$' +
                             ' - log$_{10}v_{turb,' + SED_2.replace('_','\_') + '}$')

            if mode=='NH':
                plt.suptitle(r'log$_{10}NH_{' + SED_1.replace('_','\_') + '}$' +
                             ' - log$_{10}NH_{' + SED_2.replace('_','\_') + '}$')

        else:

            SEDs_arr=np.array(list(SEDs.keys()))

            color_SED_1=np.array(base_cmap)[np.argwhere(SEDs_arr==SED_1)[0]][0]
            color_SED_2=np.array(base_cmap)[np.argwhere(SEDs_arr==SED_2)[0]][0]

            #automatic text spacing
            n_delta=(len(color_SED_1)+len(color_SED_2)+3)/100*1.2

            plt.figtext(0.59, 0.24, '(', fontdict={'color':  'black','weight': 'normal','size': 10,})
            plt.figtext(0.6+len(color_SED_1)*0.1/100, 0.24,
                        color_SED_1.replace('powderblue','blue').replace('maroon','brown').replace('forestgreen','green'),
                        fontdict={'color':  color_SED_1,'weight': 'normal','size': 10,})
            plt.figtext(0.6+len(color_SED_1)*1.5/100, 0.24, '-', fontdict={'color':  'black','weight': 'normal','size': 10,})
            plt.figtext(0.6++len(color_SED_2)*0.1/100+len(color_SED_1)*1.5/100+0.015, 0.24,
                        color_SED_2.replace('powderblue','blue').replace('maroon','brown').replace('forestgreen','green'),
                        fontdict={'color':  color_SED_2,'weight': 'normal','size': 10,})
            plt.figtext(0.6+(len(color_SED_1)+len(color_SED_2))*1.5/100+0.02, 0.24, ')',
                        fontdict={'color':  'black','weight': 'normal','size': 10,})

    plt.xlim(range_x[0],range_x[-1])
    plt.ylim(range_y[0],range_y[-1])

    if _ax is None:
        return fig_dist
    else:
        return img

#finding all list of SED pairs:
SED_pairs = [(a, b) for idx, a in enumerate(list_SEDs_disp) for b in list_SEDs_disp[idx + 1:]]


def plot_distance_all(list_SEDs_disp,mode='nR²', write_names=False, interpolate_nr2=True):

    n_seds=len(list_SEDs_disp)
    # finding all list of SED pairs:
    SED_pairs = [(a, b) for idx, a in enumerate(list_SEDs_disp) for b in list_SEDs_disp[idx + 1:]]

    fig_corner, ax_corner = plt.subplots(n_seds, n_seds, figsize=(10, 8), sharey=True, sharex=True)
    fig_corner.subplots_adjust(wspace=0)
    fig_corner.subplots_adjust(hspace=0)

    plot_positions=[[a,b] for idx,a in enumerate(np.arange(n_seds)) for b in np.arange(n_seds)[idx+1:]]

    #inverting it for the axes placement
    plot_positions_draw=np.array(plot_positions.copy())
    plot_positions_draw.T[1]=n_seds-1-plot_positions_draw.T[1]
    plot_positions_draw.T[0]=n_seds-1-plot_positions_draw.T[0]

    plot_positions_draw=plot_positions_draw.tolist()

    img_list=[]
    glob_vmin=-0.0001
    glob_vmax=0.0001

    if mode=='nR²':

        fig_corner.text(0.5, 0.04, r'log$_{10}$(v$_{turb}$)', ha='center')
        fig_corner.text(0.04, 0.5, r'log$_{10}$(NH)', va='center', rotation='vertical')


    if mode=='v_turb':

        fig_corner.text(0.5, 0.04, r'log$_{10}$(nR$^2$)', ha='center')
        fig_corner.text(0.04, 0.5, r'log$_{10}$(NH)', va='center', rotation='vertical')


    if mode=='NH':
        fig_corner.text(0.5, 0.04, r'log$_{10}$(nR$^2$)', ha='center')
        fig_corner.text(0.04, 0.5, r'log$_{10}$(v_turb)', va='center', rotation='vertical')



    for i_x in range(n_seds):
        for i_y in range(n_seds):

            if [i_x,i_y] in plot_positions_draw:

                #a bit weird but works well
                combi_index_prel=np.argwhere(np.sum(np.array(plot_positions_draw)==[i_x,i_y],1)==2)[0][0]

                plot_positions_true=plot_positions[combi_index_prel]

                breakpoint()

                combi_index=np.argwhere(np.sum(np.array(plot_positions_draw)==plot_positions_true,1)==2)[0][0]

                breakpoint()

                SED_combi_1=SED_pairs[combi_index][0]
                SED_combi_2=SED_pairs[combi_index][1]

                img=plot_distance(ax_corner[i_x][i_y],SED_combi_1,SED_combi_2,mode=mode,write_names=write_names,
                              interpolate_nr2=interpolate_nr2,)

                if img.norm.vmin<glob_vmin:
                    glob_vmin=img.norm.vmin
                    glob.vmax=img.norm.vmax

                img_list+=[img]

                if i_x==0:

                    '''
                    Note: the x axis is labeled as the SED 2, the y axis as the SED 1
                    '''

                    SEDs_arr = np.array(list(SEDs.keys()))

                    color_SED_2 = np.array(base_cmap)[np.argwhere(SEDs_arr == SED_combi_2)[0]][0]

                    if write_names:
                        label_x_str=SED_combi_2.replace('_', '\_')
                    else:

                        label_x_str=color_SED_2.replace('powderblue', 'blue').replace('maroon', 'brown').replace(
                                        'forestgreen', 'green')

                    label_x=ax_corner[i_x][i_y].set_xlabel(label_x_str)

                    label_x.set_color(color_SED_2)


                if i_y==0:

                    '''
                    Note: the y axis is labeled as the SED 2, the y axis as the SED 1
                    '''

                    SEDs_arr = np.array(list(SEDs.keys()))

                    color_SED_1 = np.array(base_cmap)[np.argwhere(SEDs_arr == SED_combi_2)[0]][0]

                    if write_names:
                        label_y_str = SED_combi_1.replace('_', '\_')
                    else:

                        label_y_str = color_SED_1.replace('powderblue', 'blue').replace('maroon', 'brown').replace(
                            'forestgreen', 'green')

                    label_y = ax_corner[i_x][i_y].set_ylabel(label_y_str)

                    label_y.set_color(color_SED_1)

            else:

                ax_corner[i_x][i_y].remove()

    #climming everything to the extremal bounds
    for elem_img in img_list:
        elem_img.norm.vmin=glob_vmin
        elem_img.norm.vmax=glob_vmax

    #plotting the common elements
    #one global colorbar


    cm_ticks = np.linspace(glob_vmin,0, 6, endpoint=True).tolist() + \
               np.linspace(0, glob_vmax, 6, endpoint=True).tolist()


    cb=fig_corner.colorbar(img_list[0],location='bottom', orientation='horizontal',spacing='proportional',
                     pad=0.16 if write_names else 0.16,
                     ticks=cm_ticks)
    #important to rescale the colorbar properly, otherwise both sides will always be 50/50
    # ('proportional' in the cb settings isn't really working for twoslopenorm)
    cb.ax.set_xscale('linear')

    if mode == 'nR²':
        cb.ax.set_title(r'$\Delta$lognR$^{2}$')
    if mode == 'v_turb':
        cb.ax.set_title(r'$\Delta$log$_{10}v_{turb}$')

    if mode == 'NH':
        cb.ax.set_title(r'$\Delta$log$_{10}NH$')

    return fig_corner

if plot_distance_SEDs:

    if combine_plot_distance:
        plot_distance_glob=plot_distance_all(list_SEDs_disp,mode=plot_distance_dim,write_names=plot_distance_names,
                                             interpolate_nr2=interpolate_nr2)
        with tab_delta:
            st.pyplot(plot_distance_glob)
    else:

        with tab_delta:
            column_list = st.columns(len(list_SEDs_disp) - 1)

        for couple in SED_pairs:

            #note: the none is here for the ax argument that must be first to avoid hashing issues with caching
            plot_couple=plot_distance(None,couple[0],couple[1],mode=plot_distance_dim,write_names=plot_distance_names,
                                      interpolate_nr2=interpolate_nr2)

            with column_list[np.argwhere(np.array(list_SEDs_disp)==couple[0])[0][0]]:
                st.pyplot(plot_couple)

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
                                    plane_type='delauney',line_mode=False,show_legend=False):
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
                                  showlegend=show_legend and i_v_turb==0)]


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
                    args_x=np.array([np.argwhere(range_nh==elem)[0][0] for elem in x_tris])
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
            ids_x_grid=np.unique(np.array([np.argwhere(range_nh==elem)[0][0] for elem in x_tris]))
            if abs(ids_x_grid[1]-ids_x_grid[0])>1:
                continue

            ids_y_grid=np.unique(np.array([np.argwhere(range_v_turb==np.log10(elem))[0][0] for elem in y_tris]))
            if abs(ids_y_grid[1]-ids_y_grid[0])>1:
                continue
                
            # ids_z_grid=np.unique(np.array([np.argwhere(range_nh==elem)[0][0] for elem in x_tris]))
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

        if i_SED in [0,1,3,6]:
            alpha_higher=8+(2 if i_SED==3 else 0)
            alpha_lower=6
        else:
            alpha_higher=6
            alpha_lower=6

        alpha_higher=str(alpha_higher+(0 if under_sampling_nh==1 and under_sampling_v_turb==1 else 3+under_sampling_v_turb+under_sampling_nh))
        alpha_lower=str(alpha_lower+(0 if under_sampling_nh==1 and under_sampling_v_turb==1 else 3+under_sampling_v_turb+under_sampling_nh))

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
                                                plane_type='nearest',show_legend=True)
            shapes+=make_shape_triangles_planes(plane_lowest,color=color,volume_str=volume_str,legendgroup=legendgroup,
                                                plane_type='nearest')

        if plot_points:
            shapes += [go.Scatter3d(x=points_for_points.T[0], y=points_for_points.T[1], z=points_for_points.T[2],
                                    mode='markers',
                                       marker=dict(size=2, color=color,opacity=0.4 if single_mode else 1.),
                                    name=volume_str, legendgroup=legendgroup, legendgrouptitle={'text': legendgroup},
                                    showlegend=not draw_surface)]

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
def make_3D_figure(SEDs_disp, SEDs_surface, cmap, plot_points=False, under_sampling_v_turb='var',
                   under_sampling_nh='var', camera=None):
    '''
    Under sampling gives how much we divide in one axis to reduce the number of vertices
    '''

    # getting all the surfaces into an array
    mult_d_surfaces = []

    single_mode = len(SEDs_surface) == 1

    n_SEDs = len(SEDs_surface)

    additional_sampling = 0 + (1 if 'diagonal_upper_mid_highE_flux' in SEDs_surface else 0) \
                          + (1 if 'diagonal_upper_high_highE_flux' in SEDs_surface else 0)

    for i_SED, elem_SED in enumerate(list(SEDs.keys())):

        if under_sampling_v_turb == 'var':

            if i_SED in [2, 4]:
                under_sampling_v_turb = 3 if single_mode else 5 + n_SEDs // 2 + 2 * additional_sampling
            else:
                under_sampling_v_turb = 1 if single_mode else 2 + n_SEDs // 2 + additional_sampling

        if under_sampling_nh == 'var':
            if i_SED in [2, 4]:
                under_sampling_nh = 3 if single_mode else 5 + n_SEDs // 2 + 2 * additional_sampling
            else:
                under_sampling_nh = 1 + n_SEDs // 6 + additional_sampling

        if elem_SED not in SEDs_disp:
            continue

        valid_volumes = valid_volumes_dict[elem_SED]

        for i_vol in range(len(valid_volumes)):
            pos_v_turb_lower = np.array([np.argwhere(np.log10(elem) == range_v_turb)[0][0]
                                         for elem in valid_volumes[i_vol][0].T[1]])
            pos_v_turb_higher = np.array([np.argwhere(np.log10(elem) == range_v_turb)[0][0]
                                          for elem in valid_volumes[i_vol][1].T[1]])

            # combining undersampling and adding the first and last v_turbs in any case to avoid missing out the edes of the surface
            mask_under_sampling_v_turb_lower = (pos_v_turb_lower % under_sampling_v_turb == 0) | (
                        pos_v_turb_lower == max(pos_v_turb_lower)) \
                                               | (pos_v_turb_lower == min(pos_v_turb_lower))

            mask_under_sampling_v_turb_higher = (pos_v_turb_higher % under_sampling_v_turb == 0) | (
                        pos_v_turb_higher == max(pos_v_turb_higher)) \
                                                | (pos_v_turb_higher == min(pos_v_turb_higher))

            pos_nh_lower = np.array([np.argwhere(elem == range_nh)[0][0]
                                     for elem in valid_volumes[i_vol][0].T[0]])
            pos_nh_higher = np.array([np.argwhere(elem == range_nh)[0][0]
                                      for elem in valid_volumes[i_vol][1].T[0]])

            # combining undersampling and adding the first and last nhs in any case to avoid missing out the edes of the surface
            mask_under_sampling_nh_lower = (pos_nh_lower % under_sampling_nh == 0) | (pos_nh_lower == max(pos_nh_lower)) \
                                           | (pos_nh_lower == min(pos_nh_lower))

            mask_under_sampling_nh_higher = (pos_nh_higher % under_sampling_nh == 0) | (
                        pos_nh_higher == max(pos_nh_higher)) \
                                            | (pos_nh_higher == min(pos_nh_higher))

            mask_under_sampling_lower = mask_under_sampling_v_turb_lower & mask_under_sampling_nh_lower
            mask_under_sampling_higher = mask_under_sampling_v_turb_higher & mask_under_sampling_nh_higher

            valid_volumes_under_sampled = [valid_volumes[i_vol][0][mask_under_sampling_lower],
                                           valid_volumes[i_vol][1][mask_under_sampling_higher]]

            mult_d_surfaces += plot_3d_surface(valid_volumes_under_sampled, color=cmap[i_SED], volume_number=i_vol + 1,
                                               legendgroup=elem_SED, i_SED=i_SED, draw_surface=elem_SED in SEDs_surface,
                                               full_planes=valid_volumes[i_vol],
                                               under_sampling_v_turb=under_sampling_v_turb,
                                               under_sampling_nh=under_sampling_nh, plot_points=plot_points,
                                               single_mode=single_mode)

    fig = go.Figure(data=mult_d_surfaces)

    # Layout settings with custom axis names
    fig.update_layout(
        scene=dict(
            xaxis_title='nH',  # Custom X-axis label
            yaxis_title='v_turb',  # Custom Y-axis label
            zaxis_title='log10(nR²)',  # Custom Z-axis label
            aspectmode="cube",
            yaxis=dict(type='log'),
            # camera=camera,

        ),
        height=1000
    )

    if setup_camera:
        fig.update_layout(scene_camera=camera,scene_dragmode='orbit')
    else:
        fig.update_layout(scene_camera=camera)

    # fig_3D.show()
    # fig_3D.show()

    #
    with tab_3D:
        st.plotly_chart(fig, use_container_width=True, theme=None)

    return fig

if plot_3D:
    fig_3D=make_3D_figure(list_SEDs_disp, list_SEDs_surface, cmap=base_cmap, plot_points=plot_points, camera=camera)

    if not online:
        with tab_3D:

            savedir='/home/parrama/Documents/Work/PhD/docs/papers/wind_4U/global/ion_visu/save_figs'
            def save_3dfig_local():

                '''
                # Saves the current maxi_graphs in a svg (i.e. with clickable points) format.
                '''

                if fig_3D is not None:
                    fig_3D.write_image(savedir+'/'+'fig3D_'+str(round(time.time()))+'.pdf',engine="kaleido",
                                       width=1000, height=1000)

            st.button('Save current 3D figure',on_click=save_3dfig_local,key='save_3dfig_local')


            with open(savedir+'/fig_3D.pkl', 'wb+') as f:
                dill.dump(fig_3D,f)
        #
        # this doesn't work
        # setup_default_3Dcam=st.toggle('Setup default camera position')

        # @st.cache_data
        # def setup_cam(setup):
        #
        # if setup_default_3Dcam:
        #
        # # if setup and 'fig_shown' not in st.session_state:
        #
        #     #requires ipywidgets installed
        #     f = go.FigureWidget(fig_3D)
        #
        #     breakpoint()
                # fig_3D.show()
                # st.session_state['fig_shown']=True
        #
        #
        # if setup_default_3Dcam:
        #     setup_cam(setup_default_3Dcam)
        #
        # def show_cam_pos():
        #     with tab_3D:
        #         st.text(fig_3D.layout['scene']['camera'])
        #
        # save_default_3Dcam=st.button('show camera position',on_click=show_cam_pos,key='show_cam_pos')

