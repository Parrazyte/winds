import numpy as np

from visual_line_tools import load_catalogs,obj_values,abslines_values,values_manip,distrib_graph,correl_graph,\
    n_infos, plot_lightcurve, hid_graph, sources_det_dic, dippers_list,telescope_list,load_integral

from fitting_tools import lines_std,lines_std_names,range_absline
from dist_mass_tools import dist_mass

catal_blackcat, catal_watchdog, catal_blackcat_obj, catal_watchdog_obj,\
    catal_maxi_df, catal_maxi_simbad, catal_bat_df, catal_bat_simbad = load_catalogs()

lineval_files=['/media/parrama/SSD/Observ/BHLMXB/NICER/time-resolved/100s/4U1630-47/03-23/bigbatch/lineplots_opt/line_values_4_10_0.05_0.01_10_500.txt']


dict_linevis = {
    'ctl_blackcat': catal_blackcat,
    'ctl_blackcat_obj': catal_blackcat_obj,
    'ctl_watchdog': catal_watchdog,
    'ctl_watchdog_obj': catal_watchdog_obj,
    'lineval_files': lineval_files,
    'obj_list': ['4U1630-47'],
    'cameras': 'all',
    'expmodes': 'all',
    'multi_obj': False,
    'range_absline': range_absline,
    'n_infos': n_infos,
    'args_cam': 'all',
    'args_line_search_e': '4 10 0.05',
    'args_line_search_norm': '0.01 10 500',
    'visual_line': True,

}

dist_obj_list, mass_obj_list = dist_mass(dict_linevis)

# distance factor for the flux conversion later on
dist_factor = 4 * np.pi * (dist_obj_list * 1e3 * 3.086e18) ** 2

# L_Edd unit factor
Edd_factor = dist_factor / (1.26e38 * mass_obj_list)

# Reading the results files
observ_list, lineval_list, lum_list, date_list, instru_list, exptime_list = obj_values(lineval_files, Edd_factor,
                                                                                       dict_linevis)
