import numpy as np
import matplotlib.pyplot as plt
import os

sudeb_file='/media/parrama/SSD/Simu/MHD/Sudeb/Sudeb_test_highres/temp_ion_fraction_details.dat'

maxime_file='/media/parrama/SSD/Simu/MHD/Sudeb/Sudeb_test_highres/xstar_output_details.dat'

maxime_summary_parfile='/media/parrama/SSD/Simu/MHD/Sudeb/Sudeb_test_highres/xstar_pars.log'

sudeb=np.loadtxt(sudeb_file).T

sudeb_rin=sudeb[2]
sudeb_rout=sudeb[3]

sudeb_xi_in=sudeb[7]
sudeb_xi_out=sudeb[8]
sudeb_n=sudeb[9]
sudeb_vobsx=sudeb[10]
sudeb_nh=sudeb[11]
sudeb_t=sudeb[12]

sudeb_nFe26=sudeb[31]
sudeb_nhFe26=sudeb[51]

max=np.loadtxt(maxime_file).T
max_summary=np.loadtxt(maxime_summary_parfile,dtype='str').T

max_r=max[2]
max_xi=max[5]

max_summary_r=np.sqrt(max_summary[3].astype(float)*1e38/(max_summary[5].astype(float)*10**(max_summary[7].astype(float))))
max_n=max_summary[5].astype(float)
max_nh=max_summary[6].astype(float)

max_v=max[7]
max_nh=max[8]
max_t=max[10]

max_nFe26=max[28]
max_nhFe26=max[48]


# #logxi
# plt.gridspec()
# plt.xscale('log')
# plt.yscale('log')
# #logxi
# plt.gridspec()
# plt.xscale('log')
# plt.yscale('log')

#density

#column density