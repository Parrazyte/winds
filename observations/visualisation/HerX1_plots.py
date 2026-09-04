import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

def plot_los(cmap='plasma'):


    '''
    projection of the Leahy & Frost 2025 model for Her X-1 accross angles
    '''

    #first: Fig. 5 count rates digitized by ChatGPT
    BAT_rate_clean= np.array([
        [0.001, 0.0585, 0.0019, 0.0020],
        [0.036, 0.0560, 0.0016, 0.0016],
        [0.052, 0.0502, 0.0015, 0.0016],
        [0.081, 0.0435, 0.0014, 0.0015],
        [0.091, 0.0410, 0.0015, 0.0015],
        [0.101, 0.0340, 0.0015, 0.0015],
        [0.113, 0.0265, 0.0015, 0.0015],
        [0.123, 0.0245, 0.0012, 0.0013],
        [0.140, 0.0200, 0.0010, 0.0011],
        [0.153, 0.0175, 0.0010, 0.0010],
        [0.161, 0.0144, 0.0010, 0.0010],
        [0.172, 0.0114, 0.0009, 0.0009],
        [0.181, 0.0090, 0.0009, 0.0009],
        [0.191, 0.0068, 0.0007, 0.0008],
        [0.201, 0.0058, 0.0007, 0.0007],
        [0.220, 0.0040, 0.0006, 0.0006],
        [0.245, 0.0028, 0.0006, 0.0006],
        [0.269, 0.0020, 0.0006, 0.0006],
        [0.299, 0.0014, 0.0005, 0.0005],
        [0.330, 0.0018, 0.0006, 0.0006],
        [0.345, 0.0015, 0.0005, 0.0005],
        [0.358, 0.0011, 0.0005, 0.0005],
        [0.375, 0.0008, 0.0005, 0.0005],
        [0.390, 0.0019, 0.0006, 0.0006],
        [0.409, 0.0014, 0.0005, 0.0005],
        [0.425, 0.0039, 0.0006, 0.0006],
        [0.457, 0.0071, 0.0007, 0.0008],
        [0.467, 0.0098, 0.0008, 0.0009],
        [0.500, 0.0119, 0.0009, 0.0009],
        [0.511, 0.0110, 0.0009, 0.0010],
        [0.523, 0.0108, 0.0009, 0.0009],
        [0.540, 0.0094, 0.0008, 0.0008],
        [0.555, 0.0090, 0.0008, 0.0008],
        [0.580, 0.0072, 0.0008, 0.0008],
        [0.603, 0.0082, 0.0009, 0.0009],
        [0.616, 0.0062, 0.0007, 0.0007],
        [0.638, 0.0046, 0.0006, 0.0006],
        [0.649, 0.0041, 0.0006, 0.0006],
        [0.674, 0.0033, 0.0006, 0.0006],
        [0.700, 0.0022, 0.0005, 0.0005],
        [0.718, 0.0018, 0.0005, 0.0005],
        [0.744, 0.0007, 0.0005, 0.0005],
        [0.764, 0.0027, 0.0007, 0.0008],
        [0.790, 0.0019, 0.0006, 0.0006],
        [0.810, 0.0016, 0.0005, 0.0005],
        [0.829, 0.0015, 0.0005, 0.0005],
        [0.842, 0.0020, 0.0006, 0.0006],
        [0.853, 0.0031, 0.0007, 0.0007],
        [0.862, 0.0045, 0.0008, 0.0008],
        [0.870, 0.0118, 0.0010, 0.0010],
        [0.880, 0.0185, 0.0012, 0.0013],
        [0.887, 0.0320, 0.0016, 0.0017],
        [0.900, 0.0360, 0.0015, 0.0016],
        [0.910, 0.0440, 0.0016, 0.0017],
        [0.920, 0.0460, 0.0015, 0.0016],
        [0.930, 0.0505, 0.0015, 0.0016],
        [0.943, 0.0550, 0.0016, 0.0017],
        [0.951, 0.0560, 0.0015, 0.0016],
        [0.963, 0.0580, 0.0015, 0.0016],
        [0.978, 0.0605, 0.0019, 0.0020],
    ])

    BAT_phase = BAT_rate_clean[:, 0]
    BAT_count_rate = BAT_rate_clean[:, 1]
    BAT_yerr_lower = BAT_rate_clean[:, 2]
    BAT_yerr_upper = BAT_rate_clean[:, 3]

    MAXI_rate_clean= np.array([
    [0.001, 0.385, 0.006, 0.006],
    [0.033, 0.360, 0.007, 0.007],
    [0.044, 0.380, 0.014, 0.014],
    [0.053, 0.335, 0.014, 0.014],
    [0.083, 0.277, 0.012, 0.013],
    [0.103, 0.230, 0.013, 0.013],
    [0.122, 0.198, 0.011, 0.011],
    [0.134, 0.178, 0.010, 0.010],
    [0.151, 0.157, 0.009, 0.009],
    [0.166, 0.136, 0.008, 0.008],
    [0.181, 0.091, 0.007, 0.007],
    [0.195, 0.073, 0.006, 0.006],
    [0.205, 0.058, 0.006, 0.006],
    [0.224, 0.030, 0.006, 0.006],
    [0.246, 0.015, 0.005, 0.005],
    [0.258, 0.010, 0.005, 0.005],
    [0.278, 0.019, 0.006, 0.006],
    [0.285, 0.016, 0.005, 0.005],
    [0.301, 0.012, 0.005, 0.005],
    [0.321, 0.013, 0.005, 0.005],
    [0.340, 0.011, 0.005, 0.005],
    [0.359, 0.012, 0.005, 0.005],
    [0.375, 0.011, 0.005, 0.005],
    [0.391, 0.015, 0.005, 0.005],
    [0.405, 0.016, 0.006, 0.006],
    [0.420, 0.020, 0.006, 0.006],
    [0.458, 0.063, 0.008, 0.008],
    [0.468, 0.071, 0.009, 0.009],
    [0.481, 0.078, 0.009, 0.010],
    [0.497, 0.086, 0.009, 0.010],
    [0.511, 0.090, 0.009, 0.010],
    [0.526, 0.082, 0.009, 0.009],
    [0.544, 0.072, 0.008, 0.009],
    [0.560, 0.074, 0.008, 0.008],
    [0.575, 0.062, 0.007, 0.008],
    [0.592, 0.052, 0.007, 0.007],
    [0.610, 0.052, 0.007, 0.007],
    [0.627, 0.043, 0.006, 0.006],
    [0.647, 0.031, 0.006, 0.006],
    [0.665, 0.024, 0.006, 0.006],
    [0.683, 0.017, 0.006, 0.006],
    [0.701, 0.013, 0.005, 0.005],
    [0.720, 0.010, 0.005, 0.005],
    [0.742, 0.017, 0.006, 0.006],
    [0.764, 0.023, 0.007, 0.007],
    [0.789, 0.021, 0.006, 0.007],
    [0.810, 0.013, 0.006, 0.006],
    [0.827, 0.016, 0.006, 0.006],
    [0.843, 0.020, 0.006, 0.006],
    [0.858, 0.040, 0.008, 0.008],
    [0.870, 0.048, 0.009, 0.009],
    [0.885, 0.142, 0.010, 0.011],
    [0.899, 0.208, 0.012, 0.012],
    [0.913, 0.252, 0.012, 0.013],
    [0.929, 0.317, 0.013, 0.014],
    [0.940, 0.332, 0.014, 0.014],
    [0.951, 0.371, 0.016, 0.016],
    [0.978, 0.378, 0.014, 0.014],
    ])

    MAXI_phase = MAXI_rate_clean[:, 0]
    MAXI_count_rate = MAXI_rate_clean[:, 1]
    MAXI_yerr_lower = MAXI_rate_clean[:, 2]
    MAXI_yerr_upper = MAXI_rate_clean[:, 3]

    #from Kosec et al. 2026 table 1 https://doi.org/10.3847/1538-4357/ae680a
    #post eclipses 1 and 2 removed for clarity
    data_wind_XRISM = np.array([
        [0.00270, 0.00156, 16.98, 0.086, 0.075, 17.92, 0.05, 0.04, -260, 30, 20, 220, 20, 30],
        [0.00673, 0.00148, 16.87, 0.10, 0.087, 18.00, 0.04, 0.04, -300, 20, 20, 220, 20, 20],
        [0.01022, 0.00126, 16.94, 0.096, 0.083, 17.91, 0.04, 0.04, -260, 30, 30, 230, 30, 30],
        [0.01275, 0.00128, 16.92, 0.096, 0.084, 17.90, 0.05, 0.06, -240, 20, 20, 150, 30, 30],
        [0.01928, 0.00257, 16.84, 0.11, 0.093, 17.88, 0.04, 0.04, -200, 30, 30, 250, 30, 30],
        [0.02421, 0.00158, 16.72, 0.14, 0.11, 17.84, 0.04, 0.04, -270, 20, 20, 230, 30, 30],
        [0.02812, 0.00158, 16.68, 0.15, 0.12, 17.72, 0.05, 0.04, -340, 40, 30, 240, 40, 40],

        #[0.04470, 0.00258, 16.88, 0.071, 0.063, 17.86, 0.03, 0.03, -300, 20, 20, 250, 20, 20],
        [0.05253, 0.00451, 16.02, 0.47, 0.12, 17.55, 0.04, 0.04, -510, 20, 20, 210, 20, 20],
        [0.06229, 0.00451, 16.10, 0.46, 0.22, 17.62, 0.03, 0.03, -390, 30, 30, 280, 30, 30],
        [0.07299, 0.00544, 16.28, 0.30, 0.19, 17.51, 0.05, 0.05, -530, 70, 110, 380, 70, 110],

        #[0.09498, 0.00307, 16.38, 0.25, 0.16, 17.65, 0.04, 0.04, -380, 40, 40, 290, 40, 50],
        [0.10326, 0.00447, 16.35, 0.20, 0.13, 17.41, 0.04, 0.04, -590, 50, 40, 270, 40, 50],
        [0.11294, 0.00435, 16.05, 0.81, 0.26, 17.39, 0.06, 0.09, -620, 90, 80, 390, 80, 260],
    ])


    columns = [
        "phase",
        "phase_err",
        "FeXXV",
        "FeXXV_err_minus",
        "FeXXV_err_plus",
        "FeXXVI",
        "FeXXVI_err_minus",
        "FeXXVI_err_plus",
        "outflow_velocity",
        "outflow_err_minus",
        "outflow_err_plus",
        "velocity_width",
        "width_err_minus",
        "width_err_plus",
    ]

    #note: phase 0 for main high
    phase_precess=np.arange(0,2.0001,1e-3)

    fig,ax=plt.subplots(1,figsize=(10,8))
    plt.xlabel('Precession phase [35 days] with 0=main high')
    plt.ylabel('angle from orbital plane (°)')
    plt.xlim(0.,2.)

    #averaged values

    # system inclination
    i_plane=85.12

    #inner node tilt
    tilt_inner=23.25

    #outer node tilt
    tilt_outer=20.5

    #relative twist difference between inner twist and outer twist (in degrees)
    twist_inner=75.5
    twist_outer=24

    #phase shift for initial evolution
    phase_shift=0.208

    #creating few linearly spaced nodes
    n_nodes=9
    tilt_node_arr = np.linspace(tilt_inner, tilt_outer, n_nodes)
    twist_node_arr=np.linspace(twist_inner,twist_outer,n_nodes)


    color_cmap = getattr(mpl.cm,cmap)
    c_norm= mpl.colors.Normalize(vmin=0,vmax=n_nodes)
    colors_func = mpl.cm.ScalarMappable(norm=c_norm, cmap=color_cmap)

    plt.axhline(90-i_plane,color='black')

    def node_angle(phase,node_tilt,node_twist,phase_shift):
        return node_tilt*np.sin(phase*2*np.pi-np.pi/2
                                                      #twist shift
                                                      +(node_twist/360)*2*np.pi
                                                      #global phase shift
                                                      -phase_shift*2*np.pi)

    for i_node,(elem_tilt_node,elem_twist_node) in enumerate(zip(tilt_node_arr,twist_node_arr)):

        indiv_color = colors_func.to_rgba(i_node)

        plt.plot(phase_precess,#precession
                               node_angle(phase_precess,elem_tilt_node,elem_twist_node,phase_shift),
                 color=indiv_color,label='innermost node' if i_node==0 else 'outermost node' if i_node==n_nodes-1 else '',alpha=0.5)

    #overplotting MAXI and BAT
    plt.errorbar(MAXI_phase.tolist()+(MAXI_phase+1.).tolist(),
                 np.array(90-i_plane+MAXI_count_rate/MAXI_count_rate.max()*(20-(90-i_plane))).tolist()*2,
                 yerr=np.array([MAXI_yerr_lower.tolist()*2,MAXI_yerr_upper.tolist()*2])/MAXI_count_rate.max()*(20-(90-i_plane)),
                 color='brown',lw=0.5,label='MAXI data')
    
    plt.errorbar(BAT_phase.tolist()+(BAT_phase+1.).tolist(),
                 np.array(90-i_plane+BAT_count_rate/BAT_count_rate.max()*(20-(90-i_plane))).tolist()*2,
                 yerr=np.array([BAT_yerr_lower.tolist()*2,BAT_yerr_upper.tolist()*2])/BAT_count_rate.max()*(20-(90-i_plane)),
                 color='green',lw=0.5,label='BAT data')

    # plt.errorbar(1+MAXI_phase,90-i_plane+MAXI_count_rate/MAXI_count_rate.max()*(20-(90-i_plane)),
    #              yerr=np.array([MAXI_yerr_lower,MAXI_yerr_upper])/MAXI_count_rate.max()*(20-(90-i_plane)),
    #              color='cyan',lw=0.5,label='')
    # plt.errorbar(BAT_phase,90-i_plane+BAT_count_rate/BAT_count_rate.max()*(20-(90-i_plane)),
    #              yerr=np.array([BAT_yerr_lower,BAT_yerr_upper])/BAT_count_rate.max()*(20-(90-i_plane)),
    #              color='green',lw=0.5,label='BAT data')
    # plt.errorbar(1+BAT_phase,90-i_plane+BAT_count_rate/BAT_count_rate.max()*(20-(90-i_plane)),
    #              yerr=np.array([BAT_yerr_lower,BAT_yerr_upper])/BAT_count_rate.max()*(20-(90-i_plane)),
    #              color='green',lw=0.5,label='')

    #offsets between the main high phase and the main rise phase
    #approximated from the Leahy and Frost 25
    def func_shift(x):
        return x+0.12
    def func_shift_inv(x):
        return x-0.12

    ax_top=ax.secondary_xaxis(location='top',functions=(func_shift,func_shift_inv))
    ax_top.set_xlabel('Precession phase [35 days] with 0=main rise (taken as main high-0.12)')

    plt.legend()

    ax_wind=plt.twinx(ax)
    ax_wind.set_ylabel('FeXXVI wind column density (log10 cm$^{-2}$)')
    #doubling compared to the actual range
    ax_wind.set_xlim(0.,2.)
    ax_wind.set_ylim(17.3,18.8)
    ax_wind.set_yticks(np.arange(17.3,18.05,0.1))

    ax_wind.errorbar(data_wind_XRISM.T[0]-0.12,y=data_wind_XRISM.T[5],
                     xerr=data_wind_XRISM.T[1],yerr=data_wind_XRISM.T[6:8],ls='',label='XRISM data',color='black')
    ax_wind.errorbar(data_wind_XRISM.T[0]-0.12+1,y=data_wind_XRISM.T[5],
                     xerr=data_wind_XRISM.T[1],yerr=data_wind_XRISM.T[6:8],ls='',label='',color='black')
    ax_wind.errorbar(data_wind_XRISM.T[0]-0.12+2,y=data_wind_XRISM.T[5],
                     xerr=data_wind_XRISM.T[1],yerr=data_wind_XRISM.T[6:8],ls='',label='',color='black')

    ax_wind.legend()
    plt.tight_layout()

    '''
    wind vs inclination plot
    '''

    #from Tomaru20 H1743 Fig. 5 top panel (aka small disk at 0.3 L_Edd)
    thermal_h1743_nh_xxvi= np.array([
        [32.393765, 1.00000000e+16],
        [33.237916, 1.09939795e+16],
        [34.063617, 1.20321267e+16],
        [34.872053, 1.31246295e+16],
        [35.664414, 1.42786691e+16],
        [36.441892, 1.55027098e+16],
        [37.205282, 1.67839856e+16],
        [37.955574, 1.81238072e+16],
        [38.693566, 1.95265005e+16],
        [39.419844, 2.10017565e+16],
        [40.135014, 2.25503626e+16],
        [40.839863, 2.41694890e+16],
        [41.534794, 2.58672864e+16],
        [42.220395, 2.76444710e+16],
        [42.897066, 2.95143581e+16],
        [43.565208, 3.14704542e+16],
        [44.225408, 3.35091813e+16],
        [44.877869, 3.56365334e+16],
        [45.522993, 3.78566605e+16],
        [46.160968, 4.01791581e+16],
        [46.792200, 4.26034563e+16],
        [47.417077, 4.51291163e+16],
        [48.035809, 4.77559765e+16],
        [48.648583, 5.05111401e+16],
        [49.255605, 5.33926746e+16],
        [49.857263, 5.63928269e+16],
        [50.453570, 5.95131640e+16],
        [51.044714, 6.27628023e+16],
        [51.631100, 6.61576093e+16],
        [52.212917, 6.96950254e+16],
        [52.789976, 7.33798801e+16],
        [53.362863, 7.72061643e+16],
        [53.931586, 8.11758917e+16],
        [54.495941, 8.53064681e+16],
        [55.056733, 8.95944291e+16],
        [55.613549, 9.40482603e+16],
        [56.166788, 9.86653538e+16],
        [56.716259, 1.03462807e+17],
        [57.262562, 1.08480432e+17],
        [57.805482, 1.13702042e+17],
        [58.345234, 1.19141137e+17],
        [58.881816, 1.24802446e+17],
        [59.415410, 1.30692897e+17],
        [59.946227, 1.36858614e+17],
        [60.474072, 1.43291972e+17],
        [60.999338, 1.49966723e+17],
        [61.521827, 1.56911031e+17],
        [62.041926, 1.64130198e+17],
        [62.559453, 1.71664165e+17],
        [63.074597, 1.79511090e+17],
        [63.587563, 1.87693742e+17],
        [64.098139, 1.96193719e+17],
        [64.606535, 2.05045291e+17],
        [65.112754, 2.14339681e+17],
        [65.616983, 2.24096399e+17],
        [66.119230, 2.34297146e+17],
        [66.619684, 2.44977210e+17],
        [67.118164, 2.56149364e+17],
        [67.614851, 2.67890737e+17],
        [68.109753, 2.80210223e+17],
        [68.603066, 2.93155813e+17],
        [69.094594, 3.06749224e+17],
        [69.584738, 3.21057819e+17],
        [70.073089, 3.36101875e+17],
        [70.560260, 3.51994033e+17],
        [71.045835, 3.68772348e+17],
        [71.530026, 3.86530811e+17],
        [72.012825, 4.05407823e+17],
        [72.494436, 4.25465861e+17],
        [72.974648, 4.46915537e+17],
        [73.453876, 4.69828144e+17],
        [73.931713, 4.94376761e+17],
        [74.408559, 5.20747356e+17],
        [74.884217, 5.49249046e+17],
        [75.359073, 5.80193944e+17],
        [75.832749, 6.13903173e+17],
        [76.305427, 6.50865014e+17],
        [76.777310, 6.91413375e+17],
        [77.248203, 7.36354611e+17],
        [77.718301, 7.86353452e+17],
        [78.187408, 8.42668846e+17],
        [78.655918, 9.04227798e+17],
        [79.123633, 9.66724805e+17],
        [79.590758, 1.03006245e+18],
        [80.057090, 1.09603557e+18],
        [80.522815, 1.16656548e+18],
        [80.987959, 1.24332491e+18],
        [81.452497, 1.32631937e+18],
        [81.916446, 1.41528437e+18],
        [82.379993, 1.51251999e+18],
        [82.843155, 1.61797905e+18],
        [83.305711, 1.73110736e+18],
        [83.767867, 1.84966461e+18],
        [84.229637, 1.96445686e+18],
        [84.690998, 2.06307647e+18],
        [85.152170, 2.14436433e+18],
        [85.613138, 2.22578082e+18],
        [86.073713, 2.32832607e+18],
        [86.534083, 2.48052141e+18],
        [86.994264, 2.70181125e+18],
        [87.454249, 3.08493436e+18],
        [87.697725, 3.29574344e+18],
        [87.724911, 3.31669218e+18],
        [87.751901, 3.34197800e+18],
        [87.778490, 3.38019926e+18],
        [87.804685, 3.41962541e+18],
        [87.830677, 3.46006830e+18],
        [87.856471, 3.50135017e+18],
        [87.881880, 3.54319301e+18],
        [87.907077, 3.58619099e+18],
        [87.931881, 3.63148144e+18],
        [87.956488, 3.67786774e+18],
        [87.980899, 3.72575198e+18],
        [88.004908, 3.77533509e+18],
        [88.028917, 3.82697611e+18],
        [88.052329, 3.88097855e+18],
        [88.075749, 3.93718132e+18],
        [88.098767, 3.99638768e+18],
        [88.121589, 4.05805172e+18],
        [88.144214, 4.12359326e+18],
        [88.166439, 4.19292240e+18],
        [88.188458, 4.26592610e+18],
        [88.210289, 4.34584573e+18],
        [88.231916, 4.42968972e+18],
        [88.253354, 4.52102369e+18],
    ]).T

    #from Tomaru20 GX 13+1 in arxiv files (unreleased in full paper nor arxiv preprint)
    # includes both R_disk = 10 R_IC (second column) and R_disk=10 R_IC (third column), both at 0.5 L_Edd
    #we remove the points at the edge of the disk to avoid showing weird things
    thermal_gx13_nh_xxvi=np.array([
    [  5.09312, 1.00004885e+16, 1.00004885e+16],
    [  8.27334, 1.39239947e+16, 1.38265026e+16],
    [ 10.78858, 1.92813640e+16, 1.90931700e+16],
    [ 12.80493, 2.37539847e+16, 2.34240831e+16],
    [ 14.54287, 2.73665456e+16, 2.68447516e+16],
    [ 16.09586, 3.08273599e+16, 3.00613959e+16],
    [ 17.51391, 3.43826906e+16, 3.33247594e+16],
    [ 18.82778, 3.80486603e+16, 3.66159919e+16],
    [ 20.05830, 4.18344581e+16, 3.99424898e+16],
    [ 21.22037, 4.57649627e+16, 4.33170922e+16],
    [ 22.32429, 4.98547232e+16, 4.67797543e+16],
    [ 23.37860, 5.41052000e+16, 5.03246490e+16],
    [ 24.38985, 5.85704240e+16, 5.39899477e+16],
    [ 25.36300, 6.32228692e+16, 5.77390481e+16],
    [ 26.30241, 6.81160822e+16, 6.15506455e+16],
    [ 27.21166, 7.32555139e+16, 6.54874288e+16],
    [ 28.09373, 7.85750255e+16, 6.95346679e+16],
    [ 28.95099, 8.41679499e+16, 7.36538100e+16],
    [ 29.78543, 9.00381841e+16, 7.79425374e+16],
    [ 30.59943, 9.61555865e+16, 8.23872128e+16],
    [ 31.39418, 1.02576127e+17, 8.69599174e+16],
    [ 32.17127, 1.09323230e+17, 9.16839919e+16],
    [ 32.93189, 1.16445562e+17, 9.65469443e+16],
    [ 33.67723, 1.23840516e+17, 1.01537799e+17],
    [ 34.40848, 1.31646372e+17, 1.06743410e+17],
    [ 35.12624, 1.39861802e+17, 1.12190756e+17],
    [ 35.83130, 1.48451205e+17, 1.17829937e+17],
    [ 36.52464, 1.57357010e+17, 1.23599352e+17],
    [ 37.20688, 1.66695603e+17, 1.29606505e+17],
    [ 37.87841, 1.76495122e+17, 1.35958038e+17],
    [ 38.54020, 1.86749379e+17, 1.42519550e+17],
    [ 39.19228, 1.97414825e+17, 1.49334094e+17],
    [ 39.83562, 2.08630279e+17, 1.56452137e+17],
    [ 40.47022, 2.20357276e+17, 1.63849526e+17],
    [ 41.09670, 2.32616123e+17, 1.71565448e+17],
    [ 41.71564, 2.45372373e+17, 1.79615504e+17],
    [ 42.32703, 2.58659934e+17, 1.88012690e+17],
    [ 42.93127, 2.72550912e+17, 1.96766411e+17],
    [ 43.52878, 2.87123696e+17, 2.05902751e+17],
    [ 44.11973, 3.02291452e+17, 2.15437087e+17],
    [ 44.70453, 3.18105548e+17, 2.25421887e+17],
    [ 45.28338, 3.34651755e+17, 2.35864887e+17],
    [ 45.85648, 3.51944086e+17, 2.46761356e+17],
    [ 46.42401, 3.69979670e+17, 2.58182366e+17],
    [ 46.98640, 3.88821159e+17, 2.70181307e+17],
    [ 47.54362, 4.08422794e+17, 2.82778103e+17],
    [ 48.09588, 4.28908421e+17, 2.95974159e+17],
    [ 48.64337, 4.50265822e+17, 3.09754655e+17],
    [ 49.18630, 4.72638855e+17, 3.24229509e+17],
    [ 49.72487, 4.95972338e+17, 3.39511680e+17],
    [ 50.25907, 5.20394708e+17, 3.55593434e+17],
    [ 50.78930, 5.45819696e+17, 3.72482487e+17],
    [ 51.31556, 5.72405800e+17, 3.90260698e+17],
    [ 51.83786, 6.00225761e+17, 4.09028643e+17],
    [ 52.35658, 6.29359517e+17, 4.28855972e+17],
    [ 52.87173, 6.60095071e+17, 4.49854441e+17],
    [ 53.38330, 6.92374166e+17, 4.72159226e+17],
    [ 53.89151, 7.26216446e+17, 4.95811562e+17],
    [ 54.39634, 7.61945153e+17, 5.20976080e+17],
    [ 54.89820, 7.99447571e+17, 5.47562920e+17],
    [ 55.39688, 8.38864062e+17, 5.75880184e+17],
    [ 55.89258, 8.80582020e+17, 6.05760757e+17],
    [ 56.38530, 9.24542941e+17, 6.37734893e+17],
    [ 56.87545, 9.70541159e+17, 6.71042435e+17],
    [ 57.36281, 1.01752542e+18, 7.05001199e+17],
    [ 57.84741, 1.06609148e+18, 7.39205408e+17],
    [ 58.32942, 1.11609091e+18, 7.74565791e+17],
    [ 58.80905, 1.16800871e+18, 8.10514154e+17],
    [ 59.28630, 1.22117480e+18, 8.48079298e+17],
    [ 59.76116, 1.27585420e+18, 8.88323289e+17],
    [ 60.23365, 1.33368573e+18, 9.30570665e+17],
    [ 60.70415, 1.39538417e+18, 9.75362485e+17],
    [ 61.17227, 1.45993854e+18, 1.02318232e+18],
    [ 61.63841, 1.52713710e+18, 1.07362998e+18],
    [ 62.10236, 1.59713705e+18, 1.12773242e+18],
    [ 62.56453, 1.67065158e+18, 1.18485028e+18],
    [ 63.02471, 1.74808078e+18, 1.24496233e+18],
    [ 63.48290, 1.82857949e+18, 1.30854902e+18],
    [ 63.93932, 1.91286239e+18, 1.37613620e+18],
    [ 64.39414, 2.00139659e+18, 1.44830377e+18],
    [ 64.84697, 2.09317821e+18, 1.52397681e+18],
    [ 65.29823, 2.18921242e+18, 1.60279032e+18],
    [ 65.74790, 2.28979199e+18, 1.68533745e+18],
    [ 66.19597, 2.39402005e+18, 1.77231538e+18],
    [ 66.64246, 2.50309493e+18, 1.86419772e+18],
    [ 67.08757, 2.61607675e+18, 1.96072531e+18],
    [ 67.53108, 2.73338074e+18, 2.06246105e+18],
    [ 67.97340, 2.85501881e+18, 2.16956191e+18],
    [ 68.41413, 2.98236843e+18, 2.28204135e+18],
    [ 68.85347, 3.11476955e+18, 2.39961914e+18],
    [ 69.29163, 3.25324470e+18, 2.52325491e+18],
    [ 69.72860, 3.39463652e+18, 2.65310080e+18],
    [ 70.16418, 3.54469084e+18, 2.78871947e+18],
    [ 70.59876, 3.70062442e+18, 2.93055871e+18],
    [ 71.03195, 3.86169519e+18, 3.07835995e+18],
    [ 71.46415, 4.02846572e+18, 3.23289456e+18],
    [ 71.89516, 4.20022072e+18, 3.39401484e+18],
    [ 72.32539, 4.37680833e+18, 3.56142851e+18],
    [ 72.75421, 4.55942985e+18, 3.73436958e+18],
    [ 73.18224, 4.74909851e+18, 3.91292551e+18],
    [ 73.60929, 4.94434214e+18, 4.09885308e+18],
    [ 74.03554, 5.14489615e+18, 4.29300716e+18],
    [ 74.46060, 5.35086956e+18, 4.49562396e+18],
    [ 74.88506, 5.55977681e+18, 4.70494198e+18],
    [ 75.30853, 5.77168368e+18, 4.91640771e+18],
    [ 75.73120, 5.98583208e+18, 5.13143476e+18],
    [ 76.15309, 6.20288226e+18, 5.35151787e+18],
    [ 76.57418, 6.42336582e+18, 5.57696562e+18],
    [ 76.99448, 6.64655435e+18, 5.80766128e+18],
    [ 77.41417, 6.87121822e+18, 6.04213023e+18],
    [ 77.83308, 7.09338215e+18, 6.27598683e+18],
    [ 78.25160, 7.31619699e+18, 6.51386868e+18],
    [ 78.66912, 7.53558808e+18, 6.75376803e+18],
    [ 79.08624, 7.75274678e+18, 6.99411546e+18],
    [ 79.50276, 7.96660561e+18, 7.23552611e+18],
    [ 79.91869, 8.17507046e+18, 7.47402069e+18],
    [ 80.33402, 8.37571177e+18, 7.70972602e+18],
    [ 80.74896, 8.56978060e+18, 7.94431070e+18],
    [ 81.16330, 8.75234112e+18, 8.17308051e+18],
    [ 81.57725, 8.92826717e+18, 8.40262916e+18],
    [ 81.86102, 9.03826667e+18, 8.54840835e+18],
    [ 82.01322, 9.08149715e+18, 8.61095151e+18],
    [ 82.16325, 9.13885487e+18, 8.68911697e+18],
    [ 82.31108, 9.19058731e+18, 8.76371898e+18],
    [ 82.45634, 9.23642697e+18, 8.83304601e+18],
    [ 82.59941, 9.28099062e+18, 8.90129101e+18],
    [ 82.74031, 9.32368486e+18, 8.96696681e+18],
    [ 82.87901, 9.36352855e+18, 9.02964427e+18],
    [ 83.01534, 9.40087387e+18, 9.09017423e+18],
    [ 83.14968, 9.43491145e+18, 9.14739424e+18],
    [ 83.28185, 9.46831287e+18, 9.20422575e+18],
    [ 83.41203, 9.49835260e+18, 9.25727072e+18],
    [ 83.54001, 9.52578354e+18, 9.30816973e+18],
    [ 83.66604, 9.55154415e+18, 9.35763457e+18],
    [ 83.79006, 9.57426436e+18, 9.40507236e+18],
    [ 83.91209, 9.59469708e+18, 9.45063783e+18],
    [ 84.03235, 9.61440229e+18, 9.49584610e+18],
    [ 84.15062, 9.63218638e+18, 9.53874739e+18],
    [ 84.26690, 9.64647464e+18, 9.57815347e+18],
    [ 84.38140, 9.65863028e+18, 9.61557265e+18],
    [ 84.49392, 9.67157687e+18, 9.65333021e+18],
    [ 84.60485, 9.68101048e+18, 9.68769773e+18],
    [ 84.71399, 9.68848574e+18, 9.72061170e+18],
    [ 84.82135, 9.69576824e+18, 9.75363195e+18],
    [ 84.92692, 9.70010394e+18, 9.78378663e+18],
    [ 85.03090, 9.70167653e+18, 9.81104294e+18],
    [ 85.13329, 9.70226701e+18, 9.83717211e+18],
    [ 85.23390, 9.70108057e+18, 9.86177767e+18],
    [ 85.33292, 9.69832750e+18, 9.88563505e+18],
    [ 85.43035, 9.69478663e+18, 9.90894705e+18],
    [ 85.52621, 9.68199620e+18, 9.93151183e+18],
    [ 85.62046, 9.66588483e+18, 9.95109368e+18],
    [ 85.71333, 9.64314466e+18, 9.96889381e+18],
    [ 85.80481, 9.62319444e+18, 9.98672579e+18],
    [ 85.89450, 9.61713701e+18, 1.00027631e+19],
    [ 85.98301, 9.61205652e+18, 1.00184160e+19],
    [ 86.06993, 9.60659076e+18, 1.00322557e+19],
    [ 86.15565, 9.60074037e+18, 1.00442804e+19],
    [ 86.23980, 9.59626349e+18, 1.00569373e+19],
    [ 86.32274, 9.58983027e+18, 1.00673595e+19],
    [ 86.40430, 9.58262735e+18, 1.00771734e+19],
    [ 86.48447, 9.57717832e+18, 1.00882305e+19],
    [ 86.56325, 9.57212439e+18, 1.01001213e+19],
    [ 86.64104, 9.56668677e+18, 1.01114049e+19],
    [ 86.71723, 9.56105646e+18, 1.01222981e+19],
    [ 86.79245, 9.55658722e+18, 1.01327880e+19],
    [ 86.86647, 9.55289704e+18, 1.01422503e+19],
    [ 86.93910, 9.55096285e+18, 1.01513231e+19],
    [ 87.01073, 9.54863256e+18, 1.01579192e+19],
    [ 87.08118, 9.54650375e+18, 1.01630801e+19],
    [ 87.15043, 9.54572729e+18, 1.01692731e+19],
    [ 87.21850, 9.54611280e+18, 1.01760893e+19],
    [ 87.28557, 9.54766585e+18, 1.01845666e+19],
    [ 87.35145, 9.55057171e+18, 1.01951211e+19],
    [ 87.41634, 9.55542403e+18, 1.02071378e+19],
    [ 87.48024, 9.56066491e+18, 1.02189537e+19],
    [ 87.54295, 9.56766085e+18, 1.02316212e+19],
    [ 87.60486, 9.57387227e+18, 1.02426380e+19],
    [ 87.66558, 9.57970628e+18, 1.02507568e+19],
    [ 87.72532, 9.58769228e+18, 1.02580477e+19],
    [ 87.78425, 9.60074037e+18, 1.02665875e+19],
    [ 87.84200, 9.61908460e+18, 1.02774316e+19],
    [ 87.89895, 9.64020515e+18, 1.02885036e+19],
    # [ 87.95490, 9.66353200e+18, 1.02991599e+19],
    # [ 88.01007, 9.68710814e+18, 1.03087835e+19],
    # [ 88.06425, 9.71310613e+18, 1.03198893e+19],
    # [ 88.11762, 9.73779462e+18, 1.03272293e+19],
    # [ 88.17002, 9.76135761e+18, 1.03261779e+19],
    # [ 88.22161, 9.78894675e+18, 1.03031383e+19],
    # [ 88.27241, 9.81741795e+18, 1.02547108e+19],
    # [ 88.32222, 9.84437038e+18, 9.84516553e+18],
    # [ 88.37143, 9.86818564e+18, 9.87339589e+18],
    # [ 88.41965, 9.89125958e+18, 9.90029927e+18],
    # [ 88.46728, 9.90734086e+18, 9.91821164e+18],
    # [ 88.51392, 9.91800855e+18, 9.92869336e+18],
    # [ 88.55996, 9.92828676e+18, 9.93736594e+18],
    # [ 88.60520, 9.92405227e+18, 9.93131411e+18],
    # [ 88.64965, 9.91015890e+18, 9.91498525e+18],
    # [ 88.69350, 9.88483100e+18, 9.88784511e+18],
    # [ 88.73657, 9.84736655e+18, 9.84676724e+18],
    # [ 88.77903, 9.79511239e+18, 9.79133568e+18],
    # [ 88.82070, 9.72396842e+18, 9.71665922e+18],
    # [ 88.86158, 9.64627711e+18, 9.63531526e+18],
    # [ 88.90206, 9.55406535e+18, 9.54165596e+18],
    # [ 88.94175, 9.45236352e+18, 9.43874932e+18],
    # [ 88.98064, 9.35269649e+18, 9.34187697e+18],
    # [ 89.01914, 9.25689687e+18, 9.25182251e+18],
    # [ 89.05685, 9.18126603e+18, 9.18051927e+18],
    # [ 89.09396, 9.14126713e+18, 9.14330558e+18],
    # [ 89.13067, 9.13829348e+18, 9.14423655e+18],
    # [ 89.16659, 9.19525674e+18, 9.20235168e+18],
    # [ 89.20191, 9.30892687e+18, 9.31403256e+18],
    # [ 89.23663, 9.47773690e+18, 9.47985577e+18],
    # [ 89.27096, 9.67727777e+18, 9.67786676e+18],
    # [ 89.30469, 9.88643916e+18, 9.88684405e+18],
    # [ 89.33784, 1.00520362e+19, 1.00518360e+19],
    # [ 89.37038, 1.01052471e+19, 1.01044309e+19],
    # [ 89.40232, 9.94785659e+18, 9.94785659e+18],
    # [ 89.43388, 9.51804401e+18, 9.51804401e+18],
    # [ 89.46484, 8.79598994e+18, 8.79598994e+18],
    # [ 89.49539, 7.84173075e+18, 7.84157464e+18],
    # [ 89.52536, 6.78704054e+18, 6.78690157e+18],
    # [ 89.55493, 5.73558313e+18, 5.73558313e+18],
    # [ 89.58410, 4.74745911e+18, 4.74765084e+18],
    # [ 89.61248, 3.84580754e+18, 3.84604161e+18],
    # [ 89.64066, 3.07037018e+18, 3.07043305e+18],
    # [ 89.66824, 2.42656915e+18, 2.42642285e+18],
    # [ 89.69543, 1.88693669e+18, 1.88674459e+18],
    # [ 89.72221, 1.43804616e+18, 1.43781306e+18],
    # [ 89.74860, 1.07336864e+18, 1.07332468e+18],
    # [ 89.77440, 7.84682489e+17, 7.84379492e+17],
    # [ 89.80001, 5.59102561e+17, 5.59613529e+17],
    # [ 89.82501, 3.92231063e+17, 3.92262967e+17],
    # [ 89.84961, 2.67734365e+17, 2.67625655e+17],
    # [ 89.87382, 1.80320802e+17, 1.80331879e+17],
    # [ 89.89764, 1.19146027e+17, 1.19146027e+17],
    # [ 89.92105, 7.47674383e+16, 7.47674383e+16],
    # [ 89.94428, 4.10601315e+16, 4.10601315e+16],
    # [ 89.96015, 1.00004885e+16, 1.00004885e+16],
    ]).T

    fig_NH,ax_NH=plt.subplots(1,figsize=(10,8))

    ax_NH.set_yscale('linear')
    ax_NH.set_xscale('linear')
    ax_NH.set_xlim(90,45)

    ax_NH.set_ylabel('FeXXVI wind column density (log10 cm$^{-2}$)')
    ax_NH.set_xlabel('Line of sight inclination angle (°, disk at 90°)')

    min_angle_her_x=[]
    max_angle_her_x=[]
    min_angle_her_y=[]
    max_angle_her_y=[]

    for i_node,(elem_tilt_node,elem_twist_node) in enumerate(zip(tilt_node_arr,twist_node_arr)):

        indiv_color = colors_func.to_rgba(i_node)

        elem_angle_sampl=-node_angle(data_wind_XRISM.T[0]-0.12,elem_tilt_node,elem_twist_node,phase_shift)

        #note: we take the negative to get the angle of the disk w.r.t. the LoS (and not the opposite)
        #these ones are compared to the orbital plane, we correct after for the LoS angle
        elem_angle_sampl_aft=-node_angle(data_wind_XRISM.T[0]-0.12+data_wind_XRISM.T[1],
                                         elem_tilt_node,elem_twist_node,phase_shift)
        elem_angle_sampl_bef=-node_angle(data_wind_XRISM.T[0]-0.12-data_wind_XRISM.T[1],
                                         elem_tilt_node,elem_twist_node,phase_shift)

        elem_angle_sampl_bounds=np.array([elem_angle_sampl_bef,elem_angle_sampl_aft]).T

        elem_angle_sampl_rawerr=(elem_angle_sampl_bounds.T - elem_angle_sampl).T

        #messy error computation to ensure that we get the right errors even if we switch angle gradient regime
        elem_angle_sampl_err=[]
        for i_dat in range(len(elem_angle_sampl)):
            elem_angle_sampl_err+=[[abs(min((elem_angle_sampl_rawerr)[i_dat].min(),0)),
                                   abs(max((elem_angle_sampl_rawerr)[i_dat].max(),0))]]
        elem_angle_sampl_err=np.array(elem_angle_sampl_err)

        #switching to the los instead of the orbital plane
        elem_angle_sampl_los=elem_angle_sampl+(90-i_plane)


        ax_NH.errorbar(90-elem_angle_sampl_los,y=data_wind_XRISM.T[5],
                       xerr=elem_angle_sampl_err.T,
                       yerr=data_wind_XRISM.T[6:8],ls='',
                       color=indiv_color,label='innermost node data' if i_node==0 else
                                                'outermost node data' if i_node==n_nodes-1 else '',
                       alpha=0.5)

        '''
        computing the innermost and outermst possible angles for the source in a non-obscured state
        depending on the node
        note: we take the max phase as 0.16 to match the intensity of the turn-on phase defined as -0.12
        according to the MAXI lightcurve
        in this way we probe the full possible range of inclination values no matter their evolution
        note: we compute this in the LoS inclination frame so the "min" angle is the closest to 90°, and the "max the furthest
        '''

        min_angle_node=90-max(min(-node_angle(np.arange(-0.12,0.1601,1e-3),elem_tilt_node,elem_twist_node,phase_shift)+(90-i_plane)),0)
        max_angle_node=90-max(max(-node_angle(np.arange(-0.12,0.1601,1e-3),elem_tilt_node,elem_twist_node,phase_shift)+(90-i_plane)),0)

        plt.xlim(min_angle_node,max_angle_node)
        #adding the regression plot - for now not done with ci or uncertainties, but can be easily implemented with lmplot_uncert
        sns.regplot(x=90-elem_angle_sampl_los, y=data_wind_XRISM.T[5], ax=ax_NH, truncate=False, ci=None,color=indiv_color,
                    line_kws=dict(ls='--',alpha=0.5),marker="None")

        #fetching the bounds of the regression to highlight them later
        node_regress_line_coord=[elem for elem in ax_NH.get_children() if type(elem)==mpl.lines.Line2D][-1]
        min_angle_her_x+=[node_regress_line_coord._xy[0][0]]
        min_angle_her_y+=[node_regress_line_coord._xy[0][1]]
        max_angle_her_x+=[node_regress_line_coord._xy[-1][0]]
        max_angle_her_y+=[node_regress_line_coord._xy[-1][1]]

        #replotting the regression plot in dotted form beyond the Her X-1 testability
        plt.xlim(90,min_angle_node)
        sns.regplot(x=90-elem_angle_sampl_los, y=data_wind_XRISM.T[5], ax=ax_NH, truncate=False, ci=None,color=indiv_color,
                    line_kws=dict(ls=':',alpha=0.5),marker="None")

        plt.xlim(max_angle_node,45)
        sns.regplot(x=90-elem_angle_sampl_los, y=data_wind_XRISM.T[5], ax=ax_NH, truncate=False, ci=None,color=indiv_color,
                    line_kws=dict(ls=':',alpha=0.5),marker="None")

    plt.plot(min_angle_her_x,min_angle_her_y,color='black',lw=1.0,ls='dashdot',
             label='main high precession angle limits')
    plt.plot(max_angle_her_x,max_angle_her_y,color='black',lw=1.0,ls='dashdot',label='')

    plt.plot([],[],color='black',lw=2.0,ls='--',label='extrapolation within precession range')
    plt.plot([],[],color='black',lw=2.0,ls=':',label='extrapolation outside precession range')

    plt.xlim(90,45)

    #note: outer disk radius 2e11cm from  Cheng+98
    #R_IC estimated at 8e10cm in Kosec+20 (careful they quote the wind launching radius which they take as 0.1 Ric)

    ax_NH.legend(title='Her X-1: 0.14 L$_{Edd}$ 1.6 M$_{\odot}$| R$_d$=2.5R$_{IC}$')
    plt.tight_layout()

    #launching mechanisms
    ax_launching=ax_NH.twinx()
    ax_launching.yaxis.set_visible(False)
    ax_launching.set_xscale(ax_NH.get_xscale())
    ax_launching.set_yscale(ax_NH.get_yscale())
    ax_launching.set_ylim(ax_NH.get_ylim())
    ax_launching.set_xlim(ax_NH.get_xlim())

    ax_launching.plot(thermal_gx13_nh_xxvi[0],np.log10(thermal_gx13_nh_xxvi[1]),color="red",alpha=0.5,
                      ls='-',
             label=r'0.5 L$_{Edd}$ | 1.4 M$_{\odot}$ | R$_d$=10   $\;$R$_{IC}$ | R$_{is}$=0.20 R$_{IC}$')

    ax_launching.plot(thermal_gx13_nh_xxvi[0],np.log10(thermal_gx13_nh_xxvi[2]),color="red",alpha=0.5,
                      ls='--',
             label=r'0.5 L$_{Edd}$ | 1.4 M$_{\odot}$ | R$_d$=1   $\;$$\;$ R$_{IC}$ | R$_{is}$=0.20 R$_{IC}$')

    ax_launching.plot(thermal_h1743_nh_xxvi[0],np.log10(thermal_h1743_nh_xxvi[1]),color="red",alpha=0.5,
                      ls='dashdot',
             label=r'0.3 L$_{Edd}$ | 8.0 M$_{\odot}$ | R$_d$=0.18 R$_{IC}$ | R$_{is}$=0.18 R$_{IC}$')

    csv_mhd_0p1Ledd=pd.read_csv('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/mhd/'+
                            'KeigoTanimoto/Solutions/0p1_Ledd/structure/monaco_Nion.csv')

    #restricting to the first value of each density jump to prepare the interpolation
    n_val=[csv_mhd_0p1Ledd['Fe01_Nion'].values.tolist().count(elem)\
           for elem in np.unique(csv_mhd_0p1Ledd['Fe01_Nion'])]
    val_nodup = np.unique(csv_mhd_0p1Ledd['Fe01_Nion'].values)[np.array(n_val) == 1]
    val_dens_dupdrop=csv_mhd_0p1Ledd['Fe01_Nion'].drop_duplicates(keep='last')

    #finishing with the densities at the first angles where the MHD solution was computed
    val_dens_startdval=val_dens_dupdrop[np.array([val_dens_dupdrop != elem for elem in val_nodup]).sum(0) == 3]

    csv_mhd_Nion_ok=csv_mhd_0p1Ledd.iloc[val_dens_startdval.index]

    #rescaling the angles to groups of 10 which is what the initial MHD computation was created with
    #(the density spreads approximately equally accross an even angle range accross each value)
    csv_MHD_angl=np.floor(csv_mhd_Nion_ok['angle']/10)*10


    #note: computing T_IC and R_IC on the deabsorbed canonical soft state SED gives:
    #T_IC=1.06keV
    #R_IC=5.19e5 Rg
    #means for the MHD simulations, R_disk=0.48 R_IC

    ax_launching.plot(csv_MHD_angl,np.log10(csv_mhd_Nion_ok['Fe01_Nion']),
                      color='dodgerblue',alpha=1.0,
                      label='0.1 L$_{Edd}$ | 8.0 M$_{\odot}$ | R$_d$=0.48 R$_{IC}$ | n$_0$=1.7 $\cdot$10$^{18}$ | p=1.2')

    ax_launching.legend(title=r'$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$',loc='lower left',
                        framealpha=1.0)

    ax_launching.axhline(np.log10(1.735e16), color='grey', alpha=0.5, )
    ax_launching.text(68,16.13, '3$\sigma$ NewAthena limit for Her X-1 main high in 50ks',
                      color='grey',alpha=1)


    #modifies the vertical alignment for saves (which move the text slightly
    v_hm=0.09

    plt.text(87, 16.325-v_hm, 'launching mechanisms:', color='black', zorder=10)

    x_hm=1.0

    plt.text(77.5+x_hm, 16.325-v_hm, 'thermal', color='red', zorder=10)

    plt.text(74.7+x_hm, 16.325-v_hm, '/', color='black', zorder=10)


    plt.text(74.3+x_hm, 16.325-v_hm, 'MHD', color='dodgerblue', zorder=10,)

    #for 2/3 of main high peak.


