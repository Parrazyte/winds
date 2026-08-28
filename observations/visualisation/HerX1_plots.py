import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


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

    # #wind vs inclination plot
    # breakpoint()
    #
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
                       color=indiv_color,label='innermost node' if i_node==0 else 'outermost node' if i_node==n_nodes-1 else '',
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

    plt.plot(min_angle_her_x,min_angle_her_y,color='black',lw=1.0,ls=':',label='precession limit')
    plt.plot(max_angle_her_x,max_angle_her_y,color='black',lw=1.0,ls=':',label='')

    plt.xlim(90,45)


    ax_NH.legend(title='Her X-1 wind origin')
    plt.tight_layout()