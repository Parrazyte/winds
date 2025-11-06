#imgs
from XRISM.XRISM_datared_tools import *
#
# mpdaf_plot_img('xa901002010xtd_p031300010_cl_SFP_img_sky_0.3_10.ds',
#                rad_crop=[124,124],target_name_list=[None,None],crop_coords=[266.424,-29.0094],
#                target_coords_list=[('17:45:40.50', '-29:00:48.10'),['17:45:35.6500', '-29:01:34.888']],
#                target_sizes_pix=[1,1],target_colors=['black','white'],
#                target_names=['MAXI J1744-294','AX J1745.6-2901'],target_names_offset=[5,5],save=False)
#
# mpdaf_plot_img('xa901002010xtd_p031300010_cl_SFP_img_sky_7_10.ds',
#                rad_crop=[124,124],target_name_list=[None,None],crop_coords=[266.424,-29.0094],
#                target_coords_list=[('17:45:40.50', '-29:00:48.10'),['17:45:35.6500', '-29:01:34.888']],
#                target_sizes_pix=[1,1],target_colors=['black','white'],
#                target_names=['MAXI J1744-294','AX J1745.6-2901'],target_names_offset=[5,5],save=False)
#
# mpdaf_plot_img('xa901002010rsl_p0px1000_cl_RTS_img_sky_2_10.ds',
#                rad_crop=[124,124],target_name_list=[None,'AXJ1745.6-2901'],crop_coords=[266.424,-29.0094],
#                target_coords_list=[('17:45:40.476', '-29:00:46.10'),None],
#                target_sizes_pix=[1,1],target_colors=['black','white'],
#                target_names=['MAXI J1744-294','auto'],target_names_offset=[5,5],save=True)
#
# mpdaf_plot_img('xa300044010rsl_p0px1000_cl_RTS_img_sky_2_10.ds',
#                rad_crop=[124,124],target_name_list=[None,'AXJ1745.6-2901'],crop_coords=[266.422,-28.9977],
#                target_coords_list=[('17:45:40.476', '-29:00:46.10'),None],
#                target_sizes_pix=[1,1],target_colors=['black','white'],
#                target_names=['MAXI J1744-294','auto'],target_names_offset=[5,5],save=True)
#
# mpdaf_plot_img('xa901002010rsl_p0px1000_cl_RTS_img_sky_7_10.ds',
#                rad_crop=[124,124],target_name_list=[None,'AXJ1745.6-2901'],crop_coords=[266.422,-28.9977],
#                target_coords_list=[('17:45:40.476', '-29:00:46.10'),None],
#                target_sizes_pix=[1,1],target_colors=['black','white'],
#                target_names=['MAXI J1744-294','auto'],target_names_offset=[5,5],save=True)
#
# #slight offet to avoid pixel offset in xtend
#
#
# # cd /media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1744-294/new_anal/fit_duo
# xPlot('eeuf',mult_factors=[1,5],group_names=['MAXI J1744-294','AX J1745.6-2901'],force_ylog_ratio=True,ylims=[5e-2,2])
#
# plt.savefig('testfig.pdf')




# center of cropped Sgr A east image (important for computations on Sag A east arf
# ('17:45:43.1813', '-29:00:22.924')
#
# MAXI J1744-294:
# ('17:45:40.476', '-29:00:46.10')
#
# AXJ1745.6-2901:
# ['17 45 35.6400', '-29 01 33.888']



#arf_compa plots


#in DDT
# os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1744-294/new_anal/901002010_reprocorr/compa_arf/pix_NS')
# os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1744-294/new_anal/901002010_reprocorr/compa_arf/pix_smallpix')


def arf_compa_plots(dir='./',arf_num_list=[],arf_denom_list=[],label_list=[],label_num='',label_denom='',save=True,skip_ncols=0):

    os.chdir(dir)

    colors_list=['black','red','purple','orange','green','blue','pink'][skip_ncols:]
    # pixNS_rsp_arf_caldb11_1745 = fits.open(
    #     'xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_POINT_1745_gfinc_donotuse_CALDB11.arf')[
    #     1].data
    # pixNS_rsp_arf_caldb12_1745 = fits.open(
    #     'xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_POINT_1745_gfinc_donotuse_CALDB12.arf')[
    #     1].data
    # pixNS_rsp_arf_caldb11_1744 = fits.open(
    #     'xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_POINT_1744_gfinc_donotuse_CALDB11.arf')[
    #     1].data
    # pixNS_rsp_arf_caldb12_1744 = fits.open(
    #     'xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_POINT_1744_gfinc_donotuse_CALDB12.arf')[
    #     1].data
    # pixNS_rsp_arf_caldb11_diffuse = fits.open(
    #     'xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_FLATCIRCLE_fr3_GCXE_gfinc_donotuse_CALDB11.arf')[
    #     1].data
    # pixNS_rsp_arf_caldb12_diffuse = fits.open(
    #     'xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_FLATCIRCLE_fr3_GCXE_gfinc_donotuse_CALDB12.arf')[
    #     1].data
    # pixNS_rsp_arf_caldb11_SgrAEast = fits.open(
    #     'xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_IMAGE_img_Chandra_data_6.6-6.8_flux_smoothed_trim_SgrAEast_gfinc_donotuse_CALDB11.arf')[
    #     1].data
    # pixNS_rsp_arf_caldb12_SgrAEast = fits.open(
    #     'xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_IMAGE_img_Chandra_data_6.6-6.8_flux_smoothed_trim_SgrAEast_gfinc_donotuse_CALDB12.arf')[
    #     1].data
    
    arf_num_data=[fits.open(elem)[1].data for elem in arf_num_list]
    arf_denom_data=[fits.open(elem)[1].data for elem in arf_denom_list]


    fig_compa, ax_compa = plt.subplots(figsize=(8, 6), layout='constrained')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Effective area (cm²)')
    plt.xlim(2., 10)
    plt.yscale('log')
    plt.ylim(2e-1, 2e2)
    
    for col,elem_arf_num,elem_label in zip(colors_list,arf_num_data,label_list):
        plt.plot(elem_arf_num['ENERG_LO'],elem_arf_num['SPECRESP'], color=col,label='',ls='--')

    for col,elem_arf_denom,elem_label in zip(colors_list,arf_denom_data,label_list):
        plt.plot(elem_arf_denom['ENERG_LO'],elem_arf_denom['SPECRESP'], color=col,label=elem_label)
        
    # for elem_arf in enumerate(arf_denom_data):
    #
    #     plt.plot(pixNS_rsp_arf_caldb11_1745['ENERG_LO'], pixNS_rsp_arf_caldb11_1745['SPECRESP'], color='red',
    #              label='AX J1745')
    #     plt.plot(pixNS_rsp_arf_caldb12_1745['ENERG_LO'], pixNS_rsp_arf_caldb12_1745['SPECRESP'], color='red', ls='--')
    #     plt.plot(pixNS_rsp_arf_caldb11_1744['ENERG_LO'], pixNS_rsp_arf_caldb11_1744['SPECRESP'], color='black',
    #              label='MAXI J1744')
    #     plt.plot(pixNS_rsp_arf_caldb12_1744['ENERG_LO'], pixNS_rsp_arf_caldb12_1744['SPECRESP'], color='black', ls='--')
    #     plt.plot(pixNS_rsp_arf_caldb11_diffuse['ENERG_LO'], pixNS_rsp_arf_caldb11_diffuse['SPECRESP'], color='purple',
    #              label='flat arf')
    #     plt.plot(pixNS_rsp_arf_caldb12_diffuse['ENERG_LO'], pixNS_rsp_arf_caldb12_diffuse['SPECRESP'], color='purple',
    #              ls='--')
    #     plt.plot(pixNS_rsp_arf_caldb11_SgrAEast['ENERG_LO'], pixNS_rsp_arf_caldb11_SgrAEast['SPECRESP'], color='orange',
    #              label='Sgr A East')
    #     plt.plot(pixNS_rsp_arf_caldb12_SgrAEast['ENERG_LO'], pixNS_rsp_arf_caldb12_SgrAEast['SPECRESP'], color='orange',
    #              ls='--')
    plt.legend(loc='lower left')
    twin_compa = plt.twinx()
    twin_compa.xaxis.set_visible(False)
    twin_compa.yaxis.set_visible(False)
    plt.plot([], [], color='gray', ls='-', label=label_denom)
    plt.plot([], [], color='gray', ls='--', label=label_num)
    plt.legend(loc='upper left')

    if save:
        plt.savefig('arf_compa_CALDB11_12_rsp.pdf')
    
    fig_ratio, ax_ratio = plt.subplots(figsize=(8, 6), layout='constrained')
    plt.xlabel('Energy (keV)')
    plt.ylabel(label_num+'/'+label_denom+' Effective area ratio')
    plt.xlim(2., 10)
    from matplotlib.ticker import AutoMinorLocator
    plt.ylim(0.5, 1.5)
    
    ax_ratio.yaxis.set_minor_locator(AutoMinorLocator())

    for col,elem_arf_num,elem_arf_denom,elem_label in zip(colors_list,arf_num_data,arf_denom_data,label_list):
        plt.plot(elem_arf_num['ENERG_LO'],elem_arf_num['SPECRESP']/elem_arf_denom['SPECRESP'], color=col,label=elem_label)

    # plt.plot(pixNS_rsp_arf_caldb11_1745['ENERG_LO'],
    #          pixNS_rsp_arf_caldb12_1745['SPECRESP'] / pixNS_rsp_arf_caldb11_1745['SPECRESP'], color='red',
    #          label='AX J1745')
    # plt.plot(pixNS_rsp_arf_caldb11_1744['ENERG_LO'],
    #          pixNS_rsp_arf_caldb12_1744['SPECRESP'] / pixNS_rsp_arf_caldb11_1744['SPECRESP'], color='black',
    #          label='MAXI J1744')
    # plt.plot(pixNS_rsp_arf_caldb11_diffuse['ENERG_LO'],
    #          pixNS_rsp_arf_caldb12_diffuse['SPECRESP'] / pixNS_rsp_arf_caldb11_diffuse['SPECRESP'], color='purple',
    #          label='flat arf')
    # plt.plot(pixNS_rsp_arf_caldb11_SgrAEast['ENERG_LO'],
    #          pixNS_rsp_arf_caldb12_SgrAEast['SPECRESP'] / pixNS_rsp_arf_caldb11_SgrAEast['SPECRESP'], color='orange',
    #          label='Sgr A East')
    plt.legend()
    for i in np.arange(0.5, 1.5, 0.05):
        plt.axhline(y=i, color='gray', alpha=0.1)

    if save:
        plt.savefig('arf_ratio_CALDB11_12_rsp.pdf')


#####################
#IN DDT 36 35 compa
#for bigpix 36 35 compa
# arf_compa_plots(arf_denom_list=['xa901002010rsl_p0px1000_ghf_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_POINT_1744.arf','xa901002010rsl_p0px1000_ghf_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_POINT_AXJ1745.arf','xa901002010rsl_p0px1000_ghf_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_FLATCIRCLE_fr3.arf',
# 'xa901002010rsl_p0px1000_ghf_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_IMAGE_img_chandra_SgrAeast_FeKa25_SgrAEast.arf'],arf_num_list=['xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_POINT_1744_gfinc_donotuse_CALDB11.arf','xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_POINT_1745_gfinc_donotuse_CALDB11.arf','xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_FLATCIRCLE_fr3_GCXE_gfinc_donotuse_CALDB11.arf',
# 'xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_IMAGE_img_Chandra_data_6.6-6.8_flux_smoothed_trim_SgrAEast_gfinc_donotuse_CALDB11.arf'],label_list=['MAXI J1744','AX J1745','flat arf','Sgr A East'],label_num='rsp (6.36)',label_denom='arf (6.35)')

#NS
# arf_compa_plots(arf_denom_list=['xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_gti_MAN_COMBISTEADY_comb_POINT_MAXIJ1744.arf','xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_gti_MAN_COMBISTEADY_comb_POINT_AXJ1745.arf','xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_gti_MAN_COMBISTEADY_comb_FLATCIRCLE_fr3.arf',                         'xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_gti_MAN_COMBISTEADY_comb_IMAGE_img_chandra_SgrAeast_FeKa25_SgrAEast.arf'],arf_num_list=['xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_POINT_1744_gfinc_donotuse_CALDB11.arf','xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_POINT_1745_gfinc_donotuse_CALDB11.arf','xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_FLATCIRCLE_fr3_GCXE_gfinc_donotuse_CALDB11.arf','xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_IMAGE_img_Chandra_data_6.6-6.8_flux_smoothed_trim_SgrAEast_gfinc_donotuse_CALDB11.arf'],label_list=['MAXI J1744','AX J1745','flat arf','Sgr A East'],label_num='rsp (6.36)',label_denom='arf (6.35)')

#smallpix
# arf_compa_plots(arf_denom_list=['xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_POINT_1744.arf','xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_POINT_AXJ1745.arf','xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_FLATCIRCLE_fr3.arf',                         'xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_IMAGE_img_chandra_SgrAeast_FeKa25_SgrAEast.arf'],arf_num_list=['xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_POINT_1744_gfinc_donotuse_CALDB11.arf','xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_POINT_1745_gfinc_donotuse_CALDB11.arf','xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_FLATCIRCLE_fr3_GCXE_gfinc_donotuse_CALDB11.arf',
#                 'xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_0_05_60000_IMAGE_img_Chandra_data_6.6-6.8_flux_smoothed_trim_SgrAEast_gfinc_donotuse_CALDB11.arf'],label_list=['MAXI J1744','AX J1745','flat arf','Sgr A East'],
#                 label_num='rsp (6.36)',label_denom='arf (6.35)')

#in PV phase
    #bigpix
 # arf_compa_plots(arf_denom_list=['xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_POINT_AXJ1745.6-2901_CALDB11.arf','xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_FLATCIRCLE_fr3_diffuse_CALDB11.arf','xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_IMAGE_img_chandra_SgrAeast_FeKa25_SgrAEast_CALDB11.arf',],
 #                 arf_num_list=['xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_POINT_AXJ1745.6-2901_CALDB12.arf','xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_FLATCIRCLE_fr3_diffuse_CALDB12.arf','xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb_IMAGE_img_chandra_SgrAeast_FeKa25_SgrAEast_CALDB12.arf'],label_list=['AX J1745','flat arf','Sgr A East'],label_num='CALDB 12',label_denom='CALDB 11',skip_ncols=1)


 #smallpix
 # arf_compa_plots(arf_denom_list=['xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0_rmf_evtbase_2_12_0_05_60000_comb_POINT_1745_CALDB11.arf','xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0_rmf_evtbase_2_12_0_05_60000_comb_FLATCIRCLE_fr3_CALDB11.arf','xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0_rmf_evtbase_2_12_0_05_60000_comb_IMAGE_img_chandra_SgrAeast_FeKa25_CALDB11.arf',],
 #                 arf_num_list=['xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0_rmf_evtbase_2_12_0_05_60000_comb_POINT_AXJ1745.6-2901_CALDB12.arf','xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0_rmf_evtbase_2_12_0_05_60000_comb_FLATCIRCLE_fr3_diffuse_CALDB12.arf','xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0_rmf_evtbase_2_12_0_05_60000_comb_IMAGE_img_chandra_SgrAeast_FeKa25_SgrAEast_CALDB12.arf'],label_list=['AX J1745','flat arf','Sgr A East'],label_num='CALDB 12',label_denom='CALDB 11',skip_ncols=1)



 #NS
 # arf_compa_plots(arf_denom_list=['xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0_rmf_evtbase_2_12_0_05_60000_comb_POINT_1745_AXJp11_CALDB11.arf','xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0_rmf_evtbase_2_12_0_05_60000_comb_FLATCIRCLE_fr3_AXJp11_CALDB11.arf','xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0_rmf_evtbase_2_12_0_05_60000_comb_IMAGE_img_chandra_SgrAeast_FeKa25_AXJp11_CALDB11.arf',],
 #                 arf_num_list=['xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0_rmf_evtbase_2_12_0_05_60000_comb_AXJp11_POINT_AXJ1745.6-2901_CALDB12.arf','xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0_rmf_evtbase_2_12_0_05_60000_comb_AXJp11_FLATCIRCLE_fr3_diffuse_CALDB12.arf','xa300044010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0_rmf_evtbase_2_12_0_05_60000_comb_AXJp11_IMAGE_img_chandra_SgrAeast_FeKa25_SgrAEast_CALDB12.arf'],label_list=['AX J1745','flat arf','Sgr A East'],label_num='CALDB 12',label_denom='CALDB 11',skip_ncols=1)


#data plots
# xPlot('ldata,delchi',group_names=['PV Phase "big" pixel region','PV Phase NS region'],auto_figsize=(12,6))
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()


######################
#region plot XTEND PV PHASE
# mpdaf_plot_img('xa300044010xtd_p030000010_cl_SFP_img_sky_2_10.ds',
#                crop_coords=['17:45:35.6400', '-29:01:33.888'],
#                target_name_list=[None,None,None],
# rad_crop=[400,400],
#                target_coords_list=[(266.4182250,-29.0133942),(266.3979339,-29.0264812),(266.4384636,-29.0264833)],
#                target_sizes_pix=[8,8,8],target_colors=['green','red','pink'],
#                target_names=['Diffuse emission background for MAXI J1744','AX J1745 source','AXJ1745 background'],
#                target_names_offset=[1.2,1.2,1.2],save=False)

#lc plot
#plot_lc(['xa901002010rsl_p0px1000_cl_RTS_pixel_branch_filter_lc_2-10_256s.lc',
# 'xa901002010xtd_p031300010_cl_SFPreg_8pix_arcsec_lc_0.3-10_256s.lc'],
# binning=256,colors=['black','red'],labels=['Resolve MAXI J1744-294 "small pixel"','Xtend MAXI J1744-294'],
# figsize=(10,5),title=False)

# #colormapping for MAXI J1744 DDT in Xtend
# xPlot('ldata,delchi',addcomp_colors='source',addcomp_ls='group',
#       addcomp_source_cmaps=['Greens','RdPu'],group_names=['MAXI J1744-294'],auto_figsize=(7,5))
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()


#Xtend complementary SFP plots
# mpdaf_plot_img('xa901002010xtd_p031300010_cl_img_DET_2p35_2p5.ds',
#                rad_crop=[120,120],target_name_list=[None,None],crop_coords=[1087,728.5],center_crop_u='pixels',rotate=True,img_scale='log',target_coords_list=[[728.5,1087]],target_sizes_pix=[8],target_names='MAXI J1744',target_colors=['red'])

