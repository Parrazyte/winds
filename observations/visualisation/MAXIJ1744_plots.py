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


#bigpix pre_lines empirical plot
# rebinv_xrism(1,20)
# rebinv_xrism(2,20)
# xPlot('ldata,delchi',auto_figsize=[10,5],addcomp_colors='source',
# addcomp_source_cmaps=['YlGn','YlOrRd_r','PuBu'],addcomp_ls='group',
# model_ls='group',group_names=['MAXI J1744-294 "big" pixel region','AX J1745.6-2901'],
# legend_sources=True,label_sources=['MAXI J1744-294','AX J1745.6-2901','diffuse'],
# label_sources_cval=[0.8,0.2,0.8],legend_addcomp_groups=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()

#bigpix pre_lines 2keV zoom
#AllData.ignore('**-2.4 2.54-**')
# rebinv_xrism(1,7)
# rebinv_xrism(2,5)
# xPlot('ldata,ratio,delchi',auto_figsize=[4,5],addcomp_colors='source',addcomp_source_cmaps=['YlGn','YlOrRd_r','PuBu'],
# addcomp_ls='group',model_ls='group',group_names=['MAXI J1744-294 "big" pixel region','AX J1745.6-2901'],
# legend_sources=False,legend_sources_loc='upper left',skip_main_legend=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()

#zoomed plots for the BH only
# rebinv_xrism(1,10)
# plt.rcParams.update({'legend.labelspacing':0.2})
# xPlot('ldata,ratio,delchi',auto_figsize=[10,5],addcomp_colors='source',addcomp_source_cmaps=['YlGn','YlOrRd_r','PuBu'],addcomp_ls='group',model_ls='group',group_names=['MAXI J1744-294 "big" pixel region'],legend_sources_loc='lower left',legend_sources=True,label_sources=['MAXI J1744-294','AX J1745.6-2901','diffuse'],
#  label_sources_cval=[0.8,0.2,0.8],legend_addcomp_groups=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()

#blindsearches
#xrism_ls_loader('post_edgeconti_readj_Ka_emi_norm_onlyBH_narrow_out_6.3_7.1_lw_5.dill',ener_show_range=[6.3,7.1],ratio_bounds=[0.5,2.],squished_mode=True)

#after all lines empi bigpix
#note: source order changed, now the actual source order in the model
# rebinv_xrism(1,20)
# rebinv_xrism(2,20)
# xPlot('ldata,delchi',auto_figsize=[10,5],addcomp_colors='source',
# addcomp_source_cmaps=['color_green','PuBu_r','YlOrRd'],addcomp_ls='group',
# model_ls='group',group_names=['MAXI J1744-294 "big" pixel region','AX J1745.6-2901'],
# legend_sources=True,label_sources=['MAXI J1744-294','diffuse','AX J1745.6-2901'],addcomp_rebin=[3,3],
# label_sources_cval=[0.8,0.8,0.8],legend_addcomp_groups=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()

#zoom iron
# rebinv_xrism(1,10)
# rebinv_xrism(2,10)
# xPlot('ldata,delchi',auto_figsize=[10,5],addcomp_colors='source',
# addcomp_source_cmaps=['color_green','PuBu_r','YlOrRd'],addcomp_ls='group',
# model_ls='group',group_names=['MAXI J1744-294 "big" pixel region','AX J1745.6-2901'],
# legend_sources=True,label_sources=['MAXI J1744-294','diffuse','AX J1745.6-2901'],addcomp_rebin=[3,3],legend_sources_loc='lower left',
# label_sources_cval=[0.8,0.8,0.8],legend_addcomp_groups=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()

#for appendix
# rebinv_xrism(1,10)
# plt.rcParams.update({'legend.labelspacing':0.2})
# xPlot('ldata,ratio,delchi',auto_figsize=[10,5],addcomp_colors='source',
#       addcomp_source_cmaps=['color_limegreen','YlOrRd_r','PuBu'],
#       addcomp_ls='group',model_ls='group',group_names=['MAXI J1744-294 "big" pixel region'],
#       legend_sources_loc='lower left',legend_sources=True,
#       label_sources=['MAXI J1744-294','AX J1745.6-2901','diffuse'],
#  label_sources_cval=[0.8,0.2,0.8],legend_addcomp_groups=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()


#smallpix

#full preline resid
# rebinv_xrism(1,20)
# rebinv_xrism(2,20)
# xPlot('ldata,delchi',auto_figsize=[10,5],addcomp_colors='source',
# addcomp_source_cmaps=['YlGn','YlOrRd','PuBu','RdPu'],addcomp_ls='group',addcomp_rebin=[3,3],
# model_ls='group',group_names=['MAXI J1744-294 "small" pixel region','AX J1745.6-2901'],
# legend_sources=True,label_sources=['MAXI J1744-294','AX J1745.6-2901','GCXE','Sgr A East'],
# label_sources_cval=[0.8,0.8,0.8,0.8],legend_addcomp_groups=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()

#zoom 2p4
# rebinv_xrism(1,5)
# rebinv_xrism(2,5)
# xPlot('ldata,ratio,delchi',auto_figsize=[4,5],addcomp_colors='source',mult_factors=[1,1],
# addcomp_source_cmaps=['YlGn','YlOrRd','PuBu','RdPu'],addcomp_ls='group',
# model_ls='group',group_names=['MAXI J1744-294 "small" pixel region','AX J1745.6-2901'],
# legend_sources=False,label_sources=['MAXI J1744-294','AX J1745.6-2901','GCXE','Sgr A East'],
# label_sources_cval=[0.8,0.8,0.8,0.8],skip_main_legend=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()

#zoom 6p3 with readj
# rebinv_xrism(1,10)
# rebinv_xrism(2,10)
# Plot.add=False
# xPlot('ldata,ratio,delchi',auto_figsize=[6,5],addcomp_colors='source',mult_factors=[1,4.1],
# addcomp_source_cmaps=['YlGn','YlOrRd','PuBu','RdPu'],addcomp_ls='group',
# model_ls='group',group_names=['MAXI J1744-294 "small" pixel region','AX J1745.6-2901'],
# legend_sources=False,label_sources=['MAXI J1744-294','AX J1745.6-2901','GCXE','Sgr A East'],
# label_sources_cval=[0.8,0.8,0.8,0.8],skip_main_legend=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()


#zoom 6p3 no mult factor
# rebinv_xrism(1,10)
# rebinv_xrism(2,10)
# xPlot('ldata,ratio,delchi',auto_figsize=[6,5],addcomp_colors='source',mult_factors=[1,1],
# addcomp_source_cmaps=['YlGn','YlOrRd','PuBu','RdPu'],addcomp_ls='group',
# model_ls='group',group_names=['MAXI J1744-294 "small" pixel region','AX J1745.6-2901'],
# legend_sources=False,label_sources=['MAXI J1744-294','AX J1745.6-2901','GCXE','Sgr A East'],
# label_sources_cval=[0.8,0.8,0.8,0.8],skip_main_legend=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()

#zoom 7p5-9 (different size to match the size of the bigpix one after cutting the x axis labels)
# rebinv_xrism(1,10)
# rebinv_xrism(2,10)
# xPlot('ldata,ratio,delchi',auto_figsize=[3.8,5],addcomp_colors='source',mult_factors=[1,1],
# addcomp_source_cmaps=['YlGn','YlOrRd','PuBu','RdPu'],addcomp_ls='group',
# model_ls='group',group_names=['MAXI J1744-294 "small" pixel region','AX J1745.6-2901'],
# legend_sources=False,label_sources=['MAXI J1744-294','AX J1745.6-2901','GCXE','Sgr A East'],
# label_sources_cval=[0.8,0.8,0.8,0.8],skip_main_legend=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()

#appendix prelines
# rebinv_xrism(1,10)
# plt.rcParams.update({'legend.labelspacing':0.2})
# xPlot('ldata,ratio,delchi',auto_figsize=[10,5],addcomp_colors='source',
# addcomp_source_cmaps=['YlGn','YlOrRd','PuBu','RdPu'],addcomp_ls='group',
# model_ls='group',group_names=['MAXI J1744-294 "small" pixel region','AX J1745.6-2901'],
# legend_sources=True,label_sources=['MAXI J1744-294','AX J1745.6-2901','GCXE','Sgr A East'],
#       addcomp_rebin=[3,3],legend_sources_loc='lower left',
# label_sources_cval=[0.8,0.8,0.8,0.8],legend_addcomp_groups=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()
#then manually resize first panel y axis to 0.01-0.5

#appendix postlines
# rebinv_xrism(1,10)
# plt.rcParams.update({'legend.labelspacing':0.2})
# xPlot('ldata,ratio,delchi',auto_figsize=[10,5],addcomp_colors='source',
# addcomp_source_cmaps=['color_green','YlOrRd','PuBu','RdPu'],addcomp_ls='group',
# model_ls='group',group_names=['MAXI J1744-294 "small" pixel region','AX J1745.6-2901'],
# legend_sources=True,label_sources=['MAXI J1744-294','AX J1745.6-2901','GCXE','Sgr A East'],
#       addcomp_rebin=[3,3],legend_sources_loc='lower left',legend_sources_bbox=[0.15,-0.03],
# label_sources_cval=[0.8,0.8,0.8,0.8],legend_addcomp_groups=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()
#then manually resize first panel y axis to 0.01-0.5


#photo bigpix
# rebinv_xrism(1,20)
# rebinv_xrism(2,20)
# xPlot('ldata,delchi',auto_figsize=[10,5],addcomp_colors='source',
# addcomp_source_cmaps=['YlGn','PuBu','YlOrRd'],addcomp_ls='group',
# model_ls='group',group_names=['MAXI J1744-294 "big" pixel region','AX J1745.6-2901'],
# legend_sources=True,label_sources=['MAXI J1744-294','diffuse','AX J1745.6-2901'],
#       addcomp_rebin=[3,3],
# label_sources_cval=[0.8,0.2,0.8],legend_addcomp_groups=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()
#
# #zoom
# rebinv_xrism(1,10)
# rebinv_xrism(2,10)
# xPlot('ldata,delchi',auto_figsize=[10,5],addcomp_colors='source',
# addcomp_source_cmaps=['YlGn','PuBu','YlOrRd'],addcomp_ls='group',
# model_ls='group',group_names=['MAXI J1744-294 "big" pixel region','AX J1745.6-2901'],
# legend_sources=True,label_sources=['MAXI J1744-294','diffuse','AX J1745.6-2901'],
#       addcomp_rebin=[3,3],legend_sources_loc='lower left',legend_sources_bbox=[0.15,-0.03],
# label_sources_cval=[0.8,0.2,0.8],legend_addcomp_groups=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()

#smallpix: same as empi works

# rebinv_xrism(1,20)
# rebinv_xrism(2,20)
# xPlot('ldata,delchi',auto_figsize=[10,5],addcomp_colors='source',
# addcomp_source_cmaps=['YlGn','YlOrRd','PuBu','RdPu'],addcomp_ls='group',addcomp_rebin=[3,3],
# model_ls='group',group_names=['MAXI J1744-294 "small" pixel region','AX J1745.6-2901'],
# legend_sources=True,label_sources=['MAXI J1744-294','AX J1745.6-2901','GCXE','Sgr A East'],
# label_sources_cval=[0.8,0.8,0.8,0.8],legend_addcomp_groups=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()

#zoom
# rebinv_xrism(1,10)
# plt.rcParams.update({'legend.labelspacing':0.2})
# xPlot('ldata,ratio,delchi',auto_figsize=[10,5],addcomp_colors='source',
# addcomp_source_cmaps=['color_green','YlOrRd','PuBu','RdPu'],addcomp_ls='group',
# model_ls='group',group_names=['MAXI J1744-294 "small" pixel region','AX J1745.6-2901'],
# legend_sources=True,label_sources=['MAXI J1744-294','AX J1745.6-2901','GCXE','Sgr A East'],
#       addcomp_rebin=[3,3],legend_sources_loc='lower left',legend_sources_bbox=[0.15,-0.03],
# label_sources_cval=[0.8,0.8,0.8,0.8],legend_addcomp_groups=True)
# plt.gca().get_figure().get_children()[1].set_title(None)
# plt.tight_layout()
#then manually resize first panel y axis to 0.01-0.5


#set_ener('thcomp',xrism=True)
#plot_comp_ratio([1],[2,3,4,5,[6,7]],6.3,7.2,
# other_addcomps_labels=["photo zone static","photo zone b1","photo zone b2",
# "photo zone b3", r'empirical FeXXVI K$\alpha$'],
# other_addcomps_colors=['green','darkturquoise','blue','cornflowerblue','plum'],
# ylims=[1.,None],figsize=(10,4))


'''
COMP PLOTS
'''


'''
EMPIRICAL
'''
'''BIGPIX'''

os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1744-294/new_anal/fit_duo_empirical_bigpix/2025_updated/comp_mod')
Xset.restore('mod_noabs.xcm')
set_ener('thcomp',xrism=True)
plot_comp_ratio([1],[None,[10,11,12,13],[8,9],None,7,5,6,[3,4]],6.38,7.13,
                other_addcomps_labels=['main',r"FeXXV K$\alpha$",
                                       r"FeXXVI K$\alpha$",
                                       'blueshifted',
r"FeXXV K$\alpha$ 1",
r"FeXXV K$\alpha$ 2",
r"FeXXV K$\alpha$ 3",
r"FeXXVI K$\alpha$"],
                other_addcomps_colors=[None,'green','rebeccapurple',None,'darkturquoise','blue','cornflowerblue','plum'],
                ylims=[1.,2.3],figsize=(6,3),minor_locator=10,ylabel_prefix='Big pixel region empirical model \n')
plt.savefig('mod_empi_bigpix_deabs_nobfekblor_comp_ratio.pdf')

'''SMALLPIX'''

os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1744-294/new_anal/fit_duo_smallpix/2025/rsp_CALDB11/comp_mod')
Xset.restore('mod_deabs.xcm')
set_ener('thcomp',xrism=True)
plot_comp_ratio([1],[None,[10,11,12,13],[8,9],[2,7],None,6,5,[3,4],],6.38,7.13,
                other_addcomps_labels=['main',r"FeXXV K$\alpha$",
                                       r"FeXXVI K$\alpha$",
                                        'Fe I',
                                       'blueshifted',
r"FeXXV K$\alpha$ 1",
r"FeXXV K$\alpha$ 2",
r"FeXXVI K$\alpha$",
                                       ],
                other_addcomps_colors=[None,'green','rebeccapurple','black',None,'darkturquoise','blue','plum'],
                ylims=[1,2.3],figsize=(6,3),minor_locator=10,ylabel_prefix='Big pixel region empirical model \n',manual_bbox=(0.045, 1))
plt.savefig('mod_empi_smallpix_deabs_nobfekblor_comp_ratio.pdf')


'''PHOTO'''
'''BIGPIX'''

os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1744-294/physical/photo/new/bigpix/comp_mod')
Xset.restore('mod_1744_deabs_nobfekblor.xcm')
set_ener('thcomp',xrism=True)
plot_comp_ratio([1],[2,3,4,5,[6,7]],6.38,7.13,
                other_addcomps_labels=["static","blue 1","blue 2", "blue 3",
                                       r'empirical'+'\n'+r'FeXXVI K$\alpha$'],
                other_addcomps_colors=['green','darkturquoise','blue','cornflowerblue','plum'],
                ylims=[1.,2.3],figsize=(6,3),minor_locator=10,ylabel_prefix='Big pixel region PIE model \n')
plt.savefig('mod_photo_bigpix_deabs_nobfekblor_comp_ratio.pdf')

'''SMALLPIX'''

os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1744-294/physical/photo/new/smallpix/comp_mod')
Xset.restore('mod_3comp_fit_highxi_noabs_2e-3.xcm')
set_ener('thcomp',xrism=True)
plot_comp_ratio([1],[2,3,4,[5,8],[6,7]],6.38,7.13,
                other_addcomps_labels=["static","blue 1","blue 2",r'empirical'+'\nFe I',
                                       r'empirical'+'\n'+r'FeXXVI K$\alpha$'],
                other_addcomps_colors=['green','darkturquoise','blue','black','plum'],
                ylims=[1.,2.3],figsize=(6,3),minor_locator=10,ylabel_prefix='small pixel region PIE model \n',
                manual_bbox=(0.045, 1))
plt.savefig('mod_photo_smallpix_deabs_nobfekblor_comp_ratio.pdf')
