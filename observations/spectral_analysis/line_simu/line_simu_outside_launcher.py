import os,sys

sys.path.extend(['/home/parrazyte/Documents/Work/PhD/Scripts/winds', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/general', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/observations/data_reduction', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/observations/spectral_analysis', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/observations/timing_analysis', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/observations/visualisation', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/observations/data_reduction/Chandra', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/observations/data_reduction/NICER', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/observations/data_reduction/NuSTAR', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/observations/data_reduction/Suzaku', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/observations/data_reduction/Swift', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/observations/data_reduction/XMM', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/observations/data_reduction/XRISM', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/observations/visualisation/ion_visu', '/home/parrazyte/Documents/Work/PhD/Scripts/winds/observations/visualisation/visual_line_tools'])

from line_simu.line_simu_tools import line_simu

os.chdir('/home/parrazyte/Observ_SA/NewAthena/NHdet')
# line_simu(outdir='warmbright',mode='NH_noise_photo',rmf_path='NewAthena_4eV_minipc',
# arf_path='NewAthena_nofilter_minipc',bkg_path='NewAthena_NXB_1arcmin_minipc',
# expos=5,n_iter=1000,flux_range='100_1_5',flux_band='0.3 10.0',
# mod_path='SED_soft_0p1Edd_2e22.xcm',set_ener_data=True,set_ener_data_str='large_canon',
# photo_mod='pion_abs_canon_soft_minipc',photo_comp_pos=3,photo_xi_range='warm_freeze',
# photo_turb_range=[100,'min',300],photo_v_range=[0,-1000,1000],photo_nsteppar_turb=10,
# photo_nsteppar_v=20,par_freeze_steppar=[3,5],n_cores=1)

#must be finished
#line_simu(outdir='warmfaint',mode='NH_noise_photo',rmf_path='NewAthena_4eV_minipc',
#arf_path='NewAthena_nofilter_minipc',bkg_path='NewAthena_NXB_1arcmin_minipc',
#expos=5,n_iter=1000,flux_range='0.316227766_1e-4_8',flux_band='0.3 10.0',
#mod_path='SED_soft_0p1Edd_2e22.xcm',set_ener_data=True,set_ener_data_str='large_canon',
#photo_mod='pion_abs_canon_soft_minipc',photo_comp_pos=3,photo_xi_range='warm_freeze',
#photo_turb_range=[100,'min',300],photo_v_range=[0,-1000,1000],photo_nsteppar_turb=10,
#photo_nsteppar_v=20,par_freeze_steppar=[3,5],n_cores=14)

#multi logxi for AMD
# for i in [4.,4.5,5.][::-1]:
#     os.chdir('/home/parrazyte/Observ_SA/NewAthena/NHdet')
# #
# #
#     line_simu(outdir='AMD_1000_100_logxi_%.2f'%i,mode='NH_noise_photo',rmf_path='NewAthena_4eV_minipc',
#      arf_path='NewAthena_nofilter_minipc',bkg_path='NewAthena_NXB_1arcmin_minipc',
#      expos=5,n_iter=1000,flux_range='100_100_1',flux_band='0.3 10.0',
#      mod_path='SED_soft_0p1Edd_2e22.xcm',set_ener_data=True,set_ener_data_str='large_canon',
#      photo_mod='pion_abs_canon_soft_minipc',photo_comp_pos=3,photo_xi_range='%.2f'%i+'_freeze',
#      photo_turb_range=[100,'min',300],photo_v_range=[0,-1000,1000],photo_nsteppar_turb=11,
#      photo_nsteppar_v=21,par_freeze_steppar=[3,5],n_cores=14)

#updating the two with lower exposures to get to lower SNR
for i in [2.]:
    os.chdir('/home/parrazyte/Observ_SA/NewAthena/NHdet')
#

    line_simu(outdir='AMD_1000_0p5ks_hard_logxi_%.2f'%i,mode='NH_noise_photo',rmf_path='NewAthena_4eV_minipc',
     arf_path='NewAthena_nofilter_minipc',bkg_path='NewAthena_NXB_1arcmin_minipc',
     expos=0.5,n_iter=1000,flux_range='0.1_10_5',flux_band='0.3 10.0',
     mod_path='SED_hard_norenorm_2e22.xcm',set_ener_data=True,set_ener_data_str='large_canon',
     photo_mod='pion_abs_canon_hard_minipc',photo_comp_pos=3,photo_xi_range='%.2f'%i+'_freeze',
     photo_turb_range=[100,'min',300],photo_v_range=[0,-1000,1000],photo_nsteppar_turb=11,
     photo_nsteppar_v=21,par_freeze_steppar=[3,5],n_cores=14)

#more points to higher end for logxi=4
for i in [4.]:
    os.chdir('/home/parrazyte/Observ_SA/NewAthena/NHdet')
#

    line_simu(outdir='AMD_1000_50ks_brighthard_logxi_%.2f'%i,mode='NH_noise_photo',rmf_path='NewAthena_4eV_minipc',
     arf_path='NewAthena_nofilter_minipc',bkg_path='NewAthena_NXB_1arcmin_minipc',
     expos=50,n_iter=1000,flux_range='1_1000_7',flux_band='0.3 10.0',
     mod_path='SED_hard_norenorm_2e22.xcm',set_ener_data=True,set_ener_data_str='large_canon',
     photo_mod='pion_abs_canon_hard_minipc',photo_comp_pos=3,photo_xi_range='%.2f'%i+'_freeze',
     photo_turb_range=[100,'min',300],photo_v_range=[0,-1000,1000],photo_nsteppar_turb=11,
     photo_nsteppar_v=21,par_freeze_steppar=[3,5],n_cores=14)

#continuing with 0
for i in [0.]:
    os.chdir('/home/parrazyte/Observ_SA/NewAthena/NHdet')
#

    line_simu(outdir='AMD_1000_0p5ks_brighthard_logxi_%.2f'%i,mode='NH_noise_photo',rmf_path='NewAthena_4eV_minipc',
     arf_path='NewAthena_nofilter_minipc',bkg_path='NewAthena_NXB_1arcmin_minipc',
     expos=0.5,n_iter=1000,flux_range='1_100_5',flux_band='0.3 10.0',
     mod_path='SED_hard_norenorm_2e22.xcm',set_ener_data=True,set_ener_data_str='large_canon',
     photo_mod='pion_abs_canon_hard_minipc',photo_comp_pos=3,photo_xi_range='%.2f'%i+'_freeze',
     photo_turb_range=[100,'min',300],photo_v_range=[0,-1000,1000],photo_nsteppar_turb=11,
     photo_nsteppar_v=21,par_freeze_steppar=[3,5],n_cores=14)


# for i in [4.,2.,0.]:
#     os.chdir('/home/parrazyte/Observ_SA/NewAthena/NHdet')
# #
#
#     line_simu(outdir='AMD_1000_50ks_brighthard_logxi_%.2f'%i,mode='NH_noise_photo',rmf_path='NewAthena_4eV_minipc',
#      arf_path='NewAthena_nofilter_minipc',bkg_path='NewAthena_NXB_1arcmin_minipc',
#      expos=50,n_iter=1000,flux_range='1_100_5',flux_band='0.3 10.0',
#      mod_path='SED_soft_0p1Edd_2e22.xcm',set_ener_data=True,set_ener_data_str='large_canon',
#      photo_mod='pion_abs_canon_hard_minipc',photo_comp_pos=3,photo_xi_range='%.2f'%i+'_freeze',
#      photo_turb_range=[100,'min',300],photo_v_range=[0,-1000,1000],photo_nsteppar_turb=11,
#      photo_nsteppar_v=21,par_freeze_steppar=[3,5],n_cores=14)

    # line_simu(outdir='hotfaint_focusedbkg',mode='NH_noise_photo',rmf_path='NewAthena_4eV_minipc',
    #  arf_path='NewAthena_nofilter_minipc',bkg_path='NewAthena_NXB_10asec_minipc',
    #  expos=5,n_iter=1000,flux_range='0.01_0.1_3',flux_band='0.3 10.0',
    #  mod_path='SED_soft_0p1Edd_2e22.xcm',set_ener_data=True,set_ener_data_str='large_canon',
    #  photo_mod='pion_abs_canon_soft_minipc',photo_comp_pos=3,photo_xi_range='%.2f'%i+'_freeze',
    #  photo_turb_range=[100,'min',300],photo_v_range=[0,-1000,1000],photo_nsteppar_turb=11,
    #  photo_nsteppar_v=21,par_freeze_steppar=[3,5],n_cores=14)
    #
    # os.chdir('/home/parrazyte/Observ_SA/NewAthena/NHdet')
#
#
    # line_simu(outdir='hotfaint_long_focusedbkg',mode='NH_noise_photo',rmf_path='NewAthena_4eV_minipc',
    #  arf_path='NewAthena_nofilter_minipc',bkg_path='NewAthena_NXB_10asec_minipc',
    #  expos=500,n_iter=1000,flux_range='0.01_1_5',flux_band='0.3 10.0',
    #  mod_path='SED_soft_0p1Edd_2e22.xcm',set_ener_data=True,set_ener_data_str='large_canon',
    #  photo_mod='pion_abs_canon_soft_minipc',photo_comp_pos=3,photo_xi_range='%.2f'%i+'_freeze',
    #  photo_turb_range=[100,'min',300],photo_v_range=[0,-1000,1000],photo_nsteppar_turb=11,
    #  photo_nsteppar_v=21,par_freeze_steppar=[3,5],n_cores=14)