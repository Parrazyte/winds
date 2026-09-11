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
# line_simu(outdir='warmfaint',mode='NH_noise_photo',rmf_path='NewAthena_4eV_minipc',
# arf_path='NewAthena_nofilter_minipc',bkg_path='NewAthena_NXB_1arcmin_minipc',
# expos=5,n_iter=1000,flux_range='0.316227766_1e-4_8',flux_band='0.3 10.0',
# mod_path='SED_soft_0p1Edd_2e22.xcm',set_ener_data=True,set_ener_data_str='large_canon',
# photo_mod='pion_abs_canon_soft_minipc',photo_comp_pos=3,photo_xi_range='warm_freeze',
# photo_turb_range=[100,'min',300],photo_v_range=[0,-1000,1000],photo_nsteppar_turb=10,
# photo_nsteppar_v=20,par_freeze_steppar=[3,5],n_cores=13)

#multi logxi for AMD
for i in [0.,0.5,1.,1.5,2.5,3.,3.5,4.,4.5,5.,5.5,6.][::-1]:
    os.chdir('/home/parrazyte/Observ_SA/NewAthena/NHdet')


    line_simu(outdir='AMD_logxi_%.2f'%i,mode='NH_noise_photo',rmf_path='NewAthena_4eV_minipc',
    arf_path='NewAthena_nofilter_minipc',bkg_path='NewAthena_NXB_1arcmin_minipc',
    expos=5,n_iter=100,flux_range='1_0.1_1',flux_band='0.3 10.0',
    mod_path='SED_soft_0p1Edd_2e22.xcm',set_ener_data=True,set_ener_data_str='large_canon',
    photo_mod='pion_abs_canon_soft_minipc',photo_comp_pos=3,photo_xi_range='%.2f'%i+'_freeze',
    photo_turb_range=[100,'min',300],photo_v_range=[0,-1000,1000],photo_nsteppar_turb=10,
    photo_nsteppar_v=20,par_freeze_steppar=[3,5],n_cores=13)

#on the main pc
for i in [0.,0.5,1.,1.5,2.5,3.,3.5,4.,4.5,5.][::-1]:
    os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/NewAthena/SpecialIssue/NHdet')


    line_simu(outdir='AMD_logxi_%.2f'%i,mode='NH_noise_photo',rmf_path='NewAthena_4eV',
    arf_path='NewAthena_nofilter',bkg_path='NewAthena_NXB_1arcmin',
    expos=5,n_iter=100,flux_range='1_1_1',flux_band='0.3 10.0',
    mod_path='SED_soft_0p1Edd_2e22.xcm',set_ener_data=True,set_ener_data_str='large_canon',
    photo_mod='pion_abs_canon_soft',photo_comp_pos=3,photo_xi_range='%.2f'%i+'_freeze',
    photo_turb_range=[100,'min',300],photo_v_range=[0,-1000,1000],photo_nsteppar_turb=11,
    photo_nsteppar_v=21,par_freeze_steppar=[3,5],n_cores=13)
