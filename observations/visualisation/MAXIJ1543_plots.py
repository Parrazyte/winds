
os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/photo/comp_plots')
plot_comp_ratio([1,2],[[1,2,3],[1,2],[1,2],[1,2],[3]],6.38,7.13,
                other_addcomps_labels=['combined','photo 1',
                                       'photo 2',
                                       'photo 3',
                                       'photo 1 em',
                                       ],
                other_addcomps_colors=['black','orange','red','pink','blue'],
                other_addcomps_alpha=[1,0.5,0.5,0.5,0.5],
                other_addcomps_type=['abs','abs','abs','abs','abs'],
                cont_addcomps_xcm='3comp1em_closersol_2-10_deabs_cont.xcm',
                other_addcomps_xcm=['3comp1em_closersol_2-10_deabs_tot.xcm','3comp1em_closersol_2-10_deabs_tot_abs1.xcm',
                          '3comp1em_closersol_2-10_deabs_tot_abs2.xcm',
                          '3comp1em_closersol_2-10_deabs_tot_abs3.xcm',
                          '3comp1em_closersol_2-10_deabs_tot.xcm',],
                ylims=[0.01,1.],figsize=(10,8),minor_locator=10,ylabel_prefix='')


os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/902003010_repro/analysis_tr/sp/comp_plots')
Xset.restore('NXB_unfrozen_lines_prep_error_forvisu.xcm')
set_ener('thcomp',xrism=True)
plot_comp_ratio([1,22],[[2,3,14,15],[4,5,6,7,8,9,16,17],[10,11],[12,13],[18,19],[20,21]],6.2,7.13,
                other_addcomps_labels=[r"main FeXXVI",
                                       r"main FeXXV",
                                       'SXVI','unknown',
                                       'red FeXXVI',
                                       'Fe satellites'],
                other_addcomps_colors=['green','blue','plum','black','red','darkturquoise'],
                ylims=[0.,1.3],figsize=(10,8),minor_locator=10,
                ylabel_prefix='MAXI J1543 region empirical model \n')


plt.savefig('mod_empi.pdf')

