
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