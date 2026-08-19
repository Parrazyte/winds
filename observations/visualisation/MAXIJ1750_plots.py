
os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1750-327/simu_DDT')

AllData.clear()
Xset.restore('basemod_2Crab_2-10_cons.xcm')
set_ener('thcomp',xrism=True)
fake50=FakeitSettings(response='rsl_Hp_5eV.rmf',
                    arf='rsl_pointsource_fwND_GVclosed.arf',
                     fileName='fake_50ks_ND.pi',
                    exposure=5e4*0.35)
AllData.fakeit(nSpectra=1,settings=fake50,applyStats=True)
AllData.ignore('**-2. 10.-**')
rebinv_xrism(1,20)

AllData.clear()
Xset.restore('basemod_2Crab_2-10_cons.xcm')
set_ener('thcomp',xrism=True)
fake10=FakeitSettings(response='rsl_Hp_5eV.rmf',
                    arf='rsl_pointsource_fwND_GVclosed.arf',
                     fileName='fake_10ks_ND.pi',
                    exposure=1e4*0.35)
AllData.fakeit(nSpectra=1,settings=fake10,applyStats=True)
AllData.ignore('**-2. 10.-**')
rebinv_xrism(1,20)

Xset.restore('mod_all_ND.xcm')
AllData.notice('all')
AllData.ignore('**-4.5 7.5-**')
rebinv_xrism(1,50)
rebinv_xrism(2,25)
xPlot('ldata',mult_factors=[1,0.7],ylims=[35,160])
plt.savefig('zoom_highE.pdf')

Xset.restore('mod_all_ND.xcm')
AllData.notice('all')
AllData.ignore('**-2. 3.-**')
rebinv_xrism(1,20)
rebinv_xrism(2,15)
xPlot('ldata',mult_factors=[1,0.7],ylims=[10,200])
plt.savefig('zoom_lowE.pdf')

AllData.notice('all')
AllData.ignore('**-2. 10.-**')
rebinv_xrism(1,50)
rebinv_xrism(2,35)
xPlot('ldata',mult_factors=[1,0.7],ylims=[10,200])
