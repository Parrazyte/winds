from xspec_config_multisp import *

fake_hard=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=5e4,fileName='fake_bhard.pi')]

fake_transi1=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=3.5e4,fileName='fake_transi1.pi')]

fake_transi2=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=1e4,fileName='fake_transi2.pi')]

fake_transi3=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=1e4,fileName='fake_transi3.pi')]

fake_transi4=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=1e4,fileName='fake_transi4.pi')]

fake_transi5=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=3.5e4,fileName='fake_transi5.pi')]

fake_bsoft=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=5e4,fileName='fake_bsoft.pi')]

AllData.clear()
set_ener('thcomp',xrism=True)
AllData.fakeit(nSpectra=1,settings=fake_hard,applyStats=True)


Xset.restore('mod_merge_v2.xcm')
AllData.ignore('**-2. 10.-**')
set_ener('thcomp',xrism=True)

for i in [1, 2, 3, 4, 5, 6,7]:
    rebinv_xrism(i, 10)

Plot.add=False

AllData.notice('all')
AllData.ignore('**-1.95 4.15-**')
xPlot('ldata',auto_figsize=(6,4),mult_factors=np.logspace(0,-3,7),
      data_colors=mpl.colormaps['plasma']((np.linspace(0,0.8,7))),model_colors='data')
plt.gca().get_legend().remove()
plt.gca().set_title('')
plt.tight_layout()

AllData.notice('all')
AllData.ignore('**-6.4 8.4-**')
xPlot('ldata',auto_figsize=(6,4),mult_factors=np.logspace(0,-3,7),
      data_colors=mpl.colormaps['plasma']((np.linspace(0,0.8,7))),model_colors='data')

AllData.notice('all')
AllData.ignore('**-2.3 2.65-**')
xPlot('ldata',auto_figsize=(2,4),mult_factors=np.logspace(0,-3,7),
      data_colors=mpl.colormaps['plasma']((np.linspace(0,0.8,7))),model_colors='data')
plt.gca().get_legend().remove()
plt.gca().set_title('')
plt.ylabel('')
plt.tight_layout()

for i in [1, 2, 3, 4, 5, 6,7]:
    rebinv_xrism(i, 10)
AllData.notice('all')
AllData.ignore('**-2.15 3.95-**')
xPlot('ldata',auto_figsize=(6,6),mult_factors=np.logspace(0,-4,7),
      data_colors=mpl.colormaps['plasma']((np.linspace(0,0.8,7))),model_colors='data')
ax=plt.gca()
ax.get_legend().remove()
ax.set_title('')
ax.set_yticks([])
ax.set_yticks([],minor=True)
ax.set_ylabel(r'normalized counts s$^{-1}$ keV${-1}$')
plt.tight_layout()
plt.savefig('simu_xrism_lowE.pdf')

for i in [1, 2, 3, 4, 5, 6,7]:
    rebinv_xrism(i, 10)
AllData.notice('all')
AllData.ignore('**-6.4 8.4-**')
xPlot('ldata',auto_figsize=(6,6),mult_factors=np.logspace(0,-4,7),
      data_colors=mpl.colormaps['plasma']((np.linspace(0,0.8,7))),model_colors='data')
ax2=plt.gca()
ax2.get_legend().remove()
ax2.set_title('')
ax2.set_yticks([])
ax2.set_yticks([],minor=True)
ax2.set_ylabel(r'normalized counts s$^{-1}$ keV${-1}$')
plt.tight_layout()
plt.savefig('simu_xrism_highE.pdf')


'''
NS
'''

fakens_soft=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=45e3,fileName='fakens_soft.pi')]

fakens_hard=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=45e3,fileName='fakens_hard.pi')]

fakens_soft_dip=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=5e3,fileName='fakens_soft_dip.pi')]

fakens_hard_dip=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=5e3,fileName='fakens_hard_dip.pi')]

AllData.clear()
AllModels.clear()
Xset.restore('mod_wind_soft_0p2Edd_8kpc.xcm')
AllData.fakeit(nSpectra=1,settings=fakens_soft,applyStats=True)

#PLOTS
os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_XRISM_AO3/NS')
Xset.restore('mod_mergens_8kpc.xcm')


#lowE
for i in [1,3]:
    rebinv_xrism(i, 10)

for i in [2,4]:
    rebinv_xrism(i, 5)

AllData.notice('all')
AllData.ignore('**-2.35 3.95-**')
Plot.add=False
xPlot('ldata',auto_figsize=(6,6),mult_factors=[10,2,5,0.5],
      data_colors=['blue','turquoise','red','pink'],model_colors='data')
ax=plt.gca()
ax.get_legend().remove()
ax.set_title('')
ax.set_yticks([])
ax.set_yticks([],minor=True)
ax.set_ylabel(r'normalized counts s$^{-1}$ keV${-1}$')
plt.tight_layout()
plt.savefig('simu_xrism_NS_lowE.pdf')

#highE
for i in [1,3]:
    rebinv_xrism(i, 20)

for i in [2,4]:
    rebinv_xrism(i, 5)
    Plot.add=False
AllData.notice('all')
AllData.ignore('**-6.15 8.55-**')
xPlot('ldata',auto_figsize=(7,6),mult_factors=[300,9,50,0.6],
      data_colors=['blue','turquoise','red','pink'],model_colors='data')
ax=plt.gca()
ax.get_legend().remove()
ax.set_title('')
ax.set_yticks([])
ax.set_yticks([],minor=True)
ax.set_ylabel(r'normalized counts s$^{-1}$ keV${-1}$')
plt.tight_layout()
plt.ylim(0.05,1500)
plt.subplots_adjust(left=0.1,right=0.9)
plt.savefig('simu_xrism_NS_highE.pdf')


'''
obscured
'''

fakeobscured_10=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=50e3,fileName='fake_obs_10.pi')]

fakeobscured_100=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=50e3,fileName='fake_obs_100.pi')]

fakeobscured_10_prog=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=50e3,fileName='fake_obs_10_prog.pi')]

fakeobscured_100_prog=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=50e3,fileName='fake_obs_100_prog.pi')]

fakeobscured_refl=[FakeitSettings(response='comb_from_1744.rmf',
                          arf='rsl_pntsrc_GVC_2025.arf',
                     exposure=50e3,fileName='fake_obs_100_refl.pi')]

AllData.clear()
AllModels.clear()
# Xset.restore('mod_wind_soft_0p2Edd_8kpc.xcm')
AllData.fakeit(nSpectra=1,settings=fakens_soft,applyStats=True)

#PLOTS
os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_XRISM_AO3/obscured')
Xset.restore('mod_merge_narrow10_refl100.xcm')


#lowE
for i in [1,2]:
    rebinv_xrism(i, 10)

AllData.notice('all')
AllData.ignore('**-2. 4.-**')
Plot.add=False
xPlot('ldata',auto_figsize=(6,6),mult_factors=[1,1],
      data_colors=['blue','red'],model_colors='data')
ax=plt.gca()
ax.get_legend().remove()
# ax.set_title('')
# ax.set_yticks([])
# ax.set_yticks([],minor=True)
# ax.set_ylabel(r'normalized counts s$^{-1}$ keV${-1}$')
plt.tight_layout()
plt.savefig('simu_xrism_obs_lowE.pdf')

#highE
for i in [1,2]:
    rebinv_xrism(i, 10)

rebinv_xrism(1,15)

AllData.notice('all')
AllData.ignore('**-6.3 8.5-**')
Plot.add=False
xPlot('ldata',auto_figsize=(6,6),mult_factors=[1,1],
      data_colors=['blue','red'],model_colors='data')
ax=plt.gca()
ax.get_legend().remove()
# ax.set_title('')
# ax.set_yticks([])
# ax.set_yticks([],minor=True)
# ax.set_ylabel(r'normalized counts s$^{-1}$ keV${-1}$')
plt.tight_layout()
plt.savefig('simu_xrism_obs_highE.pdf')

#highE
#highE refl

rebinv_xrism(1,10)
rebinv_xrism(2,15)

AllData.notice('all')
AllData.ignore('**-6.1 7.2-**')
Plot.add=False
xPlot('ldata',auto_figsize=(6,6),mult_factors=[1,1],
      data_colors=['blue','red'],model_colors='data')
ax=plt.gca()
ax.get_legend().remove()
# ax.set_title('')
# ax.set_yticks([])
# ax.set_yticks([],minor=True)
# ax.set_ylabel(r'normalized counts s$^{-1}$ keV${-1}$')
plt.tight_layout()
#plt.savefig('simu_xrism_obs_highE.pdf')
plt.savefig('simu_xrism_narrow10_refl100_highE.pdf')


'''NS XRAY CATALOG'''

catal=fits.open('/home/parrazyte/Documents/Work/PostDoc/catal/LMXB_cat.fits')
NS_mask=['XB' in elem for elem in catal[1].data['Xray_Type']]
XT_mask=['XT' in elem for elem in catal[1].data['Xray_Type']]

test=(np.array(NS_mask)) & (np.array(XT_mask))

catal_NSXT=catal[1].data[test]

catal_highi_NSXT=catal_NSXT[catal_NSXT['Incl']>=50]

'''
chararray(['EXO 0748-676', 'MXB 1659-29', 'IGR J17062-6143',
           'XTE J1710-281', 'SWIFT J1749.4-2807', 'SAX J1808.4-3658',
           'GS 1826-238', 'Swift J1858.6-0814', 'XTE J2123-058'],
          dtype='<U23')
most known or false inclination values
'''

