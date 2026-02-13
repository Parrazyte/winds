
import os,sys
import matplotlib.pyplot as plt
from xspec_config_multisp import *
from xspec import *
os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NuSTAR_AO12')

fakeall=[FakeitSettings(response='nu90002004004A01_sr.rmf',arf='nu90002004004A01_sr.arf',
                     exposure=5e4,fileName='A1'),
FakeitSettings(response='nu90002004004B01_sr.rmf',arf='nu90002004004B01_sr.arf',
                     exposure=5e4,fileName='B1'),
FakeitSettings(response='nu90002004004A01_sr.rmf',arf='nu90002004004A01_sr.arf',
                     exposure=5e4,fileName='A2'),
FakeitSettings(response='nu90002004004B01_sr.rmf',arf='nu90002004004B01_sr.arf',
                     exposure=5e4,fileName='B2'),
FakeitSettings(response='nu90002004004A01_sr.rmf',arf='nu90002004004A01_sr.arf',
                     exposure=5e4,fileName='A3'),
FakeitSettings(response='nu90002004004B01_sr.rmf',arf='nu90002004004B01_sr.arf',
                     exposure=5e4,fileName='B3')]

fake1=[FakeitSettings(response='nu90002004004A01_sr.rmf',arf='nu90002004004A01_sr.arf',
                      background='nu90102011002A01_bk.pha',
                     exposure=5e4,fileName='fakeA_bhard.pi'),
FakeitSettings(response='nu90002004004B01_sr.rmf',arf='nu90002004004B01_sr.arf',background='nu90102011002B01_bk.pha',
                     exposure=5e4,fileName='fakeB_bhard.pi')]

fake2=[FakeitSettings(response='nu90002004004A01_sr.rmf',arf='nu90002004004A01_sr.arf',
                      background='nu90102011002A01_bk.pha',
                     exposure=5e4,fileName='fakeA_bsoft.pi'),
FakeitSettings(response='nu90002004004B01_sr.rmf',arf='nu90002004004B01_sr.arf',
                     background='nu90102011002B01_bk.pha',
                     exposure=5e4,fileName='fakeB_bsoft.pi')]

fake3=[FakeitSettings(response='nu90002004004A01_sr.rmf',arf='nu90002004004A01_sr.arf',
                    background='nu90102011002A01_bk.pha',
                     exposure=5e4,fileName='fakeA_fhard.pi'),
FakeitSettings(response='nu90002004004B01_sr.rmf',arf='nu90002004004B01_sr.arf',
background='nu90102011002B01_bk.pha',
                     exposure=5e4,fileName='fakeB_fhard.pi')]

os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NuSTAR_AO12')
set_ener('thcomp',xrism=True)
#100mCrab in 1-10keV, ~200mCrab in 1-100keV, gamma=1.6 taking example on bright hard MAXIJ180
#here the disk is around 10%flux compared to the refl+corona comp in 1-10keV
bhard=Model('TBabs(relxill+diskbb)',modName='bhard',sourceNum=1,
            setPars={1:0.5,10:1.6,12:3,13:50,14:1,15:5.17e-03,16:0.5,17:524})
AllData.fakeit(nSpectra=2,settings=fake1,applyStats=True)

#100mCrab in 1-10keV
#reasonable wind parameters and disk temperature
#comptonization fraction makes the 15-50/3-6 flux at ~0.05 which is coherent with a middle of the pack for
#Fig. 2 in Parra25 on 4U
bsoft=Model('TBabs(mtable{pionabsAXJ1745.fits}*thcomp(diskbb))',modName='bsoft',sourceNum=1,
            setPars={1:0.5,2:4,3:10,4:300,6:2.3,7:150,8:5e-2,10:1.5,11:32.05})
AllData.fakeit(nSpectra=2,settings=fake2,applyStats=True)

#10mCrab in 1-10keV for return, no disk, gamma at 1.5, logxi/10, ~30mCrab in 1-100keV
fhard=Model('TBabs(relxill)',modName='fhard',sourceNum=1,
            setPars={1:0.5,10:1.7,11:2.1,12:3,13:50,14:1,15:7.70e-4,})
AllData.fakeit(nSpectra=2,settings=fake3,applyStats=True)



#for plotting
os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NuSTAR_AO12')
Xset.restore('mod_merge.xcm')
AllData.ignore('**-3. 79.-**')
for i in [1, 2, 3, 4, 5, 6]:
    rebinv_xrism(i, 3)
set_ener('thcomp', xrism=True)
xPlot('eeuf',auto_figsize=(7,5),group_names=['100mCrab hard FPMA', '100mCrab hard FPMB', '100mCrab soft FPMA', '100mCrab soft FPMB',
                           '10mCrab hard FPMA', '10mCrab hard FPMB'])
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-2,1.3)
#removing the legend
plt.gca().get_children()[-2].remove()
plt.subplots_adjust(bottom=0.08)
plt.tight_layout()
# plt.legend()
plt.gca().get_children()[-5].set_visible(False)
plt.tight_layout()
plt.savefig('fig_eeuf_NuSTAR.pdf')
plt.savefig('fig_eeuf_NuSTAR.png',dpi=400)
