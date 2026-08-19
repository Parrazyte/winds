import numpy as np
import matplotlib.pyplot as plt

def calc_branchingratios(rate, debug=False):



    """
    adapted from the XRISM version in
    https://qiita.com/yamadasuzaku/items/e7be399cfa974a86d27d
    see
    https://colab.research.google.com/drive/1uQLLy0cohZyw0ZjAURm6kqBpPAUCGo8U?usp=sharing#scrollTo=3LtLdHuUol32

    分岐比を計算する。武田さんSPIEに準じるが検証が必要な式。

    Analytical branching ratios estimates should always be supplanted by SIXTE simulatiosn since its difficult to
    estimate the count rate distribution per pixel

    """
    clock = 12500 # 12.5kHz = 80us
    #from the X-IFU bright source spectrum extraction slides given on Jul-2026

    grade_names=['VH',
                 'H',
                 'I',
                 'M',
                 'Lim',
                 'Low',
                 'Null']
    grade_dt_bef=np.array([24,
                           24,
                           12,
                           12,
                           12,
                           12])*1e-3

    grade_dt_aft=np.array([55.2,
                           26.8,
                           11.1,
                           3.16,
                           1.2,
                           0.05])*1e-3

    exp_branch_bef=np.exp(-1.0*(grade_dt_bef)*rate)
    exp_branch_aft=np.exp(-1.0*(grade_dt_aft)*rate)

    #ideal case
    branch_VH=exp_branch_bef[0]*exp_branch_aft[0]
    branch_H=exp_branch_bef[1]*exp_branch_aft[1]-branch_VH
    branch_I=exp_branch_bef[2]*exp_branch_aft[2]-branch_VH-branch_H
    branch_M=exp_branch_bef[3]*exp_branch_aft[3]-branch_VH-branch_H-branch_I
    branch_Lim=exp_branch_bef[4]*exp_branch_aft[4]-branch_VH-branch_H-branch_I-branch_M
    branch_Low=exp_branch_bef[5]*exp_branch_aft[5]-branch_VH-branch_H-branch_I-branch_M-branch_Lim
    branch_Null=1-branch_VH-branch_H-branch_I-branch_M-branch_Lim-branch_Low

    branch_tot=[branch_VH,branch_H,branch_I,branch_M,branch_Lim,branch_Low,branch_Null]
    if abs(sum(branch_tot)-1)>1e-5:
        print('Error: branching ratios not matching')
        breakpoint()

    return branch_tot

# rate の範囲（ハードウェアが見ている総イベントレート）
rate_arr = np.logspace(-2, 4, 200)  # 0.01 ～ 100 counts/s


branch_VH_arr,branch_H_arr,branch_I_arr,branch_M_arr,branch_Lim_arr,branch_Low_arr,branch_Null_arr = np.zeros((7,len(rate_arr)))

branch_tot_arr=[branch_VH_arr,branch_H_arr,branch_I_arr,branch_M_arr,branch_Lim_arr,branch_Low_arr,branch_Null_arr]

for i,r in enumerate(rate_arr):
    branch_VH_arr[i],branch_H_arr[i],branch_I_arr[i],branch_M_arr[i],branch_Lim_arr[i],branch_Low_arr[i],branch_Null_arr[i]\
        = calc_branchingratios(r)

rate_VH_arr,rate_H_arr,rate_I_arr,rate_M_arr,rate_Lim_arr,rate_Low_arr,rate_Null_arr =\
                                                    [branch_elem*rate_arr for branch_elem in branch_tot_arr]


# =========================
# プロット
# =========================
fig, axes = plt.subplots(
    2, 1,
    figsize=(9, 7),
    sharex=True,
    gridspec_kw={"hspace": 0.08}
)

# ---------
# 上段：branching ratio
# ---------
ax = axes[0]
ax.semilogx(rate_arr, branch_VH_arr, label="Very High (4ev)",color='darkgreen')
ax.semilogx(rate_arr, branch_H_arr, label="High (4ev)",color='limegreen')
ax.semilogx(rate_arr, branch_I_arr, label="Inter (4.2ev)",color='turquoise')
ax.semilogx(rate_arr, branch_M_arr, label="Medium (5ev)",color='blue')
ax.semilogx(rate_arr, branch_Lim_arr, label="Limited (7ev)",color='orange')
ax.semilogx(rate_arr, branch_Low_arr, label="Low  (30ev)",color='red')
ax.semilogx(rate_arr, branch_Null_arr, label="Null (XX eV)",color='gray',ls='--')

ax.set_ylabel("Branching ratio")
ax.set_ylim(0, 1.05)
ax.grid(True, which="both", ls=":")
ax.legend(ncol=3, fontsize=9)
ax.set_title("Resolve grade branching ratios (ideal Poisson model)")

# ---------
# 下段：absolute event rate
# ---------
ax = axes[1]
ax.semilogx(rate_arr, rate_VH_arr, label="Very High (4ev)",color='darkgreen')
ax.semilogx(rate_arr, rate_H_arr, label="High (4ev)",color='limegreen')
ax.semilogx(rate_arr, rate_I_arr, label="Inter (4.2ev)",color='turquoise')
ax.semilogx(rate_arr, rate_M_arr, label="Medium (5ev)",color='blue')
ax.semilogx(rate_arr, rate_Lim_arr, label="Limited (7ev)",color='orange')
ax.semilogx(rate_arr, rate_Low_arr, label="Low  (30ev)",color='red')
ax.semilogx(rate_arr, rate_Null_arr, label="Null (XX eV)",color='gray',ls='--')

ax.set_xlabel("Total event rate [counts/s]")
ax.set_ylabel("Event rate per grade [counts/s]")
ax.grid(True, which="both", ls=":")
ax.legend(ncol=3, fontsize=9)
ax.set_ylim(0,20)
plt.tight_layout()
plt.show()

plop=FakeitSettings(response='new_athena_xifu_mar_v2_4eV_gaussian.rmf',
                    arf='new_athena_xifu_mar_v2_no_filter.arf',
                    bkg='new_athena_xifu_mar_v2_nxb_1amin2.pha',
                    exposure=1000000,)
AllData.fakeit(settings=plop,applyStats=True)
#with SED
'''
AllModels.calcFlux("0.3 10.")
 Model Flux    1.4408 photons (7.824e-09 ergs/cm^2/s) range (0.30000 - 10.000 keV)
 
 Net count rate (cts/s) for Spectrum:1  4.528e+03 +/- 6.729e-02

'''

flux=7.824e-09/2.4e-8

count_rate=4.528e+03

'''
from Kammount18, in defocused at 1keV 27 pixels of width (13.5 diameter).
apprximately 573 pxiels covered. removing 13*6 for the main lines, we get 495 pixels.
Assuming even illumination at first approximation we get
4.528e+03/495
Out[12]: 9.147474747474748
'''

#9.2cps is approximately 0.8 throughput for grades 1 to 3 so we go ahead with that

plop2=FakeitSettings(response='new_athena_xifu_mar_v2_4eV_gaussian.rmf',
                    arf='new_athena_xifu_mar_v2_no_filter.arf',
                    background='new_athena_xifu_mar_v2_nxb_1amin2.pha',
                    exposure=5e3*0.8,)
AllData.fakeit(settings=plop2,applyStats=True)

os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue')
Xset.restore('mod_bothsim_5ks.xcm')
set_ener('thcomp',xrism=True)
rebinv_xrism(1,10)
rebinv_xrism(2,10)
Plot.add=False
xPlot('ldata',xlims=[0.8,2.05],mult_factors=[1.,0.5],ylims=[5,2600])


os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue')
Xset.restore('mod_allsim.xcm')
set_ener('thcomp',xrism=True)
rebinv_xrism(1,10)
rebinv_xrism(2,10)
rebinv_xrism(3,10)
rebinv_xrism(4,10)
AllData(3).ignore('**-1.')
AllData(4).ignore('**-1.')
Plot.add=False
xPlot('ldata',xlims=[0.8,2.05],mult_factors=[1.,0.5,0.1,0.04],
      data_colors=['darkblue','orange','darkblue','orange'],
      group_names=[r"log$\xi$=4  |  NH=$10^{23}$ cm$^{-2}$",
                   r"log$\xi$=2  |  NH=$10^{22}$ cm$^{-2}$","",""],
      model_colors=['darkblue','orange','darkblue','orange'],ylims=[8,2600])


'''''''''''''''
EEUF PLOT
'''''''''''''''

os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue')
Xset.restore('mod_allsim.xcm')
set_ener('thcomp',xrism=True)
rebinv_xrism(1,10,max_bins=60000)
rebinv_xrism(2,10,max_bins=60000)
rebinv_xrism(3,10,max_bins=60000)
rebinv_xrism(4,10,max_bins=60000)
AllData(3).ignore('**-0.8')
AllData(4).ignore('**-0.8')
Plot.add=False
xPlot('eeuf',xlims=[0.8,2.05],mult_factors=[1.,0.5,0.0008,0.0003],
      data_colors=['darkblue','orange','darkblue','orange'],
      group_names=[r"log$\xi$=4($\pm0.01$) | log$_{10}$NH=23($\pm0.01$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm3$) km s$^{-1}$ | v$_{out}$=300($\pm3$) km s$^{-1}$",
                   r"log$\xi$=2($\pm0.03$) | log$_{10}$NH=22($\pm0.004$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm1$) km s$^{-1}$ | v$_{out}$=300($\pm2$) km s$^{-1}$",
                   "",""],
      model_colors=['darkblue','orange','darkblue','orange'],ylims=[1e-5,4])
plt.yscale('log')
ax=plt.gca()
rebinv_xrism(2,5)
#axins.remove()
axins = ax.inset_axes(
    [0.57, 0.42, 0.25, 0.31],
    xlim=(1.99, 2.01), ylim=(1e-2, 1e0), xticklabels=[], yticklabels=[])
xPlot('eeuf',axes_input=[axins],mult_factors=[0.0001,0.5,0.001,0.0001],
      data_colors=['darkblue','orange','darkblue','orange'],model_colors=['darkblue','orange','darkblue','orange']
)
axins.set_ylabel('')
axins.set_xlabel('')
axins.set_title('')
axins.set_xlim(1.9975,2.0125)
axins.set_ylim(0.09,0.9)
axins.get_children()[-3].remove()
axins.get_children()[-2].remove()
ax.indicate_inset_zoom(axins, edgecolor="black",alpha=0.5)
axins.set_yscale('log')
axins.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=True,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=True,
    labeltop=False,
    direction='out')
axins.set_xticklabels([1.995,2.000,2.005,2.010])
axins.text(1.998, 1.7e-1, r'v=300$\pm$6 km s$^{-1}$', fontsize=10)
axins.text(1.998, 1.3e-1, r'$\sigma$=100$\pm$7 km s$^{-1}$', fontsize=10)
axins.text(1.998, 1.e-1, r'EW=3.18$\pm$0.04 eV', fontsize=10)
ax.legend(loc='upper left')

ax2 = ax.twinx()
ax2.errorbar([],[],xerr=[],yerr=[],color='darkblue',
             label=r"log$\xi$=4($\pm0.2$) | log$_{10}$NH=23($\pm0.2$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm30$) km s$^{-1}$ | v$_{out}$=300($\pm60$) km s$^{-1}$")
ax2.errorbar([],[],xerr=[],yerr=[],color='orange',
             label=r"log$\xi$=2($\pm0.1$) | log$_{10}$NH=22($\pm0.1$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm15$) km s$^{-1}$ | v$_{out}$=300($\pm30$) km s$^{-1}$")
ax2.get_yaxis().set_visible(False)
ax2.legend(loc='lower right')

'''''''''''''''
RATIO PLOT
'''''''''''''''

os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue')
Xset.restore('mod_allsim.xcm')
set_ener('thcomp',xrism=True)
rebinv_xrism(1,10)
rebinv_xrism(2,10)
rebinv_xrism(3,10)
rebinv_xrism(4,10)
AllData(3).ignore('**-1.')
AllData(4).ignore('**-1.')
Plot.add=False
delcomp('pion_abs')
xPlot('ratio',xlims=[0.8,2.05],mult_factors=[1.,0.55,0.1,0.04],
      data_colors=['darkblue','orange','darkblue','orange'],
      group_names=[r"log$\xi$=4  |  NH=$10^{23}$ cm$^{-2}$",
                   r"log$\xi$=2  |  NH=$10^{22}$ cm$^{-2}$","",""],
      model_colors=['darkblue','orange','darkblue','orange'],ylims=[1e-2,1.1])
plt.yscale('log')
plt.axhline(y=0.55,xmin=0,xmax=1,color='green')
plt.axhline(y=0.1,xmin=0.162,xmax=1,color='green')
plt.axhline(y=0.04,xmin=0.162,xmax=1,color='green')

os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue')
Xset.restore('mod_test_soft_highxi_1e23.xcm')
set_ener('thcomp',xrism=True)
plop2=[FakeitSettings(response='aciss_heg-1_cy28.grmf',
                    arf='aciss_heg-1_cy28.garf',
                     fileName='fake_highxi_src_5ks_heg_-1.pha',
                    exposure=5e3*1.,),
       FakeitSettings(response='aciss_heg1_cy28.grmf',
                      arf='aciss_heg1_cy28.garf',
                      fileName='fake_highxi_src_5ks_heg_1.pha',
                      exposure=5e3 * 1., )
       ]

AllData.fakeit(nSpectra=2,settings=plop2,applyStats=True)
AllData('2:2 fake_highxi_src_50ks_heg_1.pha')
rebinv_xrism(1,10)
rebinv_xrism(2,10)
set_ener('thcomp',xrism=True)
AllData.ignore('**-0.3 10.-**')
xPlot('eeuf',xlims=[0.8,2.05],ylims=[1e-5,2],mult_factors=[0.0008,0.0008])
plt.yscale('log')

Xset.restore('mod_test_soft_midxi_1e22.xcm')

os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue')
Xset.restore('mod_test_soft_midxi_1e22.xcm')
set_ener('thcomp',xrism=True)
plop2=[FakeitSettings(response='aciss_heg-1_cy28.grmf',
                    arf='aciss_heg-1_cy28.garf',
                     fileName='fake_midxi_src_50s_heg_-1.pha',
                    exposure=5e1*1.,),
       FakeitSettings(response='aciss_heg1_cy28.grmf',
                      arf='aciss_heg1_cy28.garf',
                      fileName='fake_midxi_src_50s_heg_1.pha',
                      exposure=5e1 * 1., )
       ]
AllData.fakeit(nSpectra=2,settings=plop2,applyStats=True)


os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue')
Xset.restore('mod_allsim_Chandra_highexp.xcm')
rebinv_xrism(1,10)
rebinv_xrism(2,10)
rebinv_xrism(3,10)
rebinv_xrism(4,10)
rebinv_xrism(5,10)
rebinv_xrism(6,10)
rebinv_xrism(7,10)
rebinv_xrism(8,10)
set_ener('thcomp',xrism=True)
AllData.ignore('**-0.3 10.-**')
AllData(5).ignore('**-1.')
AllData(6).ignore('**-1.')
AllData(7).ignore('**-1.')
AllData(8).ignore('**-1.')
Plot.add=False
xPlot('eeuf',xlims=[0.8,2.05],mult_factors=[1.,1.,0.5,0.5,0.0008,0.0008,0.0003,0.0003],
      data_colors=['darkblue','mediumblue','orange','gold','darkblue','mediumblue','orange','gold'],
      group_names=[r"HETG +/-1 log$\xi$=4  |  NH=$10^{23}$ cm$^{-2}$",
                   #r"HEG  1 log$\xi$=4  |  NH=$10^{23}$ cm$^{-2}$",
                   "",
                   r"HETG +/-1 log$\xi$=2  |  NH=$10^{22}$ cm$^{-2}$",
                   #r"HEG  1 log$\xi$=2  |  NH=$10^{22}$ cm$^{-2}$",
                   "",
                   "","","",""],
      model_colors=['darkblue','mediumblue','orange','gold','darkblue','mediumblue','orange','gold'],
      ylims=[1e-5,2])
plt.yscale('log')
plt.legend(ncol=1,loc='upper left')
plt.tight_layout()
plt.savefig('5Ms_50ks_compa_eeuf_Chandra.pdf')

os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue')
Xset.restore('mod_allsim_Chandra_lowexp.xcm')
rebinv_xrism(1,10)
rebinv_xrism(2,10)
rebinv_xrism(3,10)
rebinv_xrism(4,10)
rebinv_xrism(5,3)
rebinv_xrism(6,3)
rebinv_xrism(7,3)
rebinv_xrism(8,3)
set_ener('thcomp',xrism=True)
AllData.ignore('**-0.3 10.-**')
AllData(5).ignore('**-1.')
AllData(6).ignore('**-1.')
AllData(7).ignore('**-1.')
AllData(8).ignore('**-1.')
Plot.add=False
xPlot('eeuf',xlims=[0.8,2.05],mult_factors=[1.,1.,0.5,0.5,0.0008,0.0008,0.0003,0.0003],
      data_colors=['darkblue','mediumblue','orange','gold','darkblue','mediumblue','orange','gold'],
      group_names=[r"HETG +/-1 log$\xi$=4  |  NH=$10^{23}$ cm$^{-2}$",
                   #r"HEG  1 log$\xi$=4  |  NH=$10^{23}$ cm$^{-2}$",
                   "",
                   r"HETG +/-1 log$\xi$=2  |  NH=$10^{22}$ cm$^{-2}$",
                   #r"HEG  1 log$\xi$=2  |  NH=$10^{22}$ cm$^{-2}$",
                   "",
                   "","","",""],
      model_colors=['darkblue','mediumblue','orange','gold','darkblue','mediumblue','orange','gold'],
      ylims=[1e-5,2])
plt.yscale('log')
plt.legend(ncol=1,loc='upper left')
plt.tight_layout()
plt.savefig('5ks_50s_compa_eeuf_Chandra.pdf')


'''

for fitting the main line
energy average of the 2 transitions is (2.0060*2+2.0043)/3=2.0054333

'''

os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue')
Xset.restore("postfit_gaussian_SiXIVKa_5ksNA.xcm")
set_ener('thcomp',xrism=True)

width_todv(2.0054333,
           val=AllModels(1)(2).values[0]-2.0054333,
           err=abs(np.array(AllModels(1)(2).error[:-1])-AllModels(1)(2).values[0]))

'''
gives 
(296.2686172795168,
 np.float64(5.024999700833348),
 np.float64(6.244415710523015))
 at 90%
'''

'''
for the width
'''
width_todv(2.0054333,par=3)
'''
gives 
(197.56312677054,
 np.float64(8.592459583390394),
 np.float64(4.8978613996528395))
 at 90% for the width. Unsure why this is x2 compared to pion, width is lower when 2 gaussians are used but still weird.
 We divide by 2 for simplicity in the final values.
'''

'''
for the EW

we get

Additive group equiv width for Component 2:  -0.00318151 keV
Parameter distribution is derived from fit covariance matrix.
Equiv width error range:  -12.2962 - -0.0032172 keV

clearly issue for lower boundary probably from xspec routine, we take the other side of the uncertainty as symmetric

AllData(1).eqwidth
Out[12]: (-0.003181506914, -12.29623712, -0.00321719946)

'''
