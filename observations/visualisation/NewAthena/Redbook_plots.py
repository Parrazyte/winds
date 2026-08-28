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
EEUF PLOT OLD CONSTRAINTS
'''''''''''''''

os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue/Redbook/SIXTE/common')
Xset.restore('mod_highmid_NA.xcm')
set_ener('large_canon',xrism=True)
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
      model_colors=['darkblue','orange','darkblue','orange'],ylims=[5e-6,4])
plt.yscale('log')
ax=plt.gca()
rebinv_xrism(2,5)
#axins.remove()
axins = ax.inset_axes(
    [0.57, 0.45, 0.25, 0.29],
    xlim=(1.99, 2.01), ylim=(1e-2, 1e0), xticklabels=[], yticklabels=[])
xPlot('eeuf',axes_input=[axins],mult_factors=[0.0001,0.5,0.001,0.0001],
      data_colors=['darkblue','orange','darkblue','orange'],model_colors=['darkblue','orange','darkblue','orange']
)
axins.set_ylabel('')
axins.set_xlabel('')
axins.set_title('')
axins.set_xlim(1.9975,2.0125)
axins.set_ylim(0.08,0.9)
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
axins.text(1.998, 1.47e-1, r'v=300$\pm$6 km s$^{-1}$', fontsize=10)
axins.text(1.998, 1.15e-1, r'$\sigma$=100$\pm$7 km s$^{-1}$', fontsize=10)
axins.text(1.998, 0.9e-1, r'EW=3.18$\pm$0.04 eV', fontsize=10)
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
EEUF PLOT NEW CONSTRAINTS

UPDATED FIT CONSTRAINTS

highxi_5k
Fit.error('1-5')
*** Parameter 3 is not a variable model parameter and no confidence range will be calculated.
 Parameter   Confidence Range (2.706)
     1       3.9856      4.01636    (-0.0108282,0.0199298)
     2      9.26066      10.2535    (-0.403861,0.588933) (biggest err 0.03 in log space)
     4      93.8147      100.431    (-3.46,3.15675)
     5  -0.00100814 -0.000983041    (-1.26006e-05,1.24954e-05) (biggest err 5km/s)

r"log$\xi$=4($\pm0.02$) | log$_{10}$NH=23($\pm0.03$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm6$) km s$^{-1}$ | v$_{out}$=300($\pm5$) km s$^{-1}$"
                   
midxi_5k
Fit.error('1-5')
*** Parameter 3 is not a variable model parameter and no confidence range will be calculated.
 Parameter   Confidence Range (2.706)
     1      1.99575      2.00191    (-0.00300444,0.0031583)
     2     0.995404      1.01416    (-0.00993655,0.00882077)
     4      99.7929      102.244    (-1.24636,1.20499)
     5   -0.0010076 -0.000992258    (-7.18035e-06,8.15769e-06)
     
r"log$\xi$=2($\pm0.003$) | log$_{10}$NH=22($\pm0.001$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm2$) km s$^{-1}$ | v$_{out}$=300($\pm2$) km s$^{-1}$"
                   
lowxi_5k
Fit.error('1-5')
*** Parameter 3 is not a variable model parameter and no confidence range will be calculated.
 Parameter   Confidence Range (2.706)
***Warning: Parameter pegged at hard limit: 0
     1            0    0.0135866    (0,0.0135866)
     2      0.96802      1.00698    (-0.0250994,0.0138585)
     4      83.6407      116.071    (-11.3504,21.0804)
     5   -0.0011345 -0.000931838    (-0.000102046,0.000100612)
     
highxi_50s
Fit.error('1-5')
*** Parameter 3 is not a variable model parameter and no confidence range will be calculated.
 Parameter   Confidence Range (2.706)
     1      3.68796       4.0107    (-0.153881,0.168863)
     2      4.61923      13.9305    (-2.85327,6.458)
***Warning: Parameter pegged at hard limit: 30
     4            0      117.287    (-91.6321,25.6549)
     5  -0.00111196 -0.000841007    (-0.000137284,0.000133669)
     
r"log$\xi$=4($\pm0.3$) | log$_{10}$NH=23($\pm0.3$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($<120$) km s$^{-1}$ | v$_{out}$=300($\pm50$) km s$^{-1}$"
                   
midxi_50s
Fit.error('1-5')
*** Parameter 3 is not a variable model parameter and no confidence range will be calculated.
 Parameter   Confidence Range (2.706)
     1      1.97147      2.01801    (-0.0267206,0.019813)
     2     0.777882     0.972572    (-0.0962392,0.0984506)
     4      90.8211      121.476    (-14.331,16.3237)
***Warning: Number of trials exceeded before convergence.
Current trial values -0.00109946, -0.00110069
and delta statistic 2.69561, 2.78526
     5  -0.00110033 -0.000932341    (-8.7734e-05,8.02505e-05)
     
r"log$\xi$=2($\pm0.03$) | log$_{10}$NH=22($\pm0.1$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm20$) km s$^{-1}$ | v$_{out}$=300($\pm30$) km s$^{-1}$"
                   
lowxi_50s
 Null hypothesis probability of 1.00e+00 with 16375 degrees of freedom
***Warning: New best fit found, fit parameters will be set to new values.
***Warning: Parameter pegged at hard limit: 0
     1            0      0.25063    (-0.0642196,0.18641)
Apparent non-monotonicity in statistic space detected.
Current bracket values 1.1911, 1.22068
and delta stat 1.91202, 2.80029
but latest trial 1.21779 gives 3.41876
Suggest that you check this result using the steppar command.
     2     0.813673      1.20589    (-0.252051,0.140168)
     4      120.322      579.947    (-179.563,280.063)
Apparent non-monotonicity in statistic space detected.
Current bracket values 0.0008513, 0.000851339
and delta stat 2.67931, 3.32364
but latest trial 0.000851305 gives 2.67931
Suggest that you check this result using the steppar command.
     5  -0.00182273   0.00085132    (-0.00128022,0.00139383)
cannot get the right error, I think the errors cannot find other minima.
Delta-C for the component is 39 but may come from the continuum


2keV line midxi 5ks

parameters:
Fit.error('3-6')
*** Parameter 4 is not a variable model parameter and no confidence range will be calculated.
 Parameter   Confidence Range (2.706)
     3     -296.234     -284.846    (-5.70402,5.68455)
     5  0.000822865   0.00100671    (-9.48143e-05,8.90275e-05)
     6  -0.00042296 -0.000409394    (-6.80026e-06,6.76576e-06)
BUT width weirdly overestimated - may be an issue with the line shape

AllModels.eqwidth(3,err=True,number=100,level=90)
Data group number: 1
Additive group equiv width for Component 3:  -0.00103798 keV
Parameter distribution is derived from fit covariance matrix.
Equiv width error range:  -0.00105312 - -0.00102204 keV
AllModels.eqwidth(4,err=True,number=100,level=90)
Data group number: 1
Additive group equiv width for Component 4:  -0.00206403 keV
Parameter distribution is derived from fit covariance matrix.
Equiv width error range:  -0.00210348 - -0.00203517 keV

With single line:
     1      2.00742      2.00752    (-4.05243e-05,5.7381e-05)
     2   0.00117655   0.00134646    (-8.12641e-05,8.86397e-05)
width even more overestimated, velocity better constrained to +/-8 though

With pion restricted in this range (only logxi fixed)
Fit.error('4-5')
 Parameter   Confidence Range (2.706)
     4      95.0132      111.527    (-7.73489,8.77935)
     5  -0.00103057 -0.000991398    (-2.00181e-05,1.91501e-05)
for width and velocity: better constraints !

constraints to present
r'v=300$\pm$9 km s$^{-1}$'
r'$\sigma$=100$\pm$12 km s$^{-1}$'
r'EW=3.1$\pm$0.05 eV'


7keV line highxi 5ks
With pion restricted in this range (only logxi fixed)
Fit.error('4-5')
 Parameter   Confidence Range (2.706)
     4      82.9941      100.325    (-5.52019,11.8107)
     5  -0.00100552 -0.000965853    (-2.01005e-05,1.9563e-05)
     
With empi lines
AllModels.eqwidth(3,err=True,level=90,number=100)
Data group number: 1
Additive group equiv width for Component 3:  -0.00991177 keV
Parameter distribution is derived from fit covariance matrix.
Equiv width error range:  -0.0102735 - -0.00956635 keV
AllModels.eqwidth(3,err=True,level=90,number=1000)
Data group number: 1
Additive group equiv width for Component 3:  -0.00991177 keV
Parameter distribution is derived from fit covariance matrix.
Equiv width error range:  -0.0102869 - -0.00953066 keV
AllModels.eqwidth(4,err=True,level=90,number=1000)
Data group number: 1
Additive group equiv width for Component 4:  -0.0105658 keV
Parameter distribution is derived from fit covariance matrix.
Equiv width error range:  -0.0109629 - -0.0101568 keV
0.00991177+0.0105658
Out[93]: 0.02047757
0.0102869+0.00953066
Out[94]: 0.019817559999999998
0.0109629+0.0101568
Out[95]: 0.021119699999999998

'''''''''''''''


'''
With merged Chandra

'''

os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue/Redbook/SIXTE/common')
Xset.restore('mod_highmid_NA_Chandra.xcm')
set_ener('large_canon',xrism=True)

rebinv_xrism(1,10,max_bins=60000)
rebinv_xrism(2,10,max_bins=60000)
rebinv_xrism(3,10,max_bins=60000)
rebinv_xrism(4,10,max_bins=60000)
rebinv_xrism(5,7,max_bins=60000)
rebinv_xrism(6,7,max_bins=60000)
rebinv_xrism(7,3,max_bins=60000)
rebinv_xrism(8,3,max_bins=60000)

AllData(3).ignore('**-0.8')
AllData(4).ignore('**-0.8')
AllData(5).ignore('**-0.8')
AllData(6).ignore('**-0.8')
AllData(7).ignore('**-0.8')
AllData(8).ignore('**-0.8')

Plot.add=False
xPlot('eeuf',xlims=[0.8,2.05],mult_factors=[1.,0.5,0.0008,0.0003,1.,0.5,0.0008,0.0003],
      data_colors=['darkblue','orange','darkblue','orange','grey','grey','grey','grey'],
      group_names=[r"log$\xi$=4($\pm0.02$) | log$_{10}$NH=23($\pm0.03$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm6$) km s$^{-1}$ | v$_{out}$=300($\pm5$) km s$^{-1}$",
                   r"log$\xi$=2($\pm0.003$) | log$_{10}$NH=22($\pm0.001$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm2$) km s$^{-1}$ | v$_{out}$=300($\pm2$) km s$^{-1}$",
                   "","","","","",""],
      data_alpha=[1,1,1,1,0.5,0.5,0.5,0.5],
      model_colors=['darkblue','orange','darkblue','orange','None','None','None','None'],ylims=[5e-6,4])

plt.yscale('log')
ax=plt.gca()

rebinv_xrism(2,5)
rebinv_xrism(6,5)
#axins.remove()
axins = ax.inset_axes(
    [0.57, 0.45, 0.25, 0.29],
    xlim=(1.99, 2.01), ylim=(1e-2, 1e0), xticklabels=[], yticklabels=[])
xPlot('eeuf',axes_input=[axins],mult_factors=[0.0001,0.5,0.001,0.0001,0.0001,0.5,0.001,0.0001],
      data_alpha=[1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5],
      data_colors=['darkblue','orange','darkblue','orange','grey','grey','grey','grey'],
      model_colors=['darkblue','orange','darkblue','orange','None','None','None','None'])
axins.set_ylabel('')
axins.set_xlabel('')
axins.set_title('')
axins.set_xlim(1.9975,2.0125)
axins.set_ylim(0.08,0.9)
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
axins.text(1.998, 1.47e-1, r'v=300$\pm$9 km s$^{-1}$', fontsize=10)
axins.text(1.998, 1.15e-1, r'$\sigma$=100$\pm$12 km s$^{-1}$', fontsize=10)
axins.text(1.998, 0.9e-1, r'EW=3.1$\pm$0.05 eV', fontsize=10)
ax.legend(loc='upper left')

ax2 = ax.twinx()
ax2.errorbar([],[],xerr=[],yerr=[],color='darkblue',
             label=r"log$\xi$=4($\pm0.3$) | log$_{10}$NH=23($\pm0.3$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($<120$) km s$^{-1}$ | v$_{out}$=300($\pm50$) km s$^{-1}$")
ax2.errorbar([],[],xerr=[],yerr=[],color='orange',
             label=r"log$\xi$=2($\pm0.03$) | log$_{10}$NH=22($\pm0.1$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm20$) km s$^{-1}$ | v$_{out}$=300($\pm30$) km s$^{-1}$")
ax2.get_yaxis().set_visible(False)
ax2.legend(loc='lower right')
plt.tight_layout()



'''
--------------------------------------------------------------------------------------
3 MODS NO CHANDRA + OLD
'''


# xPlot('eeuf',xlims=[0.8,2.05],mult_factors=np.array([n**-5.4,n**-2.35,n**-0.5,n**-5.1,n**-1.5,n**-0]),
#       data_colors=['darkblue','orange','darkred','darkblue','orange','darkred'],
#       group_names=[r"log$\xi$=4($\pm0.01$) | log$_{10}$NH=23($\pm0.01$) cm$^{-2}$""\n"
#                    +r"v$_{turb}$=100($\pm3$) km s$^{-1}$ | v$_{out}$=300($\pm3$) km s$^{-1}$",
#                    r"log$\xi$=2($\pm0.03$) | log$_{10}$NH=22($\pm0.004$) cm$^{-2}$""\n"
#                    +r"v$_{turb}$=100($\pm1$) km s$^{-1}$ | v$_{out}$=300($\pm2$) km s$^{-1}$",
#                   "",
#                    "","",""],
#       model_colors=['darkblue','orange','darkred','darkblue','orange','darkred'],ylims=np.array([3e-9,2.5]))
# plt.yscale('log')
# ax=plt.gca()
# ax.legend(loc='lower right')
# plt.tight_layout()

os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue/Redbook/SIXTE/common')
Xset.restore('mod_full_NA.xcm')
AllData(4).ignore('**-0.8')
AllData(5).ignore('**-0.8')
AllData(6).ignore('**-0.8')
set_ener('large_canon',xrism=True)

rebinv_xrism(1,10,max_bins=60000)
rebinv_xrism(2,10,max_bins=60000)
rebinv_xrism(3,10,max_bins=60000)
rebinv_xrism(4,10,max_bins=60000)
rebinv_xrism(5,10,max_bins=60000)
rebinv_xrism(6,10,max_bins=60000)
Plot.add=False
n=15
xPlot('eeuf',xlims=[0.8,2.05],mult_factors=np.array([n**-5.4,n**-2.35,n**-0.5,n**-5.1,n**-1.5,n**-0]),
      data_colors=['darkblue','orange','darkred','darkblue','orange','darkred'],
      group_names=[r"log$\xi$=4($\pm0.01$) | log$_{10}$NH=23($\pm0.01$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm3$) km s$^{-1}$ | v$_{out}$=300($\pm3$) km s$^{-1}$",
                   r"log$\xi$=2($\pm0.03$) | log$_{10}$NH=22($\pm0.004$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm1$) km s$^{-1}$ | v$_{out}$=300($\pm2$) km s$^{-1}$",
                  "",
                   "","",""],
      model_colors=['darkblue','orange','darkred','darkblue','orange','darkred'],ylims=np.array([1e-9,2.5]))
plt.yscale('log')
ax=plt.gca()
ax.legend(loc='lower right')
plt.tight_layout()
#rebinv_xrism(2,5,max_bins=60000)
#axins.remove()
axins = ax.inset_axes(
    [0.555, 0.375, 0.25, 0.215],
    xlim=(1.99, 2.01), ylim=(1e-2, 1e0), xticklabels=[], yticklabels=[])
xPlot('eeuf',axes_input=[axins],mult_factors=[n**-5.4,n**-2.35,n**-0.5,n**-5.1,n**-1.5,n**-0],
      data_colors=['darkblue','orange','darkred','darkblue','orange','darkred'],
      model_colors=['darkblue','orange','darkred','darkblue','orange','darkred'])
axins.set_ylabel('')
axins.set_xlabel('')
axins.set_title('')
axins.set_xlim(1.9975,2.0125)
axins.set_ylim(2.5e-4,3e-3)
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
    direction='in')
axins.tick_params(
    axis='y',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    left=False,  # ticks along the bottom edge are off
    right=True,  # ticks along the top edge are off
    direction='in',labelleft=False,labelright=True)


axins.set_xticklabels([1.995,2.000,2.005,2.010])
axins.text(1.998, 6.6e-4, r'v=300$\pm$9$\,$km$\,$s$^{-1}$', fontsize=10)
axins.text(1.998, 4.5e-4, r'$\sigma$=100$\pm$12$\,$km$\,$s$^{-1}$', fontsize=10)
axins.text(1.998, 3.2e-4, r'EW=3.18$\pm$0.05$\,$eV', fontsize=10)
# ax.legend(loc='lower right')
plt.tight_layout()
ax.get_children()[-2].remove()
# ax2 = ax.twinx()
# ax2.errorbar([],[],xerr=[],yerr=[],color='darkblue',
#              label=r"log$\xi$=4($\pm0.2$) | log$_{10}$NH=23($\pm0.2$) cm$^{-2}$""\n"
#                    +r"v$_{turb}$=100($\pm30$) km s$^{-1}$ | v$_{out}$=300($\pm60$) km s$^{-1}$")
# ax2.errorbar([],[],xerr=[],yerr=[],color='orange',
#              label=r"log$\xi$=2($\pm0.1$) | log$_{10}$NH=22($\pm0.1$) cm$^{-2}$""\n"
#                    +r"v$_{turb}$=100($\pm15$) km s$^{-1}$ | v$_{out}$=300($\pm30$) km s$^{-1}$")
# ax2.get_yaxis().set_visible(False)
# ax2.legend(loc='lower right')

'''
invert blue yellow and red to get the logxi=0 above - it will make it easier to plot the high energy panel
perhaps 6.5-8.5 with zoom or 6.5-7.0 - no need to put the 50s when they cannot do anything
'''
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.patches import FancyBboxPatch


# ---------------------------------------------------------
# Dummy plot
# ---------------------------------------------------------

# fig, ax = plt.subplots(figsize=(9, 4))
#
# x = [1, 2, 3, 4, 5]
#
# ax.errorbar(
#     x, [1, 2, 1.5, 2.5, 2],
#     yerr=[0.1, 0.2, 0.15, 0.2, 0.1],
#     color="darkblue",
#     marker="+",
# )
#
# ax.errorbar(
#     x, [1.5, 1.8, 2.2, 2.0, 2.5],
#     yerr=[0.15, 0.1, 0.2, 0.15, 0.2],
#     color="orange",
#     marker="+",
# )
#
# ax.errorbar(
#     x, [2, 2.3, 2.0, 2.7, 2.4],
#     yerr=[0.1, 0.15, 0.1, 0.2, 0.15],
#     color="darkred",
#     marker="+",
# )


# ---------------------------------------------------------
# Colors
# ---------------------------------------------------------

c1 = "darkblue"
c2 = "orange"
c3 = "darkred"


# ---------------------------------------------------------
# Legend contents
# ---------------------------------------------------------

data = [
    [
        r"log$\xi$ =",
        r"4($\pm0.01$)",
        r"2($\pm0.03$)",
        r"4($\pm1.01$)",
    ],
    [
        r"log$_{10}$NH [cm$^{-2}$] =",
        r"23($\pm0.01$)",
        r"22($\pm0.004$)",
        r"23($\pm0.01$)",
    ],
    [
        r"$v_{\rm turb} $ [km s$^{-1}$]=",
        r"100($\pm3$)",
        r"100($\pm1$)",
        r"100($\pm3$)",
    ],
    [
        r"$v_{\rm out}$ km s$^{-1}$] =",
        r"300($\pm3$)",
        r"300($\pm2$)",
        r"300($\pm3$)",
    ],
]

colors = [
    ["black", c1, c2, c3],
    ["black", c1, c2, c3],
    ["black", c1, c2, c3],
    ["black", c1, c2, c3],
]


# ---------------------------------------------------------
# Position and size of the legend
# ---------------------------------------------------------
#
# All values are in axes coordinates:
# [left, bottom, width, height]
#

bbox = [0.47, 0.02, 0.44, 0.16]


# ---------------------------------------------------------
# Draw the legend border FIRST
# ---------------------------------------------------------

border = FancyBboxPatch(
    (bbox[0], bbox[1]),
    bbox[2],
    bbox[3],
    boxstyle="round,pad=0.008,rounding_size=0.008",
    transform=ax.transAxes,
    facecolor="white",
    edgecolor="grey",
    linewidth=0.8,
    zorder=2,
)

ax.add_patch(border)


# ---------------------------------------------------------
# Create the table
# ---------------------------------------------------------

table = Table(
    ax,
    bbox=bbox,
)


# ---------------------------------------------------------
# Column widths
# ---------------------------------------------------------
#
# These are relative widths. Reducing them reduces the
# horizontal space occupied by the legend.
#
# The first column is wider because it contains the labels
# and units. The three value columns are deliberately narrow.
#

col_widths = [
    0.36,   # labels
    0.22,   # dark blue
    0.22,   # orange
    0.22,   # dark red
]


# Row height
row_height = 0.11


# ---------------------------------------------------------
# Add cells
# ---------------------------------------------------------

for i, row in enumerate(data):

    for j, value in enumerate(row):

        cell = table.add_cell(
            i,
            j,
            width=col_widths[j],
            height=row_height,
            text=value,
            loc="left",
            facecolor="white",
            edgecolor="none",
        )

        cell.get_text().set_color(colors[i][j])
        cell.get_text().set_fontsize(11)

        # Reduce padding inside each cell.
        cell.PAD = 0.0


# ---------------------------------------------------------
# Make all cell borders invisible
# ---------------------------------------------------------

for cell in table.get_celld().values():
    cell.set_linewidth(0)
    cell.set_edgecolor("none")


# ---------------------------------------------------------
# Add table on top of the border
# ---------------------------------------------------------

table.set_zorder(3)
ax.add_table(table)
plt.show()

'''
WITH CHANDRA DATA
'''


os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue/Redbook/SIXTE/common')
Xset.restore('mod_full_NA_Chandra.xcm')

AllData.ignore('**-0.8')

set_ener('large_canon',xrism=True)

rebinv_xrism(1,10,max_bins=60000)
rebinv_xrism(2,10,max_bins=60000)
rebinv_xrism(3,10,max_bins=60000)
rebinv_xrism(4,10,max_bins=60000)
rebinv_xrism(5,10,max_bins=60000)
rebinv_xrism(6,10,max_bins=60000)
rebinv_xrism(7,5,max_bins=60000)
rebinv_xrism(8,5,max_bins=60000)
rebinv_xrism(9,5,max_bins=60000)
rebinv_xrism(10,3,max_bins=60000)
rebinv_xrism(11,3,max_bins=60000)
rebinv_xrism(12,3,max_bins=60000)

Plot.add=False
n=15
xPlot('eeuf',xlims=[0.8,2.05],mult_factors=np.array([n**-5.4,n**-2.35,n**-0.5,n**-5.1,n**-1.5,n**-0,
                                                     n**-5.4,n**-2.35,n**-0.5,n**-5.1,n**-1.5,n**-0]),
      data_colors=['darkblue','orange','darkred','darkblue','orange','darkred',
                   'grey','grey','grey','grey','grey','grey'],
      group_names=[r"log$\xi$=4($\pm0.01$) | log$_{10}$NH=23($\pm0.01$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm3$) km s$^{-1}$ | v$_{out}$=300($\pm3$) km s$^{-1}$",
                   r"log$\xi$=2($\pm0.03$) | log$_{10}$NH=22($\pm0.004$) cm$^{-2}$""\n"
                   +r"v$_{turb}$=100($\pm1$) km s$^{-1}$ | v$_{out}$=300($\pm2$) km s$^{-1}$",
                  "",
                   "","","",
                   "","","",
                   "","",""],
      data_alpha=[1, 1, 1, 1, 1,1, 0.5, 0.5, 0.5, 0.5,0.5,0.5],
      model_colors=['darkblue','orange','darkred','darkblue','orange','darkred',
                    'None','None','None','None','None','None'],ylims=np.array([1e-9,2.5]))
plt.yscale('log')
ax=plt.gca()
ax.legend(loc='lower right')
plt.tight_layout()
rebinv_xrism(2,5,max_bins=60000)
rebinv_xrism(8,5,max_bins=60000)

#axins.remove()
axins = ax.inset_axes(
    [0.555, 0.375, 0.25, 0.215],
    xlim=(1.99, 2.01), ylim=(1e-2, 1e0), xticklabels=[], yticklabels=[])
xPlot('eeuf',axes_input=[axins],mult_factors=[n**-5.4,n**-2.35,n**-0.5,n**-5.1,n**-1.5,n**-0,
                                              n**-5.4,n**-2.35,n**-0.5,n**-5.1,n**-1.5,n**-0],
      data_colors=['darkblue','orange','darkred','darkblue','orange','darkred',
                   'grey','grey','grey','grey','grey','grey'],
      model_colors=['darkblue','orange','darkred','darkblue','orange','darkred',
                    'None','None','None','None','None','None'],
      data_alpha=[1, 1, 1, 1, 1,1, 0.5, 0.5, 0.5, 0.5,0.5,0.5],)
axins.set_ylabel('')
axins.set_xlabel('')
axins.set_title('')
axins.set_xlim(1.9975,2.0125)
axins.set_ylim(2.5e-4,3e-3)
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
    direction='in')
axins.tick_params(
    axis='y',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    left=False,  # ticks along the bottom edge are off
    right=True,  # ticks along the top edge are off
    direction='in',labelleft=False,labelright=True)


axins.set_xticklabels([1.995,2.000,2.005,2.010])
axins.text(1.998, 6.6e-4, r'v=300$\pm$9$\,$km$\,$s$^{-1}$', fontsize=10)
axins.text(1.998, 4.5e-4, r'$\sigma$=100$\pm$12$\,$km$\,$s$^{-1}$', fontsize=10)
axins.text(1.998, 3.2e-4, r'EW=3.18$\pm$0.05$\,$eV', fontsize=10)
# ax.legend(loc='lower right')
plt.tight_layout()
ax.get_children()[-2].remove()
# ax2 = ax.twinx()
# ax2.errorbar([],[],xerr=[],yerr=[],color='darkblue',
#              label=r"log$\xi$=4($\pm0.2$) | log$_{10}$NH=23($\pm0.2$) cm$^{-2}$""\n"
#                    +r"v$_{turb}$=100($\pm30$) km s$^{-1}$ | v$_{out}$=300($\pm60$) km s$^{-1}$")
# ax2.errorbar([],[],xerr=[],yerr=[],color='orange',
#              label=r"log$\xi$=2($\pm0.1$) | log$_{10}$NH=22($\pm0.1$) cm$^{-2}$""\n"
#                    +r"v$_{turb}$=100($\pm15$) km s$^{-1}$ | v$_{out}$=300($\pm30$) km s$^{-1}$")
# ax2.get_yaxis().set_visible(False)
# ax2.legend(loc='lower right')

'''
invert blue yellow and red to get the logxi=0 above - it will make it easier to plot the high energy panel
perhaps 6.5-8.5 with zoom or 6.5-7.0 - no need to put the 50s when they cannot do anything
'''
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.patches import FancyBboxPatch



# ---------------------------------------------------------
# Colors
# ---------------------------------------------------------

c1 = "darkblue"
c2 = "orange"
c3 = "darkred"


# ---------------------------------------------------------
# Legend contents
# ---------------------------------------------------------

data = [
    [
        r"log$\xi$ =",
        r"4($\pm0.02$)",
        r"2($\pm0.003$)",
        r"0($\pm0.01$)",
    ],
    [
        r"log$_{10}$NH [cm$^{-2}$] =",
        r"23($\pm0.03$)",
        r"22($\pm0.001$)",
        r"23($\pm0.01$)",
    ],
    [
        r"$v_{\rm turb} $ [km s$^{-1}$]=",
        r"100($\pm6$)",
        r"100($\pm2$)",
        r"100($\pm16$)",
    ],
    [
        r"$v_{\rm out}$ km s$^{-1}$] =",
        r"300($\pm5$)",
        r"300($\pm2$)",
        r"300($\pm40$)",
    ],
]

colors = [
    ["black", c1, c2, c3],
    ["black", c1, c2, c3],
    ["black", c1, c2, c3],
    ["black", c1, c2, c3],
]


# ---------------------------------------------------------
# Position and size of the legend
# ---------------------------------------------------------
#
# All values are in axes coordinates:
# [left, bottom, width, height]
#

bbox = [0.50, 0.02, 0.48, 0.19]


# ---------------------------------------------------------
# Draw the legend border FIRST
# ---------------------------------------------------------

border = FancyBboxPatch(
    (bbox[0], bbox[1]),
    bbox[2],
    bbox[3],
    boxstyle="round,pad=0.008,rounding_size=0.008",
    transform=ax.transAxes,
    facecolor="white",
    edgecolor="grey",
    linewidth=0.8,
    zorder=2,
)

ax.add_patch(border)

ax.text(
    bbox[0] + 0.35,
    bbox[1] + bbox[3] - 0.008,

    "5ks NewAthena simulations",
    transform=ax.transAxes,
    color="black",
    fontsize=11,
    ha="right",
    va="top",
    zorder=3,
)

# ---------------------------------------------------------
# Create the table
# ---------------------------------------------------------

table_bbox = [
    bbox[0],
    bbox[1],
    bbox[2],
    bbox[3] - 0.035,
]
table = Table(
    ax,
    bbox=table_bbox,
)


# ---------------------------------------------------------
# Column widths
# ---------------------------------------------------------
#
# These are relative widths. Reducing them reduces the
# horizontal space occupied by the legend.
#
# The first column is wider because it contains the labels
# and units. The three value columns are deliberately narrow.
#

col_widths = [
    0.35,   # labels
    0.23,   # dark blue
    0.23,   # orange
    0.21,   # dark red
]


# Row height
row_height = 0.11


# ---------------------------------------------------------
# Add cells
# ---------------------------------------------------------

for i, row in enumerate(data):

    for j, value in enumerate(row):

        cell = table.add_cell(
            i,
            j,
            width=col_widths[j],
            height=row_height,
            text=value,
            loc="left",
            facecolor="white",
            edgecolor="none",
        )

        cell.get_text().set_color(colors[i][j])
        cell.get_text().set_fontsize(11)

        # Reduce padding inside each cell.
        cell.PAD = 0.0


# ---------------------------------------------------------
# Make all cell borders invisible
# ---------------------------------------------------------

for cell in table.get_celld().values():
    cell.set_linewidth(0)
    cell.set_edgecolor("none")


# ---------------------------------------------------------
# Add table on top of the border
# ---------------------------------------------------------

table.set_zorder(3)
ax.add_table(table)
plt.show()

'''
HIGH ENERGIES 
'''

os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_NewAthena/SpecialIssue/Redbook/SIXTE/common')
Xset.restore('mod_full_NA_Chandra.xcm')
set_ener('large_canon',xrism=True)

n=15

set_ener('large_canon',xrism=True)

rebinv_xrism(1,10,max_bins=60000)
rebinv_xrism(2,10,max_bins=60000)
rebinv_xrism(3,10,max_bins=60000)
rebinv_xrism(4,10,max_bins=60000)
rebinv_xrism(5,10,max_bins=60000)
rebinv_xrism(6,10,max_bins=60000)
rebinv_xrism(7,5,max_bins=60000)
rebinv_xrism(8,5,max_bins=60000)
rebinv_xrism(9,5,max_bins=60000)
rebinv_xrism(10,3,max_bins=60000)
rebinv_xrism(11,3,max_bins=60000)
rebinv_xrism(12,3,max_bins=60000)

xPlot('eeuf',xlims=[6.6,8.55],mult_factors=np.array([n**-2.5,n**-1.5,n**-0.5,n**-2.,n**-1.,n**-0,
                                                     n**-2.5,n**-1.5,n**-0.5,n**-2.,n**-1.,n**-0]),
      data_colors=['darkblue', 'orange', 'darkred', 'darkblue', 'orange', 'darkred',
                   'grey', 'grey', 'grey', 'grey', 'grey', 'grey'],
      group_names=[r"log$\xi$=4($\pm0.01$) | log$_{10}$NH=23($\pm0.01$) cm$^{-2}$""\n"
                   + r"v$_{turb}$=100($\pm3$) km s$^{-1}$ | v$_{out}$=300($\pm3$) km s$^{-1}$",
                   r"log$\xi$=2($\pm0.03$) | log$_{10}$NH=22($\pm0.004$) cm$^{-2}$""\n"
                   + r"v$_{turb}$=100($\pm1$) km s$^{-1}$ | v$_{out}$=300($\pm2$) km s$^{-1}$",
                   "",
                   "", "", "",
                   "", "", "",
                   "", "", ""],
      data_alpha=[1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      model_colors=['darkblue', 'orange', 'darkred', 'darkblue', 'orange', 'darkred',
                    'None', 'None', 'None', 'None', 'None', 'None'],ylims=np.array([7e-7,5]))
ax=plt.gca()
plt.yscale('log')
ax.get_children()[-2].remove()
plt.tight_layout()
ax=plt.gca()

rebinv_xrism(1,3,max_bins=60000)
rebinv_xrism(7,3,max_bins=60000)
axins = ax.inset_axes(
    [0.26, 0.05, 0.31, 0.40],
    xlim=(6.6, 8.55), ylim=(2e-6, 4), xticklabels=[], yticklabels=[])
xPlot('eeuf',axes_input=[axins],mult_factors=np.array([n**-2.5,n**-1.5,n**-0.5,n**-2.,n**-1.,n**-0,
                                                       n**-2.5,n**-1.5,n**-0.5,n**-2.,n**-1.,n**-0]),
      data_colors=['darkblue','orange','darkred','darkblue','orange','darkred',
                   'grey','grey','grey','grey','grey','grey'],
      model_colors=['darkblue','orange','darkred','darkblue','orange','darkred',
                    'None','None','None','None','None','None'],
      data_alpha=[1, 1, 1, 1, 1,1, 0.5, 0.5, 0.5, 0.5,0.5,0.5],)

axins.set_ylabel('')
axins.set_xlabel('')
axins.set_title('')
axins.set_xlim(6.945, 6.99)
axins.set_ylim(2e-6, 4e-3)
axins.get_children()[-3].remove()
axins.get_children()[-2].remove()
axins.set_yscale('log')
axins.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=True,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=True,
    labeltop=False,
    direction='out')
axins.tick_params(
    axis='y',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    left=True,  # ticks along the bottom edge are off
    right=False,  # ticks along the top edge are off
    direction='out',labelleft=True,labelright=False)


axins.set_xticklabels(['',6.95,6.96,6.97,6.98])
axins.text(6.947, 13.2e-6, r'v = 300$\pm$10 km s$^{-1}$', fontsize=11)
axins.text(6.947, 6.4e-6, r'$\sigma$ = 100$\pm$17 km s$^{-1}$', fontsize=11)
axins.text(6.947, 3.2e-6, r'EW$_{1/2+3/2}$ = 20.5$\pm$0.6 eV', fontsize=11)
inset_ind=ax.indicate_inset_zoom(axins, edgecolor="black",alpha=0.5)
inset_ind.connectors[0].set_visible(True)
inset_ind.connectors[1].set_visible(True)
inset_ind.connectors[2].set_visible(False)
inset_ind.connectors[3].set_visible(True)
plt.show()


import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.patches import FancyBboxPatch



# ---------------------------------------------------------
# Colors
# ---------------------------------------------------------

c1 = "darkblue"
c2 = "orange"
c3 = "darkred"


# ---------------------------------------------------------
# Legend contents
# ---------------------------------------------------------

data = [
    [
        r"log$\xi$ =",
        r"4($\pm0.3$)",
        r"2($\pm0.03$)",
        r"X",
    ],
    [
        r"log$_{10}$NH [cm$^{-2}$] =",
        r"23($\pm0.3$)",
        r"22($\pm0.1$)",
        r"X",
    ],
    [
        r"$v_{\rm turb} $ [km s$^{-1}$]=",
        r"100(<120)",
        r"100($\pm20$)",
        r"X",
    ],
    [
        r"$v_{\rm out}$ km s$^{-1}$] =",
        r"300($\pm50$)",
        r"300($\pm30$)",
        r"X",
    ],
]

colors = [
    ["black", c1, c2, c3],
    ["black", c1, c2, c3],
    ["black", c1, c2, c3],
    ["black", c1, c2, c3],
]


# ---------------------------------------------------------
# Position and size of the legend
# ---------------------------------------------------------
#
# All values are in axes coordinates:
# [left, bottom, width, height]
#

bbox = [0.59, 0.02, 0.39, 0.19]


# ---------------------------------------------------------
# Draw the legend border FIRST
# ---------------------------------------------------------

border = FancyBboxPatch(
    (bbox[0], bbox[1]),
    bbox[2],
    bbox[3],
    boxstyle="round,pad=0.008,rounding_size=0.008",
    transform=ax.transAxes,
    facecolor="white",
    edgecolor="grey",
    linewidth=0.8,
    zorder=2,
)

ax.add_patch(border)

ax.text(
    bbox[0] + 0.31,
    bbox[1] + bbox[3] - 0.008,

    "50s NewAthena simulations",
    transform=ax.transAxes,
    color="black",
    fontsize=11,
    ha="right",
    va="top",
    zorder=3,
)

# ---------------------------------------------------------
# Create the table
# ---------------------------------------------------------

table_bbox = [
    bbox[0],
    bbox[1],
    bbox[2],
    bbox[3] - 0.035,
]
table = Table(
    ax,
    bbox=table_bbox,
)


# ---------------------------------------------------------
# Column widths
# ---------------------------------------------------------
#
# These are relative widths. Reducing them reduces the
# horizontal space occupied by the legend.
#
# The first column is wider because it contains the labels
# and units. The three value columns are deliberately narrow.
#

col_widths = [
    0.46,   # labels
    0.31,   # dark blue
    0.27,   # orange
    0.05,   # dark red
]


# Row height
row_height = 0.11


# ---------------------------------------------------------
# Add cells
# ---------------------------------------------------------

for i, row in enumerate(data):

    for j, value in enumerate(row):

        cell = table.add_cell(
            i,
            j,
            width=col_widths[j],
            height=row_height,
            text=value,
            loc="left",
            facecolor="white",
            edgecolor="none",
        )

        cell.get_text().set_color(colors[i][j])
        cell.get_text().set_fontsize(11)

        # Reduce padding inside each cell.
        cell.PAD = 0.0


# ---------------------------------------------------------
# Make all cell borders invisible
# ---------------------------------------------------------

for cell in table.get_celld().values():
    cell.set_linewidth(0)
    cell.set_edgecolor("none")


# ---------------------------------------------------------
# Add table on top of the border
# ---------------------------------------------------------

table.set_zorder(3)
ax.add_table(table)
plt.show()




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


'''
2-10 keV flux with updated template SED
highxi_1e23:
AllModels.calcFlux("2. 10.")
 Model Flux    1.0398 photons (6.4363e-09 ergs/cm^2/s) range (2.0000 - 10.000 keV)
 
AllModels.calcFlux("2. 10.")
Model Flux     1.069 photons (6.7129e-09 ergs/cm^2/s) range (2.0000 - 10.000 keV)

lowxi_1e22:
AllModels.calcFlux("2. 10.")
 Model Flux    1.0078 photons (6.439e-09 ergs/cm^2/s) range (2.0000 - 10.000 keV)

'''