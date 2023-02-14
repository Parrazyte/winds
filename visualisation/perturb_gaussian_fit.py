import matplotlib.pyplot as plt
from numpy import exp, loadtxt, pi, sqrt
import numpy as np
from custom_pymccorrelation import perturb_values
import scipy
from tqdm import tqdm

from lmfit import Model

# data = loadtxt('model1d_gauss.dat')
x_ch= np.array([12.40280570006887, -84.02067488142659, -483.4139688969211,
              -152.87652097551972, 290.9225747358635, 23.49136053669804,
              124.83158124529744, -2639.060062961778, -948.903977133402,
              -1356.3406485107535, -302.5298351361417, 327.4877483130003,
              -137.70558336923622, 1.1727191077625907, 340.813437, 278.63,
              336.792855, 231.945, 156.873228, 346.56806059692923,
              186.13778011952184, 1187.68021572, 537.7635037595994,
              276.93458664220805, 170.849999, 625.1693181554703,
              -87.21586432705847, -110.73699981543382, -119.267999, -128.690998,
              102.977008, 780.8136704681532, 1068.0489691145126,
              708.0210633887243, 276.764156337214, 231.494, 417.43, 167.01751,
              -389.10808])

xerr_ch=np.array([[153.082, 153.087, 169.59, 248.011, 307.863, 114.736, 27.0215,
              542.754, 469.091, 556.487, 59.1957, 182.962, 649.822, 359.864,
              136.771, 135.414, 28.6262, 178.895, 255.659, 505.261, 171.917,
              98.5161, 237.574, 198.365, 28.2305, 472.317, 461.823, 418.23,
              271.694, 269.347, 138.228, 461.926, 1144.42, 296.301, 451.746,
              46.2451, 142.565, 407.868, 396.116],[140.714, 147.153, 215.419, 266.286, 320.071, 98.8966, 25.7976,
                            692.425, 405.077, 542.942, 89.2122, 166.829, 629.896, 318.244,
                            142.71, 140.704, 93.2334, 196.267, 278.182, 506.962, 267.278,
                            157.467, 106.836, 267.837, 30.2385, 511.585, 523.407, 589.057,
                            384.51, 330.017, 101.982, 500.461, 1025.56, 321.61, 808.192,
                            50.8927, 166.87, 641.708, 443.212]]).T
                                                   
x_xmm=np.array([3876.5976858994486, 4992.9741481798155, 5169.240227352179,
       6030.869999844778, 4630.238442295601, 2441.059810599271,
       -2257.7672260053396, 108.28000167497493, 159.05999996473508,
       570.3689979618697, 4059.619889414482, -291.7330000048591,
       -748.6853744803508, 1173.4799836331974, 4286.28193823,
       5818.15567685, 5227.15009441, 5884.67999996, 4259.6994659,
       3467.9243362253446, 1154.065807715878, -917.6949910154199,
       486.774603, -1057.98992338, -326.738002, 3495.34,
       -450.03899998157516, 2946.5917946196805, 1385.08602323,
       2873.38814136, 2020.31, 6599.60097845, 5048.579992495374])

xerr_xmm=np.array([[930.864, 2227.06, 169.604, 530.306, 2603.64, 1512.52, 933.269,
       760.359, 601.297, 1749.3, 1857.25, 875.203, 1146.69, 795.632,
       434.345, 1604.49, 118.031, 223.938, 950.764, 2662.64, 5832.46,
       742.052, 1200.71, 581.966, 592.478, 1447.28, 1743.79, 1143.38,
       1466.85, 1076.75, 1680.95, 3726.85, 1378.86],[781.063, 1865.4, 333.408, 537.019, 2995.31, 1472.95, 2293.58,
              587.29, 643.189, 1312.6, 1694.6, 944.406, 1586.12, 1142.82, 353.64,
              1243.99, 218.941, 266.681, 849.384, 2962.1, 6368.82, 1213.56,
              1070.41, 502.257, 480.385, 1564.12, 2432.55, 1107.05, 723.745,
              472.162, 1403.72, 2341.77, 1397.75]]).T
# y = data[:, 1]

nfakes=1000

xpert_xmm,dummy,d2=perturb_values(x_xmm,None,xerr_xmm,None,1000)
xpert_ch,dummy,d2=perturb_values(x_ch,None,xerr_ch,None,1000)

xflat_xmm=xpert_xmm.astype(float).ravel()
xflat_ch=xpert_ch.astype(float).ravel()

bins_xmm=np.arange(-5000,10001,500)
bins_ch=np.arange(-5000,10001,200)

#x points at the middle of the bin positions
x_bins_xmm=bins_xmm[:-1]+250
x_bins_ch=bins_ch[:-1]+100

fig_xmm,ax_xmm=plt.subplots()

hist_xmm=ax_xmm.hist(xflat_xmm,bins=bins_xmm,density=True)

fig_ch,ax_ch=plt.subplots()

hist_ch=ax_ch.hist(xflat_ch,bins=bins_ch,density=True)

# sig_xmm=np.zeros(1000)
# sig_ch=np.zeros(1000)

# mu_xmm=np.zeros(1000)
# mu_ch=np.zeros(1000)

# # with tqdm(total=nfakes) as pbar:
    
# #     for i in range(nfakes):
        
# #         hist_ch=plt.hist(xpert_ch[i],bins_ch)
        
# #         hist_xmm=plt.hist(xpert_xmm[i],bins_xmm)
        
# #         print(hist_ch[0])
        
# #         x_bins_ch=hist_ch[1][:-1]+100
        
# #         x_bins_xmm=hist_xmm[1][:-1]+250
# #         mu_xmm[i], sig_xmm[i] = scipy.stats.norm.fit((hist_xmm[0],x_bins_xmm))
# #         mu_ch[i], sig_ch[i]= scipy.stats.norm.fit((hist_ch[0],x_bins_ch))

# #         pbar.update(1)
        

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))


gmodel = Model(gaussian)

fit_xmm=gmodel.fit(hist_xmm[0],x=x_bins_xmm,amp=1,cen=2000,wid=2000)

print(fit_xmm.fit_report())

ax_xmm.set_title("xmm velocity shift distribution from n="+str(nfakes)+" perturbations")

# ax_xmm.plot(x_bins_xmm, fit_xmm.init_fit, '--', label='initial fit')
ax_xmm.plot(x_bins_xmm, fit_xmm.best_fit, '-', label='best fit')
ax_xmm.legend(title=r'$\mu='+str(round(fit_xmm.best_values['cen']))+'$km/s\n$\sigma='+str(round(fit_xmm.best_values['wid']))+'$km/s'
              +'\nbinning=500 km/s')


fit_ch=gmodel.fit(hist_ch[0],x=x_bins_ch,amp=1,cen=0,wid=500)

fig_xmm.savefig('./XMM_bshift_perturb.png')

print(fit_ch.fit_report())

ax_ch.set_title("chandra velocity shift distribution from n="+str(nfakes)+" perturbations")

# ax_ch.plot(x_bins_ch, fit_ch.init_fit, '--', label='initial fit')
ax_ch.plot(x_bins_ch, fit_ch.best_fit, '-', label='best fit')
ax_ch.legend(title=r'$\mu='+str(round(fit_ch.best_values['cen']))+'$km/s\n$\sigma='+str(round(fit_ch.best_values['wid']))+'$km/s'
             +'\nbinning=200 km/s')
ax_ch.set_xlim(-4000,4000)

fig_ch.savefig('./ch_bshift_perturb.png')