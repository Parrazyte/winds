import os,sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib as mpl

ap = argparse.ArgumentParser(description='Script to plot line detectability from various instruments.\n)')

'''GENERAL OPTIONS'''

#currently can be EW or bshift
ap.add_argument("-mode",nargs=1,help='plot the results of the simulations for different modes',
                default='bshift',type=str)

ap.add_argument("-fakestats",nargs=1,help='use run with or without fakeit statistical fluctuations',
                default=True,type=str)

ap.add_argument('-n_iter',nargs=1,help='number of iterations of each flux level for XRISM/XMM/Chandra',
                default=[10,100,100],type=str)

ap.add_argument('-expos',nargs=1,help='Exposure time in ks for XRISM/XMM/Chandra',default=[10,35,35],type=str)

ap.add_argument('-flux_str',nargs=1,help='flux logspace parameters for file fetching',default='1_100_10',type=str)

ap.add_argument('-mask',nargs=1,help='mask part of the values',default='1/1')

#EW arguments
ap.add_argument('-width_inter',nargs=1,help='width interval bounds',default=[5e-3,5e-3])

#bshift arguments
ap.add_argument('-bshift_EW_val',nargs=1,help='EW for which to simulate the blueshift in eVs',default=20)

ap.add_argument('-bshift_width_val',nargs=1,help='bshift for which to simulate the bshfit errors in eVs', default=0.005)

ap.add_argument('-sigmas',nargs=1,help='sigmas to plot',default=[3])


args=ap.parse_args()

mode=args.mode
fakestats=args.fakestats
n_iter=args.n_iter
expos=args.expos
flux_str=args.flux_str

mask_vals=np.array(args.mask.split('/')).astype(float)
width_inter=args.width_inter
bshift_EW_val=args.bshift_EW_val
bshift_width_val=args.bshift_width_val

width_str='_'.join([str(elem) for elem in width_inter])

#to get back to index
id_sigmas=np.array(args.sigmas)-1




if mode=='ew':

    #loading elements
    arr_XRISM=np.loadtxt('/media/parrama/SSD/Observ/highres/linedet_compa/fakes_XRISM/ew_lim_mod'
                         +('_nostat' if not fakestats else '')+
                         '_'+str(expos[0])+'ks'+
                         '_'+str(n_iter[0])+'_iter'+
                         '_flux_' + str(flux_str) +
                         '_width_'+ width_str+'.txt')

    arr_XMM=np.loadtxt('/media/parrama/SSD/Observ/highres/linedet_compa/fakes_XMM/ew_lim_mod'
                       + ('_nostat' if not fakestats else '') +
                       '_' + str(expos[1]) + 'ks' +
                       '_' + str(n_iter[1]) + '_iter' +
                       '_flux_' + str(flux_str) +
                       '_width_' + width_str + '.txt')

    arr_Chandra=np.loadtxt('/media/parrama/SSD/Observ/highres/linedet_compa/fakes_Chandra/ew_lim_mod'
                           + ('_nostat' if not fakestats else '') +
                           '_' + str(expos[2]) + 'ks' +
                           '_' + str(n_iter[2]) + '_iter' +
                           '_flux_' + str(flux_str) +
                           '_width_' + width_str + '.txt')

elif mode=='bshift':

    # loading elements
    arr_XRISM = np.loadtxt('/media/parrama/SSD/Observ/highres/linedet_compa/fakes_XRISM/bshift_err_mod'
                           + ('_nostat' if not fakestats else '') +
                           '_' + str(expos[0]) + 'ks' +
                           '_' + str(n_iter[0]) + '_iter' +
                           '_flux_' + str(flux_str) +
                            '_EW_'+str(bshift_EW_val)+
                           '_width_' + str(bshift_width_val)+ '.txt')

    arr_XMM = np.loadtxt('/media/parrama/SSD/Observ/highres/linedet_compa/fakes_XMM/bshift_err_mod'
                         + ('_nostat' if not fakestats else '') +
                         '_' + str(expos[1]) + 'ks' +
                         '_' + str(n_iter[1]) + '_iter' +
                         '_flux_' + str(flux_str) +
                         '_EW_' + str(bshift_EW_val) +
                         '_width_' +str(bshift_width_val)+ '.txt')

    arr_Chandra = np.loadtxt('/media/parrama/SSD/Observ/highres/linedet_compa/fakes_Chandra/bshift_err_mod'
                             + ('_nostat' if not fakestats else '') +
                             '_' + str(expos[2]) + 'ks' +
                             '_' + str(n_iter[2]) + '_iter' +
                             '_flux_' + str(flux_str) +
                             '_EW_' + str(bshift_EW_val) +
                             '_width_' + str(bshift_width_val)+ '.txt')

mask_vals_XRISM=[mask_vals[0]*i%mask_vals[1]==0 for i in range(len(arr_XRISM))]
mask_vals_XMM=[mask_vals[0]*i%mask_vals[1]==0 for i in range(len(arr_XMM))]
mask_vals_Chandra=[mask_vals[0]*i%mask_vals[1]==0 for i in range(len(arr_Chandra))]

arr_XRISM=arr_XRISM[mask_vals_XRISM]
arr_XMM=arr_XMM[mask_vals_XMM]
arr_Chandra=arr_Chandra[mask_vals_Chandra]

mpl.rcParams.update({'font.size': 14})

fig_EW,ax_EW=plt.subplots(figsize=(8,8))

#ax_EW.set_xscale('log')
ax_EW.set_xscale('linear')
ax_EW.set_yscale('log')

if mode=='bshift':
    ax_EW.set_xscale('log')

ax_EW.set_ylabel('Observed flux (ergs/s/cm²)')

if mode=='ew':
    ax_EW.set_xlabel('EW treshold for detection')
elif mode=='bshift':
    ax_EW.set_xlabel((str(id_sigmas[0]+1) +r'$\sigma$ ' if len(id_sigmas)==1 else '')+'velocity shift error (km/s)')

marker_sigmas=['+','x','d']
ls_sigmas=[':','--','-']


[ax_EW.plot(arr_Chandra.T[1+i_sigma],arr_Chandra.T[0],label='Chandra '+str(expos[1]) + 'ks',
               marker=marker_sigmas[i_sigma],ls=ls_sigmas[i_sigma],color='blue') for i_sigma in id_sigmas]

[ax_EW.plot(arr_XRISM.T[1+i_sigma],arr_XRISM.T[0],label='XRISM '+str(expos[2]) + 'ks',
               marker=marker_sigmas[i_sigma],ls=ls_sigmas[i_sigma],color='violet') for i_sigma in id_sigmas]

[ax_EW.plot(arr_XMM.T[1+i_sigma],arr_XMM.T[0],label='XMM '+str(expos[0]) + 'ks' ,
               marker=marker_sigmas[i_sigma],ls=ls_sigmas[i_sigma],color='red') for i_sigma in id_sigmas]



# plt.suptitle('Line EW constrains, '+(' with ' if fakestats else ' without ')+'statistical fluctuations.\n'+
#              'width interval : '+str(width_inter)+' keV '+' | '+str(n_iter)+' iterations | '+'50ks exposure')
plt.legend()
plt.rcParams['figure.constrained_layout.use'] = True

#necessary to have the constrained layout saved
plt.show()

if mode=='ew':
    plt.savefig('/media/parrama/SSD/Observ/highres/linedet_compa/EW_detec_compa'+
            ('_nostat' if not fakestats else '')+
            '_'+str(expos)+
            '_' + str(n_iter) + '_iter' +
                '_flux_' + str(flux_str) +
            '_width_' + width_str +'.png')

elif mode=='bshift':
    plt.savefig('/media/parrama/SSD/Observ/highres/linedet_compa/bshift_err_compa'+
            ('_nostat' if not fakestats else '')+
            '_' + str(expos) +
            '_' + str(n_iter) + '_iter' +
                '_flux_' + str(flux_str) +
            'EW_' + str(bshift_EW_val) +
            '_width_' + str(bshift_width_val) + '.png')