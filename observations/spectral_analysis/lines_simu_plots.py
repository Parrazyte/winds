import os,sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser(description='Script to plot line detectability from various instruments.\n)')

'''GENERAL OPTIONS'''


ap.add_argument("-fakestats",nargs=1,help='use run with or without fakeit statistical fluctuations',
                default=True,type=str)

ap.add_argument('-n_iter',nargs=1,help='number of iterations of each flux level',
                default=10,type=str)

ap.add_argument('-width_inter',nargs=1,help='width interval bounds',default=[5e-3,5e-3])

ap.add_argument('-sigmas',nargs=1,help='sigmas to plot',default=[3])

args=ap.parse_args()

n_iter=args.n_iter

width_inter=args.width_inter

width_str='_'.join([str(elem) for elem in width_inter])

#to get back to index
id_sigmas=np.array(args.sigmas)-1

fakestats=args.fakestats

arr_XRISM=np.loadtxt('/media/parrama/SSD/Observ/highres/linedet_compa/fakes_XRISM/ew_lim_mod'
                     +('_nostat' if not fakestats else '')+
                     '_'+str(n_iter)+'_iter'+
                     '_width_'+ width_str+'.txt')

arr_XMM=np.loadtxt('/media/parrama/SSD/Observ/highres/linedet_compa/fakes_XMM/ew_lim_mod'
                   + ('_nostat' if not fakestats else '') +
                   '_' + str(n_iter) + '_iter' +
                   '_width_' + width_str + '.txt')

arr_Chandra=np.loadtxt('/media/parrama/SSD/Observ/highres/linedet_compa/fakes_Chandra/ew_lim_mod'
                       + ('_nostat' if not fakestats else '') +
                       '_' + str(n_iter) + '_iter' +
                       '_width_' + width_str + '.txt')

fig_EW,ax_EW=plt.subplots(figsize=(8,6))

#ax_EW.set_xscale('log')
ax_EW.set_xscale('linear')
ax_EW.set_yscale('log')
ax_EW.set_ylabel('Observed flux (ergs/s/cm²)')
ax_EW.set_xlabel('EW treshold for detection')

marker_sigmas=['+','x','d']
ls_sigmas=[':','--','-']

[ax_EW.plot(arr_XMM.T[1+i_sigma],arr_XMM.T[0],label=str(1+i_sigma)+r' $\sigma$',
               marker=marker_sigmas[i_sigma],ls=ls_sigmas[i_sigma],color='red') for i_sigma in id_sigmas]

[ax_EW.plot(arr_Chandra.T[1+i_sigma],arr_Chandra.T[0],label='',
               marker=marker_sigmas[i_sigma],ls=ls_sigmas[i_sigma],color='blue') for i_sigma in id_sigmas]

[ax_EW.plot(arr_XRISM.T[1+i_sigma],arr_XRISM.T[0],label='',
               marker=marker_sigmas[i_sigma],ls=ls_sigmas[i_sigma],color='violet') for i_sigma in id_sigmas]

plt.suptitle('Line EW constrains, '+(' with ' if fakestats else ' without ')+'statistical fluctuations.\n'+
             'width interval : '+str(width_inter)+' keV '+' | '+str(n_iter)+' iterations | '+'50ks exposure')
plt.legend()

plt.savefig('/media/parrama/SSD/Observ/highres/linedet_compa/EW_detec_compa'+
            ('_nostat' if not fakestats else '')+
            '_' + str(n_iter) + '_iter' +
            '_width_' + width_str +'.png')