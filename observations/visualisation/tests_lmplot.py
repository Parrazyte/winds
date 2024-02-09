from lmplot_uncert import lmplot_uncert_a
import numpy as np
import matplotlib.pyplot as plt

# test=np.arange(10)+np.array([1,-6,-4,-3,7,8,-1,2,5,2])
# test2=np.arange(10)*2+np.array([-1,-6,-4,-3,7,8,-1,2,5,-2])
# test_err=np.array([5/10,4/10,3/10,2/10,1/10,1/10,2/10,3/10,4/10,5/10])+np.array([1,6,4,3,7,8,1,2,5,2])/3
#
# test_err=np.array([5/10,4/10,3/10,2/10,1/10,1/10,2/10,3/10,4/10,5/10])+np.array([np.nan,0,0,0,0,8,1,2,5,2])/3
#
# test_err=np.array([test_err,test_err])
#
# test2_err=test_err*2

#test vittoria

x=np.array([5.02, 5.38, 5.01, 4.11, 4.49, 4.76, 4.1 , 4.41, 5.17, 3.35, 2.87,
       4.33, 4.53, 3.39, 3.44, 4.8 , 4.17, 3.87, 4.5 , 3.85, 4.35, 4.86])

y=np.array([ 0.16746407,  0.10416964,  0.09299854,  0.01124055,  0.09430634,
               np.nan,         np.nan,         np.nan,         np.nan,         np.nan,
               np.nan,         np.nan,         np.nan,         np.nan, -0.19653976,
        0.11584645, -0.17916838,  0.03584625, -0.0202682 , -0.00470422,
       -0.07075814,         np.nan])

x_err=np.array([[ np.nan, 1.27, 0.38, 0.1 ,  np.nan, 0.42, 0.38, 0.08, 0.77, 0.06, 0.1 ,
       0.08, 1.15, 0.15, 0.18, 0.32, 0.21, 0.1 , 0.24, 0.38, 0.13, 0.9 ],[  np.nan, 0.44 , 0.64 , 0.17 ,   np.nan, 0.59 , 0.35 , 0.92 , 0.75 ,
       0.14 , 0.12 , 0.08 , 1.15 , 0.09 , 0.55 , 0.069, 0.17 , 0.21 ,
       0.65 , 0.18 , 0.73 , 0.24 ]])

# x_err=x*0.1

y_err=np.repeat(np.nan,len(y)*2).reshape(len(y),2).T

# y_err=np.repeat(0.1,len(y))
#
# fig,ax=plt.subplots()
# ax.errorbar(x,y,xerr=x_err,yerr=y_err,linestyle='',marker='d')
#
# a,b,c,d=lmplot_uncert_a(ax,x,y,x_err,y_err,nsim=2000,intercept_pos='auto',percent=10)
# plt.suptitle('intercept at '+str(d)
#              +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
#              +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
#              +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')
# plt.ylim((-0.5,0.5))
#
# x_err=np.array([[ 0.05, 1.27, 0.38, 0.1 , 0.05, 0.42, 0.38, 0.08, 0.77, 0.06, 0.1 ,
#        0.08, 1.15, 0.15, 0.18, 0.32, 0.21, 0.1 , 0.24, 0.38, 0.13, 0.9 ],[  0.05, 0.44 , 0.64 , 0.17 ,  0.05, 0.59 , 0.35 , 0.92 , 0.75 ,
#        0.14 , 0.12 , 0.08 , 1.15 , 0.09 , 0.55 , 0.069, 0.17 , 0.21 ,
#        0.65 , 0.18 , 0.73 , 0.24 ]])
#
# y_err=np.repeat(0.1,len(y))
#
# fig,ax=plt.subplots()
# ax.errorbar(x,y,xerr=x_err,yerr=y_err,linestyle='',marker='d')
#
# a,b,c,d=lmplot_uncert_a(ax,x,y,x_err,y_err,nsim=2000,intercept_pos='auto',percent=10)
# plt.suptitle('intercept at '+str(d)
#              +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
#              +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
#              +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')
# plt.ylim((-0.5,0.5))
#
# x_err=np.array([[ 0.5, 1.27, 0.38, 0.1 , 0.5, 0.42, 0.38, 0.08, 0.77, 0.06, 0.1 ,
#        0.08, 1.15, 0.15, 0.18, 0.32, 0.21, 0.1 , 0.24, 0.38, 0.13, 0.9 ],[  0.5, 0.44 , 0.64 , 0.17 ,  0.5, 0.59 , 0.35 , 0.92 , 0.75 ,
#        0.14 , 0.12 , 0.08 , 1.15 , 0.09 , 0.55 , 0.069, 0.17 , 0.21 ,
#        0.65 , 0.18 , 0.73 , 0.24 ]])
# y_err=np.repeat(1,len(y))
#
# fig,ax=plt.subplots()
# ax.errorbar(x,y,xerr=x_err,yerr=y_err,linestyle='',marker='d')
#
# a,b,c,d=lmplot_uncert_a(ax,x,y,x_err,y_err,nsim=2000,intercept_pos='auto',percent=10)
# plt.suptitle('intercept at '+str(d)
#              +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
#              +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
#              +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')
# plt.ylim((-0.5,0.5))
#
#
# ########################"
# x=np.arange(10)
# y=np.arange(10)
#
# x_err=np.array([[0,0,0,0,0,1,1,1,1,1],[1,1,1,1,1,0,0,0,0,0]])
#
# y_err=np.repeat(0,20).reshape(2,10)

# fig,ax=plt.subplots()
# ax.errorbar(x,y,xerr=x_err,yerr=y_err,linestyle='',marker='d')
#
# a,b,c,d=lmplot_uncert_a(ax,x,y,x_err,y_err,nsim=2000,intercept_pos='auto',percent=10)
# plt.suptitle('intercept at '+str(d)
#              +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
#              +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
#              +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')
# plt.xlim(-1,11)
# plt.ylim(-1,11)
#

# x=np.arange(10)
# y=np.arange(10)
#
# x_err=np.array([[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1]])/3
#
# y_err=np.array([[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1]])/3
#
#
# fig,ax=plt.subplots()
# ax.errorbar(x,y,xerr=x_err,yerr=y_err,linestyle='',marker='d')
#
# a,b,c,d=lmplot_uncert_a(ax,x,y,x_err,y_err,nsim=2000,intercept_pos='auto',percent=10)
# plt.suptitle('intercept at '+str(d)
#              +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
#              +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
#              +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')
# plt.xlim(-1,11)
# plt.ylim(-1,11)


x=np.arange(1,11)
y=np.arange(1,11)

# x_err=np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]])
#
#
# y_err=np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]])

# x_err=0.1*x
# y_err=0.1*y
# fig,ax=plt.subplots()
#
# plt.xscale('log')
#
# plt.yscale('log')
#
# plt.xscale('linear')
# x_err=np.repeat(0.5,10)
#
#
# ax.errorbar(x,y,xerr=x_err,yerr=y_err,linestyle='',marker='d')
#
# a,b,c,d=lmplot_uncert_a(ax,x,y,x_err,y_err,nsim=2000,intercept_pos='auto',percent=90,infer_log_scale=True,
#                         error_percent=68.26)
# plt.suptitle('intercept at '+str(d)
#              +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
#              +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
#              +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')

#maintenant tester avec just y en log
# plt.xlim(-1,11)
# plt.ylim(-1,11)














# fig,ax=plt.subplots()
# ax.errorbar(test,test2,xerr=test_err,yerr=test2_err,linestyle='')
# a,b,c,d=lmplot_uncert_a(ax,test,test2,test_err,test2_err,nsim=2000,intercept_pos=-10)
# plt.suptitle('intercept at '+str(d)
#              +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
#              +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
#              +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')
#
# fig,ax=plt.subplots()
# ax.errorbar(test,test2,xerr=test_err,yerr=test2_err,linestyle='')
# a,b,c,d=lmplot_uncert_a(ax,test,test2,test_err,test2_err,nsim=2000,intercept_pos=15)
# plt.suptitle('intercept at '+str(d)
#              +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
#              +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
#              +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')
#
#
# fig,ax=plt.subplots()
# ax.errorbar(test,test2,xerr=test_err,yerr=test2_err,linestyle='')
# a,b,c,d=lmplot_uncert_a(ax,test,test2,test_err,test2_err,nsim=2000,intercept_pos='auto')
# plt.suptitle('intercept at '+str(d)
#              +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
#              +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
#              +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')
#
#
#
# fig,ax=plt.subplots()
# ax.errorbar(test,test2,xerr=test_err,yerr=test2_err,linestyle='')
# a,b,c,d=lmplot_uncert_a(ax,test,test2,test_err,test2_err,nsim=2000,intercept_pos='best')
# plt.suptitle('intercept at '+str(d)
#              +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
#              +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
#              +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')

#
nh=np.array([23.93490814, 23.95486694, 23.28839266, 23.78931501, 24.0414426 ,
       23.853329  , 22.98799641, 21.96241761, 21.9553958 , 22.88179261,
       23.03216763, 22.70121193, 22.12518667, 23.62459709, 22.12458371,
       22.62986417, 23.36926087, 22.75400864, 22.02954341, 23.3061403 ,
       23.53417882, 23.24715018, 24.34806502, 25.0274874 , 24.29987747,
       24.92304147, 24.7454078 , 24.64184875, 24.60341439, 24.48806815,
       24.63756086, 24.86967192, 24.55946118, 24.74304147])

err_nh=np.array([[       np.nan,        np.nan, 1.00844888, 0.08027252, 0.33191499,
       0.39351571, 0.6659818 ,        np.nan,        np.nan, 0.11443332,
       0.12002962, 0.13004769,        np.nan, 0.39001031,        np.nan,
              np.nan, 0.27001278, 0.17326668,        np.nan, 0.36001696,
       0.40001572,        np.nan, 0.05781663, 0.13955817, 0.09284755,
       0.23241907, 0.21040752, 0.33464939, 0.05325816, 0.21679701,
       0.20096225, 0.15053282, 0.12734552, 0.16154618],[       np.nan,        np.nan, 0.2256683 , 0.04069652, 0.27295309,
       0.15049717, 0.32587813,        np.nan,        np.nan, 0.12326794,
       0.06005923, 0.13004769,        np.nan, 0.18002234,        np.nan,
              np.nan, 0.27001278, 0.16407724,        np.nan, 0.36001696,
       0.40001572,        np.nan, 0.05781663, 0.0947443 , 0.10159069,
       0.18843379, 0.24142347, 0.24164791, 0.06274099, 0.19729958,
       0.20053172, 0.21151955, 0.12222471, 0.07088224]])

v_out=np.array([0.108, 0.305, 0.152, 0.071, 0.13 , 0.11 , 0.237, 0.106, 0.098,
       0.128, 0.151, 0.116, 0.199, 0.285, 0.306, 0.182, 0.091, 0.085,
       0.034, 0.076, 0.142, 0.1  , 0.32 , 0.38 , 0.28 , 0.47 , 0.58 ,
       0.25 , 0.23 , 0.18 , 0.34 , 0.56 , 0.43 , 0.47 ])

err_v_out=np.array([[0.01 , 0.037, 0.023, 0.016, 0.011, 0.008, 0.011, 0.007, 0.003,
       0.063, 0.003, 0.004, 0.024, 0.003, 0.003, 0.026 , 0.003, 0.076,
       0.004, 0.004, 0.004, 0.004, 0.03 , 0.05 , 0.05 , 0.05 , 0.02 ,
       0.08 , 0.02 , 0.09 , 0.09 , 0.02 , 0.05 , 0.01 ],[0.008, 0.019, 0.02 , 0.017, 0.012, 0.015, 0.01 , 0.007, 0.003,
       0.063, 0.003, 0.004, 0.024, 0.003, 0.003, 0.26 , 0.003, 0.076,
       0.004, 0.004, 0.004, 0.004, 0.03 , 0.05 , 0.05 , 0.03 , 0.01 ,
       0.06 , 0.02 , 0.06 , 0.02 , 0.01 , 0.04 , 0.02 ]])

lvout_vals=np.array([v_out,v_out-err_v_out[0],v_out+err_v_out[1]])*299792.548

lvout=np.log10(lvout_vals)[0]

lvout_err=[lvout-np.log10(lvout_vals[1]),np.log10(lvout_vals[2])-lvout]
#
# breakpoint()


xi=np.array([5.02   , 5.38   , 5.01   , 4.11   , 4.49   , 4.76   , 4.1    ,
       4.41   , 5.17   , 3.35   , 2.87   , 4.33   , 4.53   , 3.39   ,
       3.44   , 4.8    , 4.17   , 3.87   , 4.5    , 3.85   , 4.35   ,
       4.86   , 4.49   , 4.62851, 4.38097, 5.47989, 5.18242, 5.06419,
       5.35   , 4.98308, 5.02941, 4.75942, 5.14   , 5.42   ])

err_xi=np.array([[     np.nan, 1.27    , 0.38    , 0.1     ,      np.nan, 0.42    ,
       0.38    , 0.08    , 0.77    , 0.06    , 0.1     , 0.08    ,
       1.15    , 0.15    , 0.18    , 0.32    , 0.21    , 0.1     ,
       0.24    , 0.38    , 0.13    , 0.9     , 0.1     , 0.173241,
       0.265248, 0.240721, 0.237244, 0.30613 , 0.12    , 0.395158,
       0.35802 , 0.453406, 0.15    , 0.16    ],[     np.nan, 0.44    , 0.64    , 0.17    ,      np.nan, 0.59    ,
       0.35    , 0.92    , 0.75    , 0.14    , 0.12    , 0.08    ,
       1.15    , 0.09    , 0.55    , 0.069   , 0.17    , 0.21    ,
       0.65    , 0.18    , 0.73    , 0.24    , 0.11    , 0.289017,
       0.285196, 0.38531 , 0.331887, 0.590547, 0.13    , 0.433698,
       0.384134, 0.569336, 0.25    , 0.32    ]])

lx=np.array([44.4       , 44.208     , 44.52      , 43.977     , 44.449     ,
       43.79      , 43.692     , 42.34      , 43.7       , 41.65      ,
       43.7       , 43.1       , 43.1       , 41.6       , 44.        ,
       43.97      , 43.4       , 42.73      , 43.5       , 44.3       ,
       43.2       , 43.8       , 46.25345916, 45.36172784, 44.5039934 ,
       44.76886117, 44.05435766, 44.05799195, 44.16975108, 44.1383027 ,
       44.7201593 , 44.3357921 , 43.64992061, 45.21085337])

err_lx=np.array([[0.004     , 0.007     , 0.003     , 0.009     , 0.006     ,
       0.006     , 0.008     , 0.02      , 0.05      , 0.04      ,
       0.03      , 0.01      , 0.02      , 0.07      , 0.01      ,
       0.1       , 0.03      , 0.05      , 0.01      , 0.04      ,
       0.03      , 0.02      , 0.01817132, 0.03776474, 0.00579059,
       0.02347538, 0.02554673, 0.16286043, 0.01277337, 0.01974066,
       0.06204207, 0.1002218 , 0.03776474, 0.01336291],[0.004     , 0.007     , 0.003     , 0.009     , 0.006     ,
       0.006     , 0.008     , 0.02      , 0.05      , 0.04      ,
       0.03      , 0.01      , 0.02      , 0.07      , 0.01      ,
       0.1       , 0.03      , 0.05      , 0.01      , 0.04      ,
       0.03      , 0.02      , 0.01817132, 0.09441184, 0.0028953 ,
       0.02347538, 0.02554673, 0.16286043, 0.01277337, 0.01974066,
       0.06204207, 0.13362907, 0.03776474, 0.01336291]])


fig2,ax2=plt.subplots()
ax2.errorbar(lvout,xi,xerr=lvout_err,yerr=err_xi,linestyle='')
plt.xlabel('lvout')
plt.ylabel('xi')
a,b,c,d=lmplot_uncert_a(ax2,lvout,xi,lvout_err,err_xi,nsim=2000,intercept_pos='auto',percent=68.
                        ,percent_regions=[68,95,99.7],inter_color=['red','green','lightblue'])
plt.suptitle('intercept at '+str(d)
             +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
             +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
             +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')

# plt.figure()
# plt.scatter(e,f)
fig1,ax1=plt.subplots()
ax1.errorbar(lvout,nh,xerr=lvout_err,yerr=err_nh,linestyle='')
plt.xlabel('lvout')
plt.ylabel('nh')
a,b,c,d=lmplot_uncert_a(ax1,lvout,nh,lvout_err,err_nh,nsim=2000,intercept_pos='auto',percent=68)
plt.suptitle('intercept at '+str(d)
             +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
             +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
             +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')

fig3,ax3=plt.subplots()
ax3.errorbar(xi,nh,xerr=err_xi,yerr=err_nh,linestyle='')
plt.xlabel('xi')
plt.ylabel('nh')
a,b,c,d=lmplot_uncert_a(ax3,xi,nh,err_xi,err_nh,nsim=2000,intercept_pos='auto',percent=68)
plt.suptitle('intercept at '+str(d)
             +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
             +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
             +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')


# for i_a in range(10):
#        fig4,ax4=plt.subplots()
#        ax4.errorbar(lx,v_out,xerr=err_lx,yerr=err_v_out,linestyle='')
#        plt.xlabel('lx')
#        plt.ylabel('v_out')
#        a,b,c,d=lmplot_uncert_a(ax4,lx,v_out,err_lx,err_v_out,nsim=2000,intercept_pos='auto',percent=90)
#        plt.suptitle('intercept at '+str(d)
#                     +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
#                     +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
#                     +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')