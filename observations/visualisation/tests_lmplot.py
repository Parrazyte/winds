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
# x_err=np.array([[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1]])
#
# y_err=np.repeat(0,20).reshape(2,10)
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

x_err=0.1*x
y_err=0.1*y
fig,ax=plt.subplots()

plt.xscale('log')

plt.yscale('log')

plt.xscale('linear')
x_err=np.repeat(0.5,10)


ax.errorbar(x,y,xerr=x_err,yerr=y_err,linestyle='',marker='d')

a,b,c,d=lmplot_uncert_a(ax,x,y,x_err,y_err,nsim=2000,intercept_pos='auto',percent=90,infer_log_scale=True,
                        error_percent=68.26)
plt.suptitle('intercept at '+str(d)
             +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
             +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
             +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')

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



