from lmplot_uncert import lmplot_uncert_a
import numpy as np
import matplotlib.pyplot as plt

test=np.arange(10)+np.array([1,-6,-4,-3,7,8,-1,2,5,2])
test2=np.arange(10)*2+np.array([-1,-6,-4,-3,7,8,-1,2,5,-2])
test_err=np.array([5/10,4/10,3/10,2/10,1/10,1/10,2/10,3/10,4/10,5/10])+np.array([1,6,4,3,7,8,1,2,5,2])/3

test_err=np.array([5/10,4/10,3/10,2/10,1/10,1/10,2/10,3/10,4/10,5/10])+np.array([0,0,0,0,0,8,1,2,5,2])/3

test2_err=test_err*2

fig,ax=plt.subplots()
ax.errorbar(test,test2,xerr=test_err,yerr=test2_err,linestyle='')
a,b,c,d=lmplot_uncert_a(ax,test,test2,test_err,test2_err,nsim=2000,intercept_pos=-10)
plt.suptitle('intercept at '+str(d)
             +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
             +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
             +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')

fig,ax=plt.subplots()
ax.errorbar(test,test2,xerr=test_err,yerr=test2_err,linestyle='')
a,b,c,d=lmplot_uncert_a(ax,test,test2,test_err,test2_err,nsim=2000,intercept_pos=15)
plt.suptitle('intercept at '+str(d)
             +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
             +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
             +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')


fig,ax=plt.subplots()
ax.errorbar(test,test2,xerr=test_err,yerr=test2_err,linestyle='')
a,b,c,d=lmplot_uncert_a(ax,test,test2,test_err,test2_err,nsim=2000,intercept_pos='auto')
plt.suptitle('intercept at '+str(d)
             +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
             +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
             +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')



fig,ax=plt.subplots()
ax.errorbar(test,test2,xerr=test_err,yerr=test2_err,linestyle='')
a,b,c,d=lmplot_uncert_a(ax,test,test2,test_err,test2_err,nsim=2000,intercept_pos='best')
plt.suptitle('intercept at '+str(d)
             +'\n'r'a=$%.2f'%a[0]+'_{-%.2f'%a[1]+'}^{+%.2f'%a[2]+'}$'
             +'\n'r'b=$%.2f'%b[0]+'_{-%.2f'%b[1]+'}^{+%.2f'%b[2]+'}$'
             +'\n'r'sig=$%.2f'%c[0]+'_{-%.2f'%c[1]+'}^{+%.2f'%c[2]+'}$')



