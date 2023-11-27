import pandas as pd
from numpy import array, nan_to_num, zeros, transpose, isnan
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from scipy.stats import linregress

from custom_pymccorrelation import perturb_values


def lmplot_uncert_a(ax, x, y, dx, dy, xlim=None,ylim=None, percent=90, distrib='gaussian', nsim=1000, linecolor='blue',
                    lw=1.3,intercolor=None,shade_regions=False, return_linreg=True, infer_log_scale=False):
    # bandcolor variable
    if intercolor == None:
        bandcolor = 'lightgrey'
    else:
        bandcolor = intercolor

    if infer_log_scale:
        log_x=ax.get_xscale()=='log'
        log_y=ax.get_yscale()=='log'
    else:
        log_x=False
        log_y=False

    # switching the format to array to compute the perturbations
    x_arr = array(x)
    y_arr = array(y)
    dx_arr = array(nan_to_num(dx)).T * (percent / 90)
    dy_arr = array(nan_to_num(dy)).T * (percent / 90)

    if log_x:
        x_arr=np.log10(x_arr)

        #computing the +/- dx effect in log space
        dx_arr=np.array([np.log10(array(x)+array(dx_arr.T[0])),np.log10(array(x)-array(dx_arr.T[1]))])

        #and applying it
        dx_arr=np.array([dx_arr[0]-x_arr,x_arr-dx_arr[1]]).T


    if log_y:
        y_arr = np.log10(y_arr)

        # computing the +/- dx effect in log space
        dy_arr = np.array([np.log10(array(y) + array(dy_arr.T[0])), np.log10(array(y) - array(dy_arr.T[1]))])

        #and applying it
        dy_arr=np.array([dy_arr[0]-y_arr,y_arr-dy_arr[1]]).T


    # computing perturbations
    x_pert, y_pert = perturb_values(x_arr, y_arr, dx_arr, dy_arr, xlim=xlim,ylim=ylim,Nperturb=nsim)[:2]
    x_pert = x_pert.astype(float)
    y_pert = y_pert.astype(float)
    # storing the elements already in the axe children at the start
    ax_children_init = ax.get_children()

    # plotting a first regression with no perturbations for the central line
    sns.regplot(x=x_arr, y=y_arr, ax=ax, truncate=False, ci=90)

    # # fetching the absciss of the intercept point we're gonna use
    # x_intercept = plt.gca().xaxis.get_ticklocs()
    # x_intercept = x_intercept[int(len(x_intercept) / 2)]
    x_intercept=0

    # fetching the newly added elements to the axis list
    ax_children_regplot = [elem for elem in ax.get_children() if elem not in ax_children_init]

    # deleting the line interval and the points
    for elem_children in ax_children_regplot:
        elem_children.remove()

    #we make this one in linear to ensure no issue for the sampling created from regplot
    fig_new,ax_new=plt.subplots()

    if log_x:
        ax_new.set_xlim(np.log10(ax.get_xlim()))
    if log_y:
        ax_new.set_ylim(np.log10(ax.get_ylim()))

    # updating the list of children to be preserved
    ax_children_init = ax_new.get_children()

    slope_vals = zeros(nsim)
    intercept_vals = zeros(nsim)

    # dy_lims=np.array([elem if not np.nan(elem) else 0 for elem in dx_arr])
    #
    # dy_lims=np.array([elem if not np.nan(elem) else 0 for elem in dx_arr])

    # loop on nsim iterations

    if np.ndim(dx_arr)==2:
        dx_arr_lim=dx_arr.max(1)
    else:
        dx_arr_lim=dx_arr

    if np.ndim(dy_arr)==2:
        dy_arr_lim=dy_arr.max(1)
    else:
        dy_arr_lim=dy_arr

    plt.xlim((np.nanmin(x_arr-dx_arr_lim),np.nanmax(x_arr+dx_arr_lim)))
    plt.ylim((np.nanmin(y_arr-dy_arr_lim),np.nanmax(y_arr+dy_arr_lim)))

    bound_inter = array([None] * nsim)

    with tqdm(total=nsim) as pbar:
        for i in range(nsim):

            # computing the linreg values
            mask_nonan = ~(isnan(x_pert[i]) | isnan(y_pert[i]))
            curr_regress = linregress(x_pert[i][mask_nonan], y_pert[i][mask_nonan])
            slope_vals[i] = curr_regress.slope
            intercept_vals[i] = curr_regress.intercept

            # computing a dataframe set from an iteration of perturbed values
            df_pert = pd.DataFrame(data=array([x_pert[i], y_pert[i]]).T, columns=['x_pert', 'y_pert'])
            # computing the regression plot on the current axis
            sns.regplot(x=x_pert[i], y=y_pert[i], ax=ax_new, truncate=False, ci=90)
            # fetching the newly added elements to the axis list
            ax_children_regplot = [elem for elem in ax_new.get_children() if elem not in ax_children_init]

            for elem_children in ax_children_regplot:
                # removing everything but the line interval
                if type(elem_children) != mpl.collections.PolyCollection:
                    elem_children.remove()
                else:
                    if shade_regions:
                        # lowering the alpha of the line interval
                        elem_children.set_alpha(1 / (min(nsim, 255)))
                        elem_children.set_color(bandcolor)
                    else:
                        # storing the points of this interval
                        points_inter = elem_children.get_paths()[0].to_polygons()[0].T

                        # computing the sampling fo the polygons (number of abscisses used as boundaries)
                        # note: the start and finish point are doubled, so we need to take them off
                        reg_sampling = int(len(points_inter[0]) / 2 - 1)
                        # storing the abscisses of the interval at the first iteration
                        if i == 0:
                            abs_inter = points_inter[0][1:1 + reg_sampling]
                        # storing the points for the top and bottom boundaries without repetitions and with the correct order
                        bound_inter[i] = array([points_inter[1][1:1 + reg_sampling],
                                                points_inter[1][2 + reg_sampling:][::-1]])
                        # removing the unwanted children
                        elem_children.remove()
            pbar.update()

    plt.close(fig_new)

    if not shade_regions:
        # now that we have the array, we re-organize it into something regular, then transpose and sort it to get the distribution
        # of each boundary
        bound_inter = array([elem for elem in bound_inter])
        # transposing into #low-high curve / point / iteration
        bound_inter = transpose(bound_inter, (1, 2, 0))
        # and sorting on the iterations
        bound_inter.sort(2)
        # selecting the nth percentile of each (low percentile for the lower curve, upper percentile for the higher curve)
        low_curve = array([bound_inter[0][i][round((1 - percent / 100) * nsim)] for i in range(reg_sampling)])
        high_curve = array([bound_inter[1][i][round((percent / 100) * nsim)] for i in range(reg_sampling)])
        # filling the region

        if log_x:
            abs_inter=10**abs_inter

        if log_y:
            low_curve=10**low_curve
            high_curve=10**high_curve

        ax.fill_between(abs_inter, low_curve, high_curve, color=bandcolor,zorder=0)

    uncert_arr = array([[None, None, None]] * 2)

    #in this case the linear regression actually computes the intercept at 0 because the pivot point is at 0
    intercept_at_x_vals = slope_vals * x_intercept + intercept_vals

    # sorting the values to pick out the percentiles
    slope_vals.sort()
    intercept_at_x_vals.sort()

    # storing the main medians in the array
    uncert_arr[0][0] = slope_vals[round(nsim * 0.5)]
    uncert_arr[1][0] = intercept_at_x_vals[round(nsim * 0.5)]

    # lower uncertainties
    uncert_arr[0][1] = uncert_arr[0][0] - slope_vals[round(nsim * (1 - percent / 100))]
    uncert_arr[1][1] = uncert_arr[1][0] - intercept_at_x_vals[round(nsim * (1 - percent / 100))]

    # upper uncertainties
    uncert_arr[0][2] = slope_vals[round(nsim * percent / 100)] - uncert_arr[0][0]
    uncert_arr[1][2] = intercept_at_x_vals[round(nsim * percent / 100)] - uncert_arr[1][0]

    slope_arr=uncert_arr[0]
    intercept_arr=uncert_arr[1]

    if xlim is None:
        xlim_mask=np.repeat(False,len(x))
    else:
        xlim_mask=xlim

    if ylim is None:
        ylim_mask=np.repeat(False,len(x))
    else:
        ylim_mask=ylim

    tot_nonlin_mask=~ ((xlim_mask) & (ylim_mask))

    #computing the intrinsic scatter (standard deviation)
    sigma_vals=np.array([np.sqrt(np.nansum((y_pert[id][tot_nonlin_mask]-\
                                   (x_pert[id][tot_nonlin_mask]*slope_vals[id]+\
                                    intercept_vals[id]))**2))\
                            for id in range(nsim)])

    sigma_vals.sort()
    sigma_med=sigma_vals[round(nsim*0.5)]
    sigma_arr=np.array([sigma_med,sigma_med-sigma_vals[round(nsim * (1 - percent / 100))],
                                  sigma_vals[round(nsim * percent / 100)]-sigma_med])

    #plotting the median line with the median value of the intercept and coefficient

    #fetching a sample of points from the limits of the ax to create the line in between
    #note
    if log_x:
        x_line=np.linspace(np.log10(ax.get_xlim()[0]),np.log10(ax.get_xlim()[1]),500)
    else:
        x_line=np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],500)

    #locking the ax to avoid resizing when plotting the next line
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    y_line=(x_line-x_intercept)*uncert_arr[0][0]+uncert_arr[1][0]

    #converting to powers if in log space
    if log_x:
        x_line_plot=10**x_line
    else:
        x_line_plot=x_line

    if log_y:
        y_line_plot=10**y_line
    else:
        y_line_plot=y_line

    #plotting the line
    plt.plot(x_line_plot,y_line_plot,lw=lw,color=linecolor)

    if return_linreg:
        return slope_arr,intercept_arr,sigma_arr
