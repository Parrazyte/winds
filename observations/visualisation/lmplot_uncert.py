import pandas as pd
from numpy import array, nan_to_num, zeros, transpose, isnan
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from scipy.stats import linregress
from scipy.stats import norm as scinorm

from custom_pymccorrelation import perturb_values


def lmplot_uncert_a(ax, x, y, dx, dy, xlim=None,ylim=None, percent=68.26,percent_regions='auto',nsim=100,
                    intercept_pos='auto',
                    return_linreg=True, infer_log_scale=False,log_sampling=False,nanzero_err=True,
                    error_percent=68.26,
                    xbounds='auto',ybounds='auto',
                    line_color='blue',lw=1.3, inter_color='lightgrey',check_indiv=False):

    '''
    Computes a linear regression, confidence bands and errors on its interval using bootstrapping
    We use the perturbation from custom_pymmcorrelation to consider asymmetric uncertainties and upper limits

    We are faced with a problem of lisibility: a "correct" (or close to it) mathematical description would
    require returning the equation with the covariance between the slope and the intercept, which would also need
    to be quoted with uncertainties due to the perturbation.
     or most purposes this is not necessary, so we simply return the slope and intercept computed where desired

    Notes:
        -the "central" regression and scatter values are computed from the median of the perturbated values

    Method:

    0) compute the central line from a linear regression without perturbation
        (used for plot and the central values of the distribution)

    1) compute nsim linear regression bootstraps with intercept at 0

    2) get the confidence interval from these bootstraps from the distribution of y values of all individual LR lines
       (and plot it)

    3) compute the x value with the lowest ci errors (aka where best to shift the LR to get the intercept error value)

    4) Choose an x_intercept value according to intercept_pos (see above) and recompute the LR at these values

    5) extract the percentages of slope, intercept, and scatter and return them if necessary

    Note: the scatte is returned as the sqrt of the variance of each point compared to the linear regression
    ax: ax where the plots will be displayed

    -x,y:               data

    -dx,dy:             uncertainties. can be of size [2,N] for asymmetric

    -xlim,ylim:         upper limit information. Either None or a N-size mask with a True for upper/lower limits.
                        In this case, the values will be perturbated between x and xlim with a uniform distribution
                        independantly of whether the limit is upper or lower,
                         x should always be set at the bottom treshold,
                         and xlim should be set as the upper limit interval

                         For now, upper and lower limits are not considered in the scatter computation

    -percent:           percent of the posterior distributions returned (the main value is the median so 50%)
                        Default value is for 1 sigma

    -percent_regions:   percent or list of percent of the line distributions to use for the shaded regions.
                        use "auto" to keep the same value as percent

    -intercept_pos:     method of shifting x_value for the LR computation to get the info of the intercept error

                        "auto": takes the closest xtick to the lowest intercept errors
                        "best": takes the actual lowest intercept error
                        value:  forced at that value

    -return_linreg:     returns the slope, intercept,scatter, and x_intercept (otherwise returns nothing)

    -infer_log_scale:   automatically adjust the linear regression to log-linear or log-log linear depending on whether
                        the axis are in log scale

    -log_sampling:      Force X axis log scale sampling when creating the regression curves

    -nanzero_err:       consider uncertainty values of nan as if there was no error

    -error_percent:     the error range at which the dx and dy values are provided. Used to rescale the errors
                        (assuming gaussianity) to 1 sigma to correctly perturbate the values afterwards
                        (normalized to 68.26)

    -xbounds,ybounds:   resizes the graph manually instead of automatically
                        -'auto': auto resize
                        -values: resize manually to these values
                        -None: no resize

    -line_color, lw:     displays for the main plotted line

    -inter_color:        confidence interval region color. If list like, use the different colors
                         for different intervals but should be the same size as the percent_regions list

    -check_indiv:
           -True         plot a few elements of the perturbation, to check results
           -combined     plot the elements on the same graph
    '''

    if infer_log_scale:
        log_x=ax.get_xscale()=='log'
        log_y=ax.get_yscale()=='log'
    else:
        log_x=False
        log_y=False

    # switching the format to array to compute the perturbations and removing the nans

    x_arr = array(x)
    y_arr = array(y)

    mask_values=(~np.isnan(y_arr)) & (~np.isnan(x_arr))

    x_arr=x_arr[mask_values]
    y_arr=y_arr[mask_values]

    #replacing nans with zero if asked

    #note: we need to resize the uncertainties according to the gaussian distribution
    #so we're using the z-score
    if nanzero_err:
        dx_arr = array(nan_to_num(dx)).T * 1/scinorm.ppf((1 + error_percent/100) / 2)
        dy_arr = array(nan_to_num(dy)).T * 1/scinorm.ppf((1 + error_percent/100) / 2)
    else:
        dx_arr = array(dx).T * 1/scinorm.ppf((1 + error_percent/100) / 2)
        dy_arr = array(dy).T * 1/scinorm.ppf((1 + error_percent/100) / 2)

    #reshaping simmetrical uncertainties to avoid issues later
    if np.ndim(dx_arr)==1:
        dx_arr=np.array([dx_arr,dx_arr]).T
    if np.ndim(dy_arr)==1:
        dy_arr = np.array([dy_arr, dy_arr]).T

    dx_arr=dx_arr[mask_values]
    dy_arr=dy_arr[mask_values]

    #computing a mask of where upper limits are in at least one ax of the points
    if xlim is None:
        xlim_mask=np.repeat(False,len(x))
        xlim_arr=None
    else:
        xlim_mask=xlim
        xlim_arr = xlim[mask_values]

    if ylim is None:
        ylim_mask=np.repeat(False,len(x))
        ylim_arr=None
    else:
        ylim_mask=ylim
        ylim_arr = ylim[mask_values]

    xlim_arr_mask=xlim_mask[mask_values]
    ylim_arr_mask=ylim_mask[mask_values]

    tot_nonlim_mask=~ ((xlim_arr_mask) | (ylim_arr_mask))

    if log_x:

        #computing the +/- dx effect in log space
        dx_arr=np.array([np.log10(array(x)-array(dx_arr.T[0])),np.log10(array(x)+array(dx_arr.T[1]))])

        x_arr=np.log10(x_arr)

        #and applying it
        dx_arr=np.array([x_arr-dx_arr[0],dx_arr[1]-x_arr]).T


    if log_y:

        # computing the +/- dx effect in log space
        dy_arr = np.array([np.log10(array(y) - array(dy_arr.T[0])), np.log10(array(y) + array(dy_arr.T[1]))])

        y_arr = np.log10(y_arr)

        #and applying it
        dy_arr=np.array([y_arr-dy_arr[0],dy_arr[1]-y_arr]).T

    # computing perturbations
    x_pert, y_pert = perturb_values(x_arr, y_arr, dx_arr, dy_arr, xlim=xlim_arr,ylim=ylim_arr,Nperturb=nsim)[:2]
    x_pert = x_pert.astype(float)
    y_pert = y_pert.astype(float)

    #first regplot just to get the ax limits

    # storing the elements already in the axe children at the start
    ax_children_init = ax.get_children()

    if intercept_pos=='auto':
        # plotting a first regression with no perturbations for the central line
        sns.regplot(x=x_arr, y=y_arr, ax=ax, truncate=False, ci=90)

        # # fetching the absciss of the intercept ticks for late
        x_intercept_ticks = ax.xaxis.get_ticklocs()

        # fetching the newly added elements to the axis list
        ax_children_regplot = [elem for elem in ax.get_children() if elem not in ax_children_init]

        # deleting the line interval and the points
        for elem_children in ax_children_regplot:
            elem_children.remove()

    #main array creation
    slope_vals = zeros(nsim)
    intercept_vals = zeros(nsim)

    #correctly preparing the graph size
    if np.ndim(dx_arr)==2:
        dx_arr_lim=dx_arr.max(1)
    else:
        dx_arr_lim=dx_arr

    if np.ndim(dy_arr)==2:
        dy_arr_lim=dy_arr.max(1)
    else:
        dy_arr_lim=dy_arr

    if xbounds is not None:

        if xbounds=='auto':
            if log_x:
                ax.set_xlim(10**(np.nanmin(x_arr - dx_arr_lim)), 10**(np.nanmax(x_arr + dx_arr_lim)))
            else:
                ax.set_xlim((np.nanmin(x_arr - dx_arr_lim)),(np.nanmax(x_arr + dx_arr_lim)))
        else:
            ax.set_xlim(xbounds)

    if ybounds is not None:

        if ybounds=='auto':
            if log_y:
                ax.set_ylim(10**(np.nanmin(y_arr - dy_arr_lim)), 10**(np.nanmax(y_arr + dy_arr_lim)))
            else:
                ax.set_ylim((np.nanmin(y_arr - dy_arr_lim)), (np.nanmax(y_arr + dy_arr_lim)))
        else:
            ax.set_ylim(ybounds)

    #first loop with intercept at 0 on nsim iterations
    bound_inter = array([None] * nsim)
    for i in range(nsim):

        # computing the linreg values
        mask_nonan = ~(isnan(x_pert[i]) | isnan(y_pert[i]))

        curr_regress = linregress(x_pert[i][mask_nonan], y_pert[i][mask_nonan])

        slope_vals[i] = curr_regress.slope
        intercept_vals[i] = curr_regress.intercept


    # for i in range(len(x_arr)):
    #     fig,ax=plt.subplots()
    #     ax.hist(x_pert.T[i])
    #     ax.axvline(np.median(x_pert.T[i]),zorder=1000,color='red')
    #
    #     ax.errorbar(x_arr[i],nsim*0.5,xerr=dx_arr[i] if type(dx_arr[i])==np.float64 else [[dx_arr[i].T[0]],[dx_arr[i].T[1]]],
    #                 marker='d')
    #
    # plt.figure()


    init_reg=linregress(x_arr,y_arr)
    slope=init_reg.slope
    inter=init_reg.intercept

    #computing the intrinsic scatter (standard deviation) (here because not affected by change in intercept
    #main sigma from non perturbated values
    sigma_vals=np.array([np.sqrt(np.nansum((y_pert[id][tot_nonlim_mask]-\
                                   ((x_pert[id][tot_nonlim_mask])*slope_vals[id]+\
                                    intercept_vals[id]))**2)/sum(tot_nonlim_mask))\
                            for id in range(nsim)])

    #fetching a sample of points from the limits of the ax to create the lines
    if log_x or log_sampling:
        x_line=np.logspace(np.log10(ax.get_xlim()[0]),np.log10(ax.get_xlim()[1]),2*nsim)

    else:
        x_line=np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],2*nsim)

    #saving the first set of perturbated values to allow checks if needed
    # the slope should be the same in the second perturbation round
    slope_vals_save=slope_vals.copy()
    intercept_vals_save=intercept_vals.copy()

    #list of y position of all perturbated lines
    # if log_y and not log_x:
    if log_y:
        y_line_pert=10**(np.array([(np.log10(x_line) if log_x else x_line)*slope_vals_save[i]+intercept_vals_save[i] for i in range(nsim)]).T)
    else:
        y_line_pert =np.array([(np.log10(x_line) if log_x else x_line) * slope_vals_save[i] + intercept_vals_save[i] for i in range(nsim)]).T

    # #to check the lines if needed
    if check_indiv:
        if check_indiv=='merge':
            plt.figure()
            if log_x:
                plt.xscale('log')
            if log_y:
                plt.yscale('log')

            plt.errorbar(x,y, xerr=dx, yerr=dy, linestyle='', marker='d', color='red',alpha=0.05)

        for i in range(nsim//20):
            if check_indiv is True:
                plt.figure()
                if log_x:
                    plt.xscale('log')
                if log_y:
                    plt.yscale('log')
                plt.errorbar(x,y, xerr=dx, yerr=dy, linestyle='', marker='d', color='red')

            plt.scatter(10**(x_pert[i]),10**(y_pert[i]),color='blue',alpha=0.05 if check_indiv=='merge' else 1)

            plt.plot(x_line,y_line_pert.T[i],alpha=0.05 if check_indiv=='merge' else 0.5,color='green')

    y_line_pert.sort()

    #makign an iterable for percent_regions
    if percent_regions=='auto':
        percent_regions_use=np.array([percent])
    else:
        if type(percent_regions) not in (tuple,np.ndarray,list):
            percent_regions_use=np.array([percent_regions])
        else:
            percent_regions_use=np.array(percent_regions)

    #making an iterable for inter_color
    if type(inter_color) not in (tuple,np.ndarray,list):
        inter_color_use=np.repeat(inter_color,len(percent_regions_use))
    else:
        inter_color_use=np.array(inter_color)

    assert len(percent_regions_use)==len(inter_color_use),\
        'The length of the percent regions and inter_color iterables do not match'

    #re-ordering the colors before sorting the percent regions array
    inter_color_use=inter_color_use[percent_regions_use.argsort()]

    #sorting the percent regions array
    percent_regions_use.sort()

    #reversing the percent ordering to avoid hiding the center regions with the broader ones
    for i_region,elem_percent_region in enumerate(percent_regions_use[::-1]):

        y_line_low=y_line_pert.T[round(nsim * (0.5-elem_percent_region/200))]
        y_line_high=y_line_pert.T[round(nsim* (0.5+elem_percent_region/200))]

        ax.fill_between(x_line, y_line_low, y_line_high, color=inter_color_use[i_region], zorder=0,
                        alpha=1)

    #keeping the same percent as the perturbation here to be used below
    y_line_low_std = y_line_pert.T[round(nsim * (0.5 - percent / 200))]
    y_line_high_std = y_line_pert.T[round(nsim * (0.5 + percent / 200))]

    #computing the best possible intercept
    x_intercept_best=x_line[(y_line_high_std-y_line_low_std).argsort()[0]]

    if intercept_pos=='best':
        x_intercept=x_intercept_best
    elif intercept_pos=='auto':
        x_intercept=x_intercept_ticks[abs(x_intercept_ticks-x_intercept_best).argsort()[0]]
    else:
        x_intercept=intercept_pos

    # #recomputing the LR with the intercept this time
    # #base regression without upper limits
    #
    # base_regress=linregress(x_arr[tot_nonlim_mask]-x_intercept,y_arr[tot_nonlim_mask])
    #
    # base_regress_slope=base_regress.slope
    # base_regress_intercept=base_regress.intercept

    for i in range(nsim):

        # computing the linreg values
        mask_nonan = ~(isnan(x_pert[i]) | isnan(y_pert[i]))
        curr_regress = linregress(x_pert[i][mask_nonan]-x_intercept, y_pert[i][mask_nonan])

        slope_vals[i] = curr_regress.slope
        intercept_vals[i] = curr_regress.intercept

    #tests to check individual realizations
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    # for i in np.arange(0,len(x_pert),100):
    #     plt.figure()
    #     plt.scatter(x_pert[i],y_pert[i],color='red')
    #     plt.errorbar(x,y,dx,dy,color='blue',ls='')
    #     curr_regr=((np.log10(x_line) if log_x else x_line)-x_intercept)*slope_vals[i]+intercept_vals[i]
    #     plt.plot(x_line,curr_regr,color='orange')
    #
    # plt.figure()

    uncert_arr = array([[None, None, None]] * 2)

    # sorting the values to pick out the percentiles
    slope_vals_copy=slope_vals.copy()
    slope_vals.sort()

    # storing the median value as the central position for the array
    uncert_arr[0][0] = slope_vals[round(nsim * (0.5))]
    uncert_arr[0][1] = uncert_arr[0][0] - slope_vals[round(nsim * (0.5-percent/200))]
    uncert_arr[0][2] = slope_vals[round(nsim * (0.5+percent/200))] - uncert_arr[0][0]

    slope_arr=uncert_arr[0]

    #re-centering the intercept array
    intercept_vals_copy=intercept_vals.copy()
    intercept_vals.sort()

    uncert_arr[1][0] = intercept_vals[round(nsim * (0.5))]
    uncert_arr[1][1] = uncert_arr[1][0] - intercept_vals[round(nsim * (0.5-percent/200))]
    uncert_arr[1][2] = intercept_vals[round(nsim * (0.5+percent/200))] - uncert_arr[1][0]

    intercept_arr=uncert_arr[1]

    #main sigma value
    sigma_vals.sort()

    #main value being the median
    sigma_main=sigma_vals[len(sigma_vals)//2]

    #and the uncertainties
    sigma_arr=np.array([sigma_main,max(sigma_main-sigma_vals[round(nsim * (0.5-percent/200))],0),
                                  max(sigma_vals[round(nsim*(0.5+percent/200))]-sigma_main,0)])

    base_regress_line=((np.log10(x_line) if log_x else x_line)-x_intercept)*slope_arr[0]+intercept_arr[0]

    # converting to powers if in log space
    # if log_y and not log_x:
    if log_y :
        y_line_plot=10**(base_regress_line)
    else:
        y_line_plot=base_regress_line

    #plotting the base line
    ax.plot(x_line,y_line_plot,lw=lw,color=line_color)

    #output info
    if return_linreg:
        return slope_arr,intercept_arr,sigma_arr,x_intercept

