import pandas as pd
from numpy import array, nan_to_num, zeros, transpose, isnan
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from scipy.stats import linregress

from custom_pymccorrelation import perturb_values


def lmplot_uncert_a(ax, x, y, dx, dy, percent=90, distrib='gaussian', nsim=255, linecolor='blue', lw=1.3,
                    intercolor=None,
                    shade_regions=False, return_intercept=True):
    # bandcolor variable
    if intercolor == None:
        bandcolor = 'light' + linecolor
    else:
        bandcolor = intercolor
    # switching the format to array to compute the perturbations
    x_arr = array(x)
    y_arr = array(y)
    dx_arr = array(nan_to_num(dx)).T * (percent / 90)
    dy_arr = array(nan_to_num(dy)).T * (percent / 90)
    # computing perturbations
    x_pert, y_pert = perturb_values(x_arr, y_arr, dx_arr, dy_arr, Nperturb=nsim)[:2]
    x_pert = x_pert.astype(float)
    y_pert = y_pert.astype(float)
    # storing the elements already in the axe children at the start
    ax_children_init = ax.get_children()

    # plotting a first regression with no perturbations for the central line
    sns.regplot(x=x_arr, y=y_arr, ax=ax, truncate=False, ci=90)

    # fetching the abscix of the intercept point we're gonna use
    x_intercept = plt.gca().xaxis.get_ticklocs()
    x_intercept = x_intercept[int(len(x_intercept) / 2)]

    # fetching the newly added elements to the axis list
    ax_children_regplot = [elem for elem in ax.get_children() if elem not in ax_children_init]

    # deleting the line interval and the points
    for elem_children in ax_children_regplot:
        elem_children.remove()

    # updating the list of children to be preserved
    ax_children_init = ax.get_children()

    slope_vals = zeros(nsim)
    intercept_vals = zeros(nsim)

    # loop on nsim iterations
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    bound_inter = array([None] * nsim)
    with tqdm(total=nsim) as pbar:
        for i in range(nsim):

            # computing the intercept if asked to
            if return_intercept:
                mask_nonan = ~(isnan(x_pert[i]) | isnan(y_pert[i]))
                curr_regress = linregress(x_pert[i][mask_nonan], y_pert[i][mask_nonan])
                slope_vals[i] = curr_regress.slope
                intercept_vals[i] = curr_regress.intercept

            # computing a dataframe set from an iteration of perturbed values
            df_pert = pd.DataFrame(data=array([x_pert[i], y_pert[i]]).T, columns=['x_pert', 'y_pert'])
            # computing the regression plot on the current axis
            sns.regplot(x=x_pert[i], y=y_pert[i], ax=ax, truncate=False, ci=90)
            # fetching the newly added elements to the axis list
            ax_children_regplot = [elem for elem in ax.get_children() if elem not in ax_children_init]

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

    if not shade_regions:
        # now that we ahve the array, we re-organize it into something regular, then transpose and sort it to get the distribution
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
        ax.fill_between(abs_inter, low_curve, high_curve, color=bandcolor)

    uncert_arr = array([[None, None, None]] * 2)

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

    #plotting the median line with the median value of the intercept and coefficient

    #fetching two points to create the line in between
    x_line=ax.get_xlim()

    #locking the ax to avoid resizing when plotting the next line
    ax.set_xlim(ax.get_xlim())
    ax.set_ylum(ax.get_ylim())

    #plotting the line
    plt.plot(x_line,x_line*uncert_arr+(x_line-x_intercept),lw=lw,color=linecolor)

    if return_intercept:
        return uncert_arr, x_intercept
