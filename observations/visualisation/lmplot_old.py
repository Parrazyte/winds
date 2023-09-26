#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:40:06 2022

@author: parrama
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from scipy.stats import linregress

from custom_pymccorrelation import perturb_values

'''
DEPRECATED, USE LMPLOT_VITTORIA INSTEAD
'''

def lmplot_uncert(ax,x,y,dx,dy,data,data_d,percent=90,distrib='gaussian',nsim=1000,linecolor='blue',intercolor=None,
                  shade_regions=False,return_intercept=True):
    
    '''
    plots line regression of a dataset with uncertainties, with an argument format similar to lmplot. 
    
    percent: the percentile values of the provided errors.
    distrib: error distributions. Gaussian is the only one implemented as of now
    nsim: number of performed simulations
    maincolor: color of the main line
    intercolor: color of the interval region. If set to None, uses light+linecolor (works for most basic colors)
    
    behavior: computes nsim regplot on the ax using perturbed versions of the x and y variables, while deleting the points and line at each
    attempt. The final line displayed is the one without uncertainties added (it is displayed first).

    if return_intercept is set to True, returns the perturbated slope and intercept using stats.linregress
    if shade_regions is set to true:
    
    -The line 'interval' colored regions are drawn using a very low alpha so as to create a visible distribution after nsim.
    Note : the lowest alpha value possible is 1/255, so using nsim>255 will lead to oversaturation of the colors
    
    else:
    -Computes the distribution of each individual regions from the perturbation to draw the 90% exterior of the intersection
    between the shapes    
    
    '''
    
    #bandcolor variable
    if intercolor==None:
        bandcolor='light'+linecolor
    else:
        bandcolor=intercolor
        
    #switching the format to array to compute the perturbations
    x_arr=np.array(data[x])
    y_arr=np.array(data[y])
    dx_arr=np.array(data_d[dx].replace(np.nan,0))*(percent/90)
    dy_arr=np.array(data_d[dy].replace(np.nan,0))*(percent/90)
    
    #computing perturbations
    x_pert,y_pert=perturb_values(x_arr,y_arr,dx_arr,dy_arr,Nperturb=nsim)[:2]
    
    x_pert=x_pert.astype(float)
    y_pert=y_pert.astype(float)

    #storing the elements already in the axe children at the start
    ax_children_init=ax.get_children()
    
    #plotting a first regression with no perturbations for the central line
    sns.regplot(x=x,y=y,data=data,ax=ax,truncate=False,ci=90)
    
    #fetching the newly added elements to the axis list
    ax_children_regplot=[elem for elem in ax.get_children() if elem not in ax_children_init]
    
    #deleting the line interval and the points
    for elem_children in ax_children_regplot:
        if type(elem_children) in [mpl.collections.PolyCollection,mpl.collections.PathCollection]:
                elem_children.remove()
        
        #changing the color of the points and main line
        if type(elem_children)==mpl.lines.Line2D:
            elem_children.set_color(linecolor)

                
    #updating the list of children to be preserved
    ax_children_init=ax.get_children()

    slope_vals=np.zeros(nsim)
    intercept_vals=np.zeros(nsim)

    #loop on nsim iterations
    
    bound_inter=np.array([None]*nsim)
    
    with tqdm(total=nsim) as pbar:
        for i in range(nsim):

            #computing the intercept if asked to
            if return_intercept:
                curr_regress=linregress(x_pert[i],y_pert[i])
                slope_vals[i]=curr_regress.slope
                intercept_vals[i]=curr_regress.intercept

            #computing a dataframe set from an iteration of perturbed values
            df_pert=pd.DataFrame(data=np.array([x_pert[i],y_pert[i]]).T,columns=['x_pert','y_pert'])
            
            #computing the regression plot on the current axis
            sns.regplot(x='x_pert',y='y_pert',data=df_pert,ax=ax,truncate=False,ci=90)
            
            #fetching the newly added elements to the axis list
            ax_children_regplot=[elem for elem in ax.get_children() if elem not in ax_children_init]
            
            for elem_children in ax_children_regplot:
                
                #removing everything but the line interval
                if type(elem_children)!=mpl.collections.PolyCollection:
                    elem_children.remove()
                else:    
                    
                    if shade_regions:
                        #lowering the alpha of the line interval
                        elem_children.set_alpha(1/(min(nsim,255)))
                        elem_children.set_color(bandcolor)
                    
                    else:
                        
                        '''
                        in order to compute the 90% regions, we store the top and bottom "line" boundaries of each 
                        individual regions. 
                        Note that the polygons as stored as increasing x for the lower side,
                        then decreasing x for the upper side (with one additional point at the beginning of each curve)
                        we then delete the children because we don't care about it
                        '''
                        
                        #storing the points of this interval
                        points_inter=elem_children.get_paths()[0].to_polygons()[0].T
                        
                        #computing the sampling fo the polygons (number of abscisses used as boundaries)
                        #note: the start and finish point are doubled, so we need to take them off
                        reg_sampling=int(len(points_inter[0])/2-1)
                        
                        #storing the abscisses of the interval at the first iteration
                        if i==0:
                            abs_inter=points_inter[0][1:1+reg_sampling]
                        
                        #storing the points for the top and bottom boundaries without repetitions and with the correct order
                        bound_inter[i]=np.array([points_inter[1][1:1+reg_sampling],
                                                points_inter[1][2+reg_sampling:][::-1]])
                        
                        #removing the unwanted children
                        elem_children.remove()
                    
            pbar.update()
        
    if not shade_regions:
        #now that we have the array with the distribution of all the perturbations, we re-organize it into something regular,       
        #then transpose and sort it to get the distribution of each boundary
        bound_inter=np.array([elem for elem in bound_inter])
        
        #transposing into #low-high curve / point / iteration
        bound_inter=np.transpose(bound_inter,(1,2,0))
        
        #and sorting on the iterations
        bound_inter.sort(2)
        
        #selecting the nth percentile of each (low percentile for the lower curve, upper percentile for the higher curve)
        low_curve=np.array([bound_inter[0][i][round((1-percent/100)*nsim)] for i in range(reg_sampling)])
        high_curve=np.array([bound_inter[1][i][round((percent/100)*nsim)] for i in range(reg_sampling)])
        
        #filling the region
        ax.fill_between(abs_inter,low_curve,high_curve,color=bandcolor)

    uncert_arr=np.array([[None,None,None]]*2)


    if return_intercept:

        #sorting the values to pick out the percentiles
        slope_vals.sort()
        intercept_vals.sort()

        #storing the main medians in the array
        uncert_arr[0][0]=slope_vals[int(nsim*0.5)]
        uncert_arr[1][0] = intercept_vals[int(nsim * 0.5)]

        #lower uncertainties
        uncert_arr[0][1]=uncert_arr[0][0]-slope_vals[int(nsim*(50-percent/2)/100)]
        uncert_arr[1][1] = uncert_arr[1][0] - intercept_vals[int(nsim * (50 - percent / 2) / 100)]

        #upper uncertainties
        uncert_arr[0][2] = slope_vals[int(nsim * percent/100)]-uncert_arr[0][0]
        uncert_arr[1][2] = intercept_vals[int(nsim * percent / 100)] - uncert_arr[1][0]

        return uncert_arr
