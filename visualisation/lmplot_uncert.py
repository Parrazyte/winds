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

from custom_pymccorrelation import perturb_values

def lmplot_uncert(ax,x,y,dx,dy,data,data_d,percent=90,distrib='gaussian',nsim=255,linecolor='blue',intercolor=None):
    
    '''
    plots line regression of a dataset with uncertainties, with an argument format similar to lmplot. 
    
    percent: the percentile values of the provided errors.
    distrib: error distributions. Gaussian is the only one implemented as of now
    nsim: number of performed simulations
    maincolor: color of the main line
    intercolor: color of the interval region. If set to None, uses light+linecolor (works for most basic colors)
    
    behavior: computes nsim regplot on the ax using perturbed versions of the x and y variables, while deleting the points and line at each
    attempt. The final line displayed is the one without uncertainties added (it is displayed first).
    The line 'interval' colored regions are drawn using a very low alpha so as to create a visible distribution after nsim.
    
    Note : the lowest alpha value possible is 1/255, so using nsim>255 will lead to oversaturation of the colors
    
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
    sns.regplot(x=x,y=y,data=data,ax=ax)
    
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
    
    #loop on nsim iterations
    
    with tqdm(total=nsim) as pbar:
        for i in range(nsim):
            
            #computing a dataframe set from an iteration of perturbed values
            df_pert=pd.DataFrame(data=np.array([x_pert[i],y_pert[i]]).T,columns=['x_pert','y_pert'])
            
            #computing the regression plot on the current axis
            sns.regplot(x='x_pert',y='y_pert',data=df_pert,ax=ax)
            
            #fetching the newly added elements to the axis list
            ax_children_regplot=[elem for elem in ax.get_children() if elem not in ax_children_init]
            
            for elem_children in ax_children_regplot:
                
                #removing everything but the line interval
                if type(elem_children)!=mpl.collections.PolyCollection:
                    elem_children.remove()
                else:    
                    #lowering the alpha of the line interval
                    elem_children.set_alpha(1/(min(nsim,255)))
                    elem_children.set_color(bandcolor)
                        
            pbar.update()