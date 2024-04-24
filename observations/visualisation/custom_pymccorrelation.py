'''
#Changes:
    -added halfnormal distributions combinations with each part of dx and dy if they are size 2
       (and changing the tests so that we can use the function)
    
    -added "pegged" upper limit contributions with a boundary to a physical limit. 
    In this case the uncertainty are considered as an uniform distribution instead
    #### IN PROGRESS
    
    for spearman:
    -added barycenter estimation of each percentile from the perturbation in order to be able to draw the lines
    -added an argument to plot the lines with the uncertainties over a graph
'''

"""
pymccorrelation.py
Python implementation of Curran (2014) method for calculating Spearman's
rank correlation coefficient with uncertainties. Extended to also calculate
Kendall's Tau.
Kendall tau implementation follow Isobe, Feigelson & Nelson (1986) method for
calculating the correlation coefficient with uncertainties on censored data
(upper/lowerlimit).
Copyright 2019-2021 George C. Privon, Yiqing Song
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

__version__ = '0.2.4'

import numpy as _np
import scipy.stats as _st
from scipy.stats import pearsonr as _pearsonr
from scipy.stats import spearmanr as _spearmanr
from scipy.stats import kendalltau as _kendalltau
import matplotlib.pyplot  as plt
# import _sys as sys

def valid_lims(lims):
    """
    Check limits to ensure all values are in the set [0, 1, -1]
    If limit array is valid, return True. Otherwise return False
    """

    unilims = _np.unique(lims)

    for l in unilims:
        if l not in [0, 1, -1]:
            return False
    return True


def validate_inputs(a, b, c, d):
    """
    Make sure the data arrays are all the same length
    """

    assert len(a) == len(b), "x and y must be the same length"
    assert len(c) == len(d), "the error/limit arrays must be the same length"
    assert len(a) == len(c), "the data and error/limit arrays must be the same length"

    return True


def perturb_values(x, y, dx, dy,xlim=None,ylim=None, Nperturb=10000):
    """
    For input points (x, y) with errors (dx, dy) return Nperturb sets of
    values draw from Gaussian distributions centered at x+-dx and y+-dy.

    dx/dy can be of dimension [x,2] to put non-symmetrical uncertainties

    ####EDIT: ylim added, will change the perturbation from gaussian to uniform in the interval
    """

    if y is not None:
        validate_inputs(x, y, dx, dy)
    else:
        validate_inputs(x,dx,x,dx)
        

    Nvalues = len(x)

    rng = _np.random.default_rng()

    if _np.ndim(dx)==2:
        #drawing two independant samples with gaussian shape on each side (or None if there is no uncertainty)
        #Note : we take the absolute values of the uncertainties to make sure they are valid scale parameters
        
        xp_disjointed=_np.array([[((-1)**(j+1)*abs(rng.normal(loc=0,scale=abs(dx[i][j]),size=(Nperturb)))\
                                       if (xlim is None or xlim[i]==0) else\
                                   rng.uniform(low=0,high=x[i][1],size=Nperturb)) if dx[i][1]!=0\
                                      else _np.repeat(0,Nperturb)\
                                  for i in range(len(dx))] for j in [0,1]],dtype=object)

        #different transpositions depending on if the array ends up with Nones or not
        if _np.ndim(xp_disjointed)==2:
            xp_disjointed=xp_disjointed.T
        else:
            xp_disjointed=_np.transpose(xp_disjointed,axes=(1,0,2))

        '''    
        concatenating each observation, or entirely on the side which has uncertainties,
        or just putting zeros depending on which value actually has uncertainties
        For concatening the two probability distributions, we randomly draw in each half-gaussian 
        '''

        gauss_draw = rng.integers(0, 2, (Nperturb, Nvalues))

        xp = _np.array([_np.repeat(0, Nperturb) if elem[0] is None and elem[1] is None else \
                            _np.array([elem.T[i_pert][gauss_draw[i_pert][i_elem]] for i_pert in range(Nperturb)]) \
                                if elem[0] is not None and elem[1] is not None else \
                                elem[0] if elem[0] is not None else elem[1] for i_elem, elem in
                        enumerate(xp_disjointed)])

        xp=xp.T+x
    else:

        xp = rng.normal(loc=x,
                        scale=dx,
                        size=(Nperturb, Nvalues))

    if y is not None:
        #same thing for y:
        if _np.ndim(dy)==2:
            #drawing two independant samples with gaussian shape on each side (or None if there is no uncertainty)
            #Note : we take the absolute values of the uncertainties to make sure they are valid scale parameters
            yp_disjointed=_np.array([[(((-1)**(j+1)*abs(rng.normal(loc=0,scale=abs(dy[i][j]),size=(Nperturb))))\
                                           if (ylim is None or ylim[i]==0) else\
                                       rng.uniform(low=0,high=dy[i][1],size=Nperturb)) if dy[i][1]!=0\
                                         else _np.repeat(0,Nperturb)\
                                      for i in range(len(dy))] for j in [0,1]],dtype=object)

            #different transpositions depending on if the array ends up with Nones or not
            if _np.ndim(yp_disjointed)==2:
                yp_disjointed=yp_disjointed.T
            else:
                yp_disjointed=_np.transpose(yp_disjointed,axes=(1,0,2))

            '''    
            concatenating each observation, or entirely on the side which has uncertainties,
            or just putting zeros depending on which value actually has uncertainties
            For concatening the two probability distributions, we randomly draw in each half-gaussian 
            '''

            #won't be the same as the previous one
            gauss_draw = rng.integers(0, 2, (Nperturb,Nvalues))

            yp=_np.array([_np.repeat(0,Nperturb) if elem[0] is None and elem[1] is None else\
                _np.array([elem.T[i_pert][gauss_draw[i_pert][i_elem]] for i_pert in range(Nperturb)])\
                    if elem[0] is not None and elem[1] is not None else\
                elem[0] if elem[0] is not None else elem[1] for i_elem,elem in enumerate(yp_disjointed)])

            yp=yp.T+y
                
        else:
            yp = rng.normal(loc=y,
                            scale=dy,
                            size=(Nperturb, Nvalues))
        
        if Nperturb == 1:
            xp = xp.flatten()
            yp = yp.flatten()

        fake_barycenter=_np.array([xp.mean(1),yp.mean(1)]).T
    else:
        yp=None
        fake_barycenter=None
        
    return xp, yp,fake_barycenter


def kendall(x, y,
            xlim=None, ylim=None):
    """
    Kendall tau wrapper function to determine if we need to handle censoring.
    If there is censoring, hand it off to the IFN 1986 generalized function.
    """

    if xlim is None and ylim is None:
        return _kendalltau(x, y)

    # Ensure the limit arrays are valid
    assert valid_lims(xlim), "x limit flags are not all valid"
    assert valid_lims(ylim), "y limit flags are not all valid"

    return kendall_IFN86(x, y, xlim, ylim)


def kendall_IFN86(x, y,
                  xlim, ylim):
    """
    Generalized kendall tau test described in Isobe, Feigelson & Nelson 1986
    ApJ 306, 490-507.
    Parameters:
        x: independent variable
        y: dependent variable
        xlim/ylim: censoring information for the variables. Values of
            (-1, 1, 0) correspond to (lower limit, upper limit, detection)
    Note that both x and y can be censored.
    """

    #TODO: vectorize this function, very slow

    validate_inputs(x, y, xlim, ylim)

    num = len(x)
    #set up pair counters
    a = _np.zeros((num, num))
    b = _np.zeros((num, num))

    for i in range(num):
        for j in range(num):
            if x[i] == x[j]:
                a[i, j] = 0
            elif x[i] > x[j]: #if x[i] is definitely > x[j]
                if (xlim[i] == 0 or xlim[i] == -1) and (xlim[j] == 0 or xlim[j] == 1):
                    a[i, j] = -1
            else: #if x[i] is definitely < x[j], all other uncertain cases have aij=0
                if (xlim[i] == 0 or xlim[i] == 1) and (xlim[j] == 0 or xlim[j] == -1):
                    a[i, j] = 1

    for i in range(num):
        for j in range(num):
            if y[i] == y[j]:
                b[i, j] = 0
            elif y[i] > y[j]:
                if (ylim[i] == 0 or ylim[i] == -1) and (ylim[j] == 0 or ylim[j] == 1):
                    b[i, j] = -1
            else:
                if (ylim[i] == 0 or ylim[i] == 0) and (ylim[j] == 0 or ylim[j] == -1):
                    b[i, j] = 1


    S = _np.sum(a * b)
    var = (4 / (num * (num - 1) * (num - 2))) * \
          (_np.sum(a * _np.sum(a, axis=1, keepdims=True)) - _np.sum(a * a)) * \
          (_np.sum(b * _np.sum(b, axis=1, keepdims=True)) - _np.sum(b * b)) + \
          (2 / (num * (num - 1))) * \
          _np.sum(a * a) * _np.sum(b * b)
    z = S/ _np.sqrt(var)
    tau = z * _np.sqrt(2 * (2 * num + 5)) / (3 * _np.sqrt(num * (num - 1)))
    pval = _st.norm.sf(abs(z)) * 2
    return tau, pval


def compute_corr(x, y,
                 xlim=None, ylim=None,
                 coeff=None):
    """
    Wrapper function to compute the correct correlation coefficient.
    """

    # set up the correct function for computing the requested correlation
    # coefficient
    if coeff == 'spearmanr':
        # pass to scipy function
        return _spearmanr(x, y)
    elif coeff == 'kendallt':
        # pass to our kendall tau function, in case we need to handle
        # censored data
        return kendall(x, y, xlim=xlim, ylim=ylim)
    elif coeff == 'pearsonr':
        # pass to scipy function
        return _pearsonr(x, y)


def pymccorrelation(x_init, y_init,
                    dx_init=None, dy_init=None,
                    xlim_init=None, ylim_init=None,
                    Nboot=None,
                    Nperturb=None,
                    coeff=None,
                    percentiles=(16, 50, 84),
                    return_dist=False,ax_plot=None,plot_sampling=100,color_line='grey'):
    """
    Compute a correlation coefficient with uncertainties using several methods.
    Arguments:
    x: independent variable array
    y: dependent variable array
    dx: uncertainties on independent variable (assumed to be normal)
    dy: uncertainties on dependent variable (assumed to be normal)
    xlim: censoring information for independent variable to compute
        generalized Kendall tau
        (-1, 1, 0) correspond to (lower limit, upper limit, detection)
    ylim: censoring information for dependent variable to compute generalized
        Kendall tau
        (-1, 1, 0) correspond to (lower limit, upper limit, detection)
    Nboot: number of times to bootstrap (does not boostrap if =None)
    Nperturb: number of times to perturb (does not perturb if =None)
    coeff: Correlation coefficient to compute. Must be one of:
        ['spearmanr', 'kendallt', 'pearsonr']
    percentiles: list of percentiles to compute from final distribution
    return_dist: if True, return the full distribution of the correlation
        coefficient and its and p-value

    Edit: upper limits with a boundary are treated as uniform distributions within that boundary
    """

    nonan_mask= (~_np.isnan(x_init)) & (~_np.isnan(y_init))
    x=x_init[nonan_mask]
    y=y_init[nonan_mask]
    dx=dx_init[nonan_mask]
    dy=dy_init[nonan_mask]
    xlim=None if xlim_init is None else xlim_init[nonan_mask]
    ylim=None if ylim_init is None else ylim_init[nonan_mask]

    # do some checks on input array lengths and ensure the necessary data
    # is provided
    if Nperturb is not None and dx is None and dy is None:
        raise ValueError("dx or dy must be provided if perturbation is to be used.")
    if len(x) != len(y):
        raise ValueError("x and y must be the same length.")
    if dx is not None and len(dx) != len(x):
        raise ValueError("dx and x must be the same length.")
    if dy is not None and len(dy) != len(y):
        raise ValueError("dy and y must be the same length.")

    coeffs_impl = ['spearmanr', 'kendallt', 'pearsonr']
    # make sure an implemented correlation coefficient type is requested
    if coeff not in coeffs_impl:
        raise ValueError("coeff must be one of " + ', '.join(coeffs_impl))

    # censoring is only implemented for kendall's tau, return an error
    # if censored data is provided
    
    #### EDIT: this test is taken off
#     if coeff != 'kendallt' and \
#        ((xlim is not None or ylim is not None) or
#         (_np.all(xlim == 0) and _np.all(ylim == 0))):
#         raise ValueError('Censored data provided, but ' + coeff + ' does not \
# support censored data.')

    Nvalues = len(x)

    # if no bootstrapping or correlation is requested, we can just
    # report the normal correlation coefficient values
    if Nboot is None and Nperturb is None:
        import warnings as _warnings
        _warnings.warn("No bootstrapping or perturbation applied. Returning \
normal " + coeff + " output.")
        if coeff == 'spearmanr':
            return compute_corr(x, y, ylim=ylim, coeff=coeff)
        elif coeff == 'kendallt':
            # pass along the xlim/ylim arrays, and the wrapper will handle
            # the presence of censored data
            return compute_corr(x, y, xlim=xlim, ylim=ylim, coeff=coeff)
        elif coeff == 'pearsonr':
            return compute_corr(x, y, ylim=ylim, coeff=coeff)

    # if perturbing points, and we have censored data, set up an index

    if Nperturb is not None and (xlim is not None and ylim is not None):
        do_per = _np.logical_and(xlim == 0,
                                 ylim == 0)
    else:
        do_per = _np.ones(len(x),
                          dtype=bool)

    if Nboot is not None:
        coeffs = _np.zeros(Nboot)
        pvals = _np.zeros(Nboot)
        rng = _np.random.default_rng()
        # generate all the needed bootstrapping indices
        members = rng.integers(0, high=Nvalues,
                               size=(Nboot, Nvalues))
        # loop over sets of bootstrapping indices and compute
        # correlation coefficient
        for i in range(Nboot):
            xp = x[members[i, :]]
            yp = y[members[i, :]]

            # if limit flag arrays are provided, resample these to match
            if xlim is not None:
                xlimb = xlim[members[i, :]]
            else:
                xlimb = xlim()
                
            if ylim is not None:
                ylimb = ylim[members[i, :]]
            else:
                ylimb = ylim

            if Nperturb is not None:
                # perform 1 perturbation on top of the bootstrapping
                xp = x.copy()
                yp = y.copy()
                    
                xp[do_per], yp[do_per] = perturb_values(x[members[i, :]][do_per],
                                                        y[members[i, :]][do_per],
                                                        dx[members[i, :]][do_per],
                                                        dy[members[i, :]][do_per],ylim=ylim,
                                                        Nperturb=1)

            coeffs[i], pvals[i] = compute_corr(xp, yp,
                                               xlim=xlimb, ylim=ylimb,
                                               coeff=coeff)

    elif Nperturb is not None:
        coeffs = _np.zeros(Nperturb)
        pvals = _np.zeros(Nperturb)
        
        if ax_plot is not None:
            
            #computing the extremal x values to plot
            if _np.ndim(dx)==2:
                xlims=_np.array([min(x-dx.T[0]),max(x+dx.T[1])]) 
            else:
                xlims=_np.array([min(x-dx),max(x+dx)]) 
            
            #computing the sampling of the curve
            if ax_plot.get_xaxis().get_scale()=='log':
                x_sampling=_np.logspace(_np.log10(xlims[0]),_np.log10(xlims[1]),num=plot_sampling,endpoint=True)
            else:
                x_sampling=_np.linspace(xlims[0],xlims[1],num=plot_sampling,endpoint=True)
                    
            slope_sample=_np.zeros((Nperturb,plot_sampling))
                                   
        # generate Nperturb perturbed copies of the dataset
        xp = _np.repeat([x],
                        Nperturb,
                        axis=0)
        yp = _np.repeat([y],
                        Nperturb,
                        axis=0)
        
        xp[:, do_per], yp[:, do_per],fake_bcenters = perturb_values(x[do_per],
                                                      y[do_per],
                                                      dx[do_per],
                                                      dy[do_per],ylim=ylim,
                                                      Nperturb=Nperturb)
        # loop over each perturbed copy and compute the correlation
        # coefficient
        for i in range(Nperturb):
            coeffs[i], pvals[i] = compute_corr(xp[i, :], yp[i, :],
                                               xlim=xlim, ylim=ylim,
                                               coeff=coeff)
            #sampling points in the barycenter-sloped curve 
            if coeff=='spearmanr' and ax_plot is not None:
                
                #computing the lower/higher limit curves with both baricenters
                slope_sample[i]=coeffs[i]*x_sampling+(fake_bcenters[i][1]-coeffs[i]*fake_bcenters[i][0])
            
    else:
        import warnings as _warnings
        _warnings.warn("No bootstrapping or perturbation applied. Returning \
regular " + coeff + " values.")
        # return compute_corr(xp, yp,
        #                     xlim=xlim, ylim=ylim,
        #                     coeff=coeff)

    fcoeff = _np.percentile(coeffs, percentiles,interpolation='nearest')
    fpval = _np.percentile(pvals, percentiles,interpolation='nearest')
    
    if coeff=='spearmanr':
    
        main_bcenter=_np.array([x.mean(),y.mean()]).T
        #plotting the linear correlation if asked to
        if ax_plot is not None:
            
            #computing the percentiles envelopes of the slope distribution
            points_slopes_percentiles=_np.percentile(slope_sample,percentiles[1:],axis=0)
                                    
            #computing the main result curve
            points_slopes_main=fcoeff[0]*x_sampling+(main_bcenter[1]-fcoeff[0]*main_bcenter[0])
            
            #plotting the main curve
            ax_plot.plot(x_sampling,points_slopes_main,color=color_line,alpha=1)
            
            #computing the higher and lower enveloppes of the 4 overlapping lines
            ax_plot.fill_between(x_sampling,points_slopes_percentiles[0],points_slopes_percentiles[1],color=color_line,alpha=0.5)
            
    if return_dist:
        return fcoeff, fpval, coeffs, pvals
    
    return fcoeff, fpval


def pymcspearman(x, y,
                 dx=None, dy=None,
                 Nboot=None,
                 Nperturb=None,
                 percentiles=(16, 50, 84),
                 return_dist=False):
    """
    Pass-through function to maintain backward compatibility with older
    code
    """

    return pymccorrelation(x, y,
                           dx=dx, dy=dy,
                           Nboot=Nboot,
                           Nperturb=Nperturb,
                           coeff='spearmanr',
                           percentiles=percentiles,
                           return_dist=return_dist)


def pymckendall(x, y,
                xlim, ylim,
                dx=None, dy=None,
                Nboot=None,
                Nperturb=None,
                percentiles=(16, 50, 84),
                return_dist=False):
    """
    Pass-through function to maintain backward compatibility with older
    code
    """

    return pymccorrelation(x, y,
                           dx=dx, dy=dy,
                           xlim=xlim, ylim=ylim,
                           Nboot=Nboot,
                           Nperturb=Nperturb,
                           coeff='kendallt',
                           percentiles=percentiles,
                           return_dist=return_dist)


def run_tests():
    """
    Test output of pymcspearman against tabulated values from MCSpearman
    """

    from tempfile import NamedTemporaryFile as ntf
    from urllib.request import urlretrieve

    # get test data
    tfile = ntf()
    urlretrieve("https://raw.githubusercontent.com/PACurran/MCSpearman/master/test.data",
                tfile.name)
    # open temporary file
    data = _np.genfromtxt(tfile,
                          usecols=(0, 1, 2, 3),
                          dtype=[('x', float),
                                 ('dx', float),
                                 ('y', float),
                                 ('dy', float)])

    # tabulated results from a MCSpearman run with 10000 iterations
    MCSres = [(0.8308, 0.001),  # spearman only
              (0.8213, 0.0470), # bootstrap only
              (0.7764, 0.0356), # perturbation only
              (0.7654, 0.0584)] # bootstrapping and perturbation

    # spearman only
    res = pymccorrelation(data['x'], data['y'],
                          dx=data['dx'], dy=data['dy'],
                          coeff='spearmanr',
                          Nboot=None,
                          Nperturb=None,
                          return_dist=True)
    try:
        assert _np.isclose(MCSres[0][0], res[0],
                           atol=MCSres[0][1])
        _sys.stdout.write("Passed spearman check.\n")
    except AssertionError:
        _sys.stderr.write("Spearman comparison failed.\n")

    # bootstrap only
    res = pymccorrelation(data['x'], data['y'],
                          dx=data['dx'], dy=data['dy'],
                          Nboot=10000,
                          coeff='spearmanr',
                          Nperturb=None,
                          return_dist=True)
    try:
        assert _np.isclose(MCSres[1][0], _np.mean(res[2]),
                           atol=MCSres[1][1])
        _sys.stdout.write("Passed bootstrap only method check.\n")
    except AssertionError:
        _sys.stderr.write("Bootstrap only method comparison failed.\n")

    # perturbation only
    res = pymccorrelation(data['x'], data['y'],
                          dx=data['dx'], dy=data['dy'],
                          coeff='spearmanr',
                          Nboot=None,
                          Nperturb=10000,
                          return_dist=True)
    try:
        assert _np.isclose(MCSres[2][0], _np.mean(res[2]),
                           atol=MCSres[2][1])
        _sys.stdout.write("Passed perturbation only method check.\n")
    except AssertionError:
        _sys.stderr.write("Perturbation only method comparison failed.\n")

    # composite method
    res = pymccorrelation(data['x'], data['y'],
                          dx=data['dx'], dy=data['dy'],
                          coeff='spearmanr',
                          Nboot=10000,
                          Nperturb=10000,
                          return_dist=True)
    try:
        assert _np.isclose(MCSres[3][0], _np.mean(res[2]),
                           atol=MCSres[3][1])
        _sys.stdout.write("Passed composite method check.\n")
    except AssertionError:
        _sys.stderr.write("Composite method comparison failed.\n")

    # test Kendall tau IFN86 for consistency with scipy
    sres = _kendalltau(data['x'], data['y'])
    IFN86res = kendall_IFN86(data['x'], data['y'],
                             xlim=_np.zeros(len(data)),
                             ylim=_np.zeros(len(data)))
    kt_wrap_res = pymccorrelation(data['x'], data['y'],
                                  xlim=_np.zeros(len(data)),
                                  ylim=_np.zeros(len(data)),
                                  coeff='kendallt')
    try:
        assert _np.isclose(sres[0], IFN86res[0])
        assert _np.isclose(sres[1], IFN86res[1])
        _sys.stdout.write("Passed Kendall tau comparison with scipy.\n")
    except AssertionError:
        _sys.stderr.write("Kendall tau comparison with scipy failed.\n")
    try:
        assert _np.isclose(kt_wrap_res[0], IFN86res[0])
        assert _np.isclose(kt_wrap_res[1], IFN86res[1])
        _sys.stdout.write("Passed internal Kendall tau comparison.\n")
    except AssertionError:
        _sys.stderr.write("Internal Kendall tau comparison failed.\n")

    # test pearson r wrapper
    wrap_res = pymccorrelation(data['x'], data['y'],
                               coeff='pearsonr',
                               return_dist=False)
    res = _pearsonr(data['x'], data['y'])
    try:
        assert _np.isclose(wrap_res[0], res[0])
        assert _np.isclose(wrap_res[1], res[1])
        _sys.stdout.write("Passed Pearson r wrapper check.\n")
    except AssertionError:
        _sys.stderr.write("Pearson r wrapper check failed.\n")

def main():
    """
    run tests
    """

    run_tests()


if __name__ == "__main__":
    import sys as _sys
    _sys.stdout.write("\nModule run as a program. Running test suite.\n\n")
    main()