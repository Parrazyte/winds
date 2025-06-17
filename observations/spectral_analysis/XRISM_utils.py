
from xspec import Xset, Plot, Fit,AllData, AllModels
from xspec_config_multisp import freeze, allmodel_data,set_ener,store_plot
from linedet_utils import narrow_line_search,plot_line_search,rebinv_xrism
from general_tools import ang2kev
import matplotlib.pyplot as plt
import dill
import numpy as np
import glob
import os

c_0 = 299792.458


def xrism_ls(baseload,low_e,high_e,plot_suffix="",bound_around=0.1,e_step=5e-3,lw=5e-3,
             e_sat_low_indiv=[1.5,1.5],resolve_dg=1,rebinv=[20],line_search_norm=[0.01,50,700],
             force_ylog_ratio=True,ratio_bounds=[0.5,2],title=True,set_ener_str=None,set_ener_xrism=False):

    '''

    Simple wrapper to compute a line search and make an associated plot with XRISM

    baseload should be an XCM file with both the file and model

    e_sat_low_indiv here assumes that a RESOLVE and EXTEND spectra are loaded

    rebinv should be an iterable of len the amount of DGs that should be rebinned
    '''
    Plot.xLog = False
    Fit.statMethod = 'cstat'

    prev_chatter=Xset.chatter
    Xset.chatter=1

    AllData.clear()
    AllModels.clear()

    Xset.restore(baseload)
    if rebinv is not None:
        if type(rebinv) not in [list,np.ndarray]:
            rebinv_use=[rebinv]
        else:
            rebinv_use=rebinv
        for i_dg,elem_rebinv in enumerate(rebinv_use):
            rebinv_xrism(i_dg+1,sigma=elem_rebinv)

    if set_ener_str is not None:
        set_ener(set_ener_str,xrism=set_ener_xrism)
    freeze()


    if bound_around is not None:
        AllData.ignore('**-'+str(low_e-bound_around)+' '+str(high_e+bound_around)+'-**')


    mod_ls=allmodel_data()
    narrow_out_val=narrow_line_search(mod_ls,'mod_ls',
                                      e_sat_low_indiv,[low_e,high_e,e_step],line_search_norm=line_search_norm,
                                      line_cont_range=[low_e,high_e],lw=lw,scorpeon=False)

    lw_str='_lw_'+str(round(lw*1e3))

    with open(baseload[:baseload.rfind('.')]+'_narrow_out_'+str(low_e)+'_'+str(high_e)+lw_str+'.dill',
              'wb+') as f:
        dill.dump(narrow_out_val,f)

    plt.ioff()
    plot_line_search(narrow_out_val, './', 'XRISM', suffix=baseload[:baseload.rfind('.')]+'_'+lw_str, save=True,
                     epoch_observ=[plot_suffix], format='pdf',
                     force_ylog_ratio=force_ylog_ratio,ratio_bounds=ratio_bounds,title=title)
    plt.ion()

    Xset.chatter=prev_chatter
    AllModels.clear()
    AllData.clear()


def xrism_ls_loader(dump_path,ener_show_range,force_ylog_ratio=True,ratio_bounds=[0.05,10.],squished_mode=False,
                    local_chi_bounds=True,force_side_lines='none',save=False,show_indiv_transi=True,
                    minor_locator=20,show_peak_pos=True,show_peak_width=False,

                    plot_ratio_baseload='',set_ener_str=None,set_ener_xrism=False,rebinv=[]):
    '''
    More simple version to simply load the dump and make a figure out of it


    '''

    with open(dump_path,'rb') as f:
        narrow_out_val=dill.load(f)

    if plot_ratio_baseload!='':
        Xset.restore(plot_ratio_baseload)

        if rebinv is not None:
            if type(rebinv) not in [list, np.ndarray]:
                rebinv_use = [rebinv]
            else:
                rebinv_use = rebinv
            for i_dg, elem_rebinv in enumerate(rebinv_use):
                rebinv_xrism(i_dg + 1, sigma=elem_rebinv)

        if set_ener_str is not None:
            set_ener(set_ener_str, xrism=set_ener_xrism)

        narrow_out_val['plot_ratio_values']=store_plot('ratio')
        AllData.clear()
        AllModels.clear()

    # breakpoint()
    suffix_list=np.array(ratio_bounds).astype(str).tolist()

    suffix='zoom_'+'_'.join(np.array(ener_show_range).round(2).astype(str).tolist())+'_'+\
                  '_'.join(np.array(ratio_bounds).round(2).astype(str).tolist())+('_squished' if squished_mode else '')+\
                  ('_global_chi' if not local_chi_bounds else '')+\
                  '_'+dump_path[:dump_path.rfind('narrow')]

    plot_line_search(narrow_out_val, './', 'XRISM', suffix=dump_path[:dump_path.rfind('.')], save=save,
                     epoch_observ=[suffix], format='pdf',
                     force_ylog_ratio=force_ylog_ratio,ratio_bounds=ratio_bounds,ener_show_range=ener_show_range,
                     show_indiv_transi=show_indiv_transi,title=False,squished_mode=squished_mode,
                     local_chi_bounds=local_chi_bounds,
                     force_side_lines=force_side_lines,minor_locator=minor_locator,
                     show_peak_pos=show_peak_pos,show_peak_width=show_peak_width)

    plot_line_search(narrow_out_val, './', 'XRISM', suffix=dump_path[:dump_path.rfind('.')], save=save,
                     epoch_observ=[suffix], format='png',
                     force_ylog_ratio=force_ylog_ratio,ratio_bounds=ratio_bounds,ener_show_range=ener_show_range,
                     show_indiv_transi=show_indiv_transi,title=False,squished_mode=squished_mode,
                     local_chi_bounds=local_chi_bounds,
                     force_side_lines=force_side_lines,minor_locator=minor_locator,
                     show_peak_pos=show_peak_pos,show_peak_width=show_peak_width)

def rename_plots(arg='squished_global_chi_save'):
    #for renaming
    plots=[elem for elem in glob.glob('**') if arg in elem and elem.endswith('.png')]
    for elem in plots:
        elem_new = elem[elem.rfind('save'):].split('___')[0] + '_' + '_'.join(elem.split('_')[:3]) + '.png'

        os.system('mv ' + elem + ' ' + elem_new)

#redo 04 with a single component to get the values
#redo 03 to get the values 
def width_todv(ener=7.0,par=None,val=None,err=[0,0],output_err=True):

    if par is not None:
        val_use=AllModels(1)(par).values[0]
        err_use=AllModels(1)(par).error[:2]
        err_use=[val_use-err_use[0],err_use[1]-val_use]
    else:
        val_use=val
        err_use=err

    dv_ctr=val_use/ener*c_0
    dv_sides=np.array([val_use-err_use[0],val_use+err_use[1]])/ener*c_0

    if output_err:
        return dv_ctr,dv_ctr-dv_sides[0],dv_sides[1]-dv_ctr

    else:
        return dv_ctr,dv_ctr-dv_sides[0],dv_sides[1]-dv_ctr

def dv_towidth(ener=7.0,val=None,err=[0,0]):

    dv_sides=np.array([val-err[0],val+err[1]])

    width_ctr=val*ener/c_0

    width_sides=dv_sides*ener/c_0

    return width_ctr,width_ctr-width_sides[0],width_sides[1]-width_ctr


def ew_kevtoang(ener_rest=4861.3,loaded=True,val=None,err=[0,0]):
    '''
    converts an EW from kev to angstrom. ener_rest is assumed to be in angstrom
    if loaded is true, uses the equivalent width values loaded in AllData(1)
    '''

    if loaded is not None:
        val_use=AllData(1).eqwidth[0]
        err_use=[AllData(1).eqwidth[0]-AllData(1).eqwidth[1],AllData(1).eqwidth[2]-AllData(1).eqwidth[0]]

    else:
        val_use=val
        err_use=err

    ew_dv=width_todv(ang2kev(ener_rest),val=val_use,err=err_use)

    ew_ang=dv_towidth(ener_rest,val=ew_dv[0],err=ew_dv[1:])

    return ew_ang
