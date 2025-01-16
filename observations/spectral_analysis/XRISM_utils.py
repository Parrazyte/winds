
from xspec import Xset, Plot, Fit
from xspec_config_multisp import freeze, allmodel_data
from linedet_utils import narrow_line_search,plot_line_search
import dill

def rebinv_xrism(grp_number):
    Plot.setRebin(2, 5000, grp_number)
def xrism_ls(baseload,low_e,high_e,plot_suffix,e_step=5e-3,e_sat_low_indiv=[1.5,1.5],resolve_dg=1,
             force_ylog_ratio=True,ratio_bounds=[0.05,20]):

    '''
    Simple wrapper to compute a line search and make an associated plot with XRISM

    baseload should be an XCM file with both the file and model

    e_sat_low_indiv here assumes that a RESOLVE and EXTEND spectra are loaded

    '''
    Plot.xLog = False
    Fit.statMethod = 'cstat'

    prev_chatter=Xset.chatter
    Xset.chatter=1

    Xset.restore(baseload)
    rebinv_xrism(resolve_dg)
    freeze()

    mod_ls=allmodel_data()
    narrow_out_val=narrow_line_search(mod_ls,'mod_ls',e_sat_low_indiv,[low_e,high_e,e_step],
                                      line_cont_range=[low_e,high_e])


    with open(baseload[:baseload.rfind('.')]+'_narrow_out_'+str(low_e)+'_'+str(high_e)+'.dill','wb+') as f:
        dill.dump(narrow_out_val,f)

    plot_line_search(narrow_out_val, './', 'XRISM', suffix='', save=True,
                     epoch_observ=[plot_suffix], format='pdf',
                     force_ylog_ratio=force_ylog_ratio,ratio_bounds=ratio_bounds)

    Xset.chatter=prev_chatter


#xrism_ls('pion_fit_paper.xcm',1.5,2.5,'pion_fit',)