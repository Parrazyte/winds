from xspec_config_multisp import *
from linedet_utils import narrow_line_search,plot_line_search
import os
import dill

os.chdir('/media/parrama/crucial_SSD/Observ/BHLMXB/XRISM/V4641Sgr/v4641sgr')

logfile_write,logfile=xLog_rw('xrism_log2.log')

#base load
os.chdir('/media/parrama/crucial_SSD/Observ/BHLMXB/XRISM/V4641Sgr/v4641sgr')

Xset.restore('fit.xcm')
Fit.statMethod='cstat'
AllData.ignore('**-1.5 10.-**')
Plot.xLog=False
Plot.setRebin(2,5000,1)
xPlot('ldata,ratio,delchi')


def print_xlog(string, logfile_write=logfile_write):
    '''
    prints and logs info in the xspec log file, and flushed to ensure the logs are printed before the next xspec print
    Different log file from the main xspec one to (hopefully) avoid issues
    '''
    print(string)
    logfile_write.write(time.asctime() + '\n')
    logfile_write.write(string)
    # adding a line for lisibility
    logfile_write.write('\n')
    logfile_write.flush()


Xset.restore('fit.xcm')

Fit.statMethod='cstat'


AllData.ignore('**-1.5 10.-**')

AllData.ignore('2.55-2.65')
AllData.ignore('3.3-3.4')

AllData.ignore('4.05-4.15')

AllData.ignore('5.65-5.75')

AllData.ignore('6.35-6.45')

AllData.ignore('6.6-6.8')

AllData.ignore('6.9-7.15')

Plot.xLog=False
Plot.setRebin(2,5000,1)
AllModels(1)(1).values=0.25
AllModels(1)(1).frozen=True
Fit.perform()

mod_dat=allmodel_data()

AllData.notice('1.5-10.')

e_sat_low_indiv=[1.5,1.5]

from fitting_tools import model_list

mod_list=model_list('lines_XRISM_V4641_2024')

fitlines_strong=fitmod(mod_list,logfile,logfile_write)

fitlines_strong.add_allcomps(split_fit=False)

breakpoint()


# #after the Chain is loaded
# nfakes=1000
# # drawing parameters for the MC significance test later
# autofit_drawpars = np.array([None] * nfakes)
#
# print_xlog('\nDrawing parameters from the Chain...')
# for i_draw in range(nfakes):
#     curr_simpar = AllModels.simpars()
#
#     # we restrict the simpar to the initial model because we don't really care about simulating the variations
#     # of the bg since it's currently frozen
#     autofit_drawpars[i_draw] = np.array(curr_simpar)[:AllData.nGroups * AllModels(1).nParameters] \
#         .reshape(AllData.nGroups, AllModels(1).nParameters)
#
# # turning it back into a regular array
# autofit_drawpars = np.array([elem for elem in autofit_drawpars])



# Xset.chatter=1
#
# narrow_out_min=narrow_line_search(mod_dat,'mod_dat',e_sat_low_indiv,[1.5,3.5,5e-3],line_cont_range=[1.5,3.5])
# with open('narrow_out_min.dill','wb+') as f:
#     dill.dump(narrow_out_min,f)
#
# narrow_out_min['line_cont_range'] = [1.5, 3.5]
# plot_line_search(narrow_out_min, './', 'XRISM', suffix='', save=True, epoch_observ=['first_fit'], format='pdf')
#
#
# narrow_out_low=narrow_line_search(mod_dat,'mod_dat',e_sat_low_indiv,[3.5,5.5,5e-3],line_cont_range=[3.5,5.5])
#
# narrow_out_mid=narrow_line_search(mod_dat,'mod_dat',e_sat_low_indiv,[5.5,7.5,5e-3],line_cont_range=[5.5,7.5])
#
# narrow_out_high=narrow_line_search(mod_dat,'mod_dat',e_sat_low_indiv,[7.5,9.5,5e-3],line_cont_range=[7.5,9.5])


'''
ftgrouppha infile=xa901001010xtd_src_gti01.pi outfile=xa901001010xtd_src_gti01_grp_opt.pi respfile=xa901001010xtd_src_gti01.rmf grouptype=opt
ftgrouppha infile=xa901001010xtd_src_gti02.pi outfile=xa901001010xtd_src_gti02_grp_opt.pi respfile=xa901001010xtd_src_gti02.rmf grouptype=opt
ftgrouppha infile=xa901001010xtd_src_gti03.pi outfile=xa901001010xtd_src_gti03_grp_opt.pi respfile=xa901001010xtd_src_gti03.rmf grouptype=opt
ftgrouppha infile=xa901001010xtd_src_gti04.pi outfile=xa901001010xtd_src_gti04_grp_opt.pi respfile=xa901001010xtd_src_gti04.rmf grouptype=opt
ftgrouppha infile=xa901001010xtd_src_gti05.pi outfile=xa901001010xtd_src_gti05_grp_opt.pi respfile=xa901001010xtd_src_gti05.rmf grouptype=opt



'''

'''

Xset.restore('baseload_all.xcm')

rebinv_xrism(1)
rebinv_xrism(3)
rebinv_xrism(5)
rebinv_xrism(7)
rebinv_xrism(9)
AllData.ignore('**-0.5 10.-**')
Plot.xLog=False

xPlot('ldata',
data_colors=['black','black','red','red','limegreen','limegreen','blue','blue','cyan','cyan'],
group_names=['orbit 1','','orbit 2','','orbit 3','','orbit 4','','orbit 5',''])

leg=plt.gca().get_legend()
leg.remove()
plt.legend()
plt.show()
plt.subplots_adjust(bottom=0.07)
fig=plt.gca().get_figure()
fig.set_size_inches(10,5)
plt.subplots_adjust(bottom=0.1)

'''