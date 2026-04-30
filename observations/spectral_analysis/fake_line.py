from xspec import *
from xspec_config_multisp import *
import time


def mc_sim_line(baseload='mod_final_nogauss.xcm',lines='FeKa26Aabs_agaussian',
                e_min=6.5,e_max=7.5,sigma=[5e-3],base_norm_0=True,chain='',chain_columns='full',
                unfreeze_line_cont=False,anal_method='steppar',
                position='last',mod_name='',set_ener_str='',set_ener_xrism=True,
                nfakes=1000,vrange=[-3000,3000,25],parallel=1,bound_around=1):

    '''
    basic script to perform MC simulations of line photon noise

    note: steppar crashes if parallel not set at 1 in current version

    note: currently we add the lines as the very last component (outside of abs and constant factors)
    this is to speed up the computations and because we shouldn't need to care about absorption in such conditions
    (aka this will change the norm of the lines but not the delc)

    baseload: 
        xcm data/model to load
        
    lines: which line to fake. Manually defined. Can be a list of string or a string

    e_min/e_max (keV): if both set to a value and not None, restricts to this energy range before performing the computations
                on fake spectra
    sigma (keV): if not None, iterable of len the number of lines in the component
                freezes the width of the different lines to their values

    base_norm_0:
            Start with all line(s) at norm 0 to avoid issues with the fits
    nfakes:
        number of iterations
        
    vrange: 
        bounds and step of velocity shift steppar


    chain:
        fits file used to load MC posterior parameters before each fake

    chain_columns:
        ids of the columns used in the chain (different from the parameters themselves) or 'full' to use all of them
        The code will automatically apply each column to the parameter given in the name of the column in the fits file
        e.g. ('Velocity__4', '>f8'), ('Sigma__6', '>f8'), ('norm__7', '>f8')
        chain_columns=[0,2] will load column 0 for parameter 4, and column 2 for parameter 7


    anal_method: 'fit' or 'steppar'
        if fit, performs full fitting of the model instead of steppar (much longer)

    parallel:
        number of processors used during the steppar computation
        
    bound_around:
        energy range to notice the spectrum in during the steppar computation
        default is 1 kev around the rest energy of the minimum and maximum energy lines added by the line
        ignored if set to 0
    '''

    old_chatter=Xset.chatter

    Xset.chatter=0
    Xset.logChatter=10

    AllData.clear()
    AllModels.clear()

    old_parallel=Xset.parallel.steppar

    if type(lines)==str:
        n_lines=1
        lines_use=[lines]
    else:
        n_lines=len(lines)
        lines_use=lines

    arr_delc_save=np.array([[None]*nfakes]*n_lines)

    time_log=str(int(time.time()*1000))

    baseload_str=baseload.split('/')[-1][:baseload.split('/')[-1].rfind('.')]

    log=Xset.openLog('mc_sim_line_'+baseload_str+'_'+time_log+'.log')

    # preparing the chain if need be
    if chain != '':
        chain_data=fits.open(chain)[1].data
        assert len(chain_data)>=nfakes,'Error: chain smaller than the amount of simulations required'

        if type(chain_columns)==str and chain_columns=='full':
            chain_columns_use=np.arange(len(chain_data[0])-1)
        else:
            chain_columns_use=chain_columns

    with (tqdm(total=nfakes) as pbar):
        for f_ind in range(nfakes):

            Xset.restore(baseload)

            if chain!='':
                #loadind the chain in reverse to get the best convergence among the chain elements
                chain_data_fake=chain_data[-f_ind]
                for column_id in chain_columns_use:
                    column_par=int(chain_data.columns[column_id].name.split('__')[-1])
                    AllModels(1)(column_par).values=chain_data_fake[column_id]

            time.sleep(0.5)

            if baseload.endswith('mod_save_02_noline.xcm'):
                fakeset=[FakeitSettings(response=AllData(1).response.rmf,arf=AllData(1).response.arf,
                                        exposure=AllData(1).exposure,background='',fileName=AllData(1).fileName),]
                AllData(1).background=''
                # Fit.perform()
                AllData.fakeit(settings=fakeset,applyStats=True,noWrite=True)
                AllData(1).ignore('**-2. 10.-**')

            if baseload.endswith('mod_final_nogauss.xcm'):
                fakeset=[FakeitSettings(response=AllData(1).response.rmf,arf=AllData(1).response.arf,
                                        exposure=AllData(1).exposure,background='',fileName=AllData(1).fileName),
                FakeitSettings(response=AllData(2).response.rmf,arf=AllData(2).response.arf,
                                        exposure=AllData(2).exposure,background=AllData(2).background.fileName,fileName=AllData(2).fileName)]
                AllData(1).background=''
                Fit.perform()
                AllData.fakeit(settings=fakeset,applyStats=True,noWrite=True)
                AllData(1).ignore('**-2. 10.-**')
                AllData(2).ignore('**-0.3 10.-**')

            if baseload in ['mod_baseload_1group.xcm','baseload_nolines.xcm','mod_simple_oneline_forsim.xcm']:
                fakeset=[FakeitSettings(response=AllData(1).response.rmf,
                                        exposure=AllData(1).exposure,background='',fileName=AllData(1).fileName),]
                # Fit.perform()
                AllData.fakeit(settings=fakeset,applyStats=True,noWrite=True)
                AllData(1).ignore('**-2. 10.-**')

            if set_ener_str is not None:
                set_ener(set_ener_str,xrism=set_ener_xrism)



            #for Kai in 1745: we refit the continuum first, then ignore an energy range and refit the line
            if baseload=='mod_simple_oneline_forsim.xcm':

                freeze(parlist=[4,6,7])
                Fit.perform()
                freeze()
                unfreeze(parlist=[4,6,7])
                AllData.ignore('**-6.8 7.1-**')

            Fit.perform()

            freeze()

            #version where we unfreeze the line before the steppar
            if unfreeze_line_cont:
                if baseload=='mod_simple_oneline_forsim.xcm':
                    #we keep the velocity frozen to avoid issues with finding minima
                    # (assuming that the main line is much bigger than the fluctuations, this should be fine at first order)
                    unfreeze(parlist=[4, 6, 7])

            npars_base=AllModels(1).nParameters
            if baseload.endswith('mod_final_nogauss.xcm'):
                #to avoid problems with high photon noise in the xtend spectrum when actually the delta-c is in resolve
                #(but the continuum is fit using both)
                #doing it like this because the debugger isn't happy with a simple AllData-=2 and calls elsewhere
                AllData.__isub__(2)
                pass

            mod_base=allmodel_data()
            mod_base_comp=AllModels(1).componentNames

            #re-noticing if need be
            if baseload=='mod_simple_oneline_forsim.xcm':
                AllData.notice('2.-10.')

            for i_line,indiv_line in enumerate(lines_use):

                print('testing line '+indiv_line)

                mod_base.load()
                AllData.notice('all')
                if e_min is not None and e_max is not None:
                    ignore_data_indiv([e_min],[e_max])

                if baseload.endswith('mod_final_nogauss.xcm'):
                    AllData(1).ignore('**-2. 10.-**')

                if baseload.endswith('mod_save_02_noline.xcm'):
                    AllData(1).ignore('**-2. 10.-**')

                if indiv_line in ['FeKa26Aabs_gaussian','FeKa26Aabs_agaussian']:
                    par_info=addcomp(indiv_line,return_pos=True)
                    AllModels(1)(npars_base+3).values=2e-3
                    AllModels(1)(npars_base+3).frozen=True
                    AllModels(1)(npars_base+4).values = 0.0
                    AllModels(1)(npars_base+7).link = str(npars_base+4)+'*2.0'
                elif indiv_line in ['NiKa27Wabs_gaussian','FeKb25abs_gaussian']:
                    par_info=addcomp(indiv_line,return_pos=True)
                    AllModels(1)(npars_base+3).values=5e-3
                    AllModels(1)(npars_base+3).frozen=True
                    AllModels(1)(npars_base+4).values = 0.0
                elif indiv_line in ['CrKa23Wabs_gaussian','CaKa20abs_gaussian','SKa16abs_gaussian']:
                    par_info=addcomp(indiv_line,return_pos=True)
                    AllModels(1)(npars_base+3).values=2e-3
                    AllModels(1)(npars_base+3).frozen=True
                    AllModels(1)(npars_base+4).values = 0.0
                elif indiv_line=='FeKa1Aem_gaussian':
                    par_info=addcomp(indiv_line,return_pos=True)
                    AllModels(1)(npars_base+3).values=5e-3
                    AllModels(1)(npars_base+3).frozen=True
                    AllModels(1)(npars_base+4).values = 0.0
                    AllModels(1)(npars_base+7).link = str(npars_base+4)+'*2.0'

                else:
                    par_info=addcomp(indiv_line,position=position,mod_name=mod_name,return_pos=True)

                if sigma is not None:
                    for comp in [getattr(AllModels(1,mod_name),AllModels(1,mod_name).componentNames[id_comp-1]) \
                                 for id_comp in par_info[1]]:
                        if comp.name.split('_')[0] not in ['vashift','gaussian','lorentz']:
                            breakpoint()
                        if comp.name.split('_')[0]!='vashift':
                            comp.Sigma.values=sigma[0]
                            comp.Sigma.frozen=True

                if base_norm_0:
                    for comp in [getattr(AllModels(1,mod_name),AllModels(1,mod_name).componentNames[id_comp-1]) \
                                 for id_comp in par_info[1]]:
                        if comp.name.split('_')[0] not in ['vashift','gaussian','lorentz']:
                            breakpoint()

                        #also checking the link to ensure we don't remove a link
                        if comp.name.split('_')[0]!='vashift' and comp.norm.link=='':
                            comp.norm.values=0

                AllModels.show()

                vashift_par=par_info[0][0]

                if bound_around!=0:

                    #ignoring around the energies of the lowest and highest E lines added

                    #assuming we don't switch comp names here, aka only adding stuff at the end
                    new_comps=[elem for elem in AllModels(1).componentNames if elem not in mod_base_comp]

                    line_e_list=[getattr(getattr(AllModels(1),indiv_comp),'LineE').values[0] \
                                 for indiv_comp in new_comps if 'LineE' in getattr(getattr(AllModels(1),indiv_comp),'parameterNames')]

                    AllData.ignore('**-'+str(min(line_e_list)-bound_around/2)+' '+str(max(line_e_list)+bound_around/2)+'-**')

                Xset.parallel.steppar = parallel

                #ensuring the vashift par allows the bounds of the energy range
                AllModels(1,mod_name)(vashift_par).values=[0,1,vrange[0]-1,vrange[0]-1,
                                                             vrange[1]+1,vrange[1]+1]

                if anal_method=='steppar':
                    Fit.steppar('nolog '+str(vashift_par)+' '+str(vrange[0])+' '+str(vrange[1])+' '+str(int((vrange[1]-vrange[0])/vrange[2])))

                    Xset.parallel.steppar = 1

                    arr_delc_save[i_line][f_ind]=abs(np.array([min(elem, 0) for elem in Fit.stepparResults('delstat')]))
                elif anal_method=='fit':
                    #better for cases where we refit components
                    modpre_fitline=allmodel_data()
                    modpre_fitline.load()
                    fit_improve = 0
                    base_stat = Fit.statistic

                    for elem in np.arange(vrange[0], vrange[1]+1, vrange[2]):
                        modpre_fitline.load()
                        AllModels(1)(13).values = elem
                        AllModels(1)(13).frozen = True
                        Fit.perform()
                        fit_improve = max(fit_improve, base_stat - Fit.statistic)

                    print('max stat improvement:'+str(fit_improve))
                    arr_delc_save[i_line][f_ind]=fit_improve

            pbar.update(1)

    Xset.closeLog()
    Xset.chatter=old_chatter
    Xset.parallel.steppar=old_parallel

    arr_float_list=[np.array([elem for elem in arr_delc_save[i_line]]) for i_line in range(len(lines_use))]

    for i_line,indiv_line in enumerate(lines_use):

        np.save('fake_delC_save_'+baseload_str+'_'+indiv_line+'_'+str(nfakes)+'_'
                +'v_'+('_'.join(np.array(vrange,dtype=str)))+time_log,
                arr_float_list[i_line])

        line_max_per_iter=np.copy(arr_float_list[i_line].max(1))
        line_max_per_iter.sort()
        print('line '+indiv_line+' 1 sigma delc %.1f' % line_max_per_iter[int(nfakes * 0.68)-1])
        print('line '+indiv_line+' 2 sigma delc %.1f' % line_max_per_iter[int(nfakes * 0.95)-1])
        print('line '+indiv_line+' 3 sigma delc %.1f' % line_max_per_iter[int(nfakes * 0.997)-1])



#think about the fact that this may be overestimating due to the two peaks, it could also be tested with 1 peak

#also try from the base model with at least the FeKa26 line ?
#but then would need to fit the first line and add the second one
def disp_sigmas_dump(dump):

    '''
    displays the 1, 2 and 3 sigma delta c values of the dump
    '''

    dump_arr=np.load(dump)
    line_max_per_iter = np.copy(dump_arr.max(1))
    line_max_per_iter.sort()
    print('line ' + '_'.join(dump.split('_')[-4:-2]) +
          ' 1 sigma delc %.1f' % line_max_per_iter[int(float(dump.split('_')[-1].split('.')[0]) * 0.68) - 1])
    print('line ' + '_'.join(dump.split('_')[-4:-2]) +
          ' 2 sigma delc %.1f' % line_max_per_iter[int(float(dump.split('_')[-1].split('.')[0]) * 0.95) - 1])
    print('line ' + '_'.join(dump.split('_')[-4:-2]) +
          ' 3 sigma delc %.1f' % line_max_per_iter[int(float(dump.split('_')[-1].split('.')[0]) * 0.997) - 1])

def disp_pval(dump,deltac):

    '''
    displays the significance percentage of a deltac value when compared to the values of a dump
    '''

    dump_arr=np.load(dump)
    line_max_per_iter = np.copy(dump_arr.max(1))

    p_val=sum(deltac>line_max_per_iter)/1000
    return p_val

def mc_arr_fuse(dumps,output_name='auto'):

    '''
    function to split the dumps of fake mc array computations
    checks and returns an error if the dumps do not have the same names
    '''

    split_dumps=[elem.split('/')[-1].split('_')[:-1][:-4]+elem.split('/')[-1].split('_')[:-1][-2:] for elem in dumps]
    dumps_id=['_'.join(elem_split) for elem_split in split_dumps]
    assert len(np.unique(dumps_id))==1,'Error: dump names indicate different origins'

    dumps_number=np.array(['_'.join(elem.split('/')[-1].split('_')[-5]) for elem in dumps],dtype=int)

    list_arr=np.repeat(None,len(dumps))
    for i_elem,elem in enumerate(dumps):
        list_arr[i_elem]=np.load(elem)

    dump_concat=np.concatenate(list_arr)


    if output_name=='auto':
        dump_concat_name='_'.join(elem.split('/')[-1].split('_')[:-1][:-4]+\
                                  [str(sum(dumps_number))]+\
                                  elem.split('/')[-1].split('_')[:-1][-2:]+\
                                  [str(int(time.time()*1000))])


    np.save(dump_concat_name,dump_concat)
