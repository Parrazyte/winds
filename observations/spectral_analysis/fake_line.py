from xspec import *
from xspec_config_multisp import *
import time

def faker(nfakes):
    pass

def mc_sim_line(baseload='mod_final_nogauss.xcm',lines='FeKa26Aabs_gaussian',
                nfakes=1000,vrange=[-3000,3000,50],parallel=1,bound_around=1):

    '''
    basic script to perform MC simulations of line photon noise

    note: steppar crashes if parallel not set at 1 in current version

    note: currently we add the lines as the very last component (outside of abs and constant factors)
    this is to speed up the computations and because we shouldn't need to care about absorption in such conditions
    (aka this will change the norm of the lines but not the delc)

    baseload: 
        xcm data/model to load
        
    lines: which line to fake. Manually defined. Can be a list of string or a string
    
    nfakes:
        number of iterations
        
    vrange: 
        bounds and step of velocity shift steppar
    
    parallel:
        number of processors used during the steppar computation
        
    bound_around:
        energy range to notice the spectrum in during the steppar computation
        default is 1 kev around the rest energy of the minimum and maximum energy lines added by the line
        ignored if set to 0
    '''

    old_chatter=Xset.chatter

    Xset.chatter=0

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

    with tqdm(total=nfakes) as pbar:
        for f_ind in range(nfakes):
            Xset.restore(baseload)

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

            Fit.perform()
            freeze()
            npars_base=AllModels(1).nParameters
            if baseload.endswith('mod_final_nogauss.xcm'):
                #to avoid problems with high photon noise in the xtend spectrum when actually the delta-c is in resolve
                #(but the continuum is fit using both)
                #doing it like this because the debugger isn't happy with a simple AllData-=2 and calls elsewhere
                AllData.__isub__(2)
                pass

            mod_base=allmodel_data()
            mod_base_comp=AllModels(1).componentNames
            for i_line,indiv_line in enumerate(lines_use):

                print('testing line '+indiv_line)

                mod_base.load()
                AllData.notice('all')
                if baseload.endswith('mod_final_nogauss.xcm'):
                    AllData(1).ignore('**-2. 10.-**')

                if baseload.endswith('mod_save_02_noline.xcm'):
                    AllData(1).ignore('**-2. 10.-**')

                if indiv_line=='FeKa26Aabs_gaussian':
                    addcomp(indiv_line)
                    AllModels(1)(npars_base+3).values=2e-3
                    AllModels(1)(npars_base+3).frozen=True
                    AllModels(1)(npars_base+4).values = 0.0
                    AllModels(1)(npars_base+7).link = str(npars_base+4)+'*2.0'

                if indiv_line in ['NiKa27Wabs_gaussian','FeKb25abs_gaussian']:
                    addcomp(indiv_line)
                    AllModels(1)(npars_base+3).values=5e-3
                    AllModels(1)(npars_base+3).frozen=True
                    AllModels(1)(npars_base+4).values = 0.0


                if indiv_line in ['CrKa23Wabs_gaussian','CaKa20abs_gaussian','SKa16abs_gaussian']:
                    addcomp(indiv_line)
                    AllModels(1)(npars_base+3).values=2e-3
                    AllModels(1)(npars_base+3).frozen=True
                    AllModels(1)(npars_base+4).values = 0.0


                if indiv_line=='FeKa1Aem_gaussian':
                    addcomp(indiv_line)
                    AllModels(1)(npars_base+3).values=5e-3
                    AllModels(1)(npars_base+3).frozen=True
                    AllModels(1)(npars_base+4).values = 0.0
                    AllModels(1)(npars_base+7).link = str(npars_base+4)+'*2.0'

                AllModels.show()

                vashift_par=npars_base+1

                if bound_around!=0:

                    #ignoring around the energies of the lowest and highest E lines added

                    #assuming we don't switch comp names here, aka only adding stuff at the end
                    new_comps=[elem for elem in AllModels(1).componentNames if elem not in mod_base_comp]

                    line_e_list=[getattr(getattr(AllModels(1),indiv_comp),'LineE').values[0] \
                                 for indiv_comp in new_comps if 'LineE' in getattr(getattr(AllModels(1),indiv_comp),'parameterNames')]

                    AllData.ignore('**-'+str(min(line_e_list)-bound_around/2)+' '+str(max(line_e_list)+bound_around/2)+'-**')

                Xset.parallel.steppar = parallel

                Fit.steppar('nolog '+str(vashift_par)+' '+str(vrange[0])+' '+str(vrange[1])+' '+str(int((vrange[1]-vrange[0])/vrange[2])))

                Xset.parallel.steppar = 1

                arr_delc_save[i_line][f_ind]=abs(np.array([min(elem, 0) for elem in Fit.stepparResults('delstat')]))

            pbar.update(1)

    Xset.closeLog()
    Xset.chatter=old_chatter
    Xset.parallel.steppar=old_parallel

    arr_float_list=[np.array([elem for elem in arr_delc_save[i_line]]) for i_line in range(len(lines_use))]

    for i_line,indiv_line in enumerate(lines_use):

        np.save('fake_delC_save_'+baseload_str+'_'+indiv_line+'_'+str(nfakes)+'_'+time_log,
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

    dumps_id=['_'.join(elem.split('/')[-1].split('_')[:-2]) for elem in dumps]
    assert len(np.unique(dumps_id))==1,'Error: dump names indicate different origins'

    dumps_number=np.array(['_'.join(elem.split('/')[-1].split('_')[-2]) for elem in dumps],dtype=int)

    list_arr=np.repeat(None,len(dumps))
    for i_elem,elem in enumerate(dumps):
        list_arr[i_elem]=np.load(elem)

    dump_concat=np.concatenate(list_arr)

    if output_name=='auto':
        dump_concat_name=dumps_id[0]+'_'+str(sum(dumps_number))
    np.save(dump_concat_name,dump_concat)
