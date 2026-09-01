import glob
import numpy as np
import pandas as pd

def monaco_loader(folder='./',save=False):

    '''
    loads and merge an folder of MONACO angular solutions into a single dataframe
    adds an angle column with the angle as in np.linspace(0,90,n_sol)
    '''

    sol_path_arr=np.array(glob.glob('00**.tsv'))

    sol_path_arr.sort()


    sol_arr=np.array([None]*len(sol_path_arr))

    df_list=[]
    for elem in sol_path_arr:
        elem_df=pd.read_csv(elem,sep='\t')
        elem_df.insert(0,'angle',
                       #getting the right angle as (X-1)*90/(N-1) to go from 0 (X=1) to 90 (X=N)
                       round(((float(elem.split('.')[0])-1)*90/(len(sol_arr)-1)),1))
        df_list+=[elem_df]

    pd_full=pd.concat(df_list,axis=0)

    if save:
        pd_full.to_csv(folder+'monaco_mrg.csv',index=False)

    return pd_full

def monaco_Nion(pd_full=None,pd_save_path=None,abund='lpgp',save=False):

    '''
    wrapper to compute the ionic column densities of all elements accross angles from a monaco angular distribution
    '''

    elements = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V": 23,
        "Cr": 24,
        "Mn": 25,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
    }

    from xspec import Xset

    #loading the abundance values from Xspec's tables
    curr_abund=Xset.abund.split(': ')[0]
    Xset.abund=abund
    abund_vals=np.array(Xset.abund.split(' ')[1:-1],dtype=float)
    Xset.abund=curr_abund


    assert not (pd_full is None and pd_save_path is None),'Error: need at least one form of input'
    assert not (pd_full is not None and pd_save_path is not None),'Error: need only one form of input'

    if pd_full is not None:
        pd_use=pd_full
    else:
        pd_use=pd.read_csv(pd_save_path)

    pd_angle_Nion=[]
    for elem_angle in np.unique(pd_use['angle']):
        pd_angle=pd_use[pd_use['angle']==elem_angle]

        #getting the ion fractions only
        pd_angle_fractions=pd_angle[pd_angle.columns[['_fraction' in elem for elem in pd_angle.columns]]]

        #adding ionized hydrogen
        pd_angle_fractions.insert(0,'H00_fraction',1-pd_angle_fractions['H01_fraction'])


        pd_angle_Nion+=[
            pd.DataFrame([pd_angle_fractions.iloc[i]\
                                   * pd_angle['number_density'].iloc[i]*
                          (pd_angle['radius_outer']-pd_angle['radius_inner']).iloc[i] for i in range(len(pd_angle))]).sum(0)]

    pd_angle_mrg=pd.DataFrame(pd_angle_Nion)

    #multiplying by the abundances of each element
    for column in pd_angle_mrg.columns:

        #multiplying by the right index in the abund_vals, after
        abund_factor=abund_vals[elements[column.split('_')[0][:-2]]-1]

        pd_angle_mrg[column]*=abund_factor

    #adding the full hydrogen density (neutral + ionized)
    pd_angle_mrg.insert(0,'NH_tot',pd_angle_mrg[['H00_fraction','H01_fraction']].sum(1))

    #inserting back the angle
    pd_angle_mrg.insert(0,'angle',np.unique(pd_use['angle']))



    pd_angle_mrg.rename(columns={elem:elem.replace('_fraction','_Nion') for elem in pd_angle_mrg.columns},inplace=True)

    if save:
        pd_angle_mrg.to_csv('monaco_Nion.csv',index=False)

    return pd_angle_mrg