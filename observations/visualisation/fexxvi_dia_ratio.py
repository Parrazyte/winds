import numpy as np
import matplotlib.pyplot as plt
from general_tools import ang2kev

'''

adapted from the formulae of 10.1093/mnras/195.3.705


blabla from Stefano
Ok, I have to read the papers. However, my general understanding is:
[12:20 AM]A standard He-like line can be produced either by excitation or recombination
[12:21 AM]excitation: you start from FeXXVI (1 electron) at ground state (1s), you excite it to 2p (collisionally mostly), then it goes back to 1s producing the Lya (the doublet depends on the total angular momentum)
[12:22 AM]recombination: you start with FeXXVII (no electrons), you recombine a free electron, the cascade will reach 2p, then 2p-1s, same line as before, different mechanism
[12:22 AM]Another possibility is indeed dielectronic recombination
[12:25 AM]In this case, you start with FeXXVI (one electron) in ground state. You recombine a free electron to a higher level and simultaneously excite the 1s electron (through the same process) to 2p. You end up with an excited FeXXV (2 electrons: one in 2p, the other somewhere in a higher level). When the 2p electron comes back to 1s, the energy is almost the same as the FeXXVI resonant line, but with a somewhat different energy (lower) due to the presence of the other electron ('spectator'). Note that it is FeXXV in principle, but the electron levels are basically those of FeXXVI because the other electron is in a higher level
[12:26 AM]This should be the main process. Details are in the papers...
[12:26 AM]The important point here is how effective is dielectronic recombination with respect to direct recombination, because this will set how strong are these satellite lines

'''

#in eV/K
k_ev=8.617333262e-5

def F_1(Te,E_0,E_S):
    '''
    formula (3) in the paper
    gives the temperature dependent term as a function of


    E_0 (base line energy) - eV

    E_S (satellite line energy) - eV

    Te (electron temperature) -K

    '''

    val=3.038e-13*1/f(Te,E_0)*E_0/Te*np.exp((E_0-E_S)/(k_ev*Te))

    return val

def f(Te,E_0):

    '''

    Oscillator strength of the resonant line for collisional excitation

    from Equation (4) and the terms given next page

    valid up to 1.6e17 cm^-3 ?

    this one matches their table 1 experimentally

    '''
    beta=17
    delta=1
    chi=0.23

    f_val=(0.25)**(3/2)*(beta/6.27)*(E_0/(k_ev*Te)+delta)/(E_0/(k_ev*Te)+chi)

    return f_val





'''
atomic data for the line intensity ratio 
From equation (2) but must be looked at in the table

we'll use the keys in the tables as identifiers

key: int for the line id
'''

e_key={
    '1': 1.7881,
    '2': 1.7787,
    '3': 1.7861,
    '4': 1.7918,
    '5': 1.7992,
    '6': 1.7930,
    '7': 1.7832,
    '8': 1.787,
    '9': 1.7917,
    '10': 1.7882,
    '11': 1.7921,
    '12': 1.7968,
    '13': 1.789,
    '14': 1.79,
    '15': 1.7939,
    '16': 1.7986,
    '17': 1.7742,
    '18': 1.7826,
    '19': 1.7924,
    '20': 1.801,
    '21': 1.8022,
    '22': 1.8109,

}
F_2={

    '1': 6.482e13,
    '2': 1.014e13,
    '3': 3.018e14,
    '4': 5.186e13,
    '5': 1.715e12,
    '6': 1.312e13,
    '7': 8.504e11,
    '8': 1.830e14,
    '9': 6.288e14,
    '10': 1.294e14,
    '11': 2.232e14,
    '12': 6.961e13,
    '13': 2.625e7,
    '14': 1.819e7,
    '15': 3.277e7,
    '16': 1.497e6,
    '17': 1.795e10,
    '18': 3.311e13,
    '19': 2.425e12,
    '20': 3.623e10,
    '21': 3.064e13,
    '22':4.403e13
}





int_vals=\
'''1	6.14e-2	3.74e-3
2	9.61e-3	2.53e-4
3	2.86e-1	7.52e-3
4	4.92e-2	3.51e-3
5	1.62e-3	1.12e-4
6	1.24e-2	4.72e-5
7	8.07e-4	6.40e-6
8	1.74e-1	1.38e-3
9	5.96e-1	4.73e-3
10	1.23e-1	5.38e-4
11	2.11e-1	9.23e-4
12	6.60e-2	2.89e-4
13	2.49e-8	2.28e-4
14	1.72e-8	1.57e-4
15	3.11e-8	2.85e-4
16	1.42e-9	1.30e-5
17	1.71e-5	2.67e-7
18	3.15e-2	3.73e-4
19	2.29e-3	1.15e-3
20	3.43e-5	1.73e-5
21	2.91e-2	1.02e-4
22	4.17e-2	1.46e-4'''

int_tab=np.array([elem.split('\t') for elem in int_vals.split('\n')],dtype=float)

int_vals_main={int(elem[0]):elem[1] for elem in int_tab}
int_vals_cascade={int(elem[0]):elem[2] for elem in int_tab}

def I_cas(key,T):
    '''
    composite formula from the tables and the formulas

    TOO COMPLICATED
    '''

def I_s(key,T):
    pass


def I_S(Te, E_0, key):

    """
    Te in K
    E_0 in keV

    """
    return F_1(Te, E_0*1000, ang2kev(e_key[str(key)])*1000) * F_2[str(key)]



def plot_ratios(ll_F_2=1e13,E_0=0):

    '''
    ll_F_2: lower limit on the intrinsic atomic factor to plot the transition

    Done withOUT radiative cascade

    NOTE: BECAUSE I DONT KNOW WHICH EO SHOULD BE USED I INSTEAD CHEAT BY USING THE VALUES OF TABLE 5
    '''

    T_sampl=np.logspace(5,10,1000)

    for elem in F_2.keys():
        if F_2[elem]>1e13:
            #computing the values from the tab
            F_tot=F_1(T_sampl,F_2[elem])*F_1

