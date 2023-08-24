#!/usr/bin/env python
# coding: utf-8

import os, sys
#import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
#import numpy as np
#import math
import numpy as np
import time
import pexpect
from pexpect.popen_spawn import PopenSpawn
import subprocess
import glob
global hdu1,hud2,hdu3,hdu4


def main():
   """ PyXstar is a Python module to access the data in the fits files of the XSTAR modeling code."""

if __name__ == '__main__':
    main()

par = {
"cfrac":        1.0,        #"covering fraction"
"temperature":  1.0,        #"temperature (/10**4K)"
"lcpres":       0,          #"constant pressure switch (1=yes, 0=no)"
"pressure":     0.03,       #"pressure (dyne/cm**2)"
"density":      1.e+20,     #"density (cm**-3)"
"spectrum":     "pow",      #"spectrum type?"
"spectrum_file":'', #"spectrum file?"
"spectun":      0,          #"spectrum units? (0=energy, 1=photons)"
"trad":        -1.0,        #"radiation temperature or alpha?"
"rlrad38":      1.e-6,      #"luminosity (/10**38 erg/s)"
"column":       1.e+15,     #"column density (cm**-2)"
"rlogxi":       1.0,        #"log(ionization parameter) (erg cm/s)"
"habund":       1.0,        #"hydrogen abundance"
"heabund":      1.0,        #"helium abundance"
"liabund":      0.0,        #"lithium abundance"
"beabund":      0.0,        #"beryllium abundance"
"babund":       0.0,        #"boron abundance"
"cabund":       1.0,        #"carbon abundance"
"nabund":       1.0,        #"nitrogen abundance"
"oabund":       1.0,        #"oxygen abundance"  
"fabund":       0.0,        #"fluorine abundance"
"neabund":      1.0,        #"neon abundance"
"naabund":      0.0,        #"sodium abundance"
"mgabund":      1.0,        #"magnesium abundance"
"alabund":      1.0,        #"aluminum abundance"
"siabund":      1.0,        #"silicon abundance"
"pabund":       0.0,        #"phosphorus abundance"
"sabund":       1.0,        #"sulfur abundance"
"clabund":      0.0,        #"chlorine abundance"
"arabund":      1.0,        #"argon abundance"
"kabund":       0.0,        #"potassium abundance"
"caabund":      1.0,        #"calcium abundance"
"scabund":      0.0,        #"scandium abundance"
"tiabund":      0.0,        #"titanium abundance"
"vabund":       0.0,        #"vanadium abundance"
"crabund":      0.0,        #"chromium abundance"
"mnabund":      0.0,        #"manganese abundance"
"feabund":      1.0,        #"iron abundance"
"coabund":      0.0,        #"cobalt abundance"
"niabund":      1.0,        #"nickel abundance"
"cuabund":      0.0,        #"copper abundance"
"znabund":      0.0,        #"zinc abundance"
"modelname":"XSTAR_Default",#"model name"
}

# Input hpar dictionary. Do not modify values unless you know what you are doing.

hpar = {
"nsteps":     3,     #"number of steps"
"niter":      0,     #"number of iterations"
"lwrite":     0,     #"write switch (1=yes, 0=no)"
"lprint":     0,     #"print switch (1=yes, 0=no)"
"lstep":      0,     #"step size choice switch"
"emult":      0.5,   #"Courant multiplier"
"taumax":     5.0,   #"tau max for courant step"
"xeemin":     0.1,   #"minimum electron fraction"
"critf":      1.e-7, #"critical ion abundance"
"vturbi":     1.0,   #"turbulent velocity (km/s)"
"radexp":     0.,    #"density distribution power law index"
"ncn2":       9999,  #"number of continuum bins"
"loopcontrol":0,     #"loop control (0=standalone)"
"npass":      1,     #"number of passes"
"mode":       "ql"   #"mode"
}

#########################
#additions
#########################

####Copy of xstar custom and standard functions for grid overwriting

custom_enerf90=['      subroutine ener(epi,ncn2) \n',
 '!                                                                       \n',
 '!     Name: ener.f90  \n',
 '!     Description:  \n',
 '!     This routine sets up the energy grid\n',
 '!     Grid is logarithmic with two subranges:  0.1eV-40 keV, 40keV-1MeV.\n',
 '!     This structure of epi is key to operation of various other routines\n',
 '!     special version tailored for M. Parra\n',
 '!     author: T. Kallman                                                \n',
 '!     List of Parameters:\n',
 '!           Output:\n',
 '!           epi(ncn)=energy grid (ev)\n',
 '!           ncn2=length of epi\n',
 '!     Dependencies:  none\n',
 '!     Called by:  xstar\n',
 '!                                                                       \n',
 '      use globaldata\n',
 '      implicit none \n',
 '!                                                                       \n',
 '      real(8) epi(ncn) \n',
 '      integer numcon,numcon2,numcon3,ncn2,ll,ll2,mm\n',
 '      real(8) ebnd1,ebnd2,ebnd3,ebnd4,dele1,dele2,dele3,etmp\n',
 '!                                                                       \n',
 '      numcon = ncn2 \n',
 "      if (numcon.gt.ncn) stop 'ncn2 too large for arry dimension' \n",
 "      if (numcon.lt.4) write (6,*) 'in ener: numcon error' \n",
 '!\n',
 '!     set up\n',
 '!     3 regions\n',
 '      ebnd1=0.1 \n',
 '      ebnd2=100.\n',
 '      ebnd3=1.e+4\n',
 '      ebnd4=1.e+6 \n',
 '      numcon2=max(2,ncn2/50) \n',
 '      numcon3=numcon-2*numcon2 \n',
 '      dele1=(ebnd2/ebnd1)**(1./numcon2)\n',
 '      dele2=(ebnd3/ebnd2)**(1./numcon3)\n',
 '      dele3=(ebnd4/ebnd3)**(1./(numcon2-1))\n',
 '!\n',
 '!     step thru energies\n',
 '      etmp=ebnd1\n',
 '      mm=1\n',
 '      do while (mm.le.numcon)\n',
 '        epi(mm)=etmp\n',
 '        mm=mm+1\n',
 '        if (etmp.lt.ebnd2) then\n',
 '          etmp=etmp*dele1\n',
 '        else\n',
 '          if (etmp.lt.ebnd3) then\n',
 '            etmp=etmp*dele2\n',
 '          else\n',
 '            etmp=etmp*dele3\n',
 '          endif\n',
 '        endif\n',
 '        enddo\n',
 '!\n',
 '!     print out\n',
 '!      do mm=1,numcon\n',
 '!        write (6,*)mm,epi(mm)\n',
 '!        enddo\n',
 '!\n',
 '      return \n',
 '      end                                           \n']

custom_huntff90=['      subroutine huntf(xx,n,x,jlo,lpri,lun11) \n',
 '!                                                                       \n',
 '!     Name: huntf.f90  \n',
 '!     Description:  \n',
 '!           Searches in a list. \n',
 '!           assumes logarithmically spaced values\n',
 '!\n',
 '!     List of Parameters:\n',
 '!           Input: \n',
 '!           xx(n):  list to be searched\n',
 '!           n:  search length\n',
 '!           x: search value\n',
 '!           lpri:  print switch, 1=on, 0=off\n',
 '!           lun11: logical unit number to write to\n',
 '!           Output:\n',
 '!           jlo:  index of found element\n',
 '!     Dependencies:  none\n',
 '!     Called by:  nbinc\n',
 '!\n',
 '!     this version of hunt assumes equally spaced data in log           \n',
 '!     special version for Maxime Parra\n',
 '!     author:  T. Kallman                                               \n',
 '!                                                                       \n',
 '      implicit none \n',
 '!                                                                       \n',
 '      integer n,jlo,lpri,lun11 \n',
 '      real(8) xx(n),x,xtmp,tst,tst2 \n',
 '      integer numcon,numcon2,numcon3,ncn2,ll,ll2,mm\n',
 '      real(8) ebnd1,ebnd2,ebnd3,ebnd4,dele1,dele2,dele3,etmp\n',
 '      real(8) algdele1,algdele2,algdele3\n',
 '      integer lfirst\n',
 '!\n',
 '      data lfirst/1/\n',
 '!                                                                       \n',
 '      xtmp=max(x,xx(1)) \n',
 '      jlo=1\n',
 '      if ((x.lt.1.e-34).or.(xx(1).le.1.e-34).or.(xx(n).le.1.e-34)) return\n',
 '!\n',
 '!      if (lfirst.eq.1) then\n',
 '      lfirst=0\n',
 '!     set up\n',
 '!     3 regions\n',
 '      ebnd1=0.1 \n',
 '      ebnd2=100.\n',
 '      ebnd3=1.e+4\n',
 '      ebnd4=1.e+6 \n',
 '!     correct for the fact that huntf is called with numcon3 as argument\n',
 '!     but definition used is the standard xstar definition\n',
 '      numcon3=n\n',
 '      numcon=int(numcon3/(1.-1./50.))\n',
 '      ncn2=numcon\n',
 '      numcon2=max0(2,ncn2/50) \n',
 '!     now redefine numcon3\n',
 '      numcon3=numcon-2*numcon2 \n',
 '!      write (6,*)numcon,numcon2,numcon3\n',
 '!      endif\n',
 '!\n',
 '      etmp=xtmp\n',
 '      if (etmp.lt.ebnd2) then\n',
 '        dele1=(ebnd2/ebnd1)**(1./numcon2)\n',
 '        algdele1=log(dele1)\n',
 '        jlo=int(log(etmp/ebnd1)*(1.00000001)/algdele1)+1 \n',
 '      else\n',
 '        if (etmp.lt.ebnd3) then\n',
 '          dele2=(ebnd3/ebnd2)**(1./numcon3)\n',
 '          algdele2=log(dele2)\n',
 '          jlo=numcon2+int(log(etmp/ebnd2)*(1.00000001)/algdele2)\n',
 '        else\n',
 '          dele3=(ebnd4/ebnd3)**(1./(numcon2-1))\n',
 '          algdele3=log(dele3)\n',
 '          jlo=numcon2+numcon3+int(log(etmp/ebnd3)*(1.00000001)/algdele3)\n',
 '        endif\n',
 '      endif\n',
 '      jlo=max(1,jlo) \n',
 '      jlo=min(numcon,jlo) \n',
 '      if (lpri.gt.0)                                                    &\n',
 "     &  write (lun11,*)'in huntf',n,xx(1),xx(n),jlo,xx(jlo),x           \n",
 '!                                                                       \n',
 '      return \n',
 '      end                                           \n']

std_enerf90=['      subroutine ener(epi,ncn2) \n',
 '!                                                                       \n',
 '!     Name: ener.f90  \n',
 '!     Description:  \n',
 '!     This routine sets up the energy grid\n',
 '!     Grid is logarithmic with two subranges:  0.1eV-40 keV, 40keV-1MeV.\n',
 '!     This structure of epi is key to operation of various other routines\n',
 '!     author: T. Kallman                                                \n',
 '!     List of Parameters:\n',
 '!           Output:\n',
 '!           epi(ncn)=energy grid (ev)\n',
 '!           ncn2=length of epi\n',
 '!     Dependencies:  none\n',
 '!     Called by:  xstar\n',
 '!                                                                       \n',
 '      use globaldata\n',
 '      implicit none \n',
 '!                                                                       \n',
 '      real(8) epi(ncn) \n',
 '      integer numcon,numcon2,numcon3,ncn2,ll,ll2 \n',
 '      real(8) ebnd1,ebnd2,ebnd2o,dele \n',
 '!                                                                       \n',
 '      numcon = ncn2 \n',
 "      if (numcon.gt.ncn) stop 'ncn2 too large for arry dimension' \n",
 "      if (numcon.lt.4) write (6,*) 'in ener: numcon error' \n",
 '      numcon2=max(2,ncn2/50) \n',
 '      numcon3=numcon-numcon2 \n',
 '      ebnd1=0.1 \n',
 '!     nb changed energy grid for H only                                 \n',
 '      ebnd2=4.e+5 \n',
 '!      ebnd2=4.e+1                                                      \n',
 '      ebnd2o=ebnd2 \n',
 '      dele=(ebnd2/ebnd1)**(1./float(numcon3-1)) \n',
 '      epi(1)=ebnd1 \n',
 "!      write (lun11,*)'in ener',ncn2,numcon,numcon2,numcon3                 \n",
 '      do ll=2,numcon3 \n',
 '        epi(ll)=epi(ll-1)*dele \n',
 '        enddo \n',
 '      ebnd2=1.e+6 \n',
 '      ebnd1=ebnd2o \n',
 '      dele=(ebnd2/ebnd1)**(1./float(numcon2-1)) \n',
 '      do ll2=1,numcon2 \n',
 '        ll=ll2+numcon3 \n',
 '        epi(ll)=epi(ll-1)*dele \n',
 '        enddo \n',
 '!                                                                       \n',
 '      return \n',
 '      end                                           \n']

std_huntff90=['      subroutine huntf(xx,n,x,jlo,lpri,lun11) \n',
 '!                                                                       \n',
 '!     Name: huntf.f90  \n',
 '!     Description:  \n',
 '!           Searches in a list. \n',
 '!           assumes logarithmically spaced values\n',
 '!\n',
 '!     List of Parameters:\n',
 '!           Input: \n',
 '!           xx(n):  list to be searched\n',
 '!           n:  search length\n',
 '!           x: search value\n',
 '!           lpri:  print switch, 1=on, 0=off\n',
 '!           lun11: logical unit number to write to\n',
 '!           Output:\n',
 '!           jlo:  index of found element\n',
 '!     Dependencies:  none\n',
 '!     Called by:  nbinc\n',
 '!\n',
 '!     this version of hunt assumes equally spaced data in log           \n',
 '!     author:  T. Kallman                                               \n',
 '!                                                                       \n',
 '      implicit none \n',
 '!                                                                       \n',
 '      integer n,jlo,lpri,lun11 \n',
 '      real(8) xx(n),x,xtmp,tst,tst2 \n',
 '!                                                                       \n',
 '      xtmp=max(x,xx(2)) \n',
 '      jlo=1\n',
 '      if ((x.lt.1.e-34).or.(xx(1).le.1.e-34).or.(xx(n).le.1.e-34)) return\n',
 '      jlo=int((n-1)*log(xtmp/xx(1))/log(xx(n)/xx(1)))+1 \n',
 '      if (jlo.lt.n) then \n',
 '        tst=abs(log(x/(1.e-34+xx(jlo)))) \n',
 '        tst2=abs(log(x/(1.e-34+xx(jlo+1)))) \n',
 '        if (tst2.lt.tst) jlo=jlo+1 \n',
 '        endif \n',
 '      jlo=max(1,jlo) \n',
 '      jlo=min(n,jlo) \n',
 '      if (lpri.gt.0)                                                    &\n',
 "     &  write (lun11,*)'in huntf',n,xx(1),xx(n),jlo,xx(jlo),x           \n",
 '!                                                                       \n',
 '      return \n',
 '      end                                           \n']
def update_grid(file_ener,file_huntf):
    '''
    fetches the current enerf and huntf files in the currently installed xstar
    and replaces them with file_ener and file_huntf (list of lines) if necessary

    Note: still need to reinstall heasoft after to update everything
    '''

    ener_path=glob.glob(os.environ['HEADAS']+'/../**/ener.f90',recursive=True)

    assert len(ener_path) != 0, "No ener.f90 file found in the heasoft arborescence."
    assert len(ener_path) == 1, "Multiple ener.f90 files found in the heasoft arborescence."

    huntf_path = glob.glob(os.environ['HEADAS'] + '/../**/huntf.f90', recursive=True)

    assert len(huntf_path) != 0, "No huntf.f90 file found in the heasoft arborescence."
    assert len(huntf_path) == 1, "Multiple huntf.f90 files found in the heasoft arborescence."

    ener_path=ener_path[0]
    huntf_path = huntf_path[0]

    #spawning a bash process to compile
    gfort_proc=pexpect.spawn('/bin/bash',encoding='utf-8')

    gfort_proc.logfile=sys.stdout

    with open(ener_path) as curr_ener_file:
        curr_ener_lines=curr_ener_file.readlines()

    if curr_ener_lines!=file_ener:
        print('\nChanging ener.f90 file to match the desired grid...\n')
        
        with open(ener_path,'w+') as curr_ener_file:
            curr_ener_file.writelines(file_ener)


    with open(huntf_path) as curr_huntf_file:
        curr_huntf_lines = curr_huntf_file.readlines()

    if curr_huntf_lines != file_huntf:
        print('\nChanging huntf.f90 file to match the desired grid...\n')

        with open(huntf_path, 'w+') as curr_huntf_file:
            curr_huntf_file.writelines(file_huntf)


    #adding a wait to ensure the compilation have the time to run
    gfort_proc.sendline('exit')

#########################

def LoadFiles(file1='./xout_abund1.fits',file2='./xout_lines1.fits',\
        file3='./xout_rrc1.fits', file4='./xout_spect1.fits'):
    global hdu1, hdu2, hdu3, hdu4
    if os.path.isfile(file1):
        hdu1= pyfits.open(file1)
    else:
        print('*** PyXstar: ',file1,' not found')
        return -1
    
    if os.path.isfile(file2):
        hdu2= pyfits.open(file2)
    else:
        print('*** PyXstar: ',file2,' not found')
        return -1    
        
    if os.path.isfile(file3):
        hdu3= pyfits.open(file3)
    else:
        print('*** PyXstar: ',file3,' not found')
        return -1
    
    if os.path.isfile(file4):
        hdu4= pyfits.open(file4)
    else:
        print('*** PyXstar: ',file4,' not found')
        return -1

class PlasmaParameters:
    def __init__(self):
        global hdu1
        rad=[]; delt=[]; iop=[]; xe=[]; np=[]; pres=[];temp=[];frac=[]; apar=[]
        for i in range(len(hdu1[1].data)):
            rad.append(hdu1[1].data[i]['radius'])
            delt.append(hdu1[1].data[i]['delta_r'])
            iop.append(hdu1[1].data[i]['ion_parameter'])
            xe.append(hdu1[1].data[i]['x_e'])
            np.append(hdu1[1].data[i]['n_p'])
            pres.append(hdu1[1].data[i]['pressure'])
            temp.append(hdu1[1].data[i]['temperature'])
            frac.append(hdu1[1].data[i]['frac_heat_error'])
            apar.append({'step':i+1,'radius':hdu1[1].data[i]['radius'],\
                         'delta_r':hdu1[1].data[i]['delta_r'],\
                         'ion_parameter':hdu1[1].data[i]['ion_parameter'],\
                         'x_e':hdu1[1].data[i]['x_e'],\
                         'n_p':hdu1[1].data[i]['n_p'],\
                         'pressure':hdu1[1].data[i]['pressure'],\
                         'temperature':hdu1[1].data[i]['temperature'],\
                         'frac_heat_error':hdu1[1].data[i]['frac_heat_error']})
        self.radius=tuple(rad)
        self.delta_r=tuple(delt)
        self.ion_parameter=tuple(iop)
        self.x_e=tuple(xe)
        self.n_p=tuple(np)
        self.pressure=tuple(pres)        
        self.temperature=tuple(temp)
        self.frac_heat_error=tuple(frac)
        self.all=tuple(apar)
        self.units={'radius':hdu1[1].header['TUNIT1'],\
                    'delta_r':hdu1[1].header['TUNIT2'],\
                    'ion_parameter':hdu1[1].header['TUNIT3'],\
                    'x_e':'cm-3','n_p':'cm-3','pressure':hdu1[1].header['TUNIT6'],\
                    'temperature':hdu1[1].header['TUNIT7']}

def NSteps():
    global hdu1
    return len(hdu1[1].data)

def Abundances(species):
    global hdu1
    element=('p','h','he','li','be','b','c','n','o','f','ne',\
             'na','mg','al','si','p','s','cl','ar','k','ca',\
             'sc','ti','v','cr','mn','fe','co','ni','cu','zn')
    ion=('i','ii','iii','iv','v','vi','vii','viii','ix','x',\
         'xi','xii','xiii','xiv','xv','xvi','xvii','xviii','xix','xx',\
         'xxi','xxii','xxiii','xxiv','xxv','xxvi','xxvii','xxviii','xxix','xxx')
      
    fract=[]
    
    if species.find('_') >= 0:
        for i in range(9,474):
            if species == hdu1[1].header['TTYPE'+str(i)]:
                frac=[]
                for k in range(len(hdu1[1].data)):
                    frac.append(hdu1[1].data[k][species])
                return tuple(frac)
        print('*** PyXstar: ',species,' not listed.')
        return -1
    else:
        for i in range(len(element)):
            if species == element[i]:
                for k in range(len(hdu1[1].data)):
                    frac=[]
                    for j in range(i):
                         frac.append(hdu1[1].data[k][species+'_'+ion[j]])
                    fract.append(tuple(frac))
                return tuple(fract)
        print('*** PyXstar: species ',species,' not listed.')
        return -1

def Columns(species):
    global hdu1
    element=('p','h','he','li','be','b','c','n','o','f','ne',\
             'na','mg','al','si','p','s','cl','ar','k','ca',\
             'sc','ti','v','cr','mn','fe','co','ni','cu','zn')
    ion=('i','ii','iii','iv','v','vi','vii','viii','ix','x',\
         'xi','xii','xiii','xiv','xv','xvi','xvii','xviii','xix','xx',\
         'xxi','xxii','xxiii','xxiv','xxv','xxvi','xxvii','xxviii','xxix','xxx')
    
    frac=[]
    
    if species.find('_') >= 0:
        for i in range(9,474):
            if species == hdu1[1].header['TTYPE'+str(i)]:
                return hdu1[2].data[0][species]              
        print('*** PyXstar: ',species,' not listed.')
        return -1
    else:
        for i in range(len(element)):
            if species == element[i]:
                for j in range(i):
                    frac.append(hdu1[2].data[0][species+'_'+ion[j]])
                return tuple(frac)                
        print('*** PyXstar: species ',species,' not listed.')
        return -1 

def Heating(source):
    global hdu1
    s_short=('p','h','he','li','be','b','c','n','o','f','ne',\
             'na','mg','al','si','p','s','cl','ar','k','ca',\
             'sc','ti','v','cr','mn','fe','co','ni','cu','zn','compton','total')
    s_long =('proton','hydrogen','helium','lithium','berylium','boron','carbon',\
             'nitrogen','oxygen','fluorine','neon','sodium','magnesium','aluminium',\
             'silicon','phosphorus','sulfur','chlorine','argon','potassium','calcium',\
             'scandium','titanium','vanadium','chromium','manganese','iron','cobalt',\
             'nickel','copper','zinc','compton','total')    
    
    for i in range(len(s_short)):
        if source == s_short[i]:
            return hdu1[3].data[0][s_long[i]]
    # print('*** PyXstar: source ',item,' not listed.')
    return -1  

def Cooling(source):
    global hdu1
    s_short=('p','h','he','li','be','b','c','n','o','f','ne',\
             'na','mg','al','si','p','s','cl','ar','k','ca',\
             'sc','ti','v','cr','mn','fe','co','ni','cu','zn','compton','brems','total')
    s_long =('proton','hydrogen','helium','lithium','berylium','boron','carbon',\
             'nitrogen','oxygen','fluorine','neon','sodium','magnesium','aluminium',\
             'silicon','phosphorus','sulfur','chlorine','argon','potassium','calcium',\
             'scandium','titanium','vanadium','chromium','manganese','iron','cobalt',\
             'nickel','copper','zinc','compton','brems','total')    
    
    for i in range(len(s_short)):
        if source == s_short[i]:
            return hdu1[4].data[0][s_long[i]]
    # print('*** PyXstar: source ',item,' not listed.')
    return -1  

class ContSpectra:
    def __init__(self):
        global hdu4
        ene=[]; inc=[]; trans=[]; emit_in=[];emit_out=[]
        for i in range(len(hdu4[2].data)):
            ene.append(hdu4[2].data[i]['energy'])
            inc.append(hdu4[2].data[i]['incident'])
            trans.append(hdu4[2].data[i]['transmitted'])
            emit_in.append(hdu4[2].data[i]['emit_inward'])
            emit_out.append(hdu4[2].data[i]['emit_outward'])
        self.energy=tuple(ene)
        self.incident=tuple(inc)
        self.transmitted=tuple(trans)
        self.emit_inward=tuple(emit_in)
        self.emit_outward=tuple(emit_out)

def NContPoints():
    global hdu4
    return len(hdu4[2].data)

class LineSpectra:
    def __init__(self):
        global hdu2
        line_list=[]
        for i in range(len(hdu2[2].data)):
            line_list.append({'index':hdu2[2].data[i]['index'],\
                              'ion':hdu2[2].data[i]['ion'],\
                              'lower_level':hdu2[2].data[i]['lower_level'],\
                              'upper_level':hdu2[2].data[i]['upper_level'],\
                              'wavelength':hdu2[2].data[i]['wavelength'],\
                              'emit_inward':hdu2[2].data[i]['emit_inward'],\
                              'emit_outward':hdu2[2].data[i]['emit_outward'],\
                              'depth_inward':hdu2[2].data[i]['depth_inward'],\
                              'depth_outward':hdu2[2].data[i]['depth_outward']})
        self.lines=tuple(line_list)
        self.units={'wavelength':hdu2[2].header['TUNIT5'],\
                    'emit_inward':hdu2[2].header['TUNIT6'],\
                    'emit_outward':hdu2[2].header['TUNIT7']}

def NLines():
    global hdu2
    return len(hdu2[2].data)

class RRcSpectra:
    def __init__(self):
        global hdu3
        rrc_list=[]
        for i in range(len(hdu3[2].data)):
            rrc_list.append({'index':hdu3[2].data[i]['index'],\
                             'ion':hdu3[2].data[i]['ion'],\
                             'level':hdu3[2].data[i]['level'],\
                             'energy':hdu3[2].data[i]['energy'],\
                             'emit_outward':hdu3[2].data[i]['emit_outward'],\
                             'emit_inward':hdu3[2].data[i]['emit_inward'],\
                             'depth_outward':hdu3[2].data[i]['depth_outward'],\
                             'depth_inward':hdu3[2].data[i]['depth_inward']})
        self.rrc=tuple(rrc_list)
        self.units={'energy':hdu3[2].header['TUNIT4'],\
                    'emit_outward':hdu3[2].header['TUNIT5'],\
                    'emit_inward':hdu3[2].header['TUNIT6']}              

def NRRcPoints():
    global hdu3
    return len(hdu3[2].data)

####################################################################

def run_xstar(par,hpar,headas_folder=None):
# Function to create the XSTAR input file xstar.par and run the code from 
# standard Heasoft installation

    #fetching the environment variable of the specific heasoft folder if asked to
    if headas_folder is not None:
        heaproc=PopenSpawn('/bin/bash',maxread=16000,logfile=sys.stdout.buffer)

        heaproc.sendline('source ~/.bashrc')
        time.sleep(1)
        heaproc.sendline('export HEADAS='+headas_folder)
        heaproc.sendline('source $HEADAS/headas-init.sh')
        heaproc.sendline('cd '+os.getcwd())
        heaproc.sendline('export PFILES=.')

        # heaproc.sendline('env > temp_custom_xstar_env.txt')
        #
        # time.sleep(1)
        #
        # heaproc.sendline('exit')
        #
        # with open('./temp_custom_xstar_env.txt') as env_file:
        #     env_lines=env_file.readlines()
        #
        # env_copy=os.environ.copy()
        # env_copy.clear()
        #
        # for elem_line in env_lines:
        #     #skipping non-heasoft lines (assuming the user didn't change the direct name of its heasoft dir)
        #     # if 'heasoft' not in elem_line or  elem_line.split('=')[0]=='PWD':
        #     #     continue
        #     env_var=elem_line.split('=')[0]
        #     env_copy[env_var]=elem_line.split('=')[1]
        #
        #     env_copy["PFILES"]="."
        #
        # os.remove('temp_custom_xstar_env.txt')
        #
        # xstar_path=glob.glob(headas_folder+'/**/xstar',recursive=True)[0]



# Set $PFILES environment variable to run the code with the new xstar.par
    os.environ["PFILES"]="."
    
# Remove old xstar.par and output files    
    os.system("rm -f ./xstar.par")
    
####modified this to avoid removing the log files after changing them
    os.system("rm -f xout_step.log")
    
# Construct new xstar.par file
    file=open('./xstar.par','a')
    xstar1_in='#########################################################################\n\
# File: xstar.par                                                       #\n\
#                                                                       #\n\
# Parameter list for XSTAR package                                      #\n\
#                                                                       #\n\
# Note: This file is related to the xspec2xstar.par file.               #\n\
#       Exercise caution with both when making modifications to one     #\n\
#                                                                       #\n\
# Modifications:                                                        #\n\
# 1/21/1999, WTB: added loopcontrol for multiple runs                   #\n\
#                 installed lower & upper limits                        #\n\
#                                                                       #\n\
# Documentation below is based on documentation by J. Silvis            #\n\
# of RITSS.                                                             #\n\
#                                                                       #\n\
# This file provides information needed by Xanadu Parameter             #\n\
# Interface (XPI) routines to provide input to the ftool xstar.         #\n\
# The entries to a parameter file, such as this one, have               #\n\
# seven columns.\n\
#\n\
# Column   Colunm Function    Comment\n\
#  1       Parameter Name     Alpha-numeric\n\
#\n\
#  2       Data type          s-string\n\
#                             i-integer\n\
#                             r-real\n\
#                             b-boolean\n\
#\n\
#  3       Parameter mode     q-query (asks the user for a value)\n\
#                             h-hidden (does not ask user)\n\
#                             ql-query+learn (remember value\n\
#                                               from last run)\n\
#                             hl-hidden+learn\n\
#                             a-automatic (use value stored in\n\
#                                               the mode parameter)\n\
#\n\
# 4        Default value      If user hits <cr> this is entered\n\
#\n\
# 5        Lower limit        Can be left blank if none is required\n\
#\n\
# 6        Upper limit        Can be left blank if none is required\n\
#\n\
# 7        Prompt             Question printed on screen for user\n\
#\n\
# When parameter mode is set to "a" the line will use the value set by\n\
# the mode statement e.g.\n\
#\n\
# infile,s,a,"her_bfits_1_143.pha",,,"Enter the input files name"\n\
# mode,s,h,"ql",,,""\n\
# is the same as\n\
# infile,s,ql,"her_bfits_1_143.pha",,,"Enter the input files name"\n\
#\n\
# You may want to use this if you need to change several parameters at\n\
# once.\n\
#\n\
# Note on the mode statement.  This is just a regular parameter statement,\n\
# i.e. it sets the value of a string parameter (the first s)\n\
# without prompting the user (the h) and the value is ql.\n\
#\n\
################################################################\n\
# User Adjustable Parameters\n\
#\n'

    file.write(xstar1_in)

    xstar2_in='cfrac,r,h,'+str(par['cfrac'])+',0.0,1.0,"covering fraction"\n\
#\n\
temperature,r,h,'+str(par['temperature'])+',0.0,1.0e4,"temperature (/10**4K)"\n\
#\n\
lcpres,i,h,'+str(par['lcpres'])+',0,1,"constant pressure switch (1=yes, 0=no)"\n\
#\n\
pressure,r,h,'+str(par['pressure'])+',0.0,1.0,"pressure (dyne/cm**2)"\n\
#\n\
density,r,h,'+str(par['density'])+',0.,1.e21,"density (cm**-3)"\n\
#\n\
spectrum,s,h,'+str(par['spectrum'])+',,,"spectrum type?"\n\
#\n\
spectrum_file,s,h,'+str(par['spectrum_file'])+',,,"spectrum file?"\n\
#\n\
spectun,i,h,'+str(par['spectun'])+',0,1,"spectrum units? (0=energy, 1=photons)"\n\
#\n\
trad,r,h,'+str(par['trad'])+',-1.0,0.0,"radiation temperature or alpha?"\n\
#\n\
rlrad38,r,h,'+str(par['rlrad38'])+',0.0,1.e10,"luminosity (/10**38 erg/s)"\n\
#\n\
column,r,h,'+str(par['column'])+',0.0,1.e25,"column density (cm**-2)"\n\
#\n\
rlogxi,r,h,'+str(par['rlogxi'])+',-10.,+10.,"log(ionization parameter) (erg cm/s)"\n\
#\n\
habund,r,h,'+str(par['habund'])+',0.0,100.,"hydrogen abundance"\n\
#\n\
heabund,r,h,'+str(par['heabund'])+',0.0,100.,"helium abundance"\n\
#\n\
liabund,r,h,'+str(par['liabund'])+',0.0,100.,"lithium abundance"\n\
#\n\
beabund,r,h,'+str(par['beabund'])+',0.0,100.,"beryllium abundance"\n\
#\n\
babund,r,h,'+str(par['babund'])+',0.0,100.,"boron abundance"\n\
#\n\
cabund,r,h,'+str(par['cabund'])+',0.0,100.,"carbon abundance"\n\
#\n\
nabund,r,h,'+str(par['nabund'])+',0.0,100.,"nitrogen abundance"\n\
#\n\
oabund,r,h,'+str(par['oabund'])+',0.0,100.,"oxygen abundance"\n\
#\n\
fabund,r,h,'+str(par['fabund'])+',0.0,100.,"fluorine abundance"\n\
#\n\
neabund,r,h,'+str(par['neabund'])+',0.0,100.,"neon abundance"\n\
#\n\
naabund,r,h,'+str(par['naabund'])+',0.0,100.,"sodium abundance"\n\
#\n\
mgabund,r,h,'+str(par['mgabund'])+',0.0,100.,"magnesium abundance"\n\
#\n\
alabund,r,h,'+str(par['alabund'])+',0.0,100.,"aluminum abundance"\n\
#\n\
siabund,r,h,'+str(par['siabund'])+',0.0,100.,"silicon abundance"\n\
#\n\
pabund,r,h,'+str(par['pabund'])+',0.0,100.,"phosphorus abundance"\n\
#\n\
sabund,r,h,'+str(par['sabund'])+',0.0,100.,"sulfur abundance"\n\
#\n\
clabund,r,h,'+str(par['clabund'])+',0.0,100.,"chlorine abundance"\n\
#\n\
arabund,r,h,'+str(par['arabund'])+',0.0,100.,"argon abundance"\n\
#\n\
kabund,r,h,'+str(par['kabund'])+',0.0,100.,"potassium abundance"\n\
#\n\
caabund,r,h,'+str(par['caabund'])+',0.0,100.,"calcium abundance"\n\
#\n\
scabund,r,h,'+str(par['scabund'])+',0.0,100.,"scandium abundance"\n\
#\n\
tiabund,r,h,'+str(par['tiabund'])+',0.0,100.,"titanium abundance"\n\
#\n\
vabund,r,h,'+str(par['vabund'])+',0.0,100.,"vanadium abundance"\n\
#\n\
crabund,r,h,'+str(par['crabund'])+',0.0,100.,"chromium abundance"\n\
#\n\
mnabund,r,h,'+str(par['mnabund'])+',0.0,100.,"manganese abundance"\n\
#\n\
feabund,r,h,'+str(par['feabund'])+',0.0,100.,"iron abundance"\n\
#\n\
coabund,r,h,'+str(par['coabund'])+',0.0,100.,"cobalt abundance"\n\
#\n\
niabund,r,h,'+str(par['niabund'])+',0.0,100.,"nickel abundance"\n\
#\n\
cuabund,r,h,'+str(par['cuabund'])+',0.0,100.,"copper abundance"\n\
#\n\
znabund,r,h,'+str(par['znabund'])+',0.0,100.,"zinc abundance"\n\
#\n\
modelname,s,h,'+str(par['modelname'])+',,,"model name"\n\
#\n\
#####################################################\n\
#  Hidden Parameters\n\
#  Dont mess with these unless you know what you are doing!\n'

    file.write(xstar2_in)

    xstar3_in='nsteps,i,h,'+str(hpar['nsteps'])+',1,1000,"number of steps"\n\
#\n\
niter,i,h,'+str(hpar['niter'])+',0,100,"number of iterations"\n\
#\n\
lwrite,i,h,'+str(hpar['lwrite'])+',-1,1,"write switch (1=yes, 0=no)"\n\
#\n\
lprint,i,h,'+str(hpar['lprint'])+',-1,2,"print switch (1=yes, 0=no)"\n\
#\n\
lstep,i,h,'+str(hpar['lstep'])+',0,0,"step size choice switch"\n\
#\n\
emult,r,h,'+str(hpar['emult'])+',1.e-6,1.e+6,"Courant multiplier"\n\
#\n\
taumax,r,h,'+str(hpar['taumax'])+',1.,10000.,"tau max for courant step"\n\
#\n\
xeemin,r,h,'+str(hpar['xeemin'])+',1.e-6,0.5,"minimum electron fraction"\n\
#\n\
critf,r,h,'+str(hpar['critf'])+',1.e-24,0.1,"critical ion abundance"\n\
#\n\
vturbi,r,h,'+str(hpar['vturbi'])+',0.,30000.,"turbulent velocity (km/s)"\n\
#\n\
radexp,r,h,'+str(hpar['radexp'])+',-3.,3.,"density distribution power law index"\n\
#\n\
ncn2,i,h,'+str(hpar['ncn2'])+',999,999999,"number of continuum bins"\n\
#\n\
loopcontrol,i,h,'+str(hpar['loopcontrol'])+',0,30000,"loop control (0=standalone)"\n\
#\n\
npass,i,h,'+str(hpar['npass'])+',1,10000,"number of passes"\n\
#\n\
mode,s,h,'+str(hpar['mode'])+',,,"mode"\n'

    file.write(xstar3_in)
    file.close()

# Run XSTAR with new xstar.par

    if headas_folder is not None:

        heaproc.sendline('xstar')

        heaproc_status=heaproc.expect([' total time',pexpect.TIMEOUT,
                                       'Fortran runtime error','  Program aborting...'],timeout=3600)

        assert heaproc_status!=1,'xstar timeout'
        assert heaproc_status<2,'error while running xstar'

        heaproc.sendline('exit')

    else:
        with subprocess.Popen("xstar",stdout=subprocess.PIPE,bufsize=1,shell=True,\
                              universal_newlines=True) as p:
            for line in p.stdout:
                print(line,end='') # process line here
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode,p.args)

#########################################################################

def docker_run_xstar(par,hpar,\
    container="heasoft:v6.32",\
    host_account="noperm",
    identifier='xstar_output'):

    # Function to create the XSTAR input file xstar.par and run the code from
    # the XSTAR Docker container

    '''

    note: heasoft:v6.32 tarball created locally, modified to add the new binemis version

    transformed into tar with docker export name > name.tar
    copied via scp to the server (scp file.txt ipag-calc1.u-ga.fr.viaipagssh:)
    loaded via docker load < name.tar

    Custom addition:
    when there are no permissions (flag given by setting the hot account tonoperm),
    copies the parameter files inside the docker (in a directory given by the identifier folder)
    then copies the output xstar files from the docker to the main directory before closure
    '''

# Remove old xstar.par and output files from current directory
    os.system("rm -f xstar.par")
    os.system("rm -f xout*")
    
# Construct new xstar.par file
    file=open('./xstar.par','a')
    xstar1_in='#########################################################################\n\
# File: xstar.par                                                       #\n\
#                                                                       #\n\
# Parameter list for XSTAR package                                      #\n\
#                                                                       #\n\
# Note: This file is related to the xspec2xstar.par file.               #\n\
#       Exercise caution with both when making modifications to one     #\n\
#                                                                       #\n\
# Modifications:                                                        #\n\
# 1/21/1999, WTB: added loopcontrol for multiple runs                   #\n\
#                 installed lower & upper limits                        #\n\
#                                                                       #\n\
# Documentation below is based on documentation by J. Silvis            #\n\
# of RITSS.                                                             #\n\
#                                                                       #\n\
# This file provides information needed by Xanadu Parameter             #\n\
# Interface (XPI) routines to provide input to the ftool xstar.         #\n\
# The entries to a parameter file, such as this one, have               #\n\
# seven columns.\n\
#\n\
# Column   Colunm Function    Comment\n\
#  1       Parameter Name     Alpha-numeric\n\
#\n\
#  2       Data type          s-string\n\
#                             i-integer\n\
#                             r-real\n\
#                             b-boolean\n\
#\n\
#  3       Parameter mode     q-query (asks the user for a value)\n\
#                             h-hidden (does not ask user)\n\
#                             ql-query+learn (remember value\n\
#                                               from last run)\n\
#                             hl-hidden+learn\n\
#                             a-automatic (use value stored in\n\
#                                               the mode parameter)\n\
#\n\
# 4        Default value      If user hits <cr> this is entered\n\
#\n\
# 5        Lower limit        Can be left blank if none is required\n\
#\n\
# 6        Upper limit        Can be left blank if none is required\n\
#\n\
# 7        Prompt             Question printed on screen for user\n\
#\n\
# When parameter mode is set to "a" the line will use the value set by\n\
# the mode statement e.g.\n\
#\n\
# infile,s,a,"her_bfits_1_143.pha",,,"Enter the input files name"\n\
# mode,s,h,"ql",,,""\n\
# is the same as\n\
# infile,s,ql,"her_bfits_1_143.pha",,,"Enter the input files name"\n\
#\n\
# You may want to use this if you need to change several parameters at\n\
# once.\n\
#\n\
# Note on the mode statement.  This is just a regular parameter statement,\n\
# i.e. it sets the value of a string parameter (the first s)\n\
# without prompting the user (the h) and the value is ql.\n\
#\n\
################################################################\n\
# User Adjustable Parameters\n\
#\n'

    file.write(xstar1_in)

    xstar2_in='cfrac,r,h,'+str(par['cfrac'])+',0.0,1.0,"covering fraction"\n\
#\n\
temperature,r,h,'+str(par['temperature'])+',0.0,1.0e4,"temperature (/10**4K)"\n\
#\n\
lcpres,i,h,'+str(par['lcpres'])+',0,1,"constant pressure switch (1=yes, 0=no)"\n\
#\n\
pressure,r,h,'+str(par['pressure'])+',0.0,1.0,"pressure (dyne/cm**2)"\n\
#\n\
density,r,h,'+str(par['density'])+',0.,1.e21,"density (cm**-3)"\n\
#\n\
spectrum,s,h,'+str(par['spectrum'])+',,,"spectrum type?"\n\
#\n\
spectrum_file,s,h,'+str(par['spectrum_file'])+',,,"spectrum file?"\n\
#\n\
spectun,i,h,'+str(par['spectun'])+',0,1,"spectrum units? (0=energy, 1=photons)"\n\
#\n\
trad,r,h,'+str(par['trad'])+',-1.0,0.0,"radiation temperature or alpha?"\n\
#\n\
rlrad38,r,h,'+str(par['rlrad38'])+',0.0,1.e10,"luminosity (/10**38 erg/s)"\n\
#\n\
column,r,h,'+str(par['column'])+',0.0,1.e25,"column density (cm**-2)"\n\
#\n\
rlogxi,r,h,'+str(par['rlogxi'])+',-10.,+10.,"log(ionization parameter) (erg cm/s)"\n\
#\n\
habund,r,h,'+str(par['habund'])+',0.0,100.,"hydrogen abundance"\n\
#\n\
heabund,r,h,'+str(par['heabund'])+',0.0,100.,"helium abundance"\n\
#\n\
liabund,r,h,'+str(par['liabund'])+',0.0,100.,"lithium abundance"\n\
#\n\
beabund,r,h,'+str(par['beabund'])+',0.0,100.,"beryllium abundance"\n\
#\n\
babund,r,h,'+str(par['babund'])+',0.0,100.,"boron abundance"\n\
#\n\
cabund,r,h,'+str(par['cabund'])+',0.0,100.,"carbon abundance"\n\
#\n\
nabund,r,h,'+str(par['nabund'])+',0.0,100.,"nitrogen abundance"\n\
#\n\
oabund,r,h,'+str(par['oabund'])+',0.0,100.,"oxygen abundance"\n\
#\n\
fabund,r,h,'+str(par['fabund'])+',0.0,100.,"fluorine abundance"\n\
#\n\
neabund,r,h,'+str(par['neabund'])+',0.0,100.,"neon abundance"\n\
#\n\
naabund,r,h,'+str(par['naabund'])+',0.0,100.,"sodium abundance"\n\
#\n\
mgabund,r,h,'+str(par['mgabund'])+',0.0,100.,"magnesium abundance"\n\
#\n\
alabund,r,h,'+str(par['alabund'])+',0.0,100.,"aluminum abundance"\n\
#\n\
siabund,r,h,'+str(par['siabund'])+',0.0,100.,"silicon abundance"\n\
#\n\
pabund,r,h,'+str(par['pabund'])+',0.0,100.,"phosphorus abundance"\n\
#\n\
sabund,r,h,'+str(par['sabund'])+',0.0,100.,"sulfur abundance"\n\
#\n\
clabund,r,h,'+str(par['clabund'])+',0.0,100.,"chlorine abundance"\n\
#\n\
arabund,r,h,'+str(par['arabund'])+',0.0,100.,"argon abundance"\n\
#\n\
kabund,r,h,'+str(par['kabund'])+',0.0,100.,"potassium abundance"\n\
#\n\
caabund,r,h,'+str(par['caabund'])+',0.0,100.,"calcium abundance"\n\
#\n\
scabund,r,h,'+str(par['scabund'])+',0.0,100.,"scandium abundance"\n\
#\n\
tiabund,r,h,'+str(par['tiabund'])+',0.0,100.,"titanium abundance"\n\
#\n\
vabund,r,h,'+str(par['vabund'])+',0.0,100.,"vanadium abundance"\n\
#\n\
crabund,r,h,'+str(par['crabund'])+',0.0,100.,"chromium abundance"\n\
#\n\
mnabund,r,h,'+str(par['mnabund'])+',0.0,100.,"manganese abundance"\n\
#\n\
feabund,r,h,'+str(par['feabund'])+',0.0,100.,"iron abundance"\n\
#\n\
coabund,r,h,'+str(par['coabund'])+',0.0,100.,"cobalt abundance"\n\
#\n\
niabund,r,h,'+str(par['niabund'])+',0.0,100.,"nickel abundance"\n\
#\n\
cuabund,r,h,'+str(par['cuabund'])+',0.0,100.,"copper abundance"\n\
#\n\
znabund,r,h,'+str(par['znabund'])+',0.0,100.,"zinc abundance"\n\
#\n\
modelname,s,h,'+str(par['modelname'])+',,,"model name"\n\
#\n\
#####################################################\n\
#  Hidden Parameters\n\
#  Dont mess with these unless you know what you are doing!\n'

    file.write(xstar2_in)

    xstar3_in='nsteps,i,h,'+str(hpar['nsteps'])+',1,1000,"number of steps"\n\
#\n\
niter,i,h,'+str(hpar['niter'])+',0,100,"number of iterations"\n\
#\n\
lwrite,i,h,'+str(hpar['lwrite'])+',0,1,"write switch (1=yes, 0=no)"\n\
#\n\
lprint,i,h,'+str(hpar['lprint'])+',-1,2,"print switch (1=yes, 0=no)"\n\
#\n\
lstep,i,h,'+str(hpar['lstep'])+',0,0,"step size choice switch"\n\
#\n\
emult,r,h,'+str(hpar['emult'])+',1.e-6,1.e+6,"Courant multiplier"\n\
#\n\
taumax,r,h,'+str(hpar['taumax'])+',1.,10000.,"tau max for courant step"\n\
#\n\
xeemin,r,h,'+str(hpar['xeemin'])+',1.e-6,0.5,"minimum electron fraction"\n\
#\n\
critf,r,h,'+str(hpar['critf'])+',1.e-24,0.1,"critical ion abundance"\n\
#\n\
vturbi,r,h,'+str(hpar['vturbi'])+',0.,30000.,"turbulent velocity (km/s)"\n\
#\n\
radexp,r,h,'+str(hpar['radexp'])+',-3.,3.,"density distribution power law index"\n\
#\n\
ncn2,i,h,'+str(hpar['ncn2'])+',999,999999,"number of continuum bins"\n\
#\n\
loopcontrol,i,h,'+str(hpar['loopcontrol'])+',0,30000,"loop control (0=standalone)"\n\
#\n\
npass,i,h,'+str(hpar['npass'])+',1,10000,"number of passes"\n\
#\n\
mode,s,h,'+str(hpar['mode'])+',,,"mode"\n'

    file.write(xstar3_in)
    file.close()
    
# Run XSTAR with new xstar.par

    def docker_runner(command):
        '''
        wrapping the redirection to avoid repetition
        '''
        with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, \
                              universal_newlines=True) as p:
            for line in p.stdout:
                print(line, end='')  # process line here
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, p.args)

    if host_account == "noperm":

        # The docker should only be created on the first run
        #identifying the docker id
        docker_list=str(subprocess.check_output("docker ps", shell=True)).split('\\n')
        docker_list_mask=[elem.endswith('xstar_'+identifier) for elem in docker_list]

        if sum(docker_list_mask)==0:
            #calling the docker with no mounts
            print("no running xstar docker detected. Creating it...")

            subprocess.call(['docker','run','--name','xstar_'+identifier,'-dt',container,'bash'])

        #identifying the docker id
        docker_list=str(subprocess.check_output("docker ps", shell=True)).split('\\n')
        docker_list_mask=[elem.endswith('xstar_'+identifier) for elem in docker_list]

        docker_id=np.array(docker_list)[docker_list_mask][0].split()[0]

        #creating the custom folder for the current xstar run
        subprocess.call(['docker','exec','-t','xstar_'+identifier,'bash','-c',\
            'mkdir -p '+identifier])

        #copying the pfiles into that directory (with no write access but that's not an issue)
        #the docker cp prints stuff that we can use to check if things have gone correctly
        for elem_file in ['xstar.par',par['spectrum_file']]:
            #skipping empty spectrum files if there is none used
            if elem_file is not None:
                docker_runner(['docker','cp',os.path.join(os.getcwd(),elem_file),
                               docker_id+':/home/heasoft/'+identifier])

        xstar_cmd=['docker','exec','-t','xstar_'+identifier,'bash','-c',\
            'export PFILES=/home/heasoft/'+identifier+' && cd '+identifier+' && xstar']

        docker_runner(xstar_cmd)

        #list of the standard xstar output files
        files_output_list=['xout_abund1.fits','xout_rrc1.fits','xout_cont1.fits',
                           'xout_spect1.fits','xout_lines1.fits','xout_step.log']

        # copying the output files to the outside directory
        for elem_file in files_output_list:
            docker_runner(['docker','cp',docker_id+':'+os.path.join('/home/heasoft/',identifier,elem_file),
                          os.getcwd()])

        #note: we don't kill the xstar runner here to avoid issues with multiple process accessing it

    elif host_account == "nb@000.000":
        subprocess.call(['docker','run','--name','xstar','-dt','-v',\
            os.getcwd()+':/mydata','-w','/mydata',container,'bash'])
        docker_cmd=['docker','exec','-t','xstar','bash','-c',\
            'export PFILES=/mydata && xstar']

        with subprocess.Popen(docker_cmd,stdout=subprocess.PIPE,bufsize=1,\
                 universal_newlines=True) as p:
            for line in p.stdout:
                print(line,end='') # process line here
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode,p.args)

        subprocess.call(['docker','container','rm','--force','xstar'])
    else:
        subprocess.call(['ssh',host_account,'rm -f /home/claudio/pfiles/*'])
        subprocess.call(['scp','./xstar.par',\
            host_account+':/home/claudio/pfiles'])
        os.environ['DOCKER_HOST']='ssh://' + host_account
        print(os.environ.get('DOCKER_HOST'))
        subprocess.call(['docker','run','--name','xstar','-dt','-v',\
            '/home/claudio/pfiles:/mydata','-w','/mydata',container,'bash'])
        subprocess.call(['docker','exec','-t','xstar','bash','-c',\
            'export PFILES=/mydata && xstar'])
        subprocess.call(['docker','container','rm','--force','xstar'])
        subprocess.call(['scp',host_account+':/home/claudio/pfiles/xout_*','.'])
        subprocess.call(['ssh',host_account,'rm -f /home/claudio/pfiles/*'])
        del os.environ['DOCKER_HOST']
##############################################################################
