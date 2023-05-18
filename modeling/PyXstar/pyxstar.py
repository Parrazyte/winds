#!/usr/bin/env python
# coding: utf-8

import os, sys
#import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
#import numpy as np
import math
import subprocess

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
"spectrum_file":"spct.dat", #"spectrum file?"
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

def run_xstar(par,hpar):
# Function to create the XSTAR input file xstar.par and run the code from 
# standard Heasoft installation

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

    with subprocess.Popen("xstar",stdout=subprocess.PIPE,bufsize=1,\
                          universal_newlines=True) as p:
        for line in p.stdout:
            print(line,end='') # process line here
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode,p.args)

#########################################################################

def docker_run_xstar(par,hpar,\
    container="heasoft:v6.31.1",\
    host_account="nb@000.000"):

# Function to create the XSTAR input file xstar.par and run the code from
# the XSTAR Docker container

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

    if host_account == "nb@000.000":
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
