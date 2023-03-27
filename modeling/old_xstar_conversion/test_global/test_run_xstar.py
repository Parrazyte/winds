#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:08:20 2023

@author: parrama
"""

        
import gfort2py as gf
import os
import glob
import time


def make_shared_lib():
    
    '''
    builds a shared library for xstar from all .o files in the current arborescence
    excludes funpack.o which creates issues
    '''
    
    list_dotso=glob.glob('**/*.o',recursive=True)
    str_dotso=' '.join(list_dotso)
    
    #excluding this one because the compiler doesn't like it very much
    str_dotso=str_dotso.replace('cfitsio/funpack.o ','')
    
    breakpoint()
    
    os.system('gfortran -fpic -shared -o test.so '+str_dotso)
    
    os.system('cp test.so ../')
    
def compile_zlib():
    '''
    compiles all the .c files in zlib
    '''
    
    list_c=glob.glob('*.c',recursive=True)
    
    for elem in list_c:
        os.system('gcc -c -fPIC '+elem)
        time.sleep(1)
        
    
def load_lib():

    
    SHARED_LIB_NAME='./test.so'
    MOD_FILE_NAME='xstar_mod.mod'
    
    x=gf.fFort(SHARED_LIB_NAME,MOD_FILE_NAME)
    

