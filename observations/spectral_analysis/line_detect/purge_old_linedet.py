#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:52:10 2022

@author: parrama
"""

import glob
import os
import time

alldir=glob.glob('**/',recursive=True)

currdir=os.getcwd()


for elem in alldir:
    if elem.endswith('bigbatch/'):
        os.chdir(elem)
        bigbatchdirs=glob.glob('**/')
        for bigbatchdir in bigbatchdirs:
            if bigbatchdir.startswith('lineplots') and not (bigbatchdir.startswith('lineplots_multisp_pnv5')\
                    or bigbatchdir.startswith('lineplots_newcont')\
                        or bigbatchdir.startswith('lineplots_multisp_pnv6')) :
                
                print('deleting directory '+os.path.join(os.getcwd(),bigbatchdir))
                
                os.system('rm -rf '+bigbatchdir)
                time.sleep(1)
    os.chdir(currdir)
    
    