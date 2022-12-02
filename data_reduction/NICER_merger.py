#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 16:47:29 2022

@author: parrama
"""

import glob
import pexpect
import os
import numpy as np

currdir=os.getcwd()

listfiles=glob.glob('**')

obsidlist=np.unique([elem.split('_')[0][2:] for elem in listfiles if elem.startswith('ni5') ])

bashproc=pexpect.spawn('/bin/bash',encoding='utf-8')
bashproc.sendline('cd '+currdir)
bashproc.sendline('heainit')

for obsid in obsidlist:
    listfiles_sp=glob.glob('ni'+obsid+'*.grp')
    listfiles_bg=glob.glob('ni'+obsid+'*.bg3c50')
    
    
    with open('list_file.txt','w+') as file:
        file.write(' '.join(listfiles_sp))
        file.write('\n')
        file.write(' '.join(listfiles_bg))
    
    bashproc.sendline('addascaspec list_file.txt '+obsid+'.pha '+obsid+'_bg.pha')
    