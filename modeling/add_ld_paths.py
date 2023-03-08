#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:44:32 2023

@author: parrama
"""

import os

headas=os.environ['HEADAS']
path_1=headas+'/lib/libcfitsio.so'
path_2=headas+'/lib/libxanlib_6.31.so'
path_3=headas+'/lib/libape_2.9.so'

os.environ['LD_LIBRARY_PATH']+=":"+path_1
os.environ['LD_LIBRARY_PATH']+=":"+path_2
os.environ['LD_LIBRARY_PATH']+=":"+path_3