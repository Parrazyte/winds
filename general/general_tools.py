#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:59:26 2023

@author: parrama
"""

import numpy as np

import os

# def ravel_ragged_test(array,mode=None):
#
#     '''ravels a 2/3d array/list even with ragged nested sequences'''
#
#     #leaving if the array is 0d
#
#     if type(array) not in [np.ndarray,list] or len(array)==0:
#         return array
#     #or 1d
#     if type(array[0]) not in [np.ndarray,list]:
#         return array
#
#     #testing if the array is 3d
#     if type(array[0][0]) in [np.ndarray,list]:
#         return np.array([array[i][j][k] for i in range(len(array)) for j in range(len(array[i])) for k in range(len(array[i][j]))],dtype=mode)
#     else:
#         return np.array([array[i][j] for i in range(len(array)) for j in range(len(array[i]))],dtype=mode)

def ravel_ragged(array,mode=None):

    '''ravels a multi dimensional ragged nested sequence'''

    list_elem=[]

    for elem in array:
        if type(elem) in (np.ndarray,list,tuple):
            if len(elem)!=0:
                list_elem+=ravel_ragged(elem,mode=None).tolist()
            else:
                continue
        else:
            list_elem+=[elem]
    return np.array(list_elem,dtype=mode)

def interval_extract(list):

    '''
    From a list of numbers, outputs a list of the integer intervals contained inside it
    '''

    if len(list)==0:
        return list

    list = sorted(set(list))
    range_start = previous_number = list[0]

    for number in list[1:]:
        if number == previous_number + 1:
            previous_number = number
        else:
            yield [range_start, previous_number]
            range_start = previous_number = number
    yield [range_start, previous_number]

def file_edit(path,line_id,line_data,header):
    
    '''
    Edits (or create) the file given in the path and replaces/add the line(s) where the line_id str/LIST is with
    the line-content str/LIST.
    
    line_id should be included in line_content.
    
    Header is the first line (or lines if it's a list/array) of the file, with usually different informations.
    '''
    
    lines=[]
    if type(line_id)==str or type(line_id)==np.str_:
        line_identifier=[line_id]
    else:
        line_identifier=line_id
        
    if type(line_data)==str or type(line_data)==np.str_:
        line_content=[line_data]
    else:
        line_content=line_data
        
    if os.path.isfile(path):
        with open(path) as file:
            lines=file.readlines()
            
            #loop for all the lines to add
            for single_identifier,single_content in zip(line_identifier,line_content):
                line_exists=False
                if not single_content.endswith('\n'):
                    single_content+='\n'
                    
                #loop for all the lines in the file
                for l,single_line in enumerate(lines):
                    if single_line.startswith(single_identifier):
                        
                        lines[l]=single_content
                        line_exists=True
                if line_exists==False:
                    lines+=[single_content]
                
    else:
        #adding everything
        lines=line_content

    #here we use 0 as an identifier for a non line
    
    if type(header) is np.ndarray:
        header_eff=header.tolist()
    elif type(header) is list:
        header_eff=header
    else:
        header_eff=[header]
    
    with open(path,'w+') as file:
        #adding the header lines
        
        if lines[:len(header_eff)]==header_eff:
            file.writelines(lines)
        else:
            
            file.writelines(header_eff+lines)
            
def print_log(elem,logfile_io,silent=False):

    '''
    prints and logs at once
    '''

    if not silent:
        print(elem)

    if logfile_io is not None:
        logfile_io.write(str(elem)+('\n' if not str(elem).endswith('\n') else ''))