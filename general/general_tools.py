#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:59:26 2023

@author: parrama
"""

import numpy as np
from matplotlib.ticker import Locator
import time
import os

import random
import glob

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

def get_overlap(a, b,distance=False):
    #compute overlap between two intervals. if distance is set to true, returns the distance between the
    #intervals if they are disjoint
    if distance:
        return min(a[1], b[1]) - max(a[0], b[0])
    else:
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def shorten_epoch(file_ids_init):

    if file_ids_init=='auto':
        from xspec import AllData
        file_ids=[AllData(i+1).fileName.split('_sp')[0] for i in range(AllData.nGroups)]
    else:
        file_ids=file_ids_init

    # splitting obsids
    # obsids_list = [elem.split('-')[0] for elem in file_ids]

    #we split some weirder names, like bat files, to avoid splitting them incorrectly
    obsids = np.unique([elem.split('-')[0] for elem in file_ids if 'survey_point' not in elem])
    bat_survey_files=[elem for elem in file_ids if 'survey_point' in elem]

    if len(bat_survey_files)==0:
        # returning the obsids directly if there's no gtis in the obsids
        obsids_ravel = ''.join(file_ids)
        if '-' not in obsids_ravel:
            return file_ids

    #only adding one element of the bat survey file because we will add them all at once
    obsids=obsids.tolist()+bat_survey_files[:1]

    # according the gtis in a shortened way
    epoch_str_list = []
    for elem_obsid in obsids:

        if 'survey_point' in elem_obsid:

            #computing how much of the pointings can be compacted
            survey_ids = [elem.split('_survey_point_')[1].split('.')[0] for elem in bat_survey_files]

            if len(survey_ids)==1:
                epoch_str_list+=survey_ids
            else:
                survey_ids_char = np.array([[subelem for subelem in elem] for elem in survey_ids])
                survey_id_common_char=np.argwhere([len(np.unique(elem)) != 1 for elem in survey_ids_char.T]).T[0][0]

                #and compacting
                str_obsid=survey_ids[0][:survey_id_common_char]+'-'.join(elem[survey_id_common_char:] for elem in survey_ids)

                #recognizing BAT_analysis point, and adding them accordingly
                epoch_str_list+=[str_obsid]
            continue

        str_gti_add = ''

        str_obsid = elem_obsid

        if '-' in ''.join([elem for elem in file_ids if elem.startswith(elem_obsid)]):
            str_gti_add = '-' + '-'.join([elem.split('-')[1] for elem in file_ids if \
                                          elem.startswith(elem_obsid)])

        epoch_str_list += [str_obsid + str_gti_add]

    return epoch_str_list


# not needed for now
def expand_epoch(shortened_epochs):

    '''
    Takes an array as argument so split the '_' joined short_id beforehand
    '''

    if type(shortened_epochs) in (str,int):
        shorten_list=[str(shortened_epochs)]
    else:
        shorten_list=shortened_epochs
    # splitting obsids
    file_ids = []

    for short_id in shorten_list:
        if short_id.count('-') <= 1:
            file_ids += [short_id]
        else:
            obsid = short_id.split('-')[0]
            gti_ids = short_id.split('-')[1:]

            file_ids += ['-'.join([obsid, elem_gti]) for elem_gti in gti_ids]

    return file_ids

def str_orbit(i_orbit):
    '''
    return regular str expression of orbit
    '''
    return ('%3.f' % (i_orbit + 1)).replace(' ', '0')
def rescale_flex(ax,xlims,ylims,margin,std_x=None,std_y=None):

    '''

    std_x and std_y are bounds to use as default if the xlim/ylim values are smaller

    variant for negative symlog (aka with 0 as max value) not implemented
    '''


    
    if ax.get_xscale()=='linear':
        
        xrange = xlims[1] - xlims[0]

        del_xrange = xrange * margin
        
        if std_x is not None:
            ax.set_xlim((min(xlims[0]- del_xrange,std_x[0]),max(xlims[1]+ del_xrange,std_x[1])))
        else:
            ax.set_xlim((xlims[0]-del_xrange, xlims[1]+del_xrange))
            
    elif ax.get_xscale() in ['log','symlog']:

        xrange = xlims[1] / (ax.get_xticks()[1] if ax.get_xscale() == 'symlog' else xlims[0])

        del_xrange = np.log10(xrange) * margin
        
        if std_x is not None:
            ax.set_xlim((min(xlims[0]*10**(-del_xrange),std_x[0]),max(xlims[1]*10**(del_xrange),std_x[1])))
        else:
            ax.set_xlim((xlims[0] * 10 ** (-del_xrange)), xlims[1] * 10 ** (del_xrange))

    if ax.get_yscale() == 'linear':

        yrange = ylims[1] - ylims[0]

        del_yrange = yrange * margin

        if std_y is not None:
            ax.set_ylim((min(ylims[0] - del_yrange, std_y[0]), max(ylims[1] + del_yrange, std_y[1])))
        else:
            ax.set_ylim((ylims[0] - del_yrange, ylims[1] + del_yrange))

    elif ax.get_yscale() in ['log', 'symlog']:

        yrange = ylims[1] / (ax.get_yticks()[1] if ax.get_yscale() == 'symlog' else ylims[0])

        del_yrange = np.log10(yrange) * margin

        if std_y is not None:
            ax.set_ylim((min(ylims[0] * 10 ** (-del_yrange), std_y[0]), max(ylims[1] * 10 ** (del_yrange), std_y[1])))
        else:
            ax.set_ylim((ylims[0] * 10 ** (-del_yrange)), ylims[1] * 10 ** (del_yrange))

def ravel_ragged(array,mode=None,ragtuples=True):

    '''
    ravels a multi dimensional ragged nested sequence
    only considers tuples if ragtuples is set to True
    '''

    list_elem=[]

    for elem in array:
        if type(elem) in [np.ndarray,list] + ([tuple] if ragtuples else []):
            if len(elem)!=0:
                list_elem+=ravel_ragged(elem,mode=None,ragtuples=ragtuples).tolist()
            else:
                continue
        else:
            list_elem+=[elem]

    #necessary to avoid converting tuples into arrays if ragtuples is not set to True
    if np.all(type(elem)==tuple for elem in list_elem) and not ragtuples:
        return_arr=np.array([None]*len(list_elem))
        for i_elem in range(len(list_elem)):
            return_arr[i_elem]=list_elem[i_elem]
        return return_arr

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

    '''
    Complicated mess to avoid several edits to a single file with i.e. parallel processing
    '''

    path_dir = '/'.join(path.split('/')[:-1])
    file_temp = path.split('/')[-1]
    file_temp_prefix = file_temp[:file_temp.rfind('.')]

    import random
    random_n = random.random()

    path_temp = os.path.join(path_dir, file_temp_prefix + '_temp_' + str(random_n) + file_temp[file_temp.rfind('.'):])

    # creating the temp_file
    with open(path_temp, 'w+') as f:
        pass

    # waiting 1 second (so that in case of parallel procs, all the temp files are created
    time.sleep(1)

    curr_files_temp = [elem for elem in glob.glob(os.path.join(path_dir, '**')) if
                       file_temp_prefix + '_temp' in elem.split('/')[-1]]

    curr_files_temp.sort()

    iter_concu=0

    while len(curr_files_temp) > 1:

        # waiting if this is not the first of the temp file, otherwise going forward
        if path_temp == curr_files_temp[0]:
            break

        time.sleep(0.1)
        curr_files_temp = [elem for elem in glob.glob(os.path.join(path_dir, '**')) if
                           file_temp_prefix + '_temp' in elem.split('/')[-1]]

        curr_files_temp.sort()

        iter_concu+=1

        assert iter_concu<30,'Issue with '+path+' overwriting. Check for dead temps'
    #only reading the file after the wait to get the last version

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


    with open(path_temp,'w+') as file:
        #adding the header lines
        
        if lines[:len(header_eff)]==header_eff:
            file.writelines(lines)
        else:
            file.writelines(header_eff+lines)

    if os.path.isfile(path):
        os.remove(path)
    os.system('mv '+path_temp+' '+path)

def print_log(elem,logfile_io,silent=False):

    '''
    prints and logs at once
    '''

    if not silent:
        print(elem)

    if logfile_io is not None:
        logfile_io.write(str(elem)+('\n' if not str(elem).endswith('\n') else ''))

class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh, nints=10):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        dmlower = majorlocs[1] - majorlocs[0]    # major tick difference at lower end
        dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) \
                                   or (dmlower == self.linthresh and majorlocs[0] < 0)):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) \
                                    or (dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                          '%s type.' % type(self))