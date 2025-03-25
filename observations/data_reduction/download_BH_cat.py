#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:26:56 2021

Various functions to download BH catalogs data in XMM/Chandra


"""
import argparse
import numpy as np
import pandas as pd
import os
import glob
import pexpect
import sys

#pdf table reader
import tabula
import time

#astro
from mpdaf.obj import sexa2deg


#Catalogs and manipulation
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.esa.xmm_newton import XMMNewton

ap = argparse.ArgumentParser(description='Script to downlaod XMM Observations of GBH catalogs.\n)')

#general options
ap.add_argument('-startdir',nargs=1,help='directory where to list/download data',default='./',type=str)

ap.add_argument('-mode',nargs=1,help='mode to use',default='regroup',type=str)

ap.add_argument("-catal",nargs=1,help='catalog to download (blackcat or watchdog)',default='watchdog',type=str)

ap.add_argument('-telescope',nargs=1,help='telescope to create the scripts for (Chandra or XMM)',default='Chandra',type=str)

ap.add_argument('-download_obs',nargs=1,help='download XMM obsids on top of listing them',default=False,type=bool)

ap.add_argument('-watchdog_pdf_path',nargs=1,help='pdf paper of the APJ (not preprint) version of the watchdog paper',
                default='/home/parrama/Documents/Work/PhD/docs/papers/WATCHDOG/Tetarenko_2016_ApJS_222_15.pdf',type=str)

args=ap.parse_args()
mode=args.mode

catal=args.catal
startdir=args.startdir
download_obs=args.download_obs
watchdog_pdf_path=args.watchdog_pdf_path
telescope=args.telescope

exceptions_blackCAT=[]
exceptions_watchdog=['XTE J0421+560','4U 0538-641','4U 0540-697','IGR J11321-5311','IGR J17379-3747','XTE J1812-182','SS 433','4U 1956+350',
                     '4U 2030+40','MWC 656']
'''FUNCTIONS'''

'''Exporting BLACKCAT objects'''

if mode!='regroup':
    
    catal_blackcat=pd.read_html('https://www.astro.puc.cl/BlackCAT/transients.php')[0]
    catal_blackcat_obj=catal_blackcat['Name (Counterpart)'].to_numpy()

#separating the different names given to each object

def export_blackcat_ids(path='./',write=True):
    
    catal_blackcat_ids=np.array([None]*len(catal_blackcat_obj))
    
    '''
    separates the object names for each BHT in the blackCAT catalog and stores them into a file in the path directory
    '''
        
    #not perfect but doing more would be much more complicated
    
    for i in range(len(catal_blackcat_obj)):
        curr_names=catal_blackcat_obj[i]
        #'some names are splitted by an ='
        curr_names=curr_names.split(' = ')
        
        curr_ids=[]
        for names in curr_names:
            
            #others are separated by spaces
            name_parts=names.split(' ')
            
            #however since most names are themselves composed of two elements separated by a space, we need to group them 2 by 2
            if len(name_parts)>1:
                for j in (np.arange(np.floor(len(name_parts)/2))*2).astype(int):
                    
                    #we join the elements two by two, and also the third one if we are arriving to the end of the array and it is of an uneven length
                    #which in our catalog often means the last name has 3 ids separated by 2 spaces
                    full_name=' '.join((name_parts[j],name_parts[j+1],name_parts[j+2] if j==len(name_parts)-3 else ''))
                    
                    #deleting the beginning and end spaces if they exist
                    full_name=full_name.strip()
                    
                    #We add an exception for MAXI J1820 which gets a double name with this process
                    if 'MAXI J1820+070' in full_name:
                        #manual dichotomy
                        curr_ids+=[full_name[:14],full_name[15:]]
                    curr_ids+=[full_name]
            else:
                curr_ids+=name_parts
                
        catal_blackcat_ids[i]=curr_ids
    
    if write:
        with open(os.path.join(path,'blackCAT_names.csv'),'w') as catalfile:
            for i in range(len(catal_blackcat_ids)):
                catalfile.writelines('\t'.join(catal_blackcat_ids[i])+'\n')
    return catal_blackcat_ids

def export_blackcat_coords(path='./',write=True,format='radec'):
    
    '''
    Writes a file or exports an array with the coordinates of all the objects in RA/DEC format
    '''
    catal_blackcat_coords=np.array([catal_blackcat['RA [hh:mm:ss]'].to_numpy(),catal_blackcat['DEC [dd:mm:ss]'].to_numpy()]).T
    
    if write:
        with open(os.path.join(path,'blackCAT_coords.csv'),'w') as catalfile:
            for i in range(len(catal_blackcat_coords)):
                catalfile.writelines(','.join(catal_blackcat_coords[i].tolist())+'\n')
    return catal_blackcat_coords

'''Exporting WATCHDOG objects'''

#The Vizier catalog entry is cropped to 50 objects so we fetch the table from the pdf instead

'''Note : We only extract objects NOT in BlackCAT here'''

Vizier.ROW_LIMIT=-1
catal_watchdog=Vizier.get_catalogs('watchdog')[0]

def export_watchdog_ids(path='./',write=True):

    '''
    Exports the name of each Watchdog source which is NOT in BlackCAT in a file/array
    '''
        
    catal_watchdog_notBlackCAT_ids=[]
    
    catal_blackCAT_objs=np.array(catal_blackcat['Name (Counterpart)'])
    
    #testing if each watchdog id is in blackcat (being part of any blackcat id string)
    for name_watchdog in catal_watchdog['Name']:
        
        print('\nChecking source '+name_watchdog+'...')
        
        #since some of the source's main names don't register in Simbad, we also try the other names and add a fail test
        
        if Simbad.query_object(name_watchdog) is not None:
            id_watchdog=name_watchdog
        else:
            id_watchdog=Simbad.query_objects(catal_watchdog['OName'][catal_watchdog['Name']==name_watchdog][0].split(';'))[0]['main_id']
            if id_watchdog is None:
                print("\nNone of the object's names were found in Simbad. Switching to manual input.")
                breakpoint()
        
        in_blackCAT=False
        for name_blackCAT in catal_blackCAT_objs:
            #manipulating the blackCAT first names a bit so Simbad can read them
            id_blackCAT=' '.join(name_blackCAT.split('=')[0].split(' ')[:2])
            
            if Simbad.query_object(id_watchdog)[0]['main_id']==Simbad.query_object(id_blackCAT)[0]['main_id']:
                in_blackCAT=True
                print('\nSource found in BlackCAT under the name '+id_blackCAT)
                break
        
        #if it's not the case, we add them to our list
        if not in_blackCAT:
            print('\nSource not found in BlackCAT. Adding it to the list...')
            catal_watchdog_notBlackCAT_ids+=[name_watchdog]
            
    #note: this doesn't work perfectly but we check the rest by hand anyway
    
    if write:
        with open(os.path.join(path,'WATCHDOG_notblackCAT_names_unchecked.csv'),'w') as catalfile:
            for name in catal_watchdog_notBlackCAT_ids:
                catalfile.writelines(name+'\n')

    return catal_watchdog_notBlackCAT_ids

def export_watchdog_coords(catal_ids,path='./',write=True,format='radec'):
    
    '''
    Writes a file or exports an array with the coordinates of all the objects in RA/DEC format
    '''
    
    catal_watchdog_notBlackCAT_coords=np.array([[None,None]]*len(catal_ids))
    
    for ind,name in enumerate(catal_ids):
        catal_watchdog_notBlackCAT_coords[ind][0]=catal_watchdog['RAJ2000'][catal_watchdog['Name']==name][0].replace(' ',':')
        catal_watchdog_notBlackCAT_coords[ind][1]=catal_watchdog['DEJ2000'][catal_watchdog['Name']==name][0].replace(' ',':')

    if write:
        with open(os.path.join(path,'WATCHDOG_notblackCAT_coords_unchecked.csv'),'w') as catalfile:
            for ind in range(len(catal_ids)):
                catalfile.writelines(','.join(catal_watchdog_notBlackCAT_coords[ind])+'\n')

def create_arborescence(catal_names,catal_coords,path='./',maxd=18,download=False):
    
    '''
    Creates subfolders for each object in the path folder.
    Creates a csv with the list of available XMM obsids for each object.
    maxd corresponds to the maximal distance for the fov in arcminutes. 18 is the default value in the XMMSA search
    The object directory names are the first identifier for each object.
    If download is set to true, downloads the obsid ODF files in set obsid directories
    '''
    
    os.chdir(path)
    
    startdir=os.getcwd()
    
    for i_obj in range(len(catal_names)):
        if type(catal_names[i_obj])==list:
            curr_name=catal_names[i_obj][0].replace(' ','')
        else:
            curr_name=catal_names[i_obj].replace(' ','')
        #creating the object subdirectory
        os.system('mkdir -p '+curr_name)

        #since the XSA tap service only accepts decimal (ICRS) coordinates, we transform ours in this format
        #the inversion is due to mpdaf handling coordinates in [dec,ra] format
        curr_coords=sexa2deg(catal_coords[i_obj][::-1])[::-1]

        # #deleting any previous file for overwrite purposes
        if os.path.isfile(curr_name+'/obsids_list.csv'):
            os.remove(curr_name+'/obsids_list.csv')
            
        #creating the object index file
        #for this we use curl to retrieve tables from the XSA database (c.f. http://nxsa.esac.esa.int/nxsa-web/#tap)        
        curl_str='curl -o '+curr_name+'/obsids_list.csv '+\
                 '"http://nxsa.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&'+\
                 'QUERY=SELECT+observation_id+FROM+xsa.v_all_observations+WHERE+1=intersects'+\
                 '(observation_fov_scircle,circle('+"'ICRS'"+','+str(curr_coords[0])+','+str(curr_coords[1])+','+str(maxd/60)+'))"'
        os.system(curl_str)
        
        if download==True:
            with open(curr_name+'/obsids_list.csv') as curr_obsid_file:
                #reading each obsid for the object
                curr_obsid_list=curr_obsid_file.readlines()[1:]
            for line in curr_obsid_list:
    
                #formatting the actual obsid
                obsid=line[1:-2]
                
                #creating the subdirectories
                obsdir=os.path.join(curr_name,obsid)+'/'
                
                #the done file is created ad the end of the process
                if os.path.isfile(obsdir+'done'):
                    print('Obsid '+obsid+' files for object '+curr_name+' already downloaded.')
                    continue
                
                os.system('mkdir -p '+os.path.join(curr_name,obsid))
                
                #downloading the ODF data
                
                try:
                    XMMNewton.download_data(obsid,level='ODF',filename=obsdir+obsid+'_query.tar')
                    
                    #extracting the tars
                    os.chdir(obsdir)
                    os.system('tar -xvf '+obsid+'_query.tar.gz')
                    os.system('tar -xvf *.TAR')
                    os.system('rm *.TAR')
                    os.system('rm '+obsid+'_query.tar.gz')
                    os.system('cat done>done')
                    
                    os.chdir(startdir)
                except:
                    pass

def export_all_Simbad_ids(catal_name,exceptions,ids=None,path='./',write=True):
    
    '''
    Export All Simbad IDs of the BlackCAt and Watchdog sources in a file/variable to use as a query in the tgcat catalog
    (http://tgcat.mit.edu/)
    '''
    if catal_name=='blackcat':
        
        ids=np.array(catal_blackcat['Name (Counterpart)'])
        
        for i in range(len(ids)):
            #manipulating the blackCAT first names a bit so Simbad can read them
            ids[i]=' '.join(ids[i].split('=')[0].split(' ')[:2])
    
    
    Simbad_ids=[]
    for name in ids:
        if name not in exceptions:
            if Simbad.query_object(name) is not None:
                Simbad_ids+=[Simbad.query_object(name)[0]['main_id']]
            else:
                print('\nCould not find a Simbad ID for object '+name)
                
    if catal_name=='blackcat':
        idfile_str='blackCAT_Simbad_ids.csv'
    elif catal_name=='watchdog':
        idfile_str='WATCHDOG_notblackCAT_Simbad_ids.csv'
    if write:
        with open(os.path.join(path,idfile_str),'w') as catalfile:
            for Simbad_name in Simbad_ids:
                catalfile.writelines(Simbad_name+'\n')
                   

    
'''MAIN DECISION TREE'''

if mode=='regroup':
    sys.exit()

if telescope=='XMM':
    if catal=='blackcat':  
        blackCAT_ids=export_blackcat_ids(path=startdir)
        blackCAT_coords=export_blackcat_coords(path=startdir)
        if download_obs:
            create_arborescence(blackCAT_ids, blackCAT_coords,path=startdir,download=True)
    
    if catal=='watchdog':
        watchdog_ids=export_watchdog_ids(path=startdir)
        export_watchdog_coords(watchdog_ids,path=startdir)
        
        #we expect the user to have manually curated the catalog afterwards and created two new files that we can parse
        with open(startdir+'WATCHDOG_notblackCAT_names_curated.csv') as namefile:
            watchdog_ids_curated=namefile.readlines()
        
        #reading the coords file
        with open(startdir+'WATCHDOG_notblackCAT_coords_curated.csv') as namefile:
            watchdog_coords_curated=namefile.readlines()
            
        #stripping the line skips
        for ind,elem in enumerate(watchdog_ids_curated):
            watchdog_ids_curated[ind]=elem.strip('\n')
            
        #same and adding splitting RA and DEC
        for ind,elem in enumerate(watchdog_coords_curated):
            watchdog_coords_curated[ind]=elem.strip('\n').split(',')
            
        #switching the coordinates in an array
        if download_obs:
            create_arborescence(watchdog_ids_curated, watchdog_coords_curated,path=startdir,download=True)

elif telescope=='Chandra':
    if catal=='blackcat':
        export_all_Simbad_ids('blackcat',exceptions_blackCAT)
    elif catal=='watchdog':
        watchdog_ids=export_watchdog_ids(path=startdir,write=False)
        export_all_Simbad_ids('watchdog',exceptions_watchdog,watchdog_ids)
        