import os,sys
import argparse
import logging
import glob
import time
'''
creates clean Chandra arborescence by copying the ph2 files and the responses obsid names in bigbatch directories for all sources

'''

import argparse
import numpy as np
import pandas as pd
import os
import glob
import pexpect
import sys

#pdf table reader
# import tabula

import time

def arrange_Chandra_arborescence():
    
    '''
    obsid_list file can be made manually by going to the extraction table on TGCat and copying the text
    
    opens the obsid_list file, reads the columns 2 and 3 as obsid and object, and reorganises the files in individual obsid folders 
    in object folders
    '''
    
    obsid_list=pd.read_csv('obsid_list.csv',sep='\t',header=None)
    
    subdirs=np.array(glob.glob('**/',recursive=True))
    
    #fetching the obsid and sources
    for ind in range(len(obsid_list)):
        obsid=obsid_list[2][ind]
        source=obsid_list[3][ind].replace(' ','')
        os.system('mkdir -p '+source)
        download_dir=[elem for elem in subdirs if '_'+str(obsid)+'_' in elem]
        if len(download_dir)!=1:
            
            '''Manual check in this case but that shouldn't happen'''     
            
            breakpoint()
        else:
            os.system('mv '+download_dir[0]+' '+source+'/'+str(obsid))

def merge_chandra_spectra(startdir='./',mode='add_obsid'):
    
    '''
    
    Creates arborescence for Chandra files from a tgcat unzipped arborescence

    Note: need to remove the tgid before runnning this from individual obs directory names now
    '''
    
    if not os.path.isdir(startdir):
        logging.error('Invalid directory, please start over.')
        return 0
    else:
        os.chdir(startdir)
        
    startdir=os.getcwd()
    
    allfiles=glob.glob('**',recursive=True)
    
    # # if need to clean
    # alldirs=glob.glob('**/',recursive=True)
    # for direc in alldirs:
    #     if direc.endswith('bigbatch/'):
    #         print('aa')
    #         os.chdir(direc)
    #         os.system('rm -f *')
    #         os.chdir(startdir)
    
    #solving mistakes        
    # for filepath in allfiles:
    #     if not filepath.split('/')[-1].endswith('.gz'):
    #         continue
    #     filename=filepath.split('/')[-1]
    #     filename=filename.replace('heg1','heg_1').replace('heg-1','heg_-1').replace('meg1','meg_1').replace('meg-1','meg_-1')
    #     lastdir=filepath.split('/')[-2]
    #     filename=filename.replace(lastdir,'')
    #     os.rename(filepath,'/'.join(filepath.split('/')[:-1])+'/'+filename)
        
    
        
    if mode=='add_obsid':
        for elem in allfiles:

            #fetching positions of grouped files
            if elem.endswith('/pha2.gz'):
                
                #if yes, we get the obsid from the directory paths        
                if elem.rfind('/')!=-1:
                    dirpath=elem[:elem.rfind('/')]
                else :
                    dirpath='./'
                    
                lastdir=dirpath.split('/')[-1]

                #doing this to comply with both the old and new namings of the obsid files
                if '_' in lastdir:
                    dir_obsid=lastdir.split('_')[1]
                else:
                    dir_obsid=lastdir

                #now we check if those files have an obsid shape (10 numbers)
                if len(dir_obsid)<=6 and dir_obsid.isdigit():
                    print('\nFound one pha file in obsid directory: '+elem)
                    
                    #going in the directory
                    os.chdir(dirpath)
                    
                    #creating bigbatch directory along the obsid directories
                    os.system('mkdir -p ../bigbatch')
                    
                    #copying the response files
                    os.system('cp -f heg*.gz ../bigbatch')
                    #copying the pha file
                    os.system('cp -f pha2.gz ../bigbatch')
                    
                    #going to the bigbatch directory
                    os.chdir('../bigbatch')
                    
                    #untaring the files
                    taredfiles=glob.glob('*.gz')
                    
                    for file in taredfiles:
                        os.system('gunzip '+file)
                        while not os.path.isfile(file[:-3]):
                            time.sleep(1)
                            
                        if 'pha2' in file[:-3]:
                            os.system('mv '+file[:-3]+' '+lastdir+'_'+file[:-3]+'.pha')
                        else:
                            os.system('mv '+file[:-3]+' '+lastdir+'_'+file[:-3])
    
                    print('Obsid files copied and renamed')
                    
                os.chdir(startdir)
            
def regroup_grating_spectra(extension='pha2.pha',group='opt', skip_started=True):
    
    '''To be launched above the bigbatch directory'''
    
    #spawning ciao process for splitting spectra and manual grouping
    ciao_proc=pexpect.spawn('/bin/bash',encoding='utf-8')
    ciao_proc.logfile=sys.stdout
    ciao_proc.sendline('\nconda activate ciao-4.13')
    
    #spawning heasoft spectra for Kastra grouping
    heas_proc=pexpect.spawn('/bin/bash',encoding='utf-8')
    heas_proc.logfile=sys.stdout
    heas_proc.sendline('\nheainit')

    #not used anymore
    # def ciao_group(grating,value):
            
    #     '''
    #     wrapper for the command
    #     '''
    #     ciao_proc.sendline('dmgroup infile="heg_'+str(grating)+'.pha[SPECTRUM]" outfile=heg_'+str(grating)+'_grp_'+str(value)+'.pha binspec="" '+
    #                        'grouptypeval='+str(value)+' grouptype=NUM_CTS ycolumn="counts" xcolumn="channel" clobber=yes')
            
    def ft_group(file,grptype):
        
        '''wrapper for the command'''
        
        heas_proc.sendline('ftgrouppha infile='+file+' outfile='+file.replace('.','_grp_'+grptype+'.').replace('.pi','.pha')+' grouptype='+grptype+
                           ' respfile='+file.replace(file[file.rfind('.'):],'.rmf'))
        
        heas_proc.sendline('echo done')
        heas_proc.expect('done')
        
    allfiles=glob.glob('**',recursive=True)
    pha2_spectra=[elem for elem in allfiles if elem.endswith(extension) and 'bigbatch' in elem]
    
    if skip_started:
        pha2_spectra=[elem for elem in pha2_spectra if\
                    '/'.join(elem.split('/')[:-1])+('' if len(elem.split('/'))==1 else '/')+elem.split('/')[-1].split('_')[0]+'_heg_-1_grp_'+group+'.pha' not in allfiles or\
                    '/'.join(elem.split('/')[:-1])+('' if len(elem.split('/'))==1 else '/')+elem.split('/')[-1].split('_')[0]+'_heg_1_grp_'+group+'.pha' not in allfiles]
        
    pha2_dirs=os.getcwd()+'/'+np.array([elem[:elem.rfind('/')+1] for elem in pha2_spectra]).astype(object)
    

    for ind,specpath in enumerate(pha2_spectra):

        specfile=specpath.split('/')[-1]
        #sending the ciao process in the right directory
        
        ciao_proc.sendline('cd '+pha2_dirs[ind])
        
        #ungrouping the gratings
        ciao_proc.sendline('dmtype2split "'+specfile+'[#row=3]" "'+specfile.replace('obs_','hetg').split('_')[0]+'_heg_-1.pha[SPECTRUM]" clobber=yes verbose=2')
        ciao_proc.expect('Total number of columns=')
        
        time.sleep(5)
        
        ciao_proc.sendline('dmtype2split "'+specfile+'[#row=4]" "'+specfile.replace('obs_','hetg').split('_')[0]+'_heg_1.pha[SPECTRUM]" clobber=yes verbose=2')
        ciao_proc.expect('Total number of columns=')


        file_m1=specfile.split('_')[0]+'_heg_-1.pha'
        file_p1=specfile.split('_')[0]+'_heg_1.pha'
        time.sleep(5)

        #stat grouping
        
        heas_proc.sendline('cd '+pha2_dirs[ind])
        
        if group is not None:
            
            if group=='opt':
                ft_group(file_m1,grptype='opt')
                time.sleep(5)
                ft_group(file_p1,grptype='opt')
                time.sleep(5)
            # else:
            #     ciao_group(-1,5)
            #     ciao_group(-1,10)
            #     ciao_group(-1,20)
        
            #     ciao_group(1,5)
            #     ciao_group(1,10)
            #     ciao_group(1,20)
        
    heas_proc.sendline('exit')
    ciao_proc.sendline('exit')