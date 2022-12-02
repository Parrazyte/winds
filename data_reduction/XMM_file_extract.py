import os,sys
import argparse
import logging
import glob
'''
scans starting from given/current directory, gunzips and untars everything to leave a 
clean XMM arborescence
'''

ap = argparse.ArgumentParser(description='Script to uncompress XMM files.\n)')
ap.add_argument("-dir", "--startdir", nargs='?', help="staring directory. Current one if left blank", default='./', type=str)
args=ap.parse_args()

startdir=args.startdir

if not os.path.isdir(startdir):
    logging.error('Invalid directory, please start over.')
    sys.exit()
else:
    os.chdir(startdir)
    
startdir=os.getcwd()

allfiles=glob.glob('**',recursive=True)

for elem in allfiles:
    
    #are there still archives in the directories
    if elem.endswith('.tar') or elem.endswith('.TAR') or elem.endswith('.tar.gz'):
        
        #if yes, we get the filenames and directory paths
        filename=elem[elem.rfind('/')+1:]
        filename=filename[:filename.find('.')]
        
        if elem.rfind('/')!=-1:
            dirpath=elem[:elem.rfind('/')]
        else :
            dirpath='./'
            
        #now we check if those files have an obsid shape (10 numbers)
        if len(filename)==10 and filename.isdigit():
            print('\nFound one obsid file : '+filename)
            os.chdir(dirpath)
            
            #un-gunzipping if needed 
            if elem.endswith('.gz'):
                os.system('gunzip '+filename+'.tar.gz')
            
            #untarring the first level
            if filename not in glob.glob('*'):
                print('\nExtracting...\n')
                os.system('tar -xvf '+filename+'.tar --one-top-level')
                os.system('rm '+filename+'.tar')
                os.chdir(filename)
                #and the second one
                for tars in glob.glob('*.TAR')+glob.glob('*.tar'):
                    print('\nExtracting subfile '+tars+'\n')
                    os.system('tar -xvf'+tars)
                    os.system('rm '+tars)
    os.chdir(startdir)