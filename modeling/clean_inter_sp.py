'''
To be launched in the current directory.
Cleans the current arborescence of all intermediate spectra
'''

import os
import glob
import numpy as np
import time
currdir=os.getcwd()

listdirs=glob.glob('./**/',recursive=True)

#restricting to angle directories
angle_dirs=[elem for elem in listdirs if elem.split('/')[-2].startswith('angle')]

for elem_dir in angle_dirs:
    print('found directory '+elem_dir)
    os.chdir(elem_dir)

    sp_saves = glob.glob('sp_**')
    sp_saves.sort()

    sp_saves_rest = np.array([elem for elem in sp_saves if '_tr_rest_' in elem and '_final_' not in elem])
    sp_saves_rest.sort()

    breakpoint()
    # deleting the previous tr spectra when there's more than two and no intermediate save is asked
    if len(sp_saves_rest) > 2:
        for i_save_rest in range(len(sp_saves_rest) - 2):

            os.system('rm ' + sp_saves_rest[i_save_rest])
            while os.path.isfile(sp_saves_rest[i_save_rest]):
                time.sleep(0.1)

            print('Deleted sp '+sp_saves_rest[i_save_rest])


    sp_saves_incid = np.array([elem for elem in sp_saves if '_incid_' in elem and '_final_' not in elem])
    sp_saves_incid.sort()

    # deleting the previous incid spectra when there's more than one
    if len(sp_saves_incid) > 1:
        for i_save_incid in range(len(sp_saves_incid) - 1):

            os.system('rm ' + sp_saves_incid[i_save_incid])
            time.sleep(0.1)
            while os.path.isfile(sp_saves_incid[i_save_incid]):
                time.sleep(0.1)

            print('Deleted sp '+sp_saves_incid[i_save_incid])

    os.chdir(currdir)