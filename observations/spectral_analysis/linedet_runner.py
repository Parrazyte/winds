"""
execution script of linedet_loop_single, only used for parallelization
using a previously made dump file allows simple argument names and to avoid bothering about non-pickable items
"""

import argparse
import dill
import os,sys
import matplotlib as mpl

#no-backend plotting method. important to avoid crashing with messy opencv/pyqt dependencies in clusters
mpl.use('agg')

#adding the current level and general folders to the python path to avoid issues
#done dynamically from the current path of the file, which should be in the global github project
script_dir='/'.join(os.path.realpath(__file__).split('/')[:-1])
proj_dir='/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.extend([script_dir,os.path.join(proj_dir,"general"),os.path.join(proj_dir,"general")])
print(sys.path)

from linedet_loop import linedet_loop_single


ap = argparse.ArgumentParser(description='Script to perform line detection in X-ray Spectra.\n)')

ap.add_argument('-epoch_id',nargs=1,help='epoch index in epoch_list',type=int)

ap.add_argument('-arg_dict_path',nargs=1,help='path of the argument dictionnary dump',type=str)

args=ap.parse_args()

#note: we take the first element  because for some reason they are turned into list
epoch_id=args.epoch_id[0]
arg_dict_path=args.arg_dict_path[0]

with open(arg_dict_path, 'rb') as dump_file:
    arg_dict = dill.load(dump_file)

linedet_loop_single(epoch_id,arg_dict)

print('linedet_runner complete')