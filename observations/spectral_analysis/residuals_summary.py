#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:50:35 2023

@author: parrama
"""

import glob
import numpy as np
import os
from tqdm import tqdm
import time

#pdf conversion with HTML parsin
from fpdf import FPDF, HTMLMixin

class PDF(FPDF, HTMLMixin):
    pass

#pdf merging
from PyPDF2 import PdfFileMerger

#note: to launch in "Sample" subdirectories of a given instrument



#parsing the files for lineplots_opt directories
lineplots_dirs=glob.glob('**/Sample/**/lineplots_opt/',recursive=True)
lineplots_dirs.sort()

n_obsid=len([elem for elem in glob.glob('**/Sample/**/lineplots_opt/**autofit_line_comb_plot**',recursive=True) if '_full' not in elem])

#creating the merger pdf
merger=PdfFileMerger()

with tqdm(total=n_obsid) as pbar:
    for elem_dir in lineplots_dirs:
        lineplots_files=glob.glob(elem_dir+'/**',recursive=True)
        
        comb_plots=np.array([elem for elem in lineplots_files if 'line_comb_plot' in elem])
        
        comb_plots.sort()
        
        #skipping sources with no line detection performed and _full directories
        if len(comb_plots)==0 or '_full' in elem_dir:
            continue
    
        #creating a pdf
        pdf=PDF(orientation="portrait")    
    
        #adding the plots of all exposures in the pdf
        for i_expos in range(int(len(comb_plots)/2)):
            
            pdf.add_page()
            if i_expos==0:
                pdf.set_font('helvetica', 'B', 16)
            
            #adding the two plots of each given obsid
            pdf.image(comb_plots[2*i_expos+1],x=30,y=0,w=150)
            pdf.image(comb_plots[2*i_expos].replace('line_comb','components'),x=30,y=90,w=150)
            pdf.image(comb_plots[2*i_expos],x=30,y=190,w=150)
        
            pbar.update(1)
            
        #saving the pdf
        pdf.output('./'+elem_dir.split('/')[1]+'_residuals_summary.pdf')
        
        #adding a bookmark in the merger
        merger.addBookmark(elem_dir.split('/')[1],len(merger.pages))
        
        #and adding the indiv pdf file to the merger
        merger.append('./'+elem_dir.split('/')[1]+'_residuals_summary.pdf')
            
        os.remove('./'+elem_dir.split('/')[1]+'_residuals_summary.pdf')
    
merger.write(os.getcwd().split('/')[-1]+'_residuals_summary.pdf')
merger.close()