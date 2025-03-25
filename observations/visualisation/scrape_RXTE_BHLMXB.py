#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 00:11:43 2022

@author: parrama
"""

import pandas as pd
from io import StringIO

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

#select class to choose from optiosn in browser
from selenium.webdriver.support.select import Select

from astroquery.simbad import Simbad

from visual_line_tools import silent_Simbad_query

import pickle

dump_path='./RXTE_lc_dict.pickle'

#current sample is 81 objects
#this is the list of Simbad ids so it can be directly compared to the results of the RXTE fetch
sample_ids='''
MAXI J1803-298
ATLAS 19bcxp
MAXI J0637-430
MAXI J1348-630
MAXI J1631-479
MAXI J1820+070
MAXI J1813-095
SWIFT J1658.2-4242
MAXI J1535-571
IGR J17454-2919
IGR J17451-3022
Anon 122858.0-250145
SWIFT J1753.7-2544
SWIFT J174510.8-262411
MAXI J1910-057
MAXI J1305-704
MAXI J1836-194
MAXI J1543-564
CRTS J135716.8-093238
MAXI J1659-152
XTE J1752-223
XTE J1652-453
SWIFT J1539.2-6227
PBC J1842.3-1124
SWIFT J174540.2-290005
[CHM2006] Candidate 1b
[KRL2007b] 312
[KRL2007b] 239
XTE J1818-245
2MAXI J1753-013
[KRL2007b] 224
[KRL2007b] 222
V* V1228 Sco
[KRL2007b] 353
SAX J1711.6-3808
[GHJ2008] 3
V* KV UMa
V* V406 Vul
V* V4641 Sgr
INTREF  1012
AX J1748.0-2829
V* V381 Nor
INTREF   866
AX J1740.1-3102
V* V2606 Oph
INTREF   948
AX J1733.9-3112
V* V1033 Sco
V* V2293 Oph
V* MM Vel
Granat 1915+105
V* V518 Per
V* GU Mus
V* V404 Cyg
RX J1735.9-2726
V* QZ Vul
V* BW Cir
EXO 1846-031
RX J1749.8-3312
V* V2107 Oph
NAME XTE J17464-3213
1A 0620-00
V* KY TrA
V* V821 Ara
V* V4134 Sgr
V* IL Lup
X Nor X-1
X Cen X-2
XTE J1637-498
SWIFT J1713.4-4219
XMMSL1 J171900.4-353217
XTE J1719-291
[KRL2007b] 241
NAME Great Annihilator
[SKM2002] 27
CXOGC J174540.0-290031
2XMM J180112.4-254436
XTE J1817-155
[KRL2007b] 343
XTE J1901+014
V* V1408 Aql
'''.split('\n')[1:-1]

sample_SIMBAD=None

driver = webdriver.Firefox()

driver.get("http://xte.mit.edu/asmlc/sources.html")

#the first element is just an index
source_list=[elem.text for elem in driver.find_elements(By.TAG_NAME, "li")][1:]

url_source_base='http://xte.mit.edu/asmlc/srcs/'

dic_lc_sources={}

for elem in source_list:
    
    #using the 1st name of the list as the main
    elem_name_base=elem.split(',')[0]
    
    #replacing the 'x' beginning character by 4U to avoid some sources name as x... to be unrecognized by simbad
    if elem_name_base.startswith('x') and elem_name_base[1].isdigit():
        elem_name=elem_name_base[1:]
        elem_name='4U'+elem_name
    else:
        elem_name=elem_name_base
    
    elem_Simbad=silent_Simbad_query(elem_name)
        
    #skipping sources not found in Simbad
    if elem_Simbad is None:
        continue
    
    elem_Simbad_id=elem_Simbad[0]['main_id']
        
    #skipping sources not in the current sample
    if elem_Simbad_id not in sample_ids:
        continue

    #loading the RXTE source webpage
    driver.get(url_source_base+elem_name_base+'.html')
    
    #switching binning to daily
    lc_bin=Select(driver.find_element(by=By.NAME, value="tbin"))
    lc_bin.select_by_visible_text("One-Day Average Light Curve")
    
    #selecting the elements to put in the lightcurve
    lc_get=Select(driver.find_element(by=By.NAME, value="colpick"))
    
    #the 8 first elements are (on tope of the MJD) the sum in all bands, its uncertainty, band A/B/C, its uncertainty
    [lc_get.select_by_index(i) for i in range(8)]
    
    #launching the command to fetch the lightcurve
    #note: this input search is not perfect because there's another submit type in the page but this is the first one
    lc_fetch=driver.find_element(By.XPATH,value=("//input[@type='submit']"))
    lc_fetch.click()
    
    loaded_page=False
    #waiting to generate the lightcurve
    while not loaded_page:

        driver.implicitly_wait(3)
        loaded_page='<br>\n\n</body></html>' in driver.page_source    
    
    #saving the lightcurve    
    lc_text=driver.find_element(By.TAG_NAME,value='body').text
    
    #saving a header for the upcoming datframe
    lc_header=['Obs time (MJD)']+lc_text.split('\n')[4].replace('% COLUMNS :','').split(',')
    #taking off an unwanted space
    lc_header=[elem[1:] for elem in lc_header]

    #transforming the string into an io so read_csv can read it without needing to write the file
    lc_io=StringIO(lc_text)
    
    #conversion into a csv put into the dictionnary
    dic_lc_sources[elem_Simbad_id]=pd.read_csv(lc_io,sep=' ',header=None,names=lc_header,skiprows=5)
    
with open(dump_path,'wb') as dumpfile:
    pickle.dump(dic_lc_sources,dumpfile)