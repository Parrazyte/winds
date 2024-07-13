

#general imports
import os,sys
import re as re
import numpy as np

#pdf conversion with HTML parsin
#install with fpdf2 NOT FPDF otherwise HTML won't work
from fpdf import FPDF, HTMLMixin

class PDF(FPDF, HTMLMixin):
    pass

from astropy.time import Time,TimeDelta

'''Astro'''
#general astro importss
from astropy.io import fits
from astropy.time import Time

from general_tools import ravel_ragged,shorten_epoch

from fitting_tools import line_e_ranges_fullarg,file_to_obs

def pdf_summary(epoch_files,arg_dict,fit_ok=False,summary_epoch=None,e_sat_low_list=None,e_sat_high_list=None):

    sat_glob=arg_dict['sat_glob']
    megumi_files=arg_dict['megumi_files']
    pre_reduced_NICER=arg_dict['pre_reduced_NICER']
    obj_name=arg_dict['obj_name']

    glob_summary_reg=arg_dict['glob_summary_reg']
    glob_summary_sp=arg_dict['glob_summary_sp']
    outdir=arg_dict['outdir']
    broad_HID_mode=arg_dict['broad_HID_mode']
    NICER_lc_binning=arg_dict['NICER_lc_binning']

    #the arg one
    line_cont_range_arg=arg_dict['line_cont_range_arg']
    line_search_e_arg=arg_dict['line_search_e_arg']
    line_search_norm_arg=arg_dict['line_search_norm_arg']

    diff_bands_NuSTAR_NICER=arg_dict['diff_bands_NuSTAR_NICER']
    low_E_NICER=arg_dict['low_E_NICER']
    suzaku_xis_ignore=arg_dict['suzaku_xis_ignore']
    suzaku_xis_range=arg_dict['suzaku_xis_range']
    suzaku_pin_range=arg_dict['suzaku_pin_range']
    line_cont_ig_arg=arg_dict['line_cont_ig_arg']
    megumi_files=arg_dict['megumi_files']
    e_min_NuSTAR=arg_dict['e_min_NuSTAR']
    e_max_XRT=arg_dict['e_max_XRT']

    def line_e_ranges(sat,det=None):
        return line_e_ranges_fullarg(sat,sat_glob,
                                     diff_bands_NuSTAR_NICER,low_E_NICER,line_cont_ig_arg,
                                    suzaku_xis_ignore,suzaku_pin_range,suzaku_xis_range,
                                     e_min_NuSTAR=e_min_NuSTAR,e_max_XRT=e_max_XRT,
                                    det=det)


    if summary_epoch is None:
        glob_summary_linedet=arg_dict['glob_summary_linedet']

    #used to have specific energy limits for different instruments. can be modified later

    if sat_glob == 'multi':
        e_sat_low_indiv = np.repeat([None], len(epoch_files))
        e_sat_high_indiv = np.repeat([None], len(epoch_files))
        line_cont_ig_indiv = np.repeat([None], len(epoch_files))
        sat_indiv = np.repeat([None], len(epoch_files))
        det_indiv = np.repeat([None], len(epoch_files))

        for id_epoch, elem_file in enumerate(epoch_files):
            # fetching the instrument of the individual element
            # note that we replace the megumi xis0_xis3 files by the xis1 because the merged xis0_xis3 have no header
            #we also replace SUZAKU in caps by Suzaku to have a better time matching strings
            sat_indiv[id_epoch] = fits.open(elem_file.replace('xis0_xis3', 'xis1').\
                                                 replace('xis0_xis2_xis3','xis1'))[1].header['TELESCOP']\
                .replace('SUZAKU','Suzaku')

            with fits.open(elem_file) as hdul:
                if sat_indiv[id_epoch].upper() in ['SWIFT','INTEGRAL']:
                    det_indiv[id_epoch] = hdul[1].header['INSTRUME']

                elif sat_indiv[id_epoch].upper()=='SUZAKU':

                        if megumi_files:
                            # update for pin files
                            if 'PIN' in hdul[1].header['DETNAM']:
                                det_indiv[id_epoch] = 'PIN'
                                # for megumi files we always update the header


                            elif 'XIS' in hdul[1].header['INSTRUME'] or '_xis' in elem_file:
                                det_indiv[id_epoch] ='XIS'



            e_sat_low_indiv[id_epoch], e_sat_high_indiv[id_epoch], temp,line_cont_ig_indiv[id_epoch], = \
                line_e_ranges(sat_indiv[id_epoch],det=det_indiv[id_epoch])
    else:
        e_sat_low_val, e_sat_high_val, temp,line_cont_ig_val = line_e_ranges(sat_glob)
        e_sat_low_indiv = np.repeat(e_sat_low_val, len(epoch_files))
        e_sat_high_indiv = np.repeat(e_sat_high_val, len(epoch_files))
        line_cont_ig_indiv = np.repeat(line_cont_ig_val, len(epoch_files))
        sat_indiv=np.repeat(sat_glob,len(epoch_files))

    if e_sat_low_list is not None:
        e_sat_low_indiv=e_sat_low_list
    if e_sat_high_list is not None:
        e_sat_high_indiv=e_sat_high_list

    if sat_glob == 'multi':
        epoch_observ = [file_to_obs(elem_file, elem_telescope,megumi_files) for elem_file, elem_telescope in \
                        zip(epoch_files, sat_indiv)]
    else:
        epoch_observ = [file_to_obs(elem_file, sat_glob,megumi_files) for elem_file in epoch_files]

    '''PDF creation'''

    print('\nPreparing pdf summary for exposures ')
    print(epoch_observ)

    #fetching the SNRs
    epoch_SNR=[]
    for elem_observ in epoch_observ:
        if os.path.isfile(elem_observ+'_regex_results.txt'):
            with open(elem_observ+'_regex_results.txt','r') as regex_file:
                regex_lines=regex_file.readlines()
                epoch_SNR+=[float(regex_lines[3].split('\t')[1])]
        else:
            epoch_SNR+=['X']

    epoch_inf = [elem_observ.split('_') for elem_observ in epoch_observ]

    short_epoch_id= '_'.join(shorten_epoch([elem.split('_sp')[0] for elem in epoch_observ]))


    pdf=PDF(orientation="landscape")
    pdf.add_page()
    pdf.set_font('helvetica', 'B', max(10,16-(2*len(epoch_observ))//5))

    line_skip_len=max(6,10-2*len(epoch_observ)//5)

    pdf.cell(1,1,'Epoch name :'+short_epoch_id,align='C',center=True)
    pdf.ln(10)

    pdf.cell(1,1,'Spectra informations:\n',align='C',center=True)

    #line skip
    pdf.ln(10)

    if sat_glob=='XMM':
        #global flare lightcurve screen (computed in filter_evt)
        rate_name_list=[elem_observ.replace(elem_observ.split('_')[1],'rate'+elem_observ.split('_')[1]) for elem_observ in epoch_observ]

        rate_name_list=[rate[:rate.rfind('_')]+'_screen.png' for rate in rate_name_list]


    is_sp=[]
    is_cleanevt=[]
    filename_list=[]
    exposure_list=[]
    expmode_list=[]
    for i_obs,(elem_observ,elem_sat) in enumerate(zip(epoch_observ,sat_indiv)):

        #creating new pages regularly for many GTIs
        if i_obs%8==0 and i_obs!=0:
            pdf.add_page()

        if elem_sat=='XMM':
            if os.path.isfile(elem_observ+'_sp_src_grp_20.ds'):
                is_sp+=[True]
                is_cleanevt+=[True]
            elif os.path.isfile(elem_observ+'_evt_save.ds'):
                is_sp+=[False]
                is_cleanevt+=[True]
            else:
                is_sp+=[False]
                is_cleanevt+=[False]
        elif elem_sat.upper() in ['CHANDRA','NICER','Swift','SWIFT','NuSTAR']:
            is_sp+=[True]
            is_cleanevt+=[False]

            # if elem_sat=='NICER' and pre_reduced_NICER:
            #     filename_list+=[elem_observ]
            # else:

        elif elem_sat=='Suzaku':
            if megumi_files:
                is_sp += [True]


        with fits.open(epoch_files[i_obs]) as hdul:

            try:
                exposure_list+=[hdul[1].header['EXPOSURE']]
            except:
                try:
                    exposure_list+=[hdul[1].header['ONTIME']]
                except:
                    pass

            if elem_sat=='Chandra':
                epoch_grating=hdul[1].header['GRATING']
                expmode_list+=[hdul[1].header['DATAMODE']]
            else:
                expmode_list += [''] if (pre_reduced_NICER or 'DATAMODE' not in hdul[0].header.keys())\
                                else [hdul[0].header['DATAMODE']]

            if elem_sat in ['NICER','NuSTAR']:

                if elem_sat=='NICER' and pre_reduced_NICER:
                        pdf.cell(1,1,'Object: '+obj_name+' | Date: '+Time(hdul[1].header['MJDSTART'],format='mjd').isot+
                                 ' | Obsid: '+epoch_inf[i_obs][0],align='C',center=True)
                else:
                    start_obs_s = hdul[1].header['TSTART'] + (0 if elem_sat=='NuSTAR' else hdul[1].header['TIMEZERO'])
                    # saving for titles later
                    mjd_ref = Time(hdul[1].header['MJDREFI'] + hdul[1].header['MJDREFF'], format='mjd')

                    obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')

                    date_str=str(obs_start.isot)
                    pdf.cell(1,1,'Object: '+obj_name+' | Date: '+date_str+' | Obsid: '+epoch_inf[i_obs][0],
                          align='C',center=True)

            else:
                date_str=' ' if 'DATE-OBS' not in hdul[0].header.keys() else hdul[0].header['DATE-OBS'].split('T')[0]
                pdf.cell(1,1,'Object: '+obj_name+' | Date: '+date_str+' | Obsid: '+epoch_inf[i_obs][0],
                      align='C',center=True)

            pdf.ln(line_skip_len)

            if elem_sat=='XMM':
                pdf.cell(1,1,'exposure: '+epoch_inf[i_obs][2]+' | camera: '+epoch_inf[i_obs][1]+' | mode: '+epoch_inf[i_obs][3]+
                      ' | submode: '+hdul[0].header['SUBMODE']+' | clean exposure time: '+str(round(exposure_list[i_obs]))+
                      's',align='C',center=True)
            elif elem_sat=='Chandra':

                pdf.cell(1,1,'grating: '+epoch_grating+' | mode: '+expmode_list[0]+
                         ' clean exposure time: '+str(round(exposure_list[i_obs]))+'s',align='C',center=True)
            elif elem_sat in ['NICER','Suzaku','Swift','SWIFT','NuSTAR']:
                pdf.cell(1,1,'mode: '+expmode_list[0]+
                         ' clean exposure time: '+str(round(exposure_list[i_obs]))+'s',align='C',center=True)

            pdf.ln(2)

            #we only show the third line for XMM spectra with spectrum infos if there is an actual spectrum
            if epoch_SNR[i_obs]!='X':

                pdf.ln(8)

                grouping_str='SNR: '+str(round(epoch_SNR[i_obs],3))+' | Spectrum bin grouping: '+epoch_inf[i_obs][-1].split('.')[0]+' cts/bin | '
                try:
                    pileup_lines=fits.open(elem_observ+'_sp_src.ds')[0].header['PILE-UP'].split(',')
                    pdf.cell(1,1,grouping_str+'pile-up values:'+pileup_lines[-1][10:],align='C',center=True)
                except:
                    pdf.cell(1,1,grouping_str+'no pile-up values for this exposure',align='C',center=True)

                pdf.ln(2)
        pdf.ln(line_skip_len)

        #turned off for now
        # if flag_bg[i_obs]:
        #     pdf.cell(1,1,'FLAG : EMPTY BACKGROUND')


    '''Line detection infos'''
    if summary_epoch is None:

        #the replace avoid problems with using full chandra/NICER file names as epoch loggings in the summary files

        assert sat_glob!='multi', 'This is not ready yet'

        logged_epochs=[(elem.split('\t')[0] if sat_glob in ['NICER','Swift']\
                        else '_'.join([elem.split('\t')[0],elem.split('\t')[1].replace('_grp_opt','')\
                                      .replace('sp_grp_opt','').replace('.pha','').replace('.pi','')]))\
                       for elem in glob_summary_linedet]
        right_epoch_match=np.array(logged_epochs)==str(shorten_epoch(ravel_ragged(epoch_inf)))

        #storing the message
        result_epoch_val=np.array(glob_summary_linedet)[right_epoch_match][0].split('\t')[2]

        # #and repeating it for all the individual sp
        result_epoch=np.repeat(result_epoch_val,len(epoch_inf))

    else:
        result_epoch=summary_epoch

    def disp_broadband_data():

        '''
        display the different raw spectra in the epoch
        '''

        pdf.cell(1,1,'Broad band data ('+str(min(e_sat_low_indiv))+'-'+str(max(e_sat_high_indiv))+' keV)',align='C',center=True)
        sp_displayed=False

        if len(epoch_observ)==1 or sat_glob!='XMM':
            if os.path.isfile(outdir+'/'+epoch_observ[0]+'_screen_xspec_spectrum.png'):
                pdf.image(outdir+'/'+epoch_observ[0]+'_screen_xspec_spectrum.png',x=30,y=50,w=200)
                sp_displayed=True
        else:
            #displaying the spectra of all the exposures in the epochs that have one
            for i_obs in range(len(epoch_observ)):
                if os.path.isfile(outdir+'/'+epoch_observ[i_obs]+'_screen_xspec_spectrum.png'):
                    obs_cam=epoch_observ[i_obs].split('_')[1]
                    if obs_cam=='pn':
                        x_pos=0
                    elif obs_cam=='mos1':
                        x_pos=100
                    elif obs_cam=='mos2':
                        x_pos=200
                    else:
                        x_pos=50
                    pdf.image(outdir+'/'+epoch_observ[i_obs]+'_screen_xspec_spectrum.png',x=x_pos,y=50,
                              w=200 if obs_cam not in ['pn','mos1','mos2'] else 100)
                    sp_displayed=True
        if not sp_displayed:
            pdf.ln(30)
            pdf.cell(1,1,'No spectrum to display',align='C',center=True)

    if sum(is_sp)>0:

        pdf.cell(1,1,'Line detection summary:',align='C',center=True)
        pdf.ln(10)
        pdf.cell(1, 1, short_epoch_id + ': ' + result_epoch[0], align='C', center=True)

        # for i_elem,elem_result in enumerate(result_epoch):
        #     pdf.cell(1,1,epoch_observ[i_elem]+': '+elem_result,align='C',center=True)
        #     pdf.ln(10)

        pdf.add_page()

        if not fit_ok:
            #when there is no line analysis, we directly show the spectra
            pdf.ln(10)
            disp_broadband_data()

            if os.path.isfile(outdir+'/'+epoch_observ[0]+'_screen_xspec_broadband.png'):
                pdf.add_page()

        #fetching the order of the spectra in the multi-grp spectra (which we assume are ones which went through the linedet process)
        good_sp=[epoch_observ[i_obs] for i_obs in range(len(epoch_observ)) if list(result_epoch)[i_obs]=='Line detection complete.']

        def disp_multigrp(lines):
            def format_lines(lines):

                '''
                feat chatgpt, actually terrible I should have done this manually
                '''
                formatted_lines = []
                datagroup_number = 0
                in_datagroup = False

                for line in lines:
                    # Skip empty lines
                    if line.strip() == '' or line.strip()=='\n':
                        continue

                    # Check if the line contains "Data group"
                    data_group_match = re.match(r'\s*Data group:\s+(\d+)', line)
                    if data_group_match:
                        datagroup_number = int(data_group_match.group(1))
                        in_datagroup = True
                        formatted_lines.append('-' * 65+'\n')  # Add horizontal line
                        continue

                    #Putting a space before the first statistic line
                    if 'Fit statistic' in line:
                            formatted_lines.append('\n')

                    # Check if the line contains "Warning" and skip it
                    if "Warning: cstat statistic is only valid for Poisson data." in line:
                        continue
                    if "Source file is not Poisson" in line:
                        continue

                    # Add the Data group number as a vertical column
                    if in_datagroup and re.match(r'\s+\d+\s+\d+\s+', line):
                        line = f'{datagroup_number: >2}' + line

                    # Add the column title near the left of "par comp"
                    if "par  comp" in line:
                        line = ' Dg  ' + line

                    # Add the line to the formatted lines without line jumps
                    formatted_lines.append(line)

                return formatted_lines

            '''
            scans model lines for multiple data groups and displays only the first lines of each data group after the first
            (to show the constant factors)
            '''

            # removing the bg models

            model_sep_lines = np.argwhere(np.array([elem.startswith('_') for elem in lines])).T[0]
            if len(model_sep_lines) == 1:
                lines_nobg = lines
            else:
                lines_nobg = lines[:model_sep_lines[0]] + lines[model_sep_lines[-1] + 1:]

            lineid_grp_arr = np.argwhere(np.array(['Data group' in elem for elem in np.array(lines_nobg)])).T[0]

            # no need to do anything for less than two datagroups
            if len(lineid_grp_arr) < 2:
                return lines_nobg,1
            else:
                lines_cleaned = []
                # we display up to the second data group, then only non-standardly linked lines
                for i_grp, lineid_grp in enumerate(lineid_grp_arr):
                    if i_grp == 0:
                        i_begin = 0
                        i_end = lineid_grp_arr[i_grp + 1]
                        #adding the whole group
                        lines_cleaned += lines_nobg[i_begin:i_end]
                    else:

                        lines_cleaned += [lines_nobg[lineid_grp]]

                        #testing every line after that
                        i_begin = lineid_grp+1
                        #we can allow skipping the last line here because it will always be a fit display line
                        i_end = -1 if i_grp==len(lineid_grp_arr)-1 else lineid_grp_arr[i_grp + 1]

                        #computing the group size
                        grp_size=sum([False if len(elem.split())==0 else elem.split()[0].isdigit()\
                                      for elem in lines_nobg[i_begin:i_end]])

                        for i_line in range(i_begin,i_begin+grp_size):
                            elem_line=lines_nobg[i_line]
                            elem_line_status=elem_line.split()[-1]
                            elem_line_parnumber=int(elem_line.split()[0])

                            #comparing the link to the expected link value (aka degrouped parameter value)
                            elem_line_link_std=elem_line_status=='p'+str(1+(max(0,int(elem_line_parnumber-1)%grp_size)))

                            if not elem_line_link_std:
                                lines_cleaned+=[elem_line]

                # adding everything after the end of the model (besides the last line which is just a 'model not fit yet' line)
                lines_cleaned += lines_nobg[i_begin + lineid_grp_arr[1] - lineid_grp_arr[0]:-1]

                # formatting to change the position of the datagroup
                lines_formatted = format_lines(lines_cleaned)

                return lines_formatted,len(lineid_grp_arr)

        def display_fit(fit_type):

            if 'broadband' in fit_type:
                fit_title='broad band'
                fit_ener=str(min(e_sat_low_indiv))+'-'+str(max(e_sat_high_indiv))
            elif 'broadhid' in fit_type:
                fit_title='HID'
                fit_ener=str(min(e_sat_low_indiv))+'-'+str(max(e_sat_high_indiv)) if broad_HID_mode else '3.-10.'

            if 'linecont' in fit_type:
                #overwrites the broadband if
                fit_title='Line continuum'
                fit_ener=line_cont_range_arg
            if 'autofit' in fit_type:
                fit_title='Autofit'
                fit_ener=line_cont_range_arg
            if 'post_auto' in fit_type:
                fit_title+=' post autofit'
            if 'zoom' in fit_type:
                fit_title+=' zoom'

            image_id=outdir+'/'+epoch_observ[0]+'_screen_xspec_'+fit_type

            if os.path.isfile(image_id+'.png'):

                pdf.set_font('helvetica', 'B', 16)

                #selecting the image to be plotted
                image_path=image_id+'.png'
                pdf.ln(5)

                pdf.cell(1,-5,fit_title+' ('+fit_ener+' keV):',align='C',center=True)

                #no need currently since we have that inside the graph now
                # if fit_type=='broadband' and len(epoch_observ)>1 or sat_glob=='Chandra':
                #     #displaying the colors for the upcoming plots in the first fit displayed
                #     pdf.cell(1,10,'        '.join([xcolors_grp[i_good_sp]+': '+'_'.join(good_sp[i_good_sp].split('_')[1:3])\
                #                                    for i_good_sp in range(len(good_sp))]),align='C',center=True)

                pdf.image(image_path,x=0,y=50,w=150)

                #fetching the model unless in zoom mode where the model was displayed on the page before
                if 'zoom' not in fit_type:
                    #and getting the model lines from the saved file
                    with open(outdir+'/'+epoch_observ[0]+'_mod_'+fit_type+'.txt') as mod_txt:
                        fit_lines=mod_txt.readlines()

                        cleaned_lines,n_groups=disp_multigrp(fit_lines)

                        pdf.set_font('helvetica', 'B',min(8,11-len(cleaned_lines)//13))
                        pdf.multi_cell(150,2.6-(0.2*max(0,8-(min(8,11-len(cleaned_lines)//13)))),
                                       '\n'*max(0,int((30 if 'auto' in fit_type else 30)\
                                                              -1.5*len(cleaned_lines)**2/100))\
                                                          +''.join(cleaned_lines))

                    #in some cases due to the images between some fit displays there's no need to add a page
                    if 'linecont' not in fit_type:
                        pdf.add_page()
                    else:
                        #restrictingthe addition of a new page to very long models
                        if len(cleaned_lines)>35:
                            pdf.add_page()
                else:
                    pass

        display_fit('broadband_post_auto')
        display_fit('broadhid_post_auto')


        pdf.set_margins(0.5,0.5,0.5)

        display_fit('autofit')
        display_fit('autofit_zoom')

        pdf.set_margins(1.,1.,1.)

        if os.path.isfile(outdir+'/'+epoch_observ[0]+'_autofit_components_plot_'+line_search_e_arg.replace(' ','_')+'_'+\
                          line_search_norm_arg.replace(' ','_')+'.png'):
            #combined autofit component plot
            pdf.image(outdir+'/'+epoch_observ[0]+'_autofit_components_plot_'+line_search_e_arg.replace(' ','_')+'_'+\
                          line_search_norm_arg.replace(' ','_')+'.png',x=1,w=280)

        if os.path.isfile(outdir+'/'+epoch_observ[0]+'_autofit_line_comb_plot_'+line_search_e_arg.replace(' ','_')+'_'+\
                          line_search_norm_arg.replace(' ','_')+'.png'):
            #Combined plot
            pdf.image(outdir+'/'+epoch_observ[0]+'_autofit_line_comb_plot_'+line_search_e_arg.replace(' ','_')+'_'+\
                      line_search_norm_arg.replace(' ','_')+'.png',x=1,w=280)

        if os.path.isfile(outdir+'/'+epoch_observ[0]+'_abslines_table.txt'):
            with open(outdir+'/'+epoch_observ[0]+'_abslines_table.txt','r') as table_file:
                table_html=table_file.readlines()
            pdf.add_page()
            #autofit absorption lines data table
            pdf.set_font('helvetica', 'B', 9)
            pdf.write_html(''.join(table_html))
            pdf.add_page()

        pdf.set_margins(0.5,0.5,0.5)

        display_fit('broadband')
        display_fit('broadhid')
        display_fit('broadband_linecont')

        pdf.set_margins(1.,1.,1.)

        if os.path.isfile(outdir+'/'+epoch_observ[0]+'_cont_line_comb_plot_'+line_search_e_arg.replace(' ','_')+'_'+\
                          line_search_norm_arg.replace(' ','_')+'.png'):
            #Combined plot
            pdf.image(outdir+'/'+epoch_observ[0]+'_cont_line_comb_plot_'+line_search_e_arg.replace(' ','_')+'_'+\
                      line_search_norm_arg.replace(' ','_')+'.png',x=1,w=280)

            #not needed at the moment
            # pdf.image(outdir+'/'+exposid+'_line_cont_plot_'+line_search_e_arg.replace(' ','_')+'_'+line_search_norm_arg.replace(' ','_')\
            #           +'.png',x=1,w=280)
            # pdf.image(outdir+'/'+exposid+'_line_col_plot_'+line_search_e_arg.replace(' ','_')+'_'+line_search_norm_arg.replace(' ','_')\
            #           +'.png',x=1,w=280)


        if fit_ok:
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            disp_broadband_data()

    #displaying error messages for XMM epochs with no spectrum
    elif sat_glob=='XMM':

        #for the ones with no spectra, we have only one obs per epoch so not need to loop
        #displaying the reason the region computation failed if it did
        pdf.cell(1,1,'Region extraction summary:',align='C',center=True)
        pdf.ln(10)
        pdf.cell(1,1,[elem.split('\t')[2] for elem in glob_summary_reg \
                      if '_'.join([elem.split('\t')[0],elem.split('\t')[1]])==epoch_observ[0].replace('_auto','')][0],align='C',center=True)

        pdf.ln(10)

        #displaying the reason the spectrum computation failed
        pdf.cell(1,1,'Spectrum computation summary:',align='C',center=True)
        pdf.ln(10)
        pdf.cell(1,1,[elem.split('\t')[2] for elem in glob_summary_sp \
                      if '_'.join([elem.split('\t')[0],elem.split('\t')[1]])==epoch_observ[0].replace('_auto','')][0],align='C',center=True)


    shown_obsids_NICER=[]
    '''Data reduction displays'''

    if sat_glob=='multi':
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 16)
        pdf.cell(1, 10, 'Epoch matching', align='C', center=True)
        pdf.ln(10)
        try:
            pdf.image(outdir +'/'+ short_epoch_id+'_multi_matching.pdf', x=20, y=30, w=250)
        except:
            pass

    for i_obs,(elem_epoch,elem_sat) in enumerate(zip(epoch_observ,sat_indiv)):

        if elem_sat=='NICER':

            #adding the global flares curve for each obsid
            elem_obsid=elem_epoch.split('-')[0]

            if elem_obsid not in shown_obsids_NICER:
                shown_obsids_NICER+=[elem_obsid]
                pdf.add_page()
                pdf.set_font('helvetica', 'B', 16)
                pdf.cell(1,10,'Orbits for obsid '+elem_obsid,align='C',center=True)
                pdf.ln(10)
                try:
                    pdf.image(elem_obsid +'-global_flares_night.png',x=20,y=30,w=250)
                except:
                    try:
                        pdf.image(elem_obsid + '-global_flares.png', x=20, y=30, w=250)
                    except:
                        pass

            #and adding the individual GTI's flare and lightcurves
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(1,10,'GTIS and lightcurves for gti '+elem_epoch,align='C',center=True)
            pdf.ln(10)

            #recognizing time-resolved spectra
            elem_orbit=elem_epoch.split('S')[0].split('M')[0].split('F')[0].split('I')[0].split('N')[0].split('D')[0]

            if 'N' in elem_epoch.split('-')[-1]:
                orbit_night=True
                day_add='_night'
            elif 'D' in elem_epoch.split('-')[-1]:
                orbit_day=True
                day_add='_day'
            else:
                day_add=''

            try:
                pdf.image(elem_orbit+day_add+'_flares.png',x=2,y=70,w=140)
            except:
                breakpoint()
                pass

            try:
                pdf.image(elem_epoch + '_lc_3-10_bin_' + NICER_lc_binning + '.png', x=150, y=30, w=70)
            except:
                pass
            try:
                pdf.image(elem_epoch + '_hr_6-10_3-6_bin_' + NICER_lc_binning + '.png', x=220, y=30, w=70)
            except:
                try:
                    pdf.image(elem_epoch + '_hr_3-10_bin_' + NICER_lc_binning + '.png', x=220, y=30, w=70)
                except:
                    pass
            try:
                pdf.image(elem_epoch + '_lc_3-6_bin_' + NICER_lc_binning + '.png', x=150, y=120, w=70)
            except:
                pass
            try:
                pdf.image(elem_epoch + '_lc_6-10_bin_' + NICER_lc_binning + '.png', x=220, y=120, w=70)
            except:
                pass

        if elem_sat=='NuSTAR':


            elem_epoch_nogti=elem_epoch.split('-')[0]

            if 'A0' in elem_epoch.split('_')[0]:
                nudet='FPMA'
            elif 'B0' in elem_epoch.split('_')[0]:
                nudet='FPMB'

            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(1, 10, nudet+' Data reduction for observation ' + elem_epoch_nogti, align='C', center=True)
            pdf.ln(10)
            pdf.ln(10)
            pdf.cell(1, 30, 'Region definition                                               ' +
                     'BBG selection', align='C', center=True)


            pdf.image(elem_epoch_nogti + '_auto_reg_screen.png', x=2, y=50, w=140)

            pdf.image(elem_epoch_nogti + '_vis_CCD_1_crop.png', x=150, y=50, w=70)
            pdf.image(elem_epoch_nogti + '_vis_CCD_2_mask.png', x=220, y=50, w=70)
            pdf.image(elem_epoch_nogti + '_vis_CCD_3_cut.png', x=150, y=120, w=70)
            pdf.image(elem_epoch_nogti + '_vis_CCD_4_bg.png', x=220, y=120, w=70)

            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(1, 10, nudet+' Obs Lightcurves for observation ' + elem_epoch, align='C', center=True)

            elem_epoch_obsid = elem_epoch.split('-')[0][:-3]

            pdf.image(elem_epoch_obsid + '_'+nudet+'_lc_screen_3_79_bin_100.png', x=2, y=50, w=140)
            pdf.image(elem_epoch_obsid + '_'+nudet+'_hr_screen_10-50_3-10_bin_100.png', x=150, y=50, w=70)
            pdf.image(elem_epoch_obsid + '_'+nudet+'_lc_screen_3_10_bin_100.png', x=220, y=50, w=70)
            pdf.image(elem_epoch_obsid + '_'+nudet+'_lc_screen_10_50_bin_100.png', x=220, y=120, w=70)

            #adding one page for individual orbits
            if '-' in elem_epoch:
                pdf.add_page()
                pdf.set_font('helvetica', 'B', 16)
                pdf.cell(1, 10, nudet+' GTI Lightcurves for observation ' + elem_epoch, align='C', center=True)

                elem_epoch_gti=elem_epoch.split('-')[1]

                pdf.image(elem_epoch_obsid + '-' + elem_epoch_gti + '_'+nudet+'_lc_screen_3_79_bin_10.png', x=2, y=50, w=140)
                pdf.image(elem_epoch_obsid + '-' + elem_epoch_gti + '_'+nudet+'_hr_screen_10-50_3-10_bin_10.png', x=150, y=50, w=70)
                pdf.image(elem_epoch_obsid + '-' + elem_epoch_gti + '_'+nudet+'_lc_screen_3_10_bin_10.png', x=220, y=50, w=70)
                pdf.image(elem_epoch_obsid + '-' + elem_epoch_gti + '_'+nudet+'_lc_screen_10_50_bin_10.png', x=220, y=120, w=70)


        if elem_sat=='XMM':
                if is_sp[i_obs]:
                    pdf.add_page()
                    pdf.set_font('helvetica', 'B', 16)
                    pdf.cell(1,10,'Data reduction for observation '+elem_epoch,align='C',center=True)
                    pdf.ln(10)
                    pdf.cell(1,30,'Initial region definition                                        '+
                              'Post pile-up excision (if any) region definition',align='C',center=True)
                    pdf.image(elem_epoch+'_reg_screen.png',x=2,y=50,w=140)

                    if os.path.isfile(elem_epoch+'_reg_excised_screen.png'):
                        pdf.image(elem_epoch+'_reg_excised_screen.png',x=155,y=50,w=140)

                    if expmode_list[i_obs]=='IMAGING':
                        pdf.add_page()
                        pdf.image(elem_epoch+'_opti_screen.png',x=1,w=280)

                        #adding a page for the post-pileup computation if there is one
                        if os.path.isfile(elem_epoch+'_opti_excised_screen.png'):
                            pdf.add_page()
                            pdf.image(elem_epoch+'_opti_excised_screen.png',x=1,w=280)

                    elif expmode_list[i_obs]=='TIMING' or expmode_list[i_obs]=='BURST':
                        pdf.add_page()
                        pdf.cell(1,30,'SNR evolution for different source regions, first iteration',align='C',center=True)
                        pdf.image(elem_epoch+'_opti_screen.png',x=10,y=50,w=140)

                        #adding a page for the post-pileup computation if there is one
                        if os.path.isfile(elem_epoch+'_opti_excised_screen.png'):
                            pdf.image(elem_epoch+'_opti_excised_screen.png',x=150,y=50,w=140)

                elif is_cleanevt[i_obs]:
                    pdf.set_font('helvetica', 'B', 16)
                    if expmode_list[i_obs]=='IMAGING':
                        pdf.add_page()
                        pdf.cell(1,30,'Raw image                     '+'              position catalog cropping zone          '+
                                  '            cropped region zoom',align='C',center=True)
                        try:
                            pdf.image(elem_epoch+'_img_screen.png',x=1,y=70,w=90)
                            pdf.image(elem_epoch+'_catal_reg_screen.png',x=100,y=70,w=90)
                            pdf.image(elem_epoch+'_catal_crop_screen.png',x=190,y=65,w=120)
                        except:
                            pass
                    if expmode_list[i_obs]=='TIMING' or expmode_list[i_obs]=='BURST':
                        pdf.add_page()
                        pdf.cell(1,30,'Raw image',align='C',center=True)
                        try:
                            pdf.image(elem_epoch+'_img_screen.png',x=70,y=50,w=150)
                        except:
                            pass

                '''flare curves'''

                pdf.add_page()
                try:
                    #source/bg flare "first iteration" lightcurves (no flare gti cut) with flares zones highlighted
                    pdf.image(elem_observ+'_lc_comb_snr_screen.png',x=10,y=10,w=130)
                except:
                    pass
                #corresponding flare rate curve and rate limit
                try:
                    pdf.image(rate_name_list[i_obs],x=150,y=10,w=130)
                except:
                    pass

                #source/bg flare "second iteration" lightcurve
                try:
                     pdf.image(elem_observ+'_lc_comb_snr_excised_screen.png',x=10,y=105,w=130)
                except:
                    pass

                #broad band source/bg lightcurve
                try:
                    pdf.image(elem_observ+'_lc_comb_broad_screen.png',x=150,y=105,w=130)
                except:
                    pass

    #naming differently for aborted and unaborted analysis
    if not fit_ok:
        pdf.output(outdir + '/' + ('_'.join(shorten_epoch(epoch_observ))) + '_aborted_recap.pdf')
    else:
        pdf.output(outdir+'/'+('_'.join(shorten_epoch(epoch_observ)))+'_recap.pdf')
