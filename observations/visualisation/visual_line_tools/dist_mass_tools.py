import numpy as np

dist_dict={
    '4U1543-475':[7.5,0.5,0.5,1],
    '4U1630-47':[11.5,3.4,3.4,1],
    '4U1755-388':[6.5,2.5,2.5,1],
    'A0620-00':[1.06,1,1,1],
    'A1524-61':[8,0.9,0.9,1],
    'EXO1846-031':[7,0,0,0],
    'GROJ0422+32':[2.5,0.3,0.3,1],
    'GROJ1655-40':[3.2,0.2,0.2,1],
    'GRS1009-45':[3.8,0.3,0.3,1],
    'GRS1716-249':[6.9,1.1,1.1,1],
    'GRS1739-278':[7.3,1.3,1.3,1],
    'GRS1915+105':[9.4,1.6,1.6,1],
    'GS1354-64':[25,0,0,0],
    'GS2000+251':[2.7,0.7,0.7,1],
    'H1705-250':[8.6,2.1,2.1,1],
    'H1743-322':[8.5,0.8,0.8,1],
    'IGRJ17098-3628':[10.5,0,0,0],
    'MAXIJ1305-704':[7.5,1.4,1.8,1],
    'MAXIJ1348-630':[3.4,0.4,0.4,1],
    'MAXIJ1535-571':[4.1,0.5,0.6,1],
    'MAXIJ1659-152':[8.6,3.7,3.7,1],
    'MAXIJ1820+070':[2.96,0.33,0.33,1],
    'MAXIJ1836-194':[7,3,3,1],
    'MAXIJ1848-015':[3.4,0.3,0.3,1],
    'NovaMuscae1991':[5,0.7,0.7,1],
    #see http://arxiv.org/abs/2506.12387 for details
    'SwiftJ1727.8-1613':[2.6,1.1,1.7,1],
    'SwiftJ1728.9-3613':[8.4,0.8,0.8,1],
    'SwiftJ174510.8-262411':[3.7,1.1,1.1,0],
    'SwiftJ1753.5-0127':[3.9,0.7,0.7,1],
    'V404Cyg':[2.4,0.2,0.2,1],
    'V4641Sgr':[6.2,0.7,0.7,1],
    'XTEJ1118+480':[1.7,0.1,0.1,1],
    'XTEJ1550-564':[4.4,0.4,0.6,1],
    'XTEJ1650-500':[2.6,0.7,0.7,1],
    'XTEJ1720-318':[6.5,3.5,3.5,1],
    'XTEJ1752-223':[6,2,2,1],

    #here there is no estimate (the one quoted in BlackCAT is utter garbage) so we keep this at 8kpc
    'XTEJ1817-330':[8,8,8,1],

    'XTEJ1818-245':[3.6,0.8,0.8],
    'XTEJ1859+226':[12.5,1.5,1.5],
    'XTEJ1908+094':[6.5,3.5,3.5]
}

mass_dict={
    '4U1543-475':[8.4,1,1,1],
    '4U1957+115':[3,1,2.5,1],
    'A0620-00':[6.6,0.3,0.3,1],
    'A1524-61':[5.8,2.4,3,1],
    'GROJ0422+32':[2.7,0.5,0.7,1],
    'GROJ1655-40':[5.4,0.3,0.3,1],
    'GRS1716-249':[6.4,2,3.2,1],
    'GRS1915+105':[11.2,1.8,2,1],
    'GS2000+251':[7.2,1.7,1.7,1],
    'GX339-4':[5.9,3.6,3.6,1],
    'H1705-250':[5.4,1.5,1.5,1],
    'MAXIJ1305-704':[8.9,1.,1.6,1],
    'MAXIJ1820+070':[6.9,1.2,1.2,1],
    'NovaMuscae1991':[11,1.4,2.1,1],
    'SwiftJ1753.5-0127':[8.8,1.3,1.3,1],
    'SwiftJ1357.2-0933':[11.6,1.9,2.5,1],
    'V404Cyg':[9,0.6,0.2,1],
    'V4641Sgr':[6.4,0.6,0.6,1],
    'XTEJ1118+480':[7.1,0.1,0.1,1],
    'XTEJ1550-564':[11.7,3.9,3.9,1],
    'XTEJ1859+226':[8,2,2,1]}

def dist_mass_indiv(dict_linevis,obj_name,use_unsure_mass_dist=True):

    ####TODO: add Simbad matching for the names

    ctl_blackcat=dict_linevis['ctl_blackcat']
    ctl_blackcat_obj=dict_linevis['ctl_blackcat_obj']
    ctl_watchdog=dict_linevis['ctl_watchdog']
    ctl_watchdog_obj=dict_linevis['ctl_watchdog_obj']

    d_obj_indiv= 'nan'

    if obj_name in dist_dict:
        try:
            if dist_dict[obj_name][3] == 1 or use_unsure_mass_dist:
                # putting manual/updated distance values first
                d_obj_indiv = dist_dict[obj_name][0]
        except:
            breakpoint()
            pass
    else:

        obj_row = None
        # searching for the distances corresponding to the object namess in the first (most recently updated) catalog
        for elem in ctl_blackcat_obj:
            if obj_name in elem:
                obj_row = np.argwhere(ctl_blackcat_obj == elem)[0][0]
                break

        if obj_row is not None:

            obj_d_key = ctl_blackcat.iloc[obj_row]['d [kpc]']

            if not (type(obj_d_key) == str or np.isnan(obj_d_key)) and \
                    ('≥' not in obj_d_key and '>' not in obj_d_key):

                print('New measurement found in BlackCAT, not found in the biblio. Please check.')
                breakpoint()
                d_obj_indiv = ctl_blackcat.iloc[obj_row]['d [kpc]']

                # formatting : using only the main values + we do not want to use this catalog's results if they are simply upper/lower limits
                d_obj_indiv = str(d_obj_indiv)
                d_obj_indiv = d_obj_indiv.split('/')[-1].split('±')[0].split('~')[-1].split('∼')[-1]

                if '≥' in d_obj_indiv or '>' in d_obj_indiv or '<' in d_obj_indiv or '≤' in d_obj_indiv:
                    d_obj_indiv = 'nan'

                if '-' in d_obj_indiv:
                    if '+' in d_obj_indiv:
                        # taking the mean value if it's an uncertainty
                        d_obj_indiv = float(d_obj_indiv.split('+')[0].split('-')[0])
                    else:
                        # taking the mean if it's an interval
                        d_obj_indiv = (float(d_obj_indiv.split('-')[0]) + float(d_obj_indiv.split('-')[-1])) / 2

        # searching in the second catalog if nothing was found in the first one
        if d_obj_indiv == 'nan':
            if len(np.argwhere(ctl_watchdog_obj == obj_name)) != 0:

                # watchdog assigns by default 5+-3 kpc to sources with no distance estimate so we need to check for that
                # (there is no source with an actual 5kpc distance)
                watchdog_d_val = float(ctl_watchdog[np.argwhere(ctl_watchdog_obj == obj_name)[0][0]]['Dist1'])

                # these ones are false/outdated
                # here same, the lower limit quoted in WATCHDOG has been disproved in Charles19
                watchdog_d_exclu = ['SwiftJ1357.2-0933']

                if obj_name not in watchdog_d_exclu and watchdog_d_val not in [5., 8.]:
                    print('New measurement found in WATCHDOG, not found in the biblio. Please check.')
                    breakpoint()
                    d_obj_indiv = watchdog_d_val

    if d_obj_indiv == 'nan':
        # giving a default value of 8kpc to the objects for which we do not have good distance measurements
        d_obj_indiv = 8

    else:
        d_obj_indiv = float(d_obj_indiv)

    # fixing the source mass at 8 solar Masses if not in the local list since we have very few reliable estimates
    # of the BH masses anyway except for NS whose masses are in a dictionnary
    if obj_name in mass_dict and (mass_dict[obj_name][3] == 1 or use_unsure_mass_dist):
        m_obj_indiv = mass_dict[obj_name][0]
    else:
        m_obj_indiv = 8

    return d_obj_indiv,m_obj_indiv


def dist_mass(dict_linevis, use_unsure_mass_dist=True):
    '''
    Fetches local data and blackcat/watchdog to retrieve the mass and distances of sources.
    Local is > BC/WD because it is (currently) more up to date.

    -use_unsure_mass_dist: use mass and distance measurements set as unsure in the local dictionnary
        (with 0 for the last element of their measurement array)
    '''

    names = dict_linevis['obj_list']

    d_obj = np.array([None] * len(names))
    m_obj = np.array([None] * len(names))

    for i in range(len(names)):
        d_obj[i], m_obj[i] = dist_mass_indiv(dict_linevis, names[i], use_unsure_mass_dist)

    return d_obj, m_obj
