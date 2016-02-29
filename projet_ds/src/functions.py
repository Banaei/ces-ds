# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:31:52 2016

@author: abanaei
"""

def get_rsa_data(rsa, rsa_format, verbose=None):
    
    error = False
    
    rsa = rsa.replace('\n', '')
    
    index = int(rsa[rsa_format['index_sp']-1:rsa_format['index_ep']].strip())
    
    sex = int(rsa[rsa_format['sex_sp']-1:rsa_format['sex_ep']].strip())
    
    departement = rsa[rsa_format['finess_sp' ]-1:rsa_format['finess_sp']+1].strip()
    if (not check_code(departement, departement=True)):
        if verbose:
            print 'Error in departement %s, RSA ignored' % (departement)
        error = True
        
    cmd = rsa[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']].strip()
    if (not check_code(cmd, cmd=True)):
        if verbose:
            print 'Error in CMD %s, RSA ignored' % (cmd)
        error = True
    
    dp = rsa[rsa_format['dp_sp'] - 1:rsa_format['dp_ep']].strip()
    if (not check_code(dp, cim=True)):
        if verbose:
            print 'Error in DP %s, RSA ignored' % (dp)
        error = True
        
    dr = rsa[rsa_format['dr_sp'] - 1:rsa_format['dr_ep']].strip()
    if (len(dr)>0) and (not check_code(dr, cim=True)):
        if verbose:
            print 'Error in DR %s, RSA ignored' % (dr)
        error = True

    try:
        age = int(rsa[rsa_format['age_in_year_sp'] - 1:rsa_format['age_in_year_ep']].strip())
    except ValueError:
        age = 0
        
    stay_length = int(rsa[rsa_format['stay_length_sp'] - 1:rsa_format['stay_length_ep']].strip())
    
    type_ghm = rsa[rsa_format['type_ghm_sp']-1:rsa_format['type_ghm_ep']].strip()
    if (not check_code(type_ghm, type_ghm=True)):
        if verbose:
            print 'Error in TYPE GHM %s, RSA ignored' % (type_ghm)
        error = True
    
    complexity_ghm = rsa[rsa_format['complexity_ghm_sp']-1:rsa_format['complexity_ghm_ep']].strip()
    if (not check_code(complexity_ghm, complexity_ghm=True)):
        if verbose:
            print 'Error in COMPLEXITY OF GHM %s, RSA ignored' % (complexity_ghm)
        error = True
    
    fixed_zone_length = int(rsa_format['fix_zone_length'])
    nb_aut_pgv = int(rsa[rsa_format['nb_aut_pgv_sp'] - 1:rsa_format['nb_aut_pgv_ep']].strip())
    aut_pgv_length = int(rsa_format['aut_pgv_length'])
    nb_suppl_radio = int(rsa[rsa_format['nb_suppl_radio_sp'] - 1:rsa_format['nb_suppl_radio_ep']].strip())
    suppl_radio_length = int(rsa_format['suppl_radio_length'])
    nb_rum = int(rsa[rsa_format['nbrum_sp'] - 1:rsa_format['nbrum_ep']].strip())
    rum_length = int(rsa_format['rum_length'])
    nb_das = int(rsa[rsa_format['nbdas_sp'] - 1:rsa_format['nbdas_ep']].strip())
    das_length = int(rsa_format['das_length'])
    nb_zones_actes = int(rsa[rsa_format['nbactes_sp'] - 1:rsa_format['nbactes_ep']].strip())
    zone_acte_length = int(rsa_format['zone_acte_length'])
    code_ccam_offset = int(rsa_format['code_ccam_offset'])
    code_ccam_length = int(rsa_format['code_ccam_length'])
    type_um_offset = int(rsa_format['type_um_offset'])
    type_um_length = int(rsa_format['type_um_length'])
    
    rsa_length = fixed_zone_length + nb_aut_pgv*aut_pgv_length + nb_suppl_radio*suppl_radio_length+nb_rum*rum_length + nb_das*das_length + nb_zones_actes*zone_acte_length
    if (len(rsa)!=rsa_length):
        if verbose:
            print 'The RSA length ' + str(len(rsa)) + ' is different from calculated lentgh ' + str(rsa_length) + " >" + rsa + '<'
        error = True
    
    first_um_sp = fixed_zone_length + (nb_aut_pgv * aut_pgv_length) + (nb_suppl_radio * suppl_radio_length) + type_um_offset

    type_um_dict = {}
    for i in range(0, nb_rum):
        type_um = rsa[first_um_sp: first_um_sp + type_um_length].strip()
        if (not check_code(type_um, type_um=True)):
            if verbose:
                print 'Error in TYPE UM %s' % (type_um)
            error = True
        else:
            type_um_dict[type_um] = 1
        first_um_sp += rum_length
        
    first_das_sp = fixed_zone_length + nb_aut_pgv*aut_pgv_length + nb_suppl_radio*suppl_radio_length+nb_rum*rum_length
    das_dict = {}
    for i in range(0, nb_das):
        das = rsa[first_das_sp : first_das_sp + das_length].strip()
        if (not check_code(das, cim=True)):
            if verbose:
                    print 'Error in DAS %s' % (das)
            error = True
        else:
            das_dict[das] = 1
        first_das_sp += das_length
    
    
    first_act_sp = fixed_zone_length + nb_aut_pgv*aut_pgv_length + nb_suppl_radio*suppl_radio_length + nb_rum*rum_length + nb_das*das_length + code_ccam_offset
    actes_dict = {}    
    for i in range(0, nb_zones_actes):
        acte = rsa[first_act_sp : first_act_sp + code_ccam_length].strip()
        if (not check_code(acte, ccam=True)):
            if verbose:
                    print 'Error in ACTE %s' % (acte)
            error = True
        else:
            actes_dict[acte] = 1
        first_act_sp += zone_acte_length
        
    return error, {
    'index':index,
    'cmd':cmd,
    'sex':sex,
    'dpt':departement,
    'dp':dp,
    'dr':dr,
    'age':age,
    'stay_length':stay_length,
    'type_ghm':type_ghm,
    'complexity_ghm':complexity_ghm,
    'type_um':type_um_dict.keys(),
    'das':das_dict.keys(),
    'actes':actes_dict.keys(),
    'rehosp':0,
     }

def detect_rehosps(ano_file_path, ano_format, rsa_file_path, rsa_format, rehosps_file_path):
    result_dict = {}
    line_number = 1
    rehosps_list = list()
    with open(ano_file_path) as ano_file:
        with open(rsa_file_path) as rsa_file:
            while True:
                ano_line = ano_file.readline()
                rsa_line = rsa_file.readline()
                if (len(ano_line.strip())>0):
                    ano_hash = ano_line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]
    #                rsa_index = int(ano_line[ano_format['rsa_index_sp'] - 1:ano_format['rsa_index_ep']].strip())
                    sej_num = int(ano_line[ano_format['sej_num_sp'] - 1:ano_format['sej_num_ep']])
                    stay_length = int(rsa_line[rsa_format['stay_length_sp'] - 1:rsa_format['stay_length_ep']].strip()) 
                    if (ano_hash not in result_dict):
                        result_dict[ano_hash]=list()
                    result_dict[ano_hash].append({'sej_num':sej_num, 'stay_length':stay_length, 'line_number':line_number})
                if not ano_line:
                    break
                if line_number % 100000 == 0:
                        print '\rGetting sej_num, processed ', line_number, 
                line_number += 1
    print 'Results dict length ' + str(len(result_dict))
    print 'Starting rehosps detection ...'
    line_number = 1
    for k in result_dict.keys():
        if (len(result_dict[k])>1):
            result_dict[k].sort(key=lambda x:x['sej_num'], reverse=True)
            first_loop = True
            last_sej_num = 0
            last_stay_length = 0
            current_sej_num = 0
            for i in range(0,len(result_dict[k])):
                if (first_loop):
                    last_sej_num = result_dict[k][i]['sej_num']
                    last_stay_length = result_dict[k][i]['stay_length']
                    first_loop = False
                    continue
                else:
                    current_sej_num = result_dict[k][i]['sej_num']
                    if (last_sej_num + last_stay_length < current_sej_num):
                        raise Exception('Error sorting the list : last_sej_num + last_stay_length >= current_sej_num') 
                    delay = last_sej_num + last_stay_length - current_sej_num
                    if delay <= delai_rehosp:
                        rehosps_list.append([k, result_dict[k][i]['line_number'], delay])
                    last_sej_num = current_sej_num
                    last_stay_length = result_dict[k][i]['stay_length']
        if line_number % 100000 == 0:
                print '\rRehosp detection : processed ', line_number, 
        line_number += 1
    with open(rehosps_file_path, 'w') as out_file:
        pickle.dump(rehosps_list, out_file)
    print 'Rehosps saved to ' + rehosps_file_path
    return rehosps_list
