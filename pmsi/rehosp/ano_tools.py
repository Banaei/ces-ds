# Embedded file name: ano_tools.py
"""
Created on Fri Aug 21 16:41:30 2015

@author: abanaei
"""
import formats

def get_anos_from_file(ano_file_path, ano_format):
    anofile = open(ano_file_path, 'r')
    ano_list = list()
    line_number = 0
    exit_month = 0
    for line in anofile:
        code_retour = int(line[ano_format['code_retour_sp'] - 1:ano_format['code_retour_ep']])
        if code_retour > 0:
            pass
        else:
            ano = line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]
            try:
                exit_month = int(line[ano_format['exit_month_sp'] - 1:ano_format['exit_month_ep']])
            except ValueError:
                exit_month = 0

            ano_list.append((ano, exit_month, line_number))
        if line_number % 10000 == 0:
            print '\rPorcessed ', line_number,
        line_number += 1

    return ano_list


def get_anos_in_month_01_from_file(ano_file_path, ano_format):
    anofile = open(ano_file_path, 'r')
    ano_list = list()
    line_number = 0
    exit_month = 0
    for line in anofile:
        code_retour = int(line[ano_format['code_retour_sp'] - 1:ano_format['code_retour_ep']])
        if code_retour > 0:
            pass
        else:
            ano = line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]
            try:
                exit_month = int(line[ano_format['exit_month_sp'] - 1:ano_format['exit_month_ep']])
                if exit_month == 1:
                    exit_month = 13
                    ano_list.append((ano, exit_month, line_number))
                else:
                    continue
            except ValueError:
                pass

        if line_number % 10000 == 0:
            print '\rPorcessed ', line_number,
        line_number += 1

    return ano_list


def is_ano_ok(line, ano_format):
    try:
        result = int(line[ano_format['code_retour_sp'] - 1:ano_format['code_retour_ep']]) == 0
    except ValueError:
        result = 0

    return result


def get_ano(line, ano_format, index):
    ano = line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]
    try:
        exit_month = int(line[ano_format['exit_month_sp'] - 1:ano_format['exit_month_ep']])
    except ValueError:
        exit_month = 0

    return (ano, exit_month, index)


def is_ano_in_the_list(ano, the_list):
    try:
        the_list.index(ano)
        return 1
    except ValueError:
        return 0