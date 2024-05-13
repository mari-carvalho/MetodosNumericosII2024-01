# Análise da Estabilidade para o Método Explícito

import numpy as np
import matplotlib.pyplot as plt
from solvers import solvers
from analitica_dirchlet import analiticas
from FTCS import FTCS
from BTCS import BTCS
from CN import CN
from prettytable import PrettyTable

class estabilidade():
    def calculate_estabilidade():

        p0 = 19000000
        pw = 9000000
        qw = 0.01
        q0 = 100
        cc = 'pp'
        mi = 0.001
        k = 9.869233e-14
        h = 10
        phi = 0.2
        c = 2.04e-9
        L = 20
        A = 30
        x0 = 0
        xf = L
        t0 = 0
        tf = 100

        h_t = [0.2, 0.1, 0.05]
        h_x = 0.8
        j = h_x

        tempos_totais = []

        def calculate_n_t(tf, t0, i):
            n_t = (tf - t0) / (i)
            return n_t

        n_x = (xf - x0) / (h_x)

        rxn = []

        def calculate_h_t_calc_pp():
            x_calc_array_pp = []
            t_calc_array_pp = []
            p_calc_array_pp = []
            rxn_calc_array_pp = []
            for i in h_t:
                n_t = calculate_n_t(tf, t0, i)
                x_calc_pp, t_calc_pp, p_calc_pp, rxn_calc_pp = FTCS.calculate_FTCS_pp(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)

                x_calc_array_pp.append(x_calc_pp)
                p_calc_array_pp.append(p_calc_pp)
                t_calc_array_pp.append(t_calc_pp)
                rxn_calc_array_pp.append(rxn_calc_pp)


            return x_calc_array_pp, t_calc_array_pp, p_calc_array_pp, rxn_calc_array_pp

        def calculate_h_t_calc_fp():
            x_calc_array_fp = []
            t_calc_array_fp = []
            p_calc_array_fp = []
            rxn_calc_array_fp = []
            for i in h_t:
                n_t = calculate_n_t(tf, t0, i)
                x_calc_fp, t_calc_fp, p_calc_fp, rxn_calc_fp = FTCS.calculate_FTCS_fp(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)

                x_calc_array_fp.append(x_calc_fp)
                p_calc_array_fp.append(p_calc_fp)
                t_calc_array_fp.append(t_calc_fp)
                rxn_calc_array_fp.append(rxn_calc_fp)

            return x_calc_array_fp, t_calc_array_fp, p_calc_array_fp, rxn_calc_array_fp

        x_calc_array_pp, t_calc_array_pp, p_calc_array_pp, rxn_calc_array_pp = calculate_h_t_calc_pp()
        rxn.append(rxn_calc_array_pp)
        x_calc_array_fp, t_calc_array_fp, p_calc_array_fp, rxn_calc_array_fp = calculate_h_t_calc_fp()
        rxn.append(rxn_calc_array_fp)

        conditions = ['Dirchlet', 'Neumann']
        print('rxn', rxn)

        # Tabelas:
        tabela = PrettyTable(['Condição de Contorno', 'delta t', 'Estabilidade'])
        for cond_idx, condition in enumerate(conditions):
            rx_n = rxn[cond_idx]
            for h_t_idx, delta_t in enumerate(h_t):
                rxn_value = rx_n[h_t_idx]
                rounded_value = round(rxn_value, 3)
                rxn_value_str = str(rounded_value)
                tabela.add_row([condition, delta_t, rxn_value_str])

        print(tabela)
