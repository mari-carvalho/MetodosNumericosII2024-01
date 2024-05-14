# Análise do Tempo Computacional para cada Solver

import numpy as np
import matplotlib.pyplot as plt
from solvers import solvers
from analitica import analitica
from FTCS import FTCS
from BTCS import BTCS
from CN_tt import CN
from prettytable import PrettyTable
import time

class tempo_computacional_tf():
    def calculate_tempo_computacional_h_t():

        rho_list = [8.92, 9]  # g/cm³
        cp_list = [0.092, 0.080]  # cal/(g.ºC)
        k_list = [0.95, 0.92]  # cal/(cm.s.ºC)
        qw = 25
        L = 80  # cm
        T0 = 50  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        t0 = 0
        tf = 100
        x0 = 0
        xf = L

        h_t = [1, 0.5]
        h_x = 25
        j = h_x

        tempos_totais_cobre = []
        tempos_totais_al = []


        def calculate_n_t(tf, t0, i):
            n_t = (tf - t0) / (i)
            return n_t


        n_x = (xf - x0) / (h_x)

        def calculate_h_t_calc_gs(rho_list, k_list, cp_list):
            x_calc_array_tf_gs_cobre = []
            t_calc_array_tf_gs_cobre = []
            T_calc_array_tf_gs_cobre = []
            tempo_total_array_tf_gs_cobre = []
            x_calc_array_tf_gs_al = []
            t_calc_array_tf_gs_al = []
            T_calc_array_tf_gs_al = []
            tempo_total_array_tf_gs_al = []
            n_t_array = []
            for m in range(len(rho_list)):
                if m == 0:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for i in h_t:
                        inicio_gs = time.time()
                        n_t = calculate_n_t(tf, t0, i)
                        x_calc_tf_gs_cobre, t_calc_tf_gs_cobre, T_calc_tf_gs_cobre = BTCS.calculate_BTCS_tf_gs(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                        fim_gs = time.time()
                        tempo_total_gs = fim_gs - inicio_gs

                        x_calc_array_tf_gs_cobre.append(x_calc_tf_gs_cobre)
                        T_calc_array_tf_gs_cobre.append(T_calc_tf_gs_cobre)
                        t_calc_array_tf_gs_cobre.append(t_calc_tf_gs_cobre)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_gs_cobre.append(tempo_total_gs)
                elif m == 1:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for i in h_t:
                        inicio_gs = time.time()
                        n_t = calculate_n_t(tf, t0, i)
                        x_calc_tf_gs_al, t_calc_tf_gs_al, T_calc_tf_gs_al = BTCS.calculate_BTCS_tf_gs(rho, cp, k, L, Tw, T0, Te,
                                                                                             x0, xf, t0, tf, qw, i, j,
                                                                                             n_t, n_x)
                        fim_gs = time.time()
                        tempo_total_gs = fim_gs - inicio_gs

                        x_calc_array_tf_gs_al.append(x_calc_tf_gs_al)
                        T_calc_array_tf_gs_al.append(T_calc_tf_gs_al)
                        t_calc_array_tf_gs_al.append(t_calc_tf_gs_al)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_gs_al.append(tempo_total_gs)

            return x_calc_array_tf_gs_cobre, t_calc_array_tf_gs_cobre, T_calc_array_tf_gs_cobre, tempo_total_array_tf_gs_cobre, n_t_array, \
                x_calc_array_tf_gs_al, t_calc_array_tf_gs_al, T_calc_array_tf_gs_al, tempo_total_array_tf_gs_al

        def calculate_h_t_calc_tdma(rho_list, k_list, cp_list):
            x_calc_array_tf_tdma_cobre = []
            t_calc_array_tf_tdma_cobre = []
            T_calc_array_tf_tdma_cobre = []
            tempo_total_array_tf_tdma_cobre = []
            x_calc_array_tf_tdma_al = []
            t_calc_array_tf_tdma_al = []
            T_calc_array_tf_tdma_al = []
            tempo_total_array_tf_tdma_al = []
            n_t_array = []
            for m in range(len(rho_list)):
                if m == 0:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for i in h_t:
                        inicio_tdma = time.time()
                        n_t = calculate_n_t(tf, t0, i)
                        x_calc_tf_tdma_cobre, t_calc_tf_tdma_cobre, T_calc_tf_tdma_cobre = BTCS.calculate_BTCS_tf_tdma(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                        fim_tdma = time.time()
                        tempo_total_tdma = fim_tdma - inicio_tdma

                        x_calc_array_tf_tdma_cobre.append(x_calc_tf_tdma_cobre)
                        t_calc_array_tf_tdma_cobre.append(t_calc_tf_tdma_cobre)
                        T_calc_array_tf_tdma_cobre.append(T_calc_tf_tdma_cobre)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_tdma_cobre.append(tempo_total_tdma)
                elif m == 1:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for i in h_t:
                        inicio_tdma = time.time()
                        n_t = calculate_n_t(tf, t0, i)
                        x_calc_tf_tdma_al, t_calc_tf_tdma_al, T_calc_tf_tdma_al = BTCS.calculate_BTCS_tf_tdma(rho, cp, k, L, Tw,
                                                                                                     T0, Te, x0, xf, t0,
                                                                                                     tf, qw, i, j, n_t,
                                                                                                     n_x)
                        fim_tdma = time.time()
                        tempo_total_tdma = fim_tdma - inicio_tdma

                        x_calc_array_tf_tdma_al.append(x_calc_tf_tdma_al)
                        t_calc_array_tf_tdma_al.append(t_calc_tf_tdma_al)
                        T_calc_array_tf_tdma_al.append(T_calc_tf_tdma_al)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_tdma_al.append(tempo_total_tdma)
            return x_calc_array_tf_tdma_cobre, t_calc_array_tf_tdma_cobre, T_calc_array_tf_tdma_cobre, tempo_total_array_tf_tdma_cobre, n_t_array, \
                x_calc_array_tf_tdma_al, t_calc_array_tf_tdma_al, T_calc_array_tf_tdma_al, tempo_total_array_tf_tdma_al

        def calculate_h_t_calc_jac(rho_list, k_list, cp_list):
            x_calc_array_tf_jac_cobre = []
            t_calc_array_tf_jac_cobre = []
            T_calc_array_tf_jac_cobre = []
            tempo_total_array_tf_jac_cobre = []
            n_t_array = []
            x_calc_array_tf_jac_al = []
            t_calc_array_tf_jac_al = []
            T_calc_array_tf_jac_al = []
            tempo_total_array_tf_jac_al = []
            for m in range(len(rho_list)):
                if m == 0:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for i in h_t:
                        inicio_jac = time.time()
                        n_t = calculate_n_t(tf, t0, i)
                        x_calc_tf_jac_cobre, t_calc_tf_jac_cobre, p_calc_tf_jac_cobre = BTCS.calculate_BTCS_tf_jac(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                        fim_jac = time.time()
                        tempo_total_jac = fim_jac - inicio_jac

                        x_calc_array_tf_jac_cobre.append(x_calc_tf_jac_cobre)
                        T_calc_array_tf_jac_cobre.append(p_calc_tf_jac_cobre)
                        t_calc_array_tf_jac_cobre.append(t_calc_tf_jac_cobre)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_jac_cobre.append(tempo_total_jac)
                elif m == 1:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for i in h_t:
                        inicio_jac = time.time()
                        n_t = calculate_n_t(tf, t0, i)
                        x_calc_tf_jac_al, t_calc_tf_jac_al, T_calc_tf_jac_al = BTCS.calculate_BTCS_tf_jac(rho, cp, k, L, Tw,
                                                                                                     T0, Te, x0, xf, t0,
                                                                                                     tf, qw, i, j, n_t,
                                                                                                     n_x)
                        fim_jac = time.time()
                        tempo_total_jac = fim_jac - inicio_jac

                        x_calc_array_tf_jac_al.append(x_calc_tf_jac_al)
                        t_calc_array_tf_jac_al.append(t_calc_tf_jac_al)
                        T_calc_array_tf_jac_al.append(T_calc_tf_jac_al)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_jac_al.append(tempo_total_jac)
            return x_calc_array_tf_jac_cobre, t_calc_array_tf_jac_cobre, T_calc_array_tf_jac_cobre, tempo_total_array_tf_jac_cobre, n_t_array, \
                x_calc_array_tf_jac_al, t_calc_array_tf_jac_al, T_calc_array_tf_jac_al, tempo_total_array_tf_jac_al


        def calculate_h_t_calc_gsr(rho_list, k_list, cp_list):
            x_calc_array_tf_gsr_cobre = []
            t_calc_array_tf_gsr_cobre = []
            T_calc_array_tf_gsr_cobre = []
            tempo_total_array_tf_gsr_cobre = []
            n_t_array = []
            x_calc_array_tf_gsr_al = []
            t_calc_array_tf_gsr_al = []
            T_calc_array_tf_gsr_al = []
            tempo_total_array_tf_gsr_al = []
            for m in range(len(rho_list)):
                if m == 0:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for i in h_t:
                        inicio_gsr = time.time()
                        n_t = calculate_n_t(tf, t0, i)
                        x_calc_tf_gsr_cobre, t_calc_tf_gsr_cobre, T_calc_tf_gsr_cobre = BTCS.calculate_BTCS_tf_gsr(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                        fim_gsr = time.time()
                        tempo_total_gsr = fim_gsr - inicio_gsr

                        x_calc_array_tf_gsr_cobre.append(x_calc_tf_gsr_cobre)
                        T_calc_array_tf_gsr_cobre.append(T_calc_tf_gsr_cobre)
                        t_calc_array_tf_gsr_cobre.append(t_calc_tf_gsr_cobre)
                        n_t_array.append(n_t)
                        tempo_total_array_tf_gsr_cobre.append(tempo_total_gsr)
                elif m == 1:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for i in h_t:
                        inicio_gsr = time.time()
                        n_t = calculate_n_t(tf, t0, i)
                        x_calc_tf_gsr_al, t_calc_tf_gsr_al, T_calc_tf_gsr_al = BTCS.calculate_BTCS_tf_gsr(rho,
                                                                                                                   cp,
                                                                                                                   k, L,
                                                                                                                   Tw,
                                                                                                                   T0,
                                                                                                                   Te,
                                                                                                                   x0,
                                                                                                                   xf,
                                                                                                                   t0,
                                                                                                                   tf,
                                                                                                                   qw,
                                                                                                                   i, j,
                                                                                                                   n_t,
                                                                                                                   n_x)
                        fim_gsr = time.time()
                        tempo_total_gsr = fim_gsr - inicio_gsr

                        x_calc_array_tf_gsr_al.append(x_calc_tf_gsr_al)
                        T_calc_array_tf_gsr_al.append(T_calc_tf_gsr_al)
                        t_calc_array_tf_gsr_al.append(t_calc_tf_gsr_al)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_gsr_al.append(tempo_total_gsr)
            return x_calc_array_tf_gsr_cobre, t_calc_array_tf_gsr_cobre, T_calc_array_tf_gsr_cobre, tempo_total_array_tf_gsr_cobre, n_t_array, \
                x_calc_array_tf_gsr_al, t_calc_array_tf_gsr_al, T_calc_array_tf_gsr_al, tempo_total_array_tf_gsr_al

        def calculate_h_t_calc_solv(rho_list, k_list, cp_list):
            x_calc_array_tf_solv_cobre = []
            t_calc_array_tf_solv_cobre = []
            T_calc_array_tf_solv_cobre = []
            tempo_total_array_tf_solv_cobre = []
            n_t_array = []
            x_calc_array_tf_solv_al = []
            t_calc_array_tf_solv_al = []
            T_calc_array_tf_solv_al = []
            tempo_total_array_tf_solv_al = []
            for m in range(len(rho_list)):
                if m == 0:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for i in h_t:
                        inicio_solv = time.time()
                        n_t = calculate_n_t(tf, t0, i)
                        x_calc_tf_solv_cobre, t_calc_tf_solv_cobre, p_calc_tf_solv_cobre = BTCS.calculate_BTCS_tf_solv(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                        fim_solv = time.time()
                        tempo_total_solv = fim_solv - inicio_solv

                        x_calc_array_tf_solv_cobre.append(x_calc_tf_solv_cobre)
                        T_calc_array_tf_solv_cobre.append(p_calc_tf_solv_cobre)
                        t_calc_array_tf_solv_cobre.append(t_calc_tf_solv_cobre)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_solv_cobre.append(tempo_total_solv)
                elif m == 1:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for i in h_t:
                        inicio_solv = time.time()
                        n_t = calculate_n_t(tf, t0, i)
                        x_calc_tf_solv_al, t_calc_tf_solv_al, p_calc_tf_solv_al = BTCS.calculate_BTCS_tf_solv(rho, cp, k, L, Tw,
                                                                                                     T0, Te, x0, xf, t0,
                                                                                                     tf, qw, i, j, n_t,
                                                                                                     n_x)
                        fim_solv = time.time()
                        tempo_total_solv = fim_solv - inicio_solv

                        x_calc_array_tf_solv_al.append(x_calc_tf_solv_al)
                        T_calc_array_tf_solv_al.append(p_calc_tf_solv_al)
                        t_calc_array_tf_solv_al.append(t_calc_tf_solv_al)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_solv_al.append(tempo_total_solv)
            return x_calc_array_tf_solv_cobre, t_calc_array_tf_solv_cobre, T_calc_array_tf_solv_cobre, tempo_total_array_tf_solv_cobre, n_t_array, \
                x_calc_array_tf_solv_al, t_calc_array_tf_solv_al, T_calc_array_tf_solv_al, tempo_total_array_tf_solv_al


        x_calc_array_tf_gs_cobre, t_calc_array_tf_gs_cobre, T_calc_array_tf_gs_cobre, tempo_total_array_tf_gs_cobre, n_t_array, \
            x_calc_array_tf_gs_al, t_calc_array_tf_gs_al, T_calc_array_tf_gs_al, tempo_total_array_tf_gs_al = calculate_h_t_calc_gs(rho_list, k_list, cp_list)
        tempos_totais_cobre.append(tempo_total_array_tf_gs_cobre)
        tempos_totais_al.append(tempo_total_array_tf_gs_al)
        x_calc_array_tf_tdma_cobre, t_calc_array_tf_tdma_cobre, T_calc_array_tf_tdma_cobre, tempo_total_array_tf_tdma_cobre, n_t_array, \
            x_calc_array_tf_tdma_al, t_calc_array_tf_tdma_al, T_calc_array_tf_tdma_al, tempo_total_array_tf_tdma_al = calculate_h_t_calc_tdma(rho_list, k_list, cp_list)
        tempos_totais_cobre.append(tempo_total_array_tf_tdma_cobre)
        tempos_totais_al.append(tempo_total_array_tf_tdma_al)
        x_calc_array_tf_jac_cobre, t_calc_array_tf_jac_cobre, T_calc_array_tf_jac_cobre, tempo_total_array_tf_jac_cobre, n_t_array, \
            x_calc_array_tf_jac_al, t_calc_array_tf_jac_al, T_calc_array_tf_jac_al, tempo_total_array_tf_jac_al = calculate_h_t_calc_jac(
            rho_list, k_list, cp_list)
        tempos_totais_cobre.append(tempo_total_array_tf_jac_cobre)
        tempos_totais_al.append(tempo_total_array_tf_jac_al)
        x_calc_array_tf_gsr_cobre, t_calc_array_tf_gsr_cobre, T_calc_array_tf_gsr_cobre, tempo_total_array_tf_gsr_cobre, n_t_array, \
            x_calc_array_tf_gsr_al, t_calc_array_tf_gsr_al, T_calc_array_tf_gsr_al, tempo_total_array_tf_gsr_al = calculate_h_t_calc_gsr(
            rho_list, k_list, cp_list)
        tempos_totais_cobre.append(tempo_total_array_tf_gsr_cobre)
        tempos_totais_al.append(tempo_total_array_tf_gsr_al)
        x_calc_array_tf_solv_cobre, t_calc_array_tf_solv_cobre, T_calc_array_tf_solv_cobre, tempo_total_array_tf_solv_cobre, n_t_array, \
            x_calc_array_tf_solv_al, t_calc_array_tf_solv_al, T_calc_array_tf_solv_al, tempo_total_array_tf_solv_al = calculate_h_t_calc_solv(
            rho_list, k_list, cp_list)
        tempos_totais_cobre.append(tempo_total_array_tf_solv_cobre)
        tempos_totais_al.append(tempo_total_array_tf_solv_al)

        print('h_t_cobre', tempos_totais_cobre)
        print('h_t_al', tempos_totais_al)

        solvers = ['Guass Seidel', 'TDMA', 'Jacobi', 'Guass Seidel Relaxamento', 'Solver Scipy']
        print('n_x', n_x)

        # Tabelas:
        tabela = PrettyTable(['N° de Blocos', 'Steps de Tempo', 'Solver', 'Tempo Computacional [s] - Neumann - Cobre', 'Tempo Computacional [s] - Neumann - Alumínio'])
        for solver_idx, solver in enumerate(solvers):  # enumerate adiciona um contador à iteração
            tempos_solver_cobre = tempos_totais_cobre[solver_idx]  # obtida a lista de tempos correspondente ao solver da iteração
            tempos_solver_al = tempos_totais_al[solver_idx]  # obtida a lista de tempos correspondente ao solver da iteração
            for n_t_idx, delta_t in enumerate(n_t_array):  # enumerate adiciona um contador à iteração
                tempo_total_cobre = tempos_solver_cobre[n_t_idx]  # pega o tempo correspondente ao delta t
                tempo_total_al = tempos_solver_al[n_t_idx]  # pega o tempo correspondente ao delta t
                rounded_value_cobre = round(tempo_total_cobre, 3)
                rounded_value_cobre = round(tempo_total_al, 3)
                delta_t_rounded = round(delta_t, 3)
                tabela.add_row(
                    [n_x, delta_t_rounded, solver, rounded_value_cobre, rounded_value_cobre])

        print(tabela)

        return tempo_total_array_tf_gs_cobre,  tempo_total_array_tf_tdma_cobre, tempo_total_array_tf_jac_cobre, tempo_total_array_tf_gsr_cobre, tempo_total_array_tf_solv_cobre, n_t_array, n_x, \
            tempo_total_array_tf_gs_al,  tempo_total_array_tf_tdma_al, tempo_total_array_tf_jac_al, tempo_total_array_tf_gsr_al, tempo_total_array_tf_solv_al
    def calculate_tempo_computacional_h_x():

        rho_list = [8.92, 9]  # g/cm³
        cp_list = [0.092, 0.080]  # cal/(g.ºC)
        k_list = [0.95, 0.92]  # cal/(cm.s.ºC)
        qw = 25
        L = 80  # cm
        T0 = 50  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        t0 = 0
        tf = 100
        x0 = 0
        xf = L

        h_x = [25, 15]
        h_t = 0.8
        i = h_t

        tempos_totais_cobre = []
        tempos_totais_al= []

        def calculate_n_x(xf, x0, j):
            n_x = (xf - x0) / (j)
            return n_x


        n_t = (tf - t0) / (h_t)

        def calculate_h_x_calc_gs(rho_list, k_list, cp_list):
            x_calc_array_tf_gs_cobre = []
            t_calc_array_tf_gs_cobre  = []
            T_calc_array_tf_gs_cobre  = []
            tempo_total_array_tf_gs_cobre  = []
            n_x_array = []
            x_calc_array_tf_gs_al = []
            t_calc_array_tf_gs_al = []
            T_calc_array_tf_gs_al = []
            tempo_total_array_tf_gs_al = []
            for m in range(len(rho_list)):
                if m == 0:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for j in h_x:
                        inicio_gs = time.time()
                        n_x = calculate_n_x(xf, x0, j)
                        x_calc_tf_gs_cobre, t_calc_tf_gs_cobre, T_calc_tf_gs_cobre = BTCS.calculate_BTCS_tf_gs(rho, cp,
                                                                                                               k, L, Tw,
                                                                                                               T0, Te,
                                                                                                               x0, xf,
                                                                                                               t0, tf,
                                                                                                               qw, i, j,
                                                                                                               n_t, n_x)
                        fim_gs = time.time()
                        tempo_total_gs = fim_gs - inicio_gs

                        x_calc_array_tf_gs_cobre.append(x_calc_tf_gs_cobre)
                        T_calc_array_tf_gs_cobre.append(T_calc_tf_gs_cobre)
                        t_calc_array_tf_gs_cobre.append(t_calc_tf_gs_cobre)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_gs_cobre.append(tempo_total_gs)
                elif m == 1:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for j in h_x:
                        inicio_gs = time.time()
                        n_x = calculate_n_x(xf, x0, j)
                        x_calc_tf_gs_al, t_calc_tf_gs_al, T_calc_tf_gs_al = BTCS.calculate_BTCS_tf_gs(rho, cp, k, L, Tw,
                                                                                                      T0, Te,
                                                                                                      x0, xf, t0, tf,
                                                                                                      qw, i, j,
                                                                                                      n_t, n_x)
                        fim_gs = time.time()
                        tempo_total_gs = fim_gs - inicio_gs

                        x_calc_array_tf_gs_al.append(x_calc_tf_gs_al)
                        T_calc_array_tf_gs_al.append(T_calc_tf_gs_al)
                        t_calc_array_tf_gs_al.append(t_calc_tf_gs_al)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_gs_al.append(tempo_total_gs)
            return x_calc_array_tf_gs_cobre, t_calc_array_tf_gs_cobre, T_calc_array_tf_gs_cobre, tempo_total_array_tf_gs_cobre, n_t_array, \
                x_calc_array_tf_gs_al, t_calc_array_tf_gs_al, T_calc_array_tf_gs_al, tempo_total_array_tf_gs_al

        def calculate_h_x_calc_tdma(rho_list, k_list, cp_list):
            x_calc_array_tf_tdma_cobre = []
            t_calc_array_tf_tdma_cobre = []
            T_calc_array_tf_tdma_cobre = []
            tempo_total_array_tf_tdma_cobre = []
            x_calc_array_tf_tdma_al = []
            t_calc_array_tf_tdma_al = []
            T_calc_array_tf_tdma_al = []
            tempo_total_array_tf_tdma_al = []
            n_t_array = []
            for m in range(len(rho_list)):
                if m == 0:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for j in h_x:
                        inicio_tdma = time.time()
                        n_x = calculate_n_x(xf, x0, j)
                        x_calc_tf_tdma_cobre, t_calc_tf_tdma_cobre, T_calc_tf_tdma_cobre = BTCS.calculate_BTCS_tf_tdma(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                        fim_tdma = time.time()
                        tempo_total_tdma = fim_tdma - inicio_tdma

                        x_calc_array_tf_tdma_cobre.append(x_calc_tf_tdma_cobre)
                        t_calc_array_tf_tdma_cobre.append(t_calc_tf_tdma_cobre)
                        T_calc_array_tf_tdma_cobre.append(T_calc_tf_tdma_cobre)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_tdma_cobre.append(tempo_total_tdma)
                elif m == 1:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for j in h_x:
                        inicio_tdma = time.time()
                        n_x = calculate_n_x(xf, x0, j)
                        x_calc_tf_tdma_al, t_calc_tf_tdma_al, T_calc_tf_tdma_al = BTCS.calculate_BTCS_tf_tdma(rho, cp, k, L, Tw,
                                                                                                     T0, Te, x0, xf, t0,
                                                                                                     tf, qw, i, j, n_t,
                                                                                                     n_x)
                        fim_tdma = time.time()
                        tempo_total_tdma = fim_tdma - inicio_tdma

                        x_calc_array_tf_tdma_al.append(x_calc_tf_tdma_al)
                        t_calc_array_tf_tdma_al.append(t_calc_tf_tdma_al)
                        T_calc_array_tf_tdma_al.append(T_calc_tf_tdma_al)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_tdma_al.append(tempo_total_tdma)
            return x_calc_array_tf_tdma_cobre, t_calc_array_tf_tdma_cobre, T_calc_array_tf_tdma_cobre, tempo_total_array_tf_tdma_cobre, n_t_array, \
                x_calc_array_tf_tdma_al, t_calc_array_tf_tdma_al, T_calc_array_tf_tdma_al, tempo_total_array_tf_tdma_al

        def calculate_h_x_calc_jac(rho_list, k_list, cp_list):
            x_calc_array_tf_jac_cobre = []
            t_calc_array_tf_jac_cobre = []
            T_calc_array_tf_jac_cobre = []
            tempo_total_array_tf_jac_cobre = []
            n_t_array = []
            x_calc_array_tf_jac_al = []
            t_calc_array_tf_jac_al = []
            T_calc_array_tf_jac_al = []
            tempo_total_array_tf_jac_al = []
            for m in range(len(rho_list)):
                if m == 0:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for j in h_x:
                        inicio_jac = time.time()
                        n_x = calculate_n_x(xf, x0, j)
                        x_calc_tf_jac_cobre, t_calc_tf_jac_cobre, p_calc_tf_jac_cobre = BTCS.calculate_BTCS_tf_jac(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                        fim_jac = time.time()
                        tempo_total_jac = fim_jac - inicio_jac

                        x_calc_array_tf_jac_cobre.append(x_calc_tf_jac_cobre)
                        T_calc_array_tf_jac_cobre.append(p_calc_tf_jac_cobre)
                        t_calc_array_tf_jac_cobre.append(t_calc_tf_jac_cobre)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_jac_cobre.append(tempo_total_jac)
                elif m == 1:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for j in h_x:
                        inicio_jac = time.time()
                        n_x = calculate_n_x(xf, x0, j)
                        x_calc_tf_jac_al, t_calc_tf_jac_al, T_calc_tf_jac_al = BTCS.calculate_BTCS_tf_jac(rho, cp, k, L, Tw,
                                                                                                     T0, Te, x0, xf, t0,
                                                                                                     tf, qw, i, j, n_t,
                                                                                                     n_x)
                        fim_jac = time.time()
                        tempo_total_jac = fim_jac - inicio_jac

                        x_calc_array_tf_jac_al.append(x_calc_tf_jac_al)
                        t_calc_array_tf_jac_al.append(t_calc_tf_jac_al)
                        T_calc_array_tf_jac_al.append(T_calc_tf_jac_al)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_jac_al.append(tempo_total_jac)
            return x_calc_array_tf_jac_cobre, t_calc_array_tf_jac_cobre, T_calc_array_tf_jac_cobre, tempo_total_array_tf_jac_cobre, n_t_array, \
                x_calc_array_tf_jac_al, t_calc_array_tf_jac_al, T_calc_array_tf_jac_al, tempo_total_array_tf_jac_al

        def calculate_h_x_calc_gsr(rho_list, k_list, cp_list):
            x_calc_array_tf_gsr_cobre = []
            t_calc_array_tf_gsr_cobre = []
            T_calc_array_tf_gsr_cobre = []
            tempo_total_array_tf_gsr_cobre = []
            n_t_array = []
            x_calc_array_tf_gsr_al = []
            t_calc_array_tf_gsr_al = []
            T_calc_array_tf_gsr_al = []
            tempo_total_array_tf_gsr_al = []
            for m in range(len(rho_list)):
                if m == 0:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for j in h_x:
                        inicio_gsr = time.time()
                        n_x = calculate_n_x(xf, x0, j)
                        x_calc_tf_gsr_cobre, t_calc_tf_gsr_cobre, T_calc_tf_gsr_cobre = BTCS.calculate_BTCS_tf_gsr(rho,
                                                                                                                   cp,
                                                                                                                   k, L,
                                                                                                                   Tw,
                                                                                                                   T0,
                                                                                                                   Te,
                                                                                                                   x0,
                                                                                                                   xf,
                                                                                                                   t0,
                                                                                                                   tf,
                                                                                                                   qw,
                                                                                                                   i, j,
                                                                                                                   n_t,
                                                                                                                   n_x)
                        fim_gsr = time.time()
                        tempo_total_gsr = fim_gsr - inicio_gsr

                        x_calc_array_tf_gsr_cobre.append(x_calc_tf_gsr_cobre)
                        T_calc_array_tf_gsr_cobre.append(T_calc_tf_gsr_cobre)
                        t_calc_array_tf_gsr_cobre.append(t_calc_tf_gsr_cobre)
                        n_t_array.append(n_t)
                elif m == 1:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for j in h_x:
                        inicio_gsr = time.time()
                        n_x = calculate_n_x(xf, x0, j)
                        x_calc_tf_gsr_al, t_calc_tf_gsr_al, T_calc_tf_gsr_al = BTCS.calculate_BTCS_tf_gsr(rho,
                                                                                                          cp,
                                                                                                          k, L,
                                                                                                          Tw,
                                                                                                          T0,
                                                                                                          Te,
                                                                                                          x0,
                                                                                                          xf,
                                                                                                          t0,
                                                                                                          tf,
                                                                                                          qw,
                                                                                                          i, j,
                                                                                                          n_t,
                                                                                                          n_x)
                        fim_gsr = time.time()
                        tempo_total_gsr = fim_gsr - inicio_gsr

                        x_calc_array_tf_gsr_al.append(x_calc_tf_gsr_al)
                        T_calc_array_tf_gsr_al.append(T_calc_tf_gsr_al)
                        t_calc_array_tf_gsr_al.append(t_calc_tf_gsr_al)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_gsr_al.append(tempo_total_gsr)
            return x_calc_array_tf_gsr_cobre, t_calc_array_tf_gsr_cobre, T_calc_array_tf_gsr_cobre, tempo_total_array_tf_gsr_cobre, n_t_array, \
                x_calc_array_tf_gsr_al, t_calc_array_tf_gsr_al, T_calc_array_tf_gsr_al, tempo_total_array_tf_gsr_al
        def calculate_h_x_calc_solv(rho_list, k_list, cp_list):
            x_calc_array_tf_solv_cobre = []
            t_calc_array_tf_solv_cobre = []
            T_calc_array_tf_solv_cobre = []
            tempo_total_array_tf_solv_cobre = []
            n_t_array = []
            x_calc_array_tf_solv_al = []
            t_calc_array_tf_solv_al = []
            T_calc_array_tf_solv_al = []
            tempo_total_array_tf_solv_al = []
            for m in range(len(rho_list)):
                if m == 0:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for j in h_x:
                        inicio_solv = time.time()
                        n_x = calculate_n_x(xf, x0, j)
                        x_calc_tf_solv_cobre, t_calc_tf_solv_cobre, p_calc_tf_solv_cobre = BTCS.calculate_BTCS_tf_solv(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                        fim_solv = time.time()
                        tempo_total_solv = fim_solv - inicio_solv

                        x_calc_array_tf_solv_cobre.append(x_calc_tf_solv_cobre)
                        T_calc_array_tf_solv_cobre.append(p_calc_tf_solv_cobre)
                        t_calc_array_tf_solv_cobre.append(t_calc_tf_solv_cobre)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_solv_cobre.append(tempo_total_solv)
                elif m == 1:
                    rho = rho_list[m]
                    k = k_list[m]
                    cp = cp_list[m]
                    for j in h_x:
                        inicio_solv = time.time()
                        n_x = calculate_n_x(xf, x0, j)
                        x_calc_tf_solv_al, t_calc_tf_solv_al, p_calc_tf_solv_al = BTCS.calculate_BTCS_tf_solv(rho, cp, k, L, Tw,
                                                                                                     T0, Te, x0, xf, t0,
                                                                                                     tf, qw, i, j, n_t,
                                                                                                     n_x)
                        fim_solv = time.time()
                        tempo_total_solv = fim_solv - inicio_solv

                        x_calc_array_tf_solv_al.append(x_calc_tf_solv_al)
                        T_calc_array_tf_solv_al.append(p_calc_tf_solv_al)
                        t_calc_array_tf_solv_al.append(t_calc_tf_solv_al)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_solv_al.append(tempo_total_solv)
            return x_calc_array_tf_solv_cobre, t_calc_array_tf_solv_cobre, T_calc_array_tf_solv_cobre, tempo_total_array_tf_solv_cobre, n_t_array, \
                x_calc_array_tf_solv_al, t_calc_array_tf_solv_al, T_calc_array_tf_solv_al, tempo_total_array_tf_solv_al

        x_calc_array_tf_gs_cobre, t_calc_array_tf_gs_cobre, T_calc_array_tf_gs_cobre, tempo_total_array_tf_gs_cobre, n_t_array, \
            x_calc_array_tf_gs_al, t_calc_array_tf_gs_al, T_calc_array_tf_gs_al, tempo_total_array_tf_gs_al = calculate_h_x_calc_gs(rho_list, k_list, cp_list)
        tempos_totais_cobre.append(tempo_total_array_tf_gs_cobre)
        tempos_totais_al.append(tempo_total_array_tf_gs_al)
        x_calc_array_tf_tdma_cobre, t_calc_array_tf_tdma_cobre, T_calc_array_tf_tdma_cobre, tempo_total_array_tf_tdma_cobre, n_t_array, \
            x_calc_array_tf_tdma_al, t_calc_array_tf_tdma_al, T_calc_array_tf_tdma_al, tempo_total_array_tf_tdma_al = calculate_h_x_calc_tdma(rho_list, k_list, cp_list)
        tempos_totais_cobre.append(tempo_total_array_tf_tdma_cobre)
        tempos_totais_al.append(tempo_total_array_tf_tdma_al)
        x_calc_array_tf_jac_cobre, t_calc_array_tf_jac_cobre, T_calc_array_tf_jac_cobre, tempo_total_array_tf_jac_cobre, n_t_array, \
            x_calc_array_tf_jac_al, t_calc_array_tf_jac_al, T_calc_array_tf_jac_al, tempo_total_array_tf_jac_al = calculate_h_x_calc_jac(rho_list, k_list, cp_list)
        tempos_totais_cobre.append(tempo_total_array_tf_jac_cobre)
        tempos_totais_al.append(tempo_total_array_tf_jac_al)
        x_calc_array_tf_gsr_cobre, t_calc_array_tf_gsr_cobre, T_calc_array_tf_gsr_cobre, tempo_total_array_tf_gsr_cobre, n_t_array, \
            x_calc_array_tf_gsr_al, t_calc_array_tf_gsr_al, T_calc_array_tf_gsr_al, tempo_total_array_tf_gsr_al = calculate_h_x_calc_gsr(rho_list, k_list, cp_list)
        tempos_totais_cobre.append(tempo_total_array_tf_gsr_cobre)
        tempos_totais_al.append(tempo_total_array_tf_gsr_al)
        x_calc_array_tf_solv_cobre, t_calc_array_tf_solv_cobre, T_calc_array_tf_solv_cobre, tempo_total_array_tf_solv_cobre, n_t_array, \
            x_calc_array_tf_solv_al, t_calc_array_tf_solv_al, T_calc_array_tf_solv_al, tempo_total_array_tf_solv_al = calculate_h_x_calc_solv(rho_list, k_list, cp_list)
        tempos_totais_cobre.append(tempo_total_array_tf_solv_cobre)
        tempos_totais_al.append(tempo_total_array_tf_solv_al)

        solvers = ['Guass Seidel', 'TDMA', 'Jacobi', 'Guass Seidel Relaxamento', 'Solver Scipy']
        print('n_x', n_t)

        # Tabelas:
        tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'Solver', 'Tempo Computacional [s] - Dirchlet - Cobre', 'Tempo Computacional [s] - Dirchlet - Alumínio'])
        for solver_idx, solver in enumerate(solvers): # enumerate adiciona um contador à iteração
            tempos_solver_cobre = tempos_totais_cobre[solver_idx] # obtida a lista de tempos correspondente ao solver da iteração
            tempos_solver_al = tempos_totais_al[solver_idx]  # obtida a lista de tempos correspondente ao solver da iteração
            for n_x_idx, delta_x in enumerate(n_x_array): # enumerate adiciona um contador à iteração
                tempo_total_cobre = tempos_solver_cobre[n_x_idx] # pega o tempo correspondente ao delta t
                tempo_total_al = tempos_solver_al[n_x_idx]  # pega o tempo correspondente ao delta t
                rounded_value_cobre = round(tempo_total_cobre, 3)
                rounded_value_al = round(tempo_total_al, 3)
                tempo_total_str_cobre = str(rounded_value_cobre)
                tempo_total_str_al = str(rounded_value_al)
                tabela.add_row([n_t, delta_x, solver, tempo_total_str_cobre, tempo_total_str_al])

        print(tabela)

        print(tabela)

        return tempo_total_array_tf_gs_cobre,  tempo_total_array_tf_tdma_cobre, tempo_total_array_tf_jac_cobre, tempo_total_array_tf_gsr_cobre, tempo_total_array_tf_solv_cobre, n_x_array, n_t, \
            tempo_total_array_tf_gs_al,  tempo_total_array_tf_tdma_al, tempo_total_array_tf_jac_al, tempo_total_array_tf_gsr_al, tempo_total_array_tf_solv_al


tempos_totais_h_t_tf_neumann_cobre = []
tempos_totais_h_x_tf_neumann_cobre = []
tempos_totais_h_t_tf_neumann_al = []
tempos_totais_h_x_tf_neumann_al = []
tempo_total_array_tf_gs_cobre,  tempo_total_array_tf_tdma_cobre, tempo_total_array_tf_jac_cobre, tempo_total_array_tf_gsr_cobre, tempo_total_array_tf_solv_cobre, n_t_array, n_x, \
    tempo_total_array_tf_gs_al,  tempo_total_array_tf_tdma_al, tempo_total_array_tf_jac_al, tempo_total_array_tf_gsr_al, tempo_total_array_tf_solv_al = tempo_computacional_tf.calculate_tempo_computacional_h_t()
tempos_totais_h_t_tf_neumann_cobre.append(tempo_total_array_tf_gs_cobre)
tempos_totais_h_t_tf_neumann_cobre.append(tempo_total_array_tf_tdma_cobre)
tempos_totais_h_t_tf_neumann_cobre.append(tempo_total_array_tf_jac_cobre)
tempos_totais_h_t_tf_neumann_cobre.append(tempo_total_array_tf_gsr_cobre)
tempos_totais_h_t_tf_neumann_cobre.append(tempo_total_array_tf_solv_cobre)
tempos_totais_h_t_tf_neumann_al.append(tempo_total_array_tf_gs_al)
tempos_totais_h_t_tf_neumann_al.append(tempo_total_array_tf_tdma_al)
tempos_totais_h_t_tf_neumann_al.append(tempo_total_array_tf_jac_al)
tempos_totais_h_t_tf_neumann_al.append(tempo_total_array_tf_gsr_al)
tempos_totais_h_t_tf_neumann_al.append(tempo_total_array_tf_solv_al)
tempo_total_array_tf_gs_cobre,  tempo_total_array_tf_tdma_cobre, tempo_total_array_tf_jac_cobre, tempo_total_array_tf_gsr_cobre, tempo_total_array_tf_solv_cobre, n_x_array, n_t, \
            tempo_total_array_tf_gs_al,  tempo_total_array_tf_tdma_al, tempo_total_array_tf_jac_al, tempo_total_array_tf_gsr_al, tempo_total_array_tf_solv_al = tempo_computacional_tf.calculate_tempo_computacional_h_x()
tempos_totais_h_x_tf_neumann_cobre.append(tempo_total_array_tf_gs_cobre)
tempos_totais_h_x_tf_neumann_cobre.append(tempo_total_array_tf_tdma_cobre)
tempos_totais_h_x_tf_neumann_cobre.append(tempo_total_array_tf_jac_cobre)
tempos_totais_h_x_tf_neumann_cobre.append(tempo_total_array_tf_gsr_cobre)
tempos_totais_h_x_tf_neumann_cobre.append(tempo_total_array_tf_solv_cobre)
tempos_totais_h_x_tf_neumann_al.append(tempo_total_array_tf_gs_al)
tempos_totais_h_x_tf_neumann_al.append(tempo_total_array_tf_tdma_al)
tempos_totais_h_x_tf_neumann_al.append(tempo_total_array_tf_jac_al)
tempos_totais_h_x_tf_neumann_al.append(tempo_total_array_tf_gsr_al)
tempos_totais_h_x_tf_neumann_al.append(tempo_total_array_tf_solv_al)
print('h_t_cobre', tempos_totais_h_t_tf_neumann_cobre)
print('h_x_cobre', tempos_totais_h_x_tf_neumann_cobre)
print('h_t_al', tempos_totais_h_t_tf_neumann_al)
print('h_x_al', tempos_totais_h_x_tf_neumann_al)
# Tabela Tempo

solvers = ['Guass Seidel', 'TDMA', 'Jacobi', 'Guass Seidel Relaxamento', 'Solver Scipy']

# Tabelas:
tabela = PrettyTable(['N° de Blocos', 'Steps de Tempo', 'Solver', 'Tempo Computacional [s] - Neumann - Cobre', 'Tempo Computacional [s] - Neumann - Alumínio'])
for solver_idx, solver in enumerate(solvers):  # enumerate adiciona um contador à iteração
    tempos_solver_neumann_cobre = tempos_totais_h_t_tf_neumann_cobre[solver_idx]  # obtida a lista de tempos correspondente ao solver da iteração
    tempos_solver_neumann_al = tempos_totais_h_t_tf_neumann_al[solver_idx]  # obtida a lista de tempos correspondente ao solver da iteração
    for n_t_idx, delta_t in enumerate(n_t_array):  # enumerate adiciona um contador à iteração
        tempo_total_neumann_cobre = tempos_solver_neumann_cobre[n_t_idx]  # pega o tempo correspondente ao delta t
        tempo_total_neumann_al = tempos_solver_neumann_al[n_t_idx]  # pega o tempo correspondente ao delta t
        rounded_value_neumann_cobre = round(tempo_total_neumann_cobre, 3)
        rounded_value_neumann_al = round(tempo_total_neumann_al, 3)
        tempo_total_str_neumann_cobre = str(rounded_value_neumann_cobre)
        tempo_total_str_neumann_al = str(rounded_value_neumann_al)
        delta_t_rounded = round(delta_t, 3)
        tabela.add_row([n_x, delta_t_rounded, solver, tempo_total_str_neumann_cobre, tempo_total_str_neumann_al])

print(tabela)

# Tabela Malha

solvers = ['Guass Seidel', 'TDMA', 'Jacobi', 'Guass Seidel Relaxamento', 'Solver Scipy']

# Tabelas:
tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'Solver', 'Tempo Computacional [s] - Neumann - Cobre', 'Tempo Computacional [s] - Neumann - Alumínio'])
for solver_idx, solver in enumerate(solvers):  # enumerate adiciona um contador à iteração
    tempos_solver_neumann_cobre = tempos_totais_h_x_tf_neumann_cobre[solver_idx]  # obtida a lista de tempos correspondente ao solver da iteração
    tempos_solver_neumann_al = tempos_totais_h_x_tf_neumann_al[solver_idx]  # obtida a lista de tempos correspondente ao solver da iteração
    for n_x_idx, delta_x in enumerate(n_x_array):  # enumerate adiciona um contador à iteração
        tempo_total_neumann_cobre = tempos_solver_neumann_cobre[n_x_idx]  # pega o tempo correspondente ao delta t
        tempo_total_neumann_al = tempos_solver_neumann_al[n_x_idx]  # pega o tempo correspondente ao delta t
        rounded_value_neumann_cobre = round(tempo_total_neumann_cobre, 3)
        rounded_value_neumann_al = round(tempo_total_neumann_al, 3)
        tempo_total_str_neumann_cobre = str(rounded_value_neumann_cobre)
        tempo_total_str_neumann_al = str(rounded_value_neumann_al)
        delta_x_rounded = round(delta_x, 3)
        tabela.add_row([n_t, delta_x_rounded, solver, tempo_total_str_neumann_cobre, tempo_total_str_neumann_al])

print(tabela)