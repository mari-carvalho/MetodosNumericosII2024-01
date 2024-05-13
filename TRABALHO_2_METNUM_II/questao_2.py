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

        rho = [8.92, 9]  # g/cm³
        cp = [0.092, 0.080]  # cal/(g.ºC)
        k = [0.95, 0.92]  # cal/(cm.s.ºC)
        qw = 25
        L = 80  # cm
        T0 = 50  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        t0 = 0
        tf = 100
        x0 = 0
        xf = L

        h_t = [0.1, 0.05, 0.005]
        h_x = 1
        j = h_x

        tempos_totais = []

        def calculate_n_t(tf, t0, i):
            n_t = (tf - t0) / (i)
            return n_t


        n_x = (xf - x0) / (h_x)

        def calculate_h_t_calc_gs(rho):
            x_calc_array_tf_gs_cobre = []
            t_calc_array_tf_gs_cobre = []
            T_calc_array_tf_gs_cobre = []
            tempo_total_array_tf_gs_cobre = []
            x_calc_array_tf_gs_al = []
            t_calc_array_tf_gs_al = []
            T_calc_array_tf_gs_al = []
            tempo_total_array_tf_gs_al = []
            n_t_array = []
            for m in rho:
                if m == 1:
                    rho = rho[m]
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
                elif m == 2:
                    rho = rho[m]
                    for i in h_t:
                        inicio_gs = time.time()
                        n_t = calculate_n_t(tf, t0, i)
                        x_calc_tf_gs_cobre, t_calc_tf_gs_cobre, T_calc_tf_gs_cobre = BTCS.calculate_BTCS_tf_gs(rho, cp, k, L, Tw, T0, Te,
                                                                                             x0, xf, t0, tf, qw, i, j,
                                                                                             n_t, n_x)
                        fim_gs = time.time()
                        tempo_total_gs = fim_gs - inicio_gs

                        x_calc_array_tf_gs_cobre.append(x_calc_tf_gs_cobre)
                        T_calc_array_tf_gs_cobre.append(T_calc_tf_gs_cobre)
                        t_calc_array_tf_gs_cobre.append(t_calc_tf_gs_cobre)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_gs_cobre.append(tempo_total_gs)

            return x_calc_array_tf_gs_cobre, t_calc_array_tf_gs_cobre, T_calc_array_tf_gs_cobre, tempo_total_array_tf_gs_cobre, n_t_array, \
                x_calc_array_tf_gs_al, t_calc_array_tf_gs_al, T_calc_array_tf_gs_al, tempo_total_array_tf_gs_al

        def calculate_h_t_calc_tdma(rho):
            x_calc_array_tf_tdma_cobre = []
            t_calc_array_tf_tdma_cobre = []
            T_calc_array_tf_tdma_cobre = []
            tempo_total_array_tf_tdma_cobre = []
            x_calc_array_tf_tdma_al = []
            t_calc_array_tf_tdma_al= []
            T_calc_array_tf_tdma_al = []
            tempo_total_array_tf_tdma_al = []
            n_t_array = []
            for m in rho:
                if m == 1:
                    rho = rho[m]
                    for i in h_t:
                        inicio_tdma = time.time()
                        n_t = calculate_n_t(tf, t0, i)
                        x_calc_tf_tdma, t_calc_tf_tdma, T_calc_tf_tdma = BTCS.calculate_BTCS_tf_tdma(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                        fim_tdma = time.time()
                        tempo_total_tdma = fim_tdma - inicio_tdma

                        x_calc_array_tf_tdma.append(x_calc_tf_tdma)
                        t_calc_array_tf_tdma.append(t_calc_tf_tdma)
                        T_calc_array_tf_tdma.append(T_calc_tf_tdma)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_tdma.append(tempo_total_tdma)
                elif m == 2:
                    rho = rho[m]
                    for i in h_t:
                        inicio_tdma = time.time()
                        n_t = calculate_n_t(tf, t0, i)
                        x_calc_tf_tdma, t_calc_tf_tdma, T_calc_tf_tdma = BTCS.calculate_BTCS_tf_tdma(rho, cp, k, L, Tw,
                                                                                                     T0, Te, x0, xf, t0,
                                                                                                     tf, qw, i, j, n_t,
                                                                                                     n_x)
                        fim_tdma = time.time()
                        tempo_total_tdma = fim_tdma - inicio_tdma

                        x_calc_array_tf_tdma.append(x_calc_tf_tdma)
                        t_calc_array_tf_tdma.append(t_calc_tf_tdma)
                        T_calc_array_tf_tdma.append(T_calc_tf_tdma)
                        n_t_array.append(n_t)

                        tempo_total_array_tf_tdma.append(tempo_total_tdma)
            return x_calc_array_tf_tdma, t_calc_array_tf_tdma, T_calc_array_tf_tdma, tempo_total_array_tf_tdma, n_t_array

        def calculate_h_t_calc_jac():
            x_calc_array_tf_jac = []
            t_calc_array_tf_jac = []
            T_calc_array_tf_jac = []
            tempo_total_array_tf_jac= []
            n_t_array = []
            for i in h_t:
                inicio_jac = time.time()
                n_t = calculate_n_t(tf, t0, i)
                x_calc_tf_jac, t_calc_tf_jac, p_calc_tf_jac = BTCS.calculate_BTCS_tf_jac(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                fim_jac = time.time()
                tempo_total_jac = fim_jac - inicio_jac

                x_calc_array_tf_jac.append(x_calc_tf_jac)
                T_calc_array_tf_jac.append(p_calc_tf_jac)
                t_calc_array_tf_jac.append(t_calc_tf_jac)
                n_t_array.append(n_t)

                tempo_total_array_tf_jac.append(tempo_total_jac)
            return x_calc_array_tf_jac, t_calc_array_tf_jac, T_calc_array_tf_jac, tempo_total_array_tf_jac, n_t_array

        def calculate_h_t_calc_gsr():
            x_calc_array_tf_gsr = []
            t_calc_array_tf_gsr= []
            T_calc_array_tf_gsr= []
            tempo_total_array_tf_gsr = []
            n_t_array = []
            for i in h_t:
                inicio_gsr = time.time()
                n_t = calculate_n_t(tf, t0, i)
                x_calc_tf_gsr, t_calc_tf_gsr, T_calc_tf_gsr = BTCS.calculate_BTCS_tf_gsr(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                fim_gsr = time.time()
                tempo_total_gsr = fim_gsr - inicio_gsr

                x_calc_array_tf_gsr.append(x_calc_tf_gsr)
                T_calc_array_tf_gsr.append(T_calc_tf_gsr)
                t_calc_array_tf_gsr.append(t_calc_tf_gsr)
                n_t_array.append(n_t)

                tempo_total_array_tf_gsr.append(tempo_total_gsr)
            return x_calc_array_tf_gsr, t_calc_array_tf_gsr, T_calc_array_tf_gsr, tempo_total_array_tf_gsr, n_t_array
        def calculate_h_t_calc_solv():
            x_calc_array_tf_solv = []
            t_calc_array_tf_solv = []
            T_calc_array_tf_solv = []
            tempo_total_array_tf_solv = []
            n_t_array = []
            for i in h_t:
                inicio_solv = time.time()
                n_t = calculate_n_t(tf, t0, i)
                x_calc_tf_solv, t_calc_tf_solv, p_calc_tf_solv = BTCS.calculate_BTCS_tf_solv(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                fim_solv = time.time()
                tempo_total_solv = fim_solv - inicio_solv

                x_calc_array_tf_solv.append(x_calc_tf_solv)
                T_calc_array_tf_solv.append(p_calc_tf_solv)
                t_calc_array_tf_solv.append(t_calc_tf_solv)
                n_t_array.append(n_t)

                tempo_total_array_tf_solv.append(tempo_total_solv)
            return x_calc_array_tf_solv, t_calc_array_tf_solv, T_calc_array_tf_solv, tempo_total_array_tf_solv, n_t_array

        x_calc_array_tf_gs, t_calc_array_tf_gs, T_calc_array_tf_gs, tempo_total_array_tf_gs, n_t_array = calculate_h_t_calc_gs()
        tempos_totais.append(tempo_total_array_tf_gs)
        x_calc_array_tf_tdma, t_calc_array_tf_tdma, T_calc_array_tf_tdma, tempo_total_array_tf_tdma, n_t_array = calculate_h_t_calc_tdma()
        tempos_totais.append(tempo_total_array_tf_tdma)
        x_calc_array_tf_jac, t_calc_array_tf_jac, T_calc_array_tf_jac, tempo_total_array_tf_jac, n_t_array = calculate_h_t_calc_jac()
        tempos_totais.append(tempo_total_array_tf_jac)
        x_calc_array_tf_gsr, t_calc_array_tf_gsr, T_calc_array_tf_gsr, tempo_total_array_tf_gsr, n_t_array = calculate_h_t_calc_gsr()
        tempos_totais.append(tempo_total_array_tf_gsr)
        x_calc_array_tf_solv, t_calc_array_tf_solv, T_calc_array_tf_solv, tempo_total_array_tf_solv, n_t_array = calculate_h_t_calc_solv()
        tempos_totais.append(tempo_total_array_tf_solv)

        solvers = ['Guass Seidel', 'TDMA', 'Jacobi', 'Guass Seidel Relaxamento', 'Solver Scipy']
        print('tempos', tempos_totais)
        print('n_x', n_x)

        # Tabelas:
        tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'Solver', 'Tempo Computacional [s] - Dirchlet'])
        for solver_idx, solver in enumerate(solvers): # enumerate adiciona um contador à iteração
            tempos_solver = tempos_totais[solver_idx] # obtida a lista de tempos correspondente ao solver da iteração
            for n_t_idx, delta_t in enumerate(n_t_array): # enumerate adiciona um contador à iteração
                tempo_total = tempos_solver[n_t_idx] # pega o tempo correspondente ao delta t
                rounded_value = round(tempo_total, 3)
                tempo_total_str = str(rounded_value)
                tabela.add_row([delta_t, n_x, solver, tempo_total_str])

        print(tabela)

        return tempo_total_array_tf_gs,  tempo_total_array_tf_tdma, tempo_total_array_tf_jac, tempo_total_array_tf_gsr, tempo_total_array_tf_solv, n_t_array, n_x
    def calculate_tempo_computacional_h_x():

        rho = 8.92  # g/cm³
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        qw = 25
        L = 80  # cm
        T0 = 50  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        t0 = 0
        tf = 100
        x0 = 0
        xf = L

        h_x = [5, 4, 3, 2, 1]
        h_t = 0.8
        i = h_t

        tempos_totais = []

        def calculate_n_x(xf, x0, j):
            n_x = (xf - x0) / (j)
            return n_x


        n_t = (tf - t0) / (h_t)

        def calculate_h_x_calc_gs():
            x_calc_array_tf_gs = []
            t_calc_array_tf_gs = []
            T_calc_array_tf_gs = []
            tempo_total_array_tf_gs = []
            n_x_array = []
            for j in h_x:
                inicio_gs = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_tf_gs, t_calc_tf_gs, T_calc_tf_gs = BTCS.calculate_BTCS_tf_gs(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                fim_gs = time.time()
                tempo_total_gs = fim_gs - inicio_gs

                x_calc_array_tf_gs.append(x_calc_tf_gs)
                T_calc_array_tf_gs.append(T_calc_tf_gs)
                t_calc_array_tf_gs.append(t_calc_tf_gs)
                n_x_array.append(n_x)

                tempo_total_array_tf_gs.append(tempo_total_gs)
            return x_calc_array_tf_gs, t_calc_array_tf_gs, T_calc_array_tf_gs, tempo_total_array_tf_gs, n_x_array

        def calculate_h_x_calc_tdma():
            x_calc_array_tf_tdma = []
            t_calc_array_tf_tdma = []
            T_calc_array_tf_tdma = []
            tempo_total_array_tf_tdma = []
            n_x_array = []
            for j in h_x:
                inicio_tdma = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_tf_tdma, t_calc_tf_tdma, T_calc_tf_tdma = BTCS.calculate_BTCS_tf_tdma(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                fim_tdma = time.time()
                tempo_total_tdma = fim_tdma - inicio_tdma

                x_calc_array_tf_tdma.append(x_calc_tf_tdma)
                T_calc_array_tf_tdma.append(T_calc_tf_tdma)
                T_calc_array_tf_tdma.append(T_calc_tf_tdma)
                n_x_array.append(n_x)

                tempo_total_array_tf_tdma.append(tempo_total_tdma)
            return x_calc_array_tf_tdma, t_calc_array_tf_tdma, T_calc_array_tf_tdma, tempo_total_array_tf_tdma, n_x_array

        def calculate_h_x_calc_jac():
            x_calc_array_tf_jac = []
            t_calc_array_tf_jac = []
            T_calc_array_tf_jac = []
            tempo_total_array_tf_jac= []
            n_x_array = []
            for j in h_x:
                inicio_jac = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_tf_jac, t_calc_tf_jac, T_calc_tf_jac = BTCS.calculate_BTCS_tf_jac(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                fim_jac = time.time()
                tempo_total_jac = fim_jac - inicio_jac

                x_calc_array_tf_jac.append(x_calc_tf_jac)
                T_calc_array_tf_jac.append(T_calc_tf_jac)
                t_calc_array_tf_jac.append(t_calc_tf_jac)
                n_x_array.append(n_x)

                tempo_total_array_tf_jac.append(tempo_total_jac)
            return x_calc_array_tf_jac, t_calc_array_tf_jac, T_calc_array_tf_jac, tempo_total_array_tf_jac, n_x_array

        def calculate_h_x_calc_gsr():
            x_calc_array_tf_gsr = []
            t_calc_array_tf_gsr= []
            T_calc_array_tf_gsr= []
            tempo_total_array_tf_gsr = []
            n_x_array = []
            for j in h_x:
                inicio_gsr = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_tf_gsr, t_calc_tf_gsr, T_calc_tf_gsr = BTCS.calculate_BTCS_tf_gsr(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                fim_gsr = time.time()
                tempo_total_gsr = fim_gsr - inicio_gsr

                x_calc_array_tf_gsr.append(x_calc_tf_gsr)
                T_calc_array_tf_gsr.append(T_calc_tf_gsr)
                t_calc_array_tf_gsr.append(t_calc_tf_gsr)
                n_x_array.append(n_x)

                tempo_total_array_tf_gsr.append(tempo_total_gsr)
            return x_calc_array_tf_gsr, t_calc_array_tf_gsr, T_calc_array_tf_gsr, tempo_total_array_tf_gsr, n_x_array
        def calculate_h_x_calc_solv():
            x_calc_array_tf_solv = []
            t_calc_array_tf_solv = []
            T_calc_array_tf_solv = []
            tempo_total_array_tf_solv = []
            n_x_array = []
            for j in h_x:
                inicio_solv = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_tf_solv, t_calc_tf_solv, T_calc_tf_solv = BTCS.calculate_BTCS_tf_solv(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x)
                fim_solv = time.time()
                tempo_total_solv = fim_solv - inicio_solv

                x_calc_array_tf_solv.append(x_calc_tf_solv)
                T_calc_array_tf_solv.append(T_calc_tf_solv)
                t_calc_array_tf_solv.append(t_calc_tf_solv)
                n_x_array.append(n_x)

                tempo_total_array_tf_solv.append(tempo_total_solv)
            return x_calc_array_tf_solv, t_calc_array_tf_solv, T_calc_array_tf_solv, tempo_total_array_tf_solv, n_x_array

        x_calc_array_tf_gs, t_calc_array_tf_gs, T_calc_array_tf_gs, tempo_total_array_tf_gs, n_x_array = calculate_h_x_calc_gs()
        tempos_totais.append(tempo_total_array_tf_gs)
        x_calc_array_tf_tdma, t_calc_array_tf_tdma, T_calc_array_tf_tdma, tempo_total_array_tf_tdma, n_x_array = calculate_h_x_calc_tdma()
        tempos_totais.append(tempo_total_array_tf_tdma)
        x_calc_array_tf_jac, t_calc_array_tf_jac, T_calc_array_tf_jac, tempo_total_array_tf_jac, n_x_array = calculate_h_x_calc_jac()
        tempos_totais.append(tempo_total_array_tf_jac)
        x_calc_array_tf_gsr, t_calc_array_tf_gsr, T_calc_array_tf_gsr, tempo_total_array_tf_gsr, n_x_array = calculate_h_x_calc_gsr()
        tempos_totais.append(tempo_total_array_tf_gsr)
        x_calc_array_tf_solv, t_calc_array_tf_solv, T_calc_array_tf_solv, tempo_total_array_tf_solv, n_x_array = calculate_h_x_calc_solv()
        tempos_totais.append(tempo_total_array_tf_solv)

        solvers = ['Guass Seidel', 'TDMA', 'Jacobi', 'Guass Seidel Relaxamento', 'Solver Scipy']
        print('tempos', tempos_totais)
        print('n_x', n_t)

        # Tabelas:
        tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'Solver', 'Tempo Computacional [s] - Dirchlet'])
        for solver_idx, solver in enumerate(solvers): # enumerate adiciona um contador à iteração
            tempos_solver = tempos_totais[solver_idx] # obtida a lista de tempos correspondente ao solver da iteração
            for n_x_idx, delta_x in enumerate(n_x_array): # enumerate adiciona um contador à iteração
                tempo_total = tempos_solver[n_x_idx] # pega o tempo correspondente ao delta t
                rounded_value = round(tempo_total, 3)
                tempo_total_str = str(rounded_value)
                delta_x_rounded = round(delta_x, 3)
                tabela.add_row([n_t, delta_x_rounded, solver, tempo_total_str])

        print(tabela)

        return tempo_total_array_tf_gs,  tempo_total_array_tf_tdma, tempo_total_array_tf_jac, tempo_total_array_tf_gsr, tempo_total_array_tf_solv, n_x_array, n_t


tempos_totais_h_t_tf_neumann = []
tempos_totais_h_x_tf_neumann = []
tempo_total_array_tf_gs, tempo_total_array_tf_tdma, tempo_total_array_tf_jac, tempo_total_array_tf_gsr, tempo_total_array_tf_solv, n_t_array, n_x = tempo_computacional_tf.calculate_tempo_computacional_h_t()
tempos_totais_h_t_tf_neumann.append(tempo_total_array_tf_gs)
tempos_totais_h_t_tf_neumann.append(tempo_total_array_tf_tdma)
tempos_totais_h_t_tf_neumann.append(tempo_total_array_tf_jac)
tempos_totais_h_t_tf_neumann.append(tempo_total_array_tf_gsr)
tempos_totais_h_t_tf_neumann.append(tempo_total_array_tf_solv)
tempo_total_array_tf_gs, tempo_total_array_tf_tdma, tempo_total_array_tf_jac, tempo_total_array_tf_gsr, tempo_total_array_tf_solv, n_x_array, n_t = tempo_computacional_tf.calculate_tempo_computacional_h_x()
tempos_totais_h_x_tf_neumann.append(tempo_total_array_tf_gs)
tempos_totais_h_x_tf_neumann.append(tempo_total_array_tf_tdma)
tempos_totais_h_x_tf_neumann.append(tempo_total_array_tf_jac)
tempos_totais_h_x_tf_neumann.append(tempo_total_array_tf_gsr)
tempos_totais_h_x_tf_neumann.append(tempo_total_array_tf_solv)

# Tabela Tempo

solvers = ['Guass Seidel', 'TDMA', 'Jacobi', 'Guass Seidel Relaxamento', 'Solver Scipy']

# Tabelas:
tabela = PrettyTable(['N° de Blocos', 'Steps de Tempo', 'Solver', 'Tempo Computacional [s] - Neumann'])
for solver_idx, solver in enumerate(solvers):  # enumerate adiciona um contador à iteração
    tempos_solver_neumann = tempos_totais_h_t_tf_neumann[solver_idx]  # obtida a lista de tempos correspondente ao solver da iteração
    for n_t_idx, delta_t in enumerate(n_t_array):  # enumerate adiciona um contador à iteração
        tempo_total_neumann= tempos_solver_neumann[n_t_idx]  # pega o tempo correspondente ao delta t
        rounded_value_neumann = round(tempo_total_neumann, 3)
        tempo_total_str_neumann = str(rounded_value_neumann)
        delta_t_rounded = round(delta_t, 3)
        tabela.add_row([n_x, delta_t_rounded, solver, tempo_total_str_neumann])

print(tabela)

# Tabela Malha

solvers = ['Guass Seidel', 'TDMA', 'Jacobi', 'Guass Seidel Relaxamento', 'Solver Scipy']

# Tabelas:
tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'Solver', 'Tempo Computacional [s] - Neumann'])
for solver_idx, solver in enumerate(solvers):  # enumerate adiciona um contador à iteração
    tempos_solver_neumann = tempos_totais_h_x_tf_neumann[solver_idx]  # obtida a lista de tempos correspondente ao solver da iteração
    for n_x_idx, delta_x in enumerate(n_x_array):  # enumerate adiciona um contador à iteração
        tempo_total_neumann= tempos_solver_neumann[n_x_idx]  # pega o tempo correspondente ao delta t
        rounded_value_neumann = round(tempo_total_neumann, 3)
        tempo_total_str_neumann = str(rounded_value_neumann)
        delta_x_rounded = round(delta_x, 3)
        tabela.add_row([n_t, delta_x_rounded, solver, tempo_total_str_neumann])

print(tabela)