# Análise do Tempo Computacional para cada Solver

import numpy as np
import matplotlib.pyplot as plt
from solvers import solvers
from analitica import analiticas
from FTCS import FTCS
from BTCS import BTCS
from CN_tt import CN
from prettytable import PrettyTable
import time

class tempo_computacional_pp():
    def calculate_tempo_computacional_h_t():

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

        h_t = [0.2, 0.1]
        h_x = 0.8
        j = h_x

        tempos_totais = []

        def calculate_n_t(tf, t0, i):
            n_t = (tf - t0) / (i)
            return n_t


        n_x = (xf - x0) / (h_x)

        def calculate_h_t_calc_gs():
            x_calc_array_pp_gs = []
            t_calc_array_pp_gs = []
            p_calc_array_pp_gs = []
            tempo_total_array_pp_gs = []
            n_t_array = []
            for i in h_t:
                inicio_gs = time.time()
                n_t = calculate_n_t(tf, t0, i)
                x_calc_pp_gs, t_calc_pp_gs, p_calc_pp_gs = BTCS.calculate_BTCS_pp_gs(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_gs = time.time()
                tempo_total_gs = fim_gs - inicio_gs

                x_calc_array_pp_gs.append(x_calc_pp_gs)
                p_calc_array_pp_gs.append(p_calc_pp_gs)
                t_calc_array_pp_gs.append(t_calc_pp_gs)
                n_t_array.append(n_t)

                tempo_total_array_pp_gs.append(tempo_total_gs)
            return x_calc_array_pp_gs, t_calc_array_pp_gs, p_calc_array_pp_gs, tempo_total_array_pp_gs, n_t_array

        def calculate_h_t_calc_tdma():
            x_calc_array_pp_tdma = []
            t_calc_array_pp_tdma = []
            p_calc_array_pp_tdma = []
            tempo_total_array_pp_tdma = []
            n_t_array = []
            for i in h_t:
                inicio_tdma = time.time()
                n_t = calculate_n_t(tf, t0, i)
                x_calc_pp_tdma, t_calc_pp_tdma, p_calc_pp_tdma = BTCS.calculate_BTCS_pp_tdma(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_tdma = time.time()
                tempo_total_tdma = fim_tdma - inicio_tdma

                x_calc_array_pp_tdma.append(x_calc_pp_tdma)
                p_calc_array_pp_tdma.append(p_calc_pp_tdma)
                n_t_array.append(n_t)

                tempo_total_array_pp_tdma.append(tempo_total_tdma)
            return x_calc_array_pp_tdma, t_calc_array_pp_tdma, p_calc_array_pp_tdma, tempo_total_array_pp_tdma, n_t_array

        def calculate_h_t_calc_jac():
            x_calc_array_pp_jac = []
            t_calc_array_pp_jac = []
            p_calc_array_pp_jac = []
            tempo_total_array_pp_jac= []
            n_t_array = []
            for i in h_t:
                inicio_jac = time.time()
                n_t = calculate_n_t(tf, t0, i)
                x_calc_pp_jac, t_calc_pp_jac, p_calc_pp_jac = BTCS.calculate_BTCS_pp_jac(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_jac = time.time()
                tempo_total_jac = fim_jac - inicio_jac

                x_calc_array_pp_jac.append(x_calc_pp_jac)
                p_calc_array_pp_jac.append(p_calc_pp_jac)
                t_calc_array_pp_jac.append(t_calc_pp_jac)
                n_t_array.append(n_t)

                tempo_total_array_pp_jac.append(tempo_total_jac)
            return x_calc_array_pp_jac, t_calc_array_pp_jac, p_calc_array_pp_jac, tempo_total_array_pp_jac, n_t_array

        def calculate_h_t_calc_gsr():
            x_calc_array_pp_gsr = []
            t_calc_array_pp_gsr= []
            p_calc_array_pp_gsr= []
            tempo_total_array_pp_gsr = []
            n_t_array = []
            for i in h_t:
                inicio_gsr = time.time()
                n_t = calculate_n_t(tf, t0, i)
                x_calc_pp_gsr, t_calc_pp_gsr, p_calc_pp_gsr = BTCS.calculate_BTCS_pp_gsr(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_gsr = time.time()
                tempo_total_gsr = fim_gsr - inicio_gsr

                x_calc_array_pp_gsr.append(x_calc_pp_gsr)
                p_calc_array_pp_gsr.append(p_calc_pp_gsr)
                t_calc_array_pp_gsr.append(t_calc_pp_gsr)
                n_t_array.append(n_t)

                tempo_total_array_pp_gsr.append(tempo_total_gsr)
            return x_calc_array_pp_gsr, t_calc_array_pp_gsr, p_calc_array_pp_gsr, tempo_total_array_pp_gsr, n_t_array
        def calculate_h_t_calc_solv():
            x_calc_array_pp_solv = []
            t_calc_array_pp_solv = []
            p_calc_array_pp_solv = []
            tempo_total_array_pp_solv = []
            n_t_array = []
            for i in h_t:
                inicio_solv = time.time()
                n_t = calculate_n_t(tf, t0, i)
                x_calc_pp_solv, t_calc_pp_solv, p_calc_pp_solv = BTCS.calculate_BTCS_pp_solv(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_solv = time.time()
                tempo_total_solv = fim_solv - inicio_solv

                x_calc_array_pp_solv.append(x_calc_pp_solv)
                p_calc_array_pp_solv.append(p_calc_pp_solv)
                t_calc_array_pp_solv.append(t_calc_pp_solv)
                n_t_array.append(n_t)

                tempo_total_array_pp_solv.append(tempo_total_solv)
            return x_calc_array_pp_solv, t_calc_array_pp_solv, p_calc_array_pp_solv, tempo_total_array_pp_solv, n_t_array

        x_calc_array_pp_gs, t_calc_array_pp_gs, p_calc_array_pp_gs, tempo_total_array_pp_gs, n_t_array = calculate_h_t_calc_gs()
        tempos_totais.append(tempo_total_array_pp_gs)
        x_calc_array_pp_tdma, t_calc_array_pp_tdma, p_calc_array_pp_tdma, tempo_total_array_pp_tdma, n_t_array = calculate_h_t_calc_tdma()
        tempos_totais.append(tempo_total_array_pp_tdma)
        x_calc_array_pp_jac, t_calc_array_pp_jac, p_calc_array_pp_jac, tempo_total_array_pp_jac, n_t_array = calculate_h_t_calc_jac()
        tempos_totais.append(tempo_total_array_pp_jac)
        x_calc_array_pp_gsr, t_calc_array_pp_gsr, p_calc_array_pp_gsr, tempo_total_array_pp_gsr, n_t_array = calculate_h_t_calc_gsr()
        tempos_totais.append(tempo_total_array_pp_gsr)
        x_calc_array_pp_solv, t_calc_array_pp_solv, p_calc_array_pp_solv, tempo_total_array_pp_solv, n_t_array = calculate_h_t_calc_solv()
        tempos_totais.append(tempo_total_array_pp_solv)

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

        return tempo_total_array_pp_gs,  tempo_total_array_pp_tdma, tempo_total_array_pp_jac, tempo_total_array_pp_gsr, tempo_total_array_pp_solv, n_t_array, n_x
    def calculate_tempo_computacional_h_x():

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

        h_x = [5, 4, 3, 2, 1]
        h_t = 0.8
        i = h_t

        tempos_totais = []

        def calculate_n_x(xf, x0, j):
            n_x = (xf - x0) / (j)
            return n_x


        n_t = (tf - t0) / (h_t)

        def calculate_h_x_calc_gs():
            x_calc_array_pp_gs = []
            t_calc_array_pp_gs = []
            p_calc_array_pp_gs = []
            tempo_total_array_pp_gs = []
            n_x_array = []
            for j in h_x:
                inicio_gs = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_pp_gs, t_calc_pp_gs, p_calc_pp_gs = BTCS.calculate_BTCS_pp_gs(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_gs = time.time()
                tempo_total_gs = fim_gs - inicio_gs

                x_calc_array_pp_gs.append(x_calc_pp_gs)
                p_calc_array_pp_gs.append(p_calc_pp_gs)
                t_calc_array_pp_gs.append(t_calc_pp_gs)
                n_x_array.append(n_x)

                tempo_total_array_pp_gs.append(tempo_total_gs)
            return x_calc_array_pp_gs, t_calc_array_pp_gs, p_calc_array_pp_gs, tempo_total_array_pp_gs, n_x_array

        def calculate_h_x_calc_tdma():
            x_calc_array_pp_tdma = []
            t_calc_array_pp_tdma = []
            p_calc_array_pp_tdma = []
            tempo_total_array_pp_tdma = []
            n_x_array = []
            for j in h_x:
                inicio_tdma = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_pp_tdma, t_calc_pp_tdma, p_calc_pp_tdma = BTCS.calculate_BTCS_pp_tdma(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_tdma = time.time()
                tempo_total_tdma = fim_tdma - inicio_tdma

                x_calc_array_pp_tdma.append(x_calc_pp_tdma)
                p_calc_array_pp_tdma.append(p_calc_pp_tdma)
                n_x_array.append(n_x)

                tempo_total_array_pp_tdma.append(tempo_total_tdma)
            return x_calc_array_pp_tdma, t_calc_array_pp_tdma, p_calc_array_pp_tdma, tempo_total_array_pp_tdma, n_x_array

        def calculate_h_x_calc_jac():
            x_calc_array_pp_jac = []
            t_calc_array_pp_jac = []
            p_calc_array_pp_jac = []
            tempo_total_array_pp_jac= []
            n_x_array = []
            for j in h_x:
                inicio_jac = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_pp_jac, t_calc_pp_jac, p_calc_pp_jac = BTCS.calculate_BTCS_pp_jac(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_jac = time.time()
                tempo_total_jac = fim_jac - inicio_jac

                x_calc_array_pp_jac.append(x_calc_pp_jac)
                p_calc_array_pp_jac.append(p_calc_pp_jac)
                t_calc_array_pp_jac.append(t_calc_pp_jac)
                n_x_array.append(n_x)

                tempo_total_array_pp_jac.append(tempo_total_jac)
            return x_calc_array_pp_jac, t_calc_array_pp_jac, p_calc_array_pp_jac, tempo_total_array_pp_jac, n_x_array

        def calculate_h_x_calc_gsr():
            x_calc_array_pp_gsr = []
            t_calc_array_pp_gsr= []
            p_calc_array_pp_gsr= []
            tempo_total_array_pp_gsr = []
            n_x_array = []
            for j in h_x:
                inicio_gsr = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_pp_gsr, t_calc_pp_gsr, p_calc_pp_gsr = BTCS.calculate_BTCS_pp_gsr(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_gsr = time.time()
                tempo_total_gsr = fim_gsr - inicio_gsr

                x_calc_array_pp_gsr.append(x_calc_pp_gsr)
                p_calc_array_pp_gsr.append(p_calc_pp_gsr)
                t_calc_array_pp_gsr.append(t_calc_pp_gsr)
                n_x_array.append(n_x)

                tempo_total_array_pp_gsr.append(tempo_total_gsr)
            return x_calc_array_pp_gsr, t_calc_array_pp_gsr, p_calc_array_pp_gsr, tempo_total_array_pp_gsr, n_x_array
        def calculate_h_x_calc_solv():
            x_calc_array_pp_solv = []
            t_calc_array_pp_solv = []
            p_calc_array_pp_solv = []
            tempo_total_array_pp_solv = []
            n_x_array = []
            for j in h_x:
                inicio_solv = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_pp_solv, t_calc_pp_solv, p_calc_pp_solv = BTCS.calculate_BTCS_pp_solv(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_solv = time.time()
                tempo_total_solv = fim_solv - inicio_solv

                x_calc_array_pp_solv.append(x_calc_pp_solv)
                p_calc_array_pp_solv.append(p_calc_pp_solv)
                t_calc_array_pp_solv.append(t_calc_pp_solv)
                n_x_array.append(n_x)

                tempo_total_array_pp_solv.append(tempo_total_solv)
            return x_calc_array_pp_solv, t_calc_array_pp_solv, p_calc_array_pp_solv, tempo_total_array_pp_solv, n_x_array

        x_calc_array_pp_gs, t_calc_array_pp_gs, p_calc_array_pp_gs, tempo_total_array_pp_gs, n_x_array = calculate_h_x_calc_gs()
        tempos_totais.append(tempo_total_array_pp_gs)
        x_calc_array_pp_tdma, t_calc_array_pp_tdma, p_calc_array_pp_tdma, tempo_total_array_pp_tdma, n_x_array = calculate_h_x_calc_tdma()
        tempos_totais.append(tempo_total_array_pp_tdma)
        x_calc_array_pp_jac, t_calc_array_pp_jac, p_calc_array_pp_jac, tempo_total_array_pp_jac, n_x_array = calculate_h_x_calc_jac()
        tempos_totais.append(tempo_total_array_pp_jac)
        x_calc_array_pp_gsr, t_calc_array_pp_gsr, p_calc_array_pp_gsr, tempo_total_array_pp_gsr, n_x_array = calculate_h_x_calc_gsr()
        tempos_totais.append(tempo_total_array_pp_gsr)
        x_calc_array_pp_solv, t_calc_array_pp_solv, p_calc_array_pp_solv, tempo_total_array_pp_solv, n_x_array = calculate_h_x_calc_solv()
        tempos_totais.append(tempo_total_array_pp_solv)

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

        return tempo_total_array_pp_gs,  tempo_total_array_pp_tdma, tempo_total_array_pp_jac, tempo_total_array_pp_gsr, tempo_total_array_pp_solv, n_x_array, n_t
class tempo_computacional_fp():
    def calculate_tempo_computacional_h_t():

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

        h_t = [0.2, 0.1]
        h_x = 0.8
        j = h_x

        tempos_totais = []

        def calculate_n_t(tf, t0, i):
            n_t = (tf - t0) / (i)
            return n_t


        n_x = (xf - x0) / (h_x)

        def calculate_h_t_calc_gs():
            x_calc_array_fp_gs = []
            t_calc_array_fp_gs = []
            p_calc_array_fp_gs = []
            tempo_total_array_fp_gs = []
            n_t_array = []
            for i in h_t:
                inicio_gs = time.time()
                n_t = calculate_n_t(tf, t0, i)
                x_calc_fp_gs, t_calc_fp_gs, p_calc_fp_gs = BTCS.calculate_BTCS_fp_gs(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_gs = time.time()
                tempo_total_gs = fim_gs - inicio_gs

                x_calc_array_fp_gs.append(x_calc_fp_gs)
                p_calc_array_fp_gs.append(p_calc_fp_gs)
                t_calc_array_fp_gs.append(t_calc_fp_gs)
                n_t_array.append(n_t)

                tempo_total_array_fp_gs.append(tempo_total_gs)
            return x_calc_array_fp_gs, t_calc_array_fp_gs, p_calc_array_fp_gs, tempo_total_array_fp_gs, n_t_array

        def calculate_h_t_calc_tdma():
            x_calc_array_fp_tdma = []
            t_calc_array_fp_tdma = []
            p_calc_array_fp_tdma = []
            tempo_total_array_fp_tdma = []
            n_t_array = []
            for i in h_t:
                inicio_tdma = time.time()
                n_t = calculate_n_t(tf, t0, i)
                x_calc_fp_tdma, t_calc_fp_tdma, p_calc_fp_tdma = BTCS.calculate_BTCS_fp_tdma(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_tdma = time.time()
                tempo_total_tdma = fim_tdma - inicio_tdma

                x_calc_array_fp_tdma.append(x_calc_fp_tdma)
                p_calc_array_fp_tdma.append(p_calc_fp_tdma)
                t_calc_array_fp_tdma.append(t_calc_fp_tdma)
                n_t_array.append(n_t)

                tempo_total_array_fp_tdma.append(tempo_total_tdma)
            return x_calc_array_fp_tdma, t_calc_array_fp_tdma, p_calc_array_fp_tdma, tempo_total_array_fp_tdma, n_t_array

        def calculate_h_t_calc_jac():
            x_calc_array_fp_jac = []
            t_calc_array_fp_jac = []
            p_calc_array_fp_jac = []
            tempo_total_array_fp_jac= []
            n_t_array = []
            for i in h_t:
                inicio_jac = time.time()
                n_t = calculate_n_t(tf, t0, i)
                x_calc_fp_jac, t_calc_fp_jac, p_calc_fp_jac = BTCS.calculate_BTCS_fp_jac(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_jac = time.time()
                tempo_total_jac = fim_jac - inicio_jac

                x_calc_array_fp_jac.append(x_calc_fp_jac)
                p_calc_array_fp_jac.append(p_calc_fp_jac)
                t_calc_array_fp_jac.append(t_calc_fp_jac)
                n_t_array.append(n_t)

                tempo_total_array_fp_jac.append(tempo_total_jac)
            return x_calc_array_fp_jac, t_calc_array_fp_jac, p_calc_array_fp_jac, tempo_total_array_fp_jac, n_t_array

        def calculate_h_t_calc_gsr():
            x_calc_array_fp_gsr = []
            t_calc_array_fp_gsr = []
            p_calc_array_fp_gsr = []
            tempo_total_array_fp_gsr = []
            n_t_array = []
            for i in h_t:
                inicio_gsr = time.time()
                n_t = calculate_n_t(tf, t0, i)
                x_calc_fp_gsr, t_calc_fp_gsr, p_calc_fp_gsr = BTCS.calculate_BTCS_fp_gsr(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_gsr= time.time()
                tempo_total_gsr = fim_gsr - inicio_gsr

                x_calc_array_fp_gsr.append(x_calc_fp_gsr)
                p_calc_array_fp_gsr.append(p_calc_fp_gsr)
                t_calc_array_fp_gsr.append(t_calc_fp_gsr)
                n_t_array.append(n_t)

                tempo_total_array_fp_gsr.append(tempo_total_gsr)
            return x_calc_array_fp_gsr, t_calc_array_fp_gsr, p_calc_array_fp_gsr, tempo_total_array_fp_gsr, n_t_array

        def calculate_h_t_calc_solv():
            x_calc_array_fp_solv = []
            t_calc_array_fp_solv = []
            p_calc_array_fp_solv = []
            tempo_total_array_fp_solv = []
            n_t_array = []
            for i in h_t:
                inicio_solv = time.time()
                n_t = calculate_n_t(tf, t0, i)
                x_calc_fp_solv, t_calc_fp_solv, p_calc_fp_solv = BTCS.calculate_BTCS_fp_solv(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_solv = time.time()
                tempo_total_solv = fim_solv - inicio_solv

                x_calc_array_fp_solv.append(x_calc_fp_solv)
                p_calc_array_fp_solv.append(p_calc_fp_solv)
                t_calc_array_fp_solv.append(t_calc_fp_solv)
                n_t_array.append(n_t)

                tempo_total_array_fp_solv.append(tempo_total_solv)
            return x_calc_array_fp_solv, t_calc_array_fp_solv, p_calc_array_fp_solv, tempo_total_array_fp_solv, n_t_array


        x_calc_array_fp_gs, t_calc_array_fp_gs, p_calc_array_fp_gs, tempo_total_array_fp_gs, n_t_array = calculate_h_t_calc_gs()
        tempos_totais.append(tempo_total_array_fp_gs)
        x_calc_array_fp_tdma, t_calc_array_fp_tdma, p_calc_array_fp_tdma, tempo_total_array_fp_tdma, n_t_array = calculate_h_t_calc_tdma()
        tempos_totais.append(tempo_total_array_fp_tdma)
        x_calc_array_fp_jac, t_calc_array_fp_jac, p_calc_array_fp_jac, tempo_total_array_fp_jac, n_t_array = calculate_h_t_calc_jac()
        tempos_totais.append(tempo_total_array_fp_jac)
        x_calc_array_fp_gsr, t_calc_array_fp_gsr, p_calc_array_fp_gsr, tempo_total_array_fp_gsr, n_t_array = calculate_h_t_calc_gsr()
        tempos_totais.append(tempo_total_array_fp_gsr)
        x_calc_array_fp_solv, t_calc_array_fp_solv, p_calc_array_fp_solv, tempo_total_array_fp_solv, n_t_array = calculate_h_t_calc_solv()
        tempos_totais.append(tempo_total_array_fp_solv)

        solvers = ['Guass Seidel', 'TDMA', 'Jacobi', 'Guass Seidel Relaxamento', 'Solver Scipy']
        print('tempos', tempos_totais)
        print('n_x', n_x)

        # Tabelas:
        tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'Solver', 'Tempo Computacional [s] - Neumann'])
        for solver_idx, solver in enumerate(solvers): # enumerate adiciona um contador à iteração
            tempos_solver = tempos_totais[solver_idx] # obtida a lista de tempos correspondente ao solver da iteração
            for n_t_idx, delta_t in enumerate(n_t_array): # enumerate adiciona um contador à iteração
                tempo_total = tempos_solver[n_t_idx] # pega o tempo correspondente ao delta t
                rounded_value = round(tempo_total, 3)
                tempo_total_str = str(rounded_value)
                tabela.add_row([delta_t, n_x, solver, tempo_total_str])

        print(tabela)

        return tempo_total_array_fp_gs, tempo_total_array_fp_tdma, tempo_total_array_fp_jac, tempo_total_array_fp_gsr, tempo_total_array_fp_solv, n_t_array, n_x

    def calculate_tempo_computacional_h_x():

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

        h_x = [5, 4, 3, 2, 1]
        h_t = 0.8
        i = h_t

        tempos_totais = []

        def calculate_n_x(xf, x0, j):
            n_x = (xf - x0) / (j)
            return n_x


        n_t = (tf - t0) / (h_t)

        def calculate_h_x_calc_gs():
            x_calc_array_fp_gs = []
            t_calc_array_fp_gs = []
            p_calc_array_fp_gs = []
            tempo_total_array_fp_gs = []
            n_x_array = []
            for j in h_x:
                inicio_gs = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_fp_gs, t_calc_fp_gs, p_calc_fp_gs = BTCS.calculate_BTCS_fp_gs(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_gs = time.time()
                tempo_total_gs = fim_gs - inicio_gs

                x_calc_array_fp_gs.append(x_calc_fp_gs)
                p_calc_array_fp_gs.append(p_calc_fp_gs)
                t_calc_array_fp_gs.append(t_calc_fp_gs)
                n_x_array.append(n_x)

                tempo_total_array_fp_gs.append(tempo_total_gs)
            return x_calc_array_fp_gs, t_calc_array_fp_gs, p_calc_array_fp_gs, tempo_total_array_fp_gs, n_x_array

        def calculate_h_x_calc_tdma():
            x_calc_array_fp_tdma = []
            t_calc_array_fp_tdma = []
            p_calc_array_fp_tdma = []
            tempo_total_array_fp_tdma = []
            n_x_array = []
            for j in h_x:
                inicio_tdma = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_fp_tdma, t_calc_fp_tdma, p_calc_fp_tdma = BTCS.calculate_BTCS_fp_tdma(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_tdma = time.time()
                tempo_total_tdma = fim_tdma - inicio_tdma

                x_calc_array_fp_tdma.append(x_calc_fp_tdma)
                p_calc_array_fp_tdma.append(p_calc_fp_tdma)
                t_calc_array_fp_tdma.append(t_calc_fp_tdma)
                n_x_array.append(n_x)

                tempo_total_array_fp_tdma.append(tempo_total_tdma)
            return x_calc_array_fp_tdma, t_calc_array_fp_tdma, p_calc_array_fp_tdma, tempo_total_array_fp_tdma, n_x_array

        def calculate_h_x_calc_jac():
            x_calc_array_fp_jac = []
            t_calc_array_fp_jac = []
            p_calc_array_fp_jac = []
            tempo_total_array_fp_jac= []
            n_x_array = []
            for j in h_x:
                inicio_jac = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_fp_jac, t_calc_fp_jac, p_calc_fp_jac = BTCS.calculate_BTCS_fp_jac(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_jac = time.time()
                tempo_total_jac = fim_jac - inicio_jac

                x_calc_array_fp_jac.append(x_calc_fp_jac)
                p_calc_array_fp_jac.append(p_calc_fp_jac)
                t_calc_array_fp_jac.append(t_calc_fp_jac)
                n_x_array.append(n_x)

                tempo_total_array_fp_jac.append(tempo_total_jac)
            return x_calc_array_fp_jac, t_calc_array_fp_jac, p_calc_array_fp_jac, tempo_total_array_fp_jac, n_x_array

        def calculate_h_x_calc_gsr():
            x_calc_array_fp_gsr = []
            t_calc_array_fp_gsr = []
            p_calc_array_fp_gsr = []
            tempo_total_array_fp_gsr = []
            n_x_array = []
            for j in h_x:
                inicio_gsr = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_fp_gsr, t_calc_fp_gsr, p_calc_fp_gsr = BTCS.calculate_BTCS_fp_gsr(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_gsr= time.time()
                tempo_total_gsr = fim_gsr - inicio_gsr

                x_calc_array_fp_gsr.append(x_calc_fp_gsr)
                p_calc_array_fp_gsr.append(p_calc_fp_gsr)
                t_calc_array_fp_gsr.append(t_calc_fp_gsr)
                n_x_array.append(n_x)

                tempo_total_array_fp_gsr.append(tempo_total_gsr)
            return x_calc_array_fp_gsr, t_calc_array_fp_gsr, p_calc_array_fp_gsr, tempo_total_array_fp_gsr, n_x_array

        def calculate_h_x_calc_solv():
            x_calc_array_fp_solv = []
            t_calc_array_fp_solv = []
            p_calc_array_fp_solv = []
            tempo_total_array_fp_solv = []
            n_x_array = []
            for j in h_x:
                inicio_solv = time.time()
                n_x = calculate_n_x(xf, x0, j)
                x_calc_fp_solv, t_calc_fp_solv, p_calc_fp_solv = BTCS.calculate_BTCS_fp_solv(p0, pw, qw, q0, cc, mi, k, h, phi, c, L, A, x0, xf, t0, tf, i, j, n_t, n_x)
                fim_solv = time.time()
                tempo_total_solv = fim_solv - inicio_solv

                x_calc_array_fp_solv.append(x_calc_fp_solv)
                p_calc_array_fp_solv.append(p_calc_fp_solv)
                t_calc_array_fp_solv.append(t_calc_fp_solv)
                n_x_array.append(n_x)

                tempo_total_array_fp_solv.append(tempo_total_solv)
            return x_calc_array_fp_solv, t_calc_array_fp_solv, p_calc_array_fp_solv, tempo_total_array_fp_solv, n_x_array


        x_calc_array_fp_gs, t_calc_array_fp_gs, p_calc_array_fp_gs, tempo_total_array_fp_gs, n_x_array = calculate_h_x_calc_gs()
        tempos_totais.append(tempo_total_array_fp_gs)
        x_calc_array_fp_tdma, t_calc_array_fp_tdma, p_calc_array_fp_tdma, tempo_total_array_fp_tdma, n_x_array = calculate_h_x_calc_tdma()
        tempos_totais.append(tempo_total_array_fp_tdma)
        x_calc_array_fp_jac, t_calc_array_fp_jac, p_calc_array_fp_jac, tempo_total_array_fp_jac, n_x_array = calculate_h_x_calc_jac()
        tempos_totais.append(tempo_total_array_fp_jac)
        x_calc_array_fp_gsr, t_calc_array_fp_gsr, p_calc_array_fp_gsr, tempo_total_array_fp_gsr, n_x_array = calculate_h_x_calc_gsr()
        tempos_totais.append(tempo_total_array_fp_gsr)
        x_calc_array_fp_solv, t_calc_array_fp_solv, p_calc_array_fp_solv, tempo_total_array_fp_solv, n_x_array = calculate_h_x_calc_solv()
        tempos_totais.append(tempo_total_array_fp_solv)

        solvers = ['Guass Seidel', 'TDMA', 'Jacobi', 'Guass Seidel Relaxamento', 'Solver Scipy']
        print('tempos', tempos_totais)
        print('n_x', n_t)

        # Tabelas:
        tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'Solver', 'Tempo Computacional [s] - Neumann'])
        for solver_idx, solver in enumerate(solvers): # enumerate adiciona um contador à iteração
            tempos_solver = tempos_totais[solver_idx] # obtida a lista de tempos correspondente ao solver da iteração
            for n_x_idx, delta_x in enumerate(n_x_array): # enumerate adiciona um contador à iteração
                tempo_total = tempos_solver[n_x_idx] # pega o tempo correspondente ao delta t
                rounded_value = round(tempo_total, 3)
                tempo_total_str = str(rounded_value)
                delta_x_rounded = round(delta_x, 3)
                tabela.add_row([n_t, delta_x_rounded, solver, tempo_total_str])

        print(tabela)

        return tempo_total_array_fp_gs, tempo_total_array_fp_tdma, tempo_total_array_fp_jac, tempo_total_array_fp_gsr, tempo_total_array_fp_solv, n_x_array, n_t

tempos_totais_h_t_pp_dirchlet = []
tempos_totais_h_x_pp_dirchlet = []
tempos_totais_h_t_fp_neumann = []
tempos_totais_h_x_fp_neumann = []
tempo_total_array_pp_gs,  tempo_total_array_pp_tdma, tempo_total_array_pp_jac, tempo_total_array_pp_gsr, tempo_total_array_pp_solv, n_t_array, n_x = tempo_computacional_pp.calculate_tempo_computacional_h_t()
tempos_totais_h_t_pp_dirchlet.append(tempo_total_array_pp_gs)
tempos_totais_h_t_pp_dirchlet.append(tempo_total_array_pp_tdma)
tempos_totais_h_t_pp_dirchlet.append(tempo_total_array_pp_jac)
tempos_totais_h_t_pp_dirchlet.append(tempo_total_array_pp_gsr)
tempos_totais_h_t_pp_dirchlet.append(tempo_total_array_pp_solv)
tempo_total_array_pp_gs,  tempo_total_array_pp_tdma, tempo_total_array_pp_jac, tempo_total_array_pp_gsr, tempo_total_array_pp_solv, n_x_array, n_t = tempo_computacional_pp.calculate_tempo_computacional_h_x()
tempos_totais_h_x_pp_dirchlet.append(tempo_total_array_pp_gs)
tempos_totais_h_x_pp_dirchlet.append(tempo_total_array_pp_tdma)
tempos_totais_h_x_pp_dirchlet.append(tempo_total_array_pp_jac)
tempos_totais_h_x_pp_dirchlet.append(tempo_total_array_pp_gsr)
tempos_totais_h_x_pp_dirchlet.append(tempo_total_array_pp_solv)
tempo_total_array_fp_gs, tempo_total_array_fp_tdma, tempo_total_array_fp_jac, tempo_total_array_fp_gsr, tempo_total_array_fp_solv, n_t_array, n_x = tempo_computacional_fp.calculate_tempo_computacional_h_t()
tempos_totais_h_t_fp_neumann.append(tempo_total_array_fp_gs)
tempos_totais_h_t_fp_neumann.append(tempo_total_array_fp_tdma)
tempos_totais_h_t_fp_neumann.append(tempo_total_array_fp_jac)
tempos_totais_h_t_fp_neumann.append(tempo_total_array_fp_gsr)
tempos_totais_h_t_fp_neumann.append(tempo_total_array_fp_solv)
tempo_total_array_fp_gs, tempo_total_array_fp_tdma, tempo_total_array_fp_jac, tempo_total_array_fp_gsr, tempo_total_array_fp_solv, n_x_array, n_t = tempo_computacional_fp.calculate_tempo_computacional_h_x()
tempos_totais_h_x_fp_neumann.append(tempo_total_array_fp_gs)
tempos_totais_h_x_fp_neumann.append(tempo_total_array_fp_tdma)
tempos_totais_h_x_fp_neumann.append(tempo_total_array_fp_jac)
tempos_totais_h_x_fp_neumann.append(tempo_total_array_fp_gsr)
tempos_totais_h_x_fp_neumann.append(tempo_total_array_fp_solv)

# Tabela Tempo

solvers = ['Guass Seidel', 'TDMA', 'Jacobi', 'Guass Seidel Relaxamento', 'Solver Scipy']

# Tabelas:
tabela = PrettyTable(['N° de Blocos', 'Steps de Tempo', 'Solver', 'Tempo Computacional [s] - Dirchlet', 'Tempo Computacional [s] - Neumann'])
for solver_idx, solver in enumerate(solvers):  # enumerate adiciona um contador à iteração
    tempos_solver_dirchlet = tempos_totais_h_t_pp_dirchlet[solver_idx]  # obtida a lista de tempos correspondente ao solver da iteração
    tempos_solver_neumann = tempos_totais_h_t_fp_neumann[solver_idx]  # obtida a lista de tempos correspondente ao solver da iteração
    for n_t_idx, delta_t in enumerate(n_t_array):  # enumerate adiciona um contador à iteração
        tempo_total_dirchlet = tempos_solver_dirchlet [n_t_idx]  # pega o tempo correspondente ao delta t
        tempo_total_neumann= tempos_solver_neumann[n_t_idx]  # pega o tempo correspondente ao delta t
        rounded_value_dirchlet = round(tempo_total_dirchlet, 3)
        rounded_value_neumann = round(tempo_total_neumann, 3)
        tempo_total_str_dirchlet = str(rounded_value_dirchlet)
        tempo_total_str_neumann = str(rounded_value_neumann)
        delta_t_rounded = round(delta_t, 3)
        tabela.add_row([n_x, delta_t_rounded, solver, tempo_total_str_dirchlet, tempo_total_str_neumann])

print(tabela)

# Tabela Malha

solvers = ['Guass Seidel', 'TDMA', 'Jacobi', 'Guass Seidel Relaxamento', 'Solver Scipy']

# Tabelas:
tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'Solver', 'Tempo Computacional [s] - Dirchlet', 'Tempo Computacional [s] - Neumann'])
for solver_idx, solver in enumerate(solvers):  # enumerate adiciona um contador à iteração
    tempos_solver_dirchlet = tempos_totais_h_x_pp_dirchlet[solver_idx]  # obtida a lista de tempos correspondente ao solver da iteração
    tempos_solver_neumann = tempos_totais_h_x_fp_neumann[solver_idx]  # obtida a lista de tempos correspondente ao solver da iteração
    for n_x_idx, delta_x in enumerate(n_x_array):  # enumerate adiciona um contador à iteração
        tempo_total_dirchlet = tempos_solver_dirchlet[n_x_idx]  # pega o tempo correspondente ao delta t
        tempo_total_neumann= tempos_solver_neumann[n_x_idx]  # pega o tempo correspondente ao delta t
        rounded_value_dirchlet = round(tempo_total_dirchlet, 3)
        rounded_value_neumann = round(tempo_total_neumann, 3)
        tempo_total_str_dirchlet = str(rounded_value_dirchlet)
        tempo_total_str_neumann = str(rounded_value_neumann)
        delta_x_rounded = round(delta_x, 3)
        tabela.add_row([n_t, delta_x_rounded, solver, tempo_total_str_dirchlet, tempo_total_str_neumann])

print(tabela)