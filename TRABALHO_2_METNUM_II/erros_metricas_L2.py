# Função para analisar inclinação das retas do decaimento do Tempo (Linear ou Quadrática)

import numpy as np
import matplotlib.pyplot as plt
from analitica import analitica
from FTCS import FTCS
from BTCS import BTCS
from CN_tt import CN
from prettytable import PrettyTable

class erros_tt_ftcs():
    def calculate_erros_tempo():

        rho = 8.92  # g/cm³
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        qw = 25
        L = 100  # cm
        T0 = 50  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        t0 = 0
        tf = 100
        x0 = 0
        xf = L
        variancia = 'tempo'

        h_t = [3,2,1,0.5,0.05]
        h_x = 4
        j = h_x


        def calculate_n_t(tf,t0,i):

            n_t = (tf - t0) / (i)
            return n_t

        n_x = (xf - x0) / (h_x)

        def calculate_h_t_ex(variancia):
            x_ex_array = []
            t_ex_array = []
            T_ex_array = []
            n_t_array = []
            for i in h_t:

                n_t = calculate_n_t(tf, t0, i)
                x_ex, t_ex, T_ex = analitica.calculate_analitica(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_ex_array.append(x_ex)
                T_ex_array.append(T_ex)
                t_ex_array.append(t_ex)
                n_t_array.append(n_t)

            return x_ex_array, t_ex_array, T_ex_array, n_t_array


        x_ex_array, t_ex_array, T_ex_array, n_t_array = calculate_h_t_ex(variancia)


        def calculate_h_t_calc(variancia):
            x_calc_array = []
            t_calc_array = []
            T_calc_array = []
            n_t_array = []
            for i in h_t:

                n_t = calculate_n_t(tf, t0, i)
                x_calc, t_calc, T_calc = FTCS.calculate_FTCS_tt(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_calc_array.append(x_calc)
                T_calc_array.append(T_calc)
                t_calc_array.append(t_calc)
                n_t_array.append(n_t)

            return x_calc_array, t_calc_array, T_calc_array, n_t_array


        x_calc_array, t_calc_array, T_calc_array, n_t_array = calculate_h_t_calc(variancia)

        # Cálculo do Erro:

        # Norma L2
        L2_list_tempo_tt_ftcs = []
        for i in range(len(T_calc_array)): # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            sum = 0
            y_calc = T_calc[13] # selecinar a coluna
            y_ex = T_ex[13] # selecinar a coluna
            for k in range(len(y_ex)): # acesso a cada linha
                sum = sum + ((abs((y_ex[k]-y_calc[k])/(y_ex[k])))**2)
            L2 = np.sqrt((1/(n_x**2))*sum)
            L2_list_tempo_tt_ftcs.append(L2)
        print('L2', L2_list_tempo_tt_ftcs)

        L2_log_list = np.log(L2_list_tempo_tt_ftcs)

        # Norma E_inf
        E_inf_depois_list_tempo_tt_ftcs = []

        for i in range(len(T_calc_array)): # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[13]
            y_ex = T_ex[13]
            E_inf_antes_list = []
            for k in range(len(y_ex)):
                E_inf_antes = abs((y_ex[k] - y_calc[k]))
                E_inf_antes_list.append(E_inf_antes)
            E_inf_depois = max(E_inf_antes_list)
            E_inf_depois_list_tempo_tt_ftcs.append(E_inf_depois)

        E_inf_depois_log_list = np.log(E_inf_depois_list_tempo_tt_ftcs)

        # Norma E_rel
        err_rel_total_list_tempo_tt_ftcs = []

        for i in range(len(T_calc_array)): # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[13]
            y_ex = T_ex[13]
            err_rel_list = []
            sum = 0
            for k in range(len(y_ex)):
                err_rel = abs((y_ex[k] - y_calc[k])/(y_ex[k]))
                err_rel_list.append(err_rel)
            for j in range(len(err_rel_list)):
                sum = sum + err_rel_list[j]
            err_rel_total = 1/n_x * sum
            err_rel_total_list_tempo_tt_ftcs.append(err_rel_total)

        err_rel_total_log_list = np.log(err_rel_total_list_tempo_tt_ftcs)

        h_t_log_list = np.log(h_t)

        # Plotagem:
        plt.plot(h_t_log_list, L2_list_tempo_tt_ftcs, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('L2')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_t_log_list, L2_log_list, grau) # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs) # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_t_log_list), max(h_t_log_list), 100) # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed', label='Linha de Tendência Linear')
        plt.plot(h_t_log_list, L2_log_list, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('ln (L2)')
        plt.show()

        # Plotagem:
        plt.plot(h_t_log_list, E_inf_depois_list_tempo_tt_ftcs, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('E$ \infty$')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_t_log_list, E_inf_depois_log_list, grau) # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs) # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_t_log_list), max(h_t_log_list), 100) # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed', label='Linha de Tendência Linear')
        plt.plot(h_t_log_list, E_inf_depois_log_list, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('ln (E$ \infty$)')
        plt.show()

        # Plotagem:
        plt.plot(h_t_log_list, err_rel_total_list_tempo_tt_ftcs, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('L1')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_t_log_list, err_rel_total_log_list, grau) # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs) # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_t_log_list), max(h_t_log_list), 100) # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed', label='Linha de Tendência Linear')
        plt.plot(h_t_log_list, err_rel_total_log_list, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('ln (L1)')
        plt.show()

        # Tabelas:
        tabela = PrettyTable(['N° de Blocos', 'Steps de Tempo', 'Norma Euclidiana - L2', 'Erro Absoluto Máximo - E_inf', 'Erro Relativo - Norma L1'])

        for n_t_val, L2_val, E_inf_val, err_rel_val in zip(n_t_array, L2_list_tempo_tt_ftcs, E_inf_depois_list_tempo_tt_ftcs, err_rel_total_list_tempo_tt_ftcs):
            rounded_L2_val = round(L2_val, 3)
            rounded_E_inf_val = round(E_inf_val, 3)
            rounded_err_rel_val = round(err_rel_val, 3)
            rounded_n_t_val = round(n_t_val, 3)
            tabela.add_row([n_x, rounded_n_t_val, rounded_L2_val, rounded_E_inf_val, rounded_err_rel_val])

        print(tabela)

        return L2_list_tempo_tt_ftcs, E_inf_depois_list_tempo_tt_ftcs, err_rel_total_list_tempo_tt_ftcs, n_t_array, n_x

    def calculate_erros_malha():

        rho = 8.92  # g/cm³
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        qw = 25
        L = 100  # cm
        T0 = 50  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        t0 = 0
        tf = 100
        x0 = 0
        xf = L
        variancia = 'malha'

        h_x = [4, 3, 2, 1, 0.5]
        h_t = 0.05
        i = h_t


        def calculate_n_x(xf, x0, j):

            n_x = (xf - x0) / (j)
            return n_x

        n_t = (tf - t0) / (h_t)

        def calculate_h_t_ex(variancia):
            x_ex_array = []
            t_ex_array = []
            T_ex_array = []
            n_x_array = []
            for j in h_x:
                n_x = calculate_n_x(xf, x0, j)
                x_ex, t_ex, T_ex = analitica.calculate_analitica(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_ex_array.append(x_ex)
                T_ex_array.append(T_ex)
                t_ex_array.append(t_ex)
                n_x_array.append(n_x)

            return x_ex_array, t_ex_array, T_ex_array, n_x_array

        x_ex_array, t_ex_array, T_ex_array, n_x_array = calculate_h_t_ex(variancia)

        def calculate_h_t_calc(variancia):
            x_calc_array = []
            t_calc_array = []
            T_calc_array = []
            n_x_array = []
            for j in h_x:
                n_x = calculate_n_x(xf, x0, j)
                x_calc, t_calc, T_calc = FTCS.calculate_FTCS_tt(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_calc_array.append(x_calc)
                T_calc_array.append(T_calc)
                t_calc_array.append(t_calc)
                n_x_array.append(n_x)

            return x_calc_array, t_calc_array, T_calc_array, n_x_array

        x_calc_array, t_calc_array, T_calc_array, n_x_array = calculate_h_t_calc(variancia)

        # Cálculo do Erro:

        # Norma L2
        L2_list_malha_tt_ftcs = []
        for i in range(len(T_calc_array)):  # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            sum = 0
            y_calc = T_calc[:, 2]  # selecinar a coluna
            y_ex = T_ex[:, 2]  # selecinar a coluna
            n_x = n_x_array[i]
            for k in range(1, len(y_ex)):  # acesso a cada linha
                sum = sum + ((abs((y_ex[k] - y_calc[k]) / (y_ex[k]))) ** 2)
                print('sum', sum)
            L2 = np.sqrt((1 / (n_x ** 2)) * sum)
            L2_list_malha_tt_ftcs.append(L2)
        print('L2', L2_list_malha_tt_ftcs)

        L2_log_list = np.log(L2_list_malha_tt_ftcs)

        # Norma E_inf
        E_inf_depois_list_malha_tt_ftcs = []

        for i in range(len(T_calc_array)):  # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[:, 2]
            y_ex = T_ex[:, 2]
            E_inf_antes_list = []
            n_x = n_x_array[i]
            for k in range(len(y_ex)):
                E_inf_antes = abs((y_ex[k] - y_calc[k]))
                E_inf_antes_list.append(E_inf_antes)
            E_inf_depois = max(E_inf_antes_list)
            E_inf_depois_list_malha_tt_ftcs.append(E_inf_depois)

        E_inf_depois_log_list = np.log(E_inf_depois_list_malha_tt_ftcs)

        # Norma E_rel
        err_rel_total_list_malha_tt_ftcs = []

        for i in range(len(T_calc_array)):  # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[:, 2]
            y_ex = T_ex[:, 2]
            err_rel_list = []
            sum = 0
            n_x = n_x_array[i]
            for k in range(1, len(y_ex)):
                err_rel = abs((y_ex[k] - y_calc[k]) / (y_ex[k]))
                err_rel_list.append(err_rel)
            for j in range(len(err_rel_list)):
                sum = sum + err_rel_list[j]
            err_rel_total = 1 / n_x * sum
            err_rel_total_list_malha_tt_ftcs.append(err_rel_total)

        err_rel_total_log_list = np.log(err_rel_total_list_malha_tt_ftcs)

        h_x_log_list = []
        # Log de h_t
        for i in range(len(h_x)):
            h_x_novo = (h_x[i]) ** 2
            h_x_novo2 = np.log(h_x_novo)
            h_x_log_list.append(h_x_novo2)
        print('h_t_log', h_x_log_list)

        # Plotagem:
        plt.plot(h_x_log_list, L2_list_malha_tt_ftcs, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('L2')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_x_log_list, L2_log_list,
                            grau)  # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs)  # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_x_log_list), max(h_x_log_list),
                                  100)  # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed',
                 label='Linha de Tendência Linear')
        plt.plot(h_x_log_list, L2_log_list, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('ln (L2)')
        plt.show()

        # Plotagem:
        plt.plot(h_x_log_list, E_inf_depois_list_malha_tt_ftcs, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('E$ \infty$')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_x_log_list, E_inf_depois_log_list,
                            grau)  # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs)  # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_x_log_list), max(h_x_log_list),
                                  100)  # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed',
                 label='Linha de Tendência Linear')
        plt.plot(h_x_log_list, E_inf_depois_log_list, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('ln (E$ \infty$)')
        plt.show()

        # Plotagem:
        plt.plot(h_x_log_list, err_rel_total_list_malha_tt_ftcs, linestyle='none', marker='o', color="#FF007F",label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('L1')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_x_log_list, err_rel_total_log_list,
                            grau)  # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs)  # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_x_log_list), max(h_x_log_list),
                                  100)  # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed',
                 label='Linha de Tendência Linear')
        plt.plot(h_x_log_list, err_rel_total_log_list, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('ln (L1)')
        plt.show()

        # Tabelas:
        tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'Norma Euclidiana - L2', 'Erro Absoluto Máximo - E_inf', 'Erro Relativo - Norma L1'])

        for n_x_val, L2_val, E_inf_val, err_rel_val in zip(n_x_array, L2_list_malha_tt_ftcs, E_inf_depois_list_malha_tt_ftcs, err_rel_total_list_malha_tt_ftcs):
            rounded_L2_val = round(L2_val, 3)
            rounded_E_inf_val = round(E_inf_val, 3)
            rounded_err_rel_val = round(err_rel_val, 3)
            rounded_n_x_val = round(n_x_val, 3)
            tabela.add_row([n_t, rounded_n_x_val, rounded_L2_val, rounded_E_inf_val, rounded_err_rel_val])

        print(tabela)

        return L2_list_malha_tt_ftcs, E_inf_depois_list_malha_tt_ftcs, err_rel_total_list_malha_tt_ftcs, n_x_array, n_t

class erros_tt_btcs():
    def calculate_erros_tempo():

        rho = 8.92  # g/cm³
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        qw = 25
        L = 100  # cm
        T0 = 50  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        t0 = 0
        tf = 100
        x0 = 0
        xf = L
        variancia = 'tempo'

        h_t = [3,2,1,0.5,0.05]
        h_x = 4
        j = h_x


        def calculate_n_t(tf,t0,i):

            n_t = (tf - t0) / (i)
            return n_t

        n_x = (xf - x0) / (h_x)

        def calculate_h_t_ex(variancia):
            x_ex_array = []
            t_ex_array = []
            T_ex_array = []
            n_t_array = []
            for i in h_t:

                n_t = calculate_n_t(tf, t0, i)
                x_ex, t_ex, T_ex = analitica.calculate_analitica(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_ex_array.append(x_ex)
                T_ex_array.append(T_ex)
                t_ex_array.append(t_ex)
                n_t_array.append(n_t)

            return x_ex_array, t_ex_array, T_ex_array, n_t_array


        x_ex_array, t_ex_array, T_ex_array, n_t_array = calculate_h_t_ex(variancia)


        def calculate_h_t_calc(variancia):
            x_calc_array = []
            t_calc_array = []
            T_calc_array = []
            n_t_array = []
            for i in h_t:

                n_t = calculate_n_t(tf, t0, i)
                x_calc, t_calc, T_calc = BTCS.calculate_BTCS_tt(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_calc_array.append(x_calc)
                T_calc_array.append(T_calc)
                t_calc_array.append(t_calc)
                n_t_array.append(n_t)

            return x_calc_array, t_calc_array, T_calc_array, n_t_array


        x_calc_array, t_calc_array, T_calc_array, n_t_array = calculate_h_t_calc(variancia)

        # Cálculo do Erro:

        # Norma L2
        L2_list_tempo_tt_btcs = []

        for i in range(len(T_calc_array)): # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            sum = 0
            y_calc = T_calc[13] # selecinar a coluna
            y_ex = T_ex[13] # selecinar a coluna
            for k in range(len(y_ex)): # acesso a cada linha
                sum = sum + ((abs((y_ex[k]-y_calc[k])/(y_ex[k])))**2)
            L2 = np.sqrt((1/(n_x**2))*sum)
            L2_list_tempo_tt_btcs.append(L2)

        L2_log_list = np.log(L2_list_tempo_tt_btcs )

        # Norma E_inf
        E_inf_depois_list_tempo_tt_btcs  = []

        for i in range(len(T_calc_array)): # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[13]
            y_ex = T_ex[13]
            E_inf_antes_list = []
            for k in range(len(y_ex)):
                E_inf_antes = abs((y_ex[k] - y_calc[k]))
                E_inf_antes_list.append(E_inf_antes)
            E_inf_depois = max(E_inf_antes_list)
            E_inf_depois_list_tempo_tt_btcs.append(E_inf_depois)

        E_inf_depois_log_list = np.log(E_inf_depois_list_tempo_tt_btcs)

        # Norma E_rel
        err_rel_total_list_tempo_tt_btcs  = []

        for i in range(len(T_calc_array)): # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[13]
            y_ex = T_ex[13]
            err_rel_list = []
            sum = 0
            for k in range(len(y_ex)):
                err_rel = abs((y_ex[k] - y_calc[k])/(y_ex[k]))
                err_rel_list.append(err_rel)
            for j in range(len(err_rel_list)):
                sum = sum + err_rel_list[j]
            err_rel_total = 1/n_x * sum
            err_rel_total_list_tempo_tt_btcs.append(err_rel_total)

        err_rel_total_log_list = np.log(err_rel_total_list_tempo_tt_btcs)

        h_t_log_list = np.log(h_t)

        # Plotagem:
        plt.plot(h_t_log_list, L2_list_tempo_tt_btcs , linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('L2')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_t_log_list, L2_log_list, grau) # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs) # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_t_log_list), max(h_t_log_list), 100) # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed', label='Linha de Tendência Linear')
        plt.plot(h_t_log_list, L2_log_list, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('ln (L2)')
        plt.show()

        # Plotagem:
        plt.plot(h_t_log_list, E_inf_depois_list_tempo_tt_btcs , linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('E$ \infty$')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_t_log_list, E_inf_depois_log_list, grau) # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs) # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_t_log_list), max(h_t_log_list), 100) # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed', label='Linha de Tendência Linear')
        plt.plot(h_t_log_list, E_inf_depois_log_list, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('ln (E$ \infty$)')
        plt.show()

        # Plotagem:
        plt.plot(h_t_log_list, err_rel_total_list_tempo_tt_btcs , linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('L1')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_t_log_list, err_rel_total_log_list, grau) # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs) # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_t_log_list), max(h_t_log_list), 100) # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed', label='Linha de Tendência Linear')
        plt.plot(h_t_log_list, err_rel_total_log_list, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('ln (L1)')
        plt.show()

        # Tabelas:
        tabela = PrettyTable(['N° de Blocos', 'Steps de Tempo', 'Norma Euclidiana - L2', 'Erro Absoluto Máximo - E_inf', 'Erro Relativo - Norma L1'])

        for n_t_val, L2_val, E_inf_val, err_rel_val in zip(n_t_array, L2_list_tempo_tt_btcs , E_inf_depois_list_tempo_tt_btcs , err_rel_total_list_tempo_tt_btcs ):
            rounded_L2_val = round(L2_val, 3)
            rounded_E_inf_val = round(E_inf_val, 3)
            rounded_err_rel_val = round(err_rel_val, 3)
            rounded_n_t_val = round(n_t_val, 3)
            tabela.add_row([n_x, rounded_n_t_val, rounded_L2_val, rounded_E_inf_val, rounded_err_rel_val])

        print(tabela)

        return L2_list_tempo_tt_btcs , E_inf_depois_list_tempo_tt_btcs, err_rel_total_list_tempo_tt_btcs , n_t_array, n_x

    def calculate_erros_malha():

        rho = 8.92  # g/cm³
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        qw = 25
        L = 100  # cm
        T0 = 50  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        t0 = 0
        tf = 100
        x0 = 0
        xf = L
        variancia = 'malha'

        h_x = [4, 3, 2, 1, 0.5]
        h_t = 0.05
        i = h_t


        def calculate_n_x(xf, x0, j):

            n_x = (xf - x0) / (j)
            return n_x

        n_t = (tf - t0) / (h_t)

        def calculate_h_t_ex(variancia):
            x_ex_array = []
            t_ex_array = []
            T_ex_array = []
            n_x_array = []
            for j in h_x:
                n_x = calculate_n_x(xf, x0, j)
                x_ex, t_ex, T_ex = analitica.calculate_analitica(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_ex_array.append(x_ex)
                T_ex_array.append(T_ex)
                t_ex_array.append(t_ex)
                n_x_array.append(n_x)

            return x_ex_array, t_ex_array, T_ex_array, n_x_array

        x_ex_array, t_ex_array, T_ex_array, n_x_array = calculate_h_t_ex(variancia)

        def calculate_h_t_calc(variancia):
            x_calc_array = []
            t_calc_array = []
            T_calc_array = []
            n_x_array = []
            for j in h_x:
                n_x = calculate_n_x(xf, x0, j)
                x_calc, t_calc, T_calc = BTCS.calculate_BTCS_tt(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_calc_array.append(x_calc)
                T_calc_array.append(T_calc)
                t_calc_array.append(t_calc)
                n_x_array.append(n_x)

            return x_calc_array, t_calc_array, T_calc_array, n_x_array

        x_calc_array, t_calc_array, T_calc_array, n_x_array = calculate_h_t_calc(variancia)

        # Cálculo do Erro:

        # Norma L2
        L2_list_malha_tt_btcs  = []

        for i in range(len(T_calc_array)):  # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            sum = 0
            y_calc = T_calc[:, 2]  # selecinar a coluna
            y_ex = T_ex[:, 2]  # selecinar a coluna
            n_x = n_x_array[i]
            for k in range(1, len(y_ex)):  # acesso a cada linha
                sum = sum + ((abs((y_ex[k] - y_calc[k]) / (y_ex[k]))) ** 2)
            L2 = np.sqrt((1 / (n_x ** 2)) * sum)
            L2_list_malha_tt_btcs.append(L2)
        print('L2', L2_list_malha_tt_btcs)

        L2_log_list = np.log(L2_list_malha_tt_btcs)

        # Norma E_inf
        E_inf_depois_list_malha_tt_btcs  = []

        for i in range(len(T_calc_array)):  # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[:, 2]
            y_ex = T_ex[:, 2]
            E_inf_antes_list = []
            n_x = n_x_array[i]
            for k in range(len(y_ex)):
                E_inf_antes = abs((y_ex[k] - y_calc[k]))
                E_inf_antes_list.append(E_inf_antes)
            E_inf_depois = max(E_inf_antes_list)
            E_inf_depois_list_malha_tt_btcs.append(E_inf_depois)

        E_inf_depois_log_list = np.log(E_inf_depois_list_malha_tt_btcs )

        # Norma E_rel
        err_rel_total_list_malha_tt_btcs  = []

        for i in range(len(T_calc_array)):  # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[:, 2]
            y_ex = T_ex[:, 2]
            err_rel_list = []
            sum = 0
            n_x = n_x_array[i]
            for k in range(1, len(y_ex)):
                err_rel = abs((y_ex[k] - y_calc[k]) / (y_ex[k]))
                err_rel_list.append(err_rel)
            for j in range(len(err_rel_list)):
                sum = sum + err_rel_list[j]
            err_rel_total = 1 / n_x * sum
            err_rel_total_list_malha_tt_btcs.append(err_rel_total)

        err_rel_total_log_list = np.log(err_rel_total_list_malha_tt_btcs)

        h_x_log_list = []
        # Log de h_t
        for i in range(len(h_x)):
            h_x_novo = (h_x[i]) ** 2
            h_x_novo2 = np.log(h_x_novo)
            h_x_log_list.append(h_x_novo2)
        print('h_t_log', h_x_log_list)

        # Plotagem:
        plt.plot(h_x_log_list, L2_list_malha_tt_btcs , linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('L2')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_x_log_list, L2_log_list,
                            grau)  # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs)  # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_x_log_list), max(h_x_log_list),
                                  100)  # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed',
                 label='Linha de Tendência Linear')
        plt.plot(h_x_log_list, L2_log_list, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('ln (L2)')
        plt.show()

        # Plotagem:
        plt.plot(h_x_log_list, E_inf_depois_list_malha_tt_btcs , linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('E$ \infty$')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_x_log_list, E_inf_depois_log_list,
                            grau)  # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs)  # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_x_log_list), max(h_x_log_list),
                                  100)  # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed',
                 label='Linha de Tendência Linear')
        plt.plot(h_x_log_list, E_inf_depois_log_list, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('ln (E$ \infty$)')
        plt.show()

        # Plotagem:
        plt.plot(h_x_log_list, err_rel_total_list_malha_tt_btcs , linestyle='none', marker='o', color="#FF007F",label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('L1')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_x_log_list, err_rel_total_log_list,
                            grau)  # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs)  # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_x_log_list), max(h_x_log_list),
                                  100)  # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed',
                 label='Linha de Tendência Linear')
        plt.plot(h_x_log_list, err_rel_total_log_list, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('ln (L1)')
        plt.show()

        # Tabelas:
        tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'Norma Euclidiana - L2', 'Erro Absoluto Máximo - E_inf', 'Erro Relativo - Norma L1'])

        for n_x_val, L2_val, E_inf_val, err_rel_val in zip(n_x_array, L2_list_malha_tt_btcs , E_inf_depois_list_malha_tt_btcs , err_rel_total_list_malha_tt_btcs ):
            rounded_L2_val = round(L2_val, 3)
            rounded_E_inf_val = round(E_inf_val, 3)
            rounded_err_rel_val = round(err_rel_val, 3)
            rounded_n_x_val = round(n_x_val, 3)
            tabela.add_row([n_t, rounded_n_x_val, rounded_L2_val, rounded_E_inf_val, rounded_err_rel_val])

        print(tabela)

        return L2_list_malha_tt_btcs, E_inf_depois_list_malha_tt_btcs, err_rel_total_list_malha_tt_btcs, n_x_array, n_t

class erros_tt_cn():
    def calculate_erros_tempo():

        rho = 8.92  # g/cm³
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        qw = 25
        L = 100  # cm
        T0 = 50  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        t0 = 0
        tf = 100
        x0 = 0
        xf = L
        variancia = 'tempo'

        h_t = [3,2,1,0.5,0.05]
        h_x = 4
        j = h_x

        def calculate_n_t(tf,t0,i):

            n_t = (tf - t0) / (i)
            return n_t

        n_x = (xf - x0) / (h_x)

        def calculate_h_t_ex(variancia):
            x_ex_array = []
            t_ex_array = []
            T_ex_array = []
            n_t_array = []
            for i in h_t:

                n_t = calculate_n_t(tf, t0, i)
                x_ex, t_ex, T_ex = analitica.calculate_analitica(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_ex_array.append(x_ex)
                T_ex_array.append(T_ex)
                t_ex_array.append(t_ex)
                n_t_array.append(n_t)

            return x_ex_array, t_ex_array, T_ex_array, n_t_array


        x_ex_array, t_ex_array, T_ex_array, n_t_array = calculate_h_t_ex(variancia)


        def calculate_h_t_calc(variancia):
            x_calc_array = []
            t_calc_array = []
            T_calc_array = []
            n_t_array = []
            for i in h_t:

                n_t = calculate_n_t(tf, t0, i)
                x_calc, t_calc, T_calc = CN.calculate_CN_tt(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_calc_array.append(x_calc)
                T_calc_array.append(T_calc)
                t_calc_array.append(t_calc)
                n_t_array.append(n_t)

            return x_calc_array, t_calc_array, T_calc_array, n_t_array


        x_calc_array, t_calc_array, T_calc_array, n_t_array = calculate_h_t_calc(variancia)

        # Cálculo do Erro:

        # Norma L2
        L2_list_tempo_tt_cn = []

        for i in range(len(T_calc_array)): # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            sum = 0
            y_calc = T_calc[13] # selecinar a coluna
            y_ex = T_ex[13] # selecinar a coluna
            for k in range(len(y_ex)): # acesso a cada linha
                sum = sum + ((abs((y_ex[k]-y_calc[k])/(y_ex[k])))**2)
            L2 = np.sqrt((1/(n_x**2))*sum)
            L2_list_tempo_tt_cn.append(L2)
        print('L2', L2_list_tempo_tt_cn)

        L2_log_list = np.log(L2_list_tempo_tt_cn)

        # Norma E_inf
        E_inf_depois_list_tempo_tt_cn = []

        for i in range(len(T_calc_array)): # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[13]
            y_ex = T_ex[13]
            E_inf_antes_list = []
            for k in range(len(y_ex)):
                E_inf_antes = abs((y_ex[k] - y_calc[k]))
                E_inf_antes_list.append(E_inf_antes)
            E_inf_depois = max(E_inf_antes_list)
            E_inf_depois_list_tempo_tt_cn.append(E_inf_depois)

        E_inf_depois_log_list = np.log(E_inf_depois_list_tempo_tt_cn)

        # Norma E_rel
        err_rel_total_list_tempo_tt_cn = []

        for i in range(len(T_calc_array)): # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[13]
            y_ex = T_ex[13]
            err_rel_list = []
            sum = 0
            for k in range(len(y_ex)):
                err_rel = abs((y_ex[k] - y_calc[k])/(y_ex[k]))
                err_rel_list.append(err_rel)
            for j in range(len(err_rel_list)):
                sum = sum + err_rel_list[j]
            err_rel_total = 1/n_x * sum
            err_rel_total_list_tempo_tt_cn.append(err_rel_total)

        err_rel_total_log_list = np.log(err_rel_total_list_tempo_tt_cn)

        h_t_log_list = np.log(h_t)

        # Plotagem:
        plt.plot(h_t_log_list, L2_list_tempo_tt_cn, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('L2')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_t_log_list, L2_log_list, grau) # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs) # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_t_log_list), max(h_t_log_list), 100) # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed', label='Linha de Tendência Linear')
        plt.plot(h_t_log_list, L2_log_list, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('ln (L2)')
        plt.show()

        # Plotagem:
        plt.plot(h_t_log_list, E_inf_depois_list_tempo_tt_cn, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('E$ \infty$')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_t_log_list, E_inf_depois_log_list, grau) # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs) # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_t_log_list), max(h_t_log_list), 100) # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed', label='Linha de Tendência Linear')
        plt.plot(h_t_log_list, E_inf_depois_log_list, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('ln (E$ \infty$)')
        plt.show()

        # Plotagem:
        plt.plot(h_t_log_list, err_rel_total_list_tempo_tt_cn, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('L1')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_t_log_list, err_rel_total_log_list, grau) # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs) # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_t_log_list), max(h_t_log_list), 100) # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed', label='Linha de Tendência Linear')
        plt.plot(h_t_log_list, err_rel_total_log_list, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('ln (L1)')
        plt.show()

        # Tabelas:
        tabela = PrettyTable(['N° de Blocos', 'Steps de Tempo', 'Norma Euclidiana - L2', 'Erro Absoluto Máximo - E_inf', 'Erro Relativo - Norma L1'])

        for n_t_val, L2_val, E_inf_val, err_rel_val in zip(n_t_array, L2_list_tempo_tt_cn, E_inf_depois_list_tempo_tt_cn, err_rel_total_list_tempo_tt_cn):
            rounded_L2_val = round(L2_val, 3)
            rounded_E_inf_val = round(E_inf_val, 3)
            rounded_err_rel_val = round(err_rel_val, 3)
            rounded_n_t_val = round(n_t_val, 3)
            tabela.add_row([n_x, rounded_n_t_val, rounded_L2_val, rounded_E_inf_val, rounded_err_rel_val])

        print(tabela)

        return L2_list_tempo_tt_cn, E_inf_depois_list_tempo_tt_cn, err_rel_total_list_tempo_tt_cn, n_t_array, n_x

    def calculate_erros_malha():

        rho = 8.92  # g/cm³
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        qw = 25
        L = 100  # cm
        T0 = 50  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        t0 = 0
        tf = 100
        x0 = 0
        xf = L
        variancia = 'malha'

        h_x = [4, 3, 2, 1, 0.5]
        h_t = 0.05
        i = h_t

        def calculate_n_x(xf, x0, j):

            n_x = (xf - x0) / (j)
            return n_x

        n_t = (tf - t0) / (h_t)

        def calculate_h_t_ex(variancia):
            x_ex_array = []
            t_ex_array = []
            T_ex_array = []
            n_x_array = []
            for j in h_x:
                n_x = calculate_n_x(xf, x0, j)
                x_ex, t_ex, T_ex = analitica.calculate_analitica(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_ex_array.append(x_ex)
                T_ex_array.append(T_ex)
                t_ex_array.append(t_ex)
                n_x_array.append(n_x)

            return x_ex_array, t_ex_array, T_ex_array, n_x_array

        x_ex_array, t_ex_array, T_ex_array, n_x_array = calculate_h_t_ex(variancia)

        def calculate_h_t_calc(variancia):
            x_calc_array = []
            t_calc_array = []
            T_calc_array = []
            n_x_array = []
            for j in h_x:
                n_x = calculate_n_x(xf, x0, j)
                x_calc, t_calc, T_calc = CN.calculate_CN_tt(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_calc_array.append(x_calc)
                T_calc_array.append(T_calc)
                t_calc_array.append(t_calc)
                n_x_array.append(n_x)

            return x_calc_array, t_calc_array, T_calc_array, n_x_array

        x_calc_array, t_calc_array, T_calc_array, n_x_array = calculate_h_t_calc(variancia)

        # Cálculo do Erro:

        # Norma L2
        L2_list_malha_tt_cn = []

        for i in range(len(T_calc_array)):  # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            sum = 0
            y_calc = T_calc[:, 2]  # selecinar a coluna
            y_ex = T_ex[:, 2]  # selecinar a coluna
            n_x = n_x_array[i]
            for k in range(1, len(y_ex)):  # acesso a cada linha
                sum = sum + ((abs((y_ex[k] - y_calc[k]) / (y_ex[k]))) ** 2)
            L2 = np.sqrt((1 / (n_x ** 2)) * sum)
            L2_list_malha_tt_cn.append(L2)
        print('L2', L2_list_malha_tt_cn)

        L2_log_list = np.log(L2_list_malha_tt_cn)

        # Norma E_inf
        E_inf_depois_list_malha_tt_cn = []

        for i in range(len(T_calc_array)):  # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[:, 2]
            y_ex = T_ex[:, 2]
            E_inf_antes_list = []
            n_x = n_x_array[i]
            for k in range(len(y_ex)):
                E_inf_antes = abs((y_ex[k] - y_calc[k]))
                E_inf_antes_list.append(E_inf_antes)
            E_inf_depois = max(E_inf_antes_list)
            E_inf_depois_list_malha_tt_cn.append(E_inf_depois)

        E_inf_depois_log_list = np.log(E_inf_depois_list_malha_tt_cn)

        # Norma E_rel
        err_rel_total_list_malha_tt_cn = []

        for i in range(len(T_calc_array)):  # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[:, 2]
            y_ex = T_ex[:, 2]
            err_rel_list = []
            sum = 0
            n_x = n_x_array[i]
            for k in range(1, len(y_ex)):
                err_rel = abs((y_ex[k] - y_calc[k]) / (y_ex[k]))
                err_rel_list.append(err_rel)
            for j in range(len(err_rel_list)):
                sum = sum + err_rel_list[j]
            err_rel_total = 1 / n_x * sum
            err_rel_total_list_malha_tt_cn.append(err_rel_total)

        err_rel_total_log_list = np.log(err_rel_total_list_malha_tt_cn)

        h_x_log_list = []
        # Log de h_t
        for i in range(len(h_x)):
            h_x_novo = (h_x[i]) ** 2
            h_x_novo2 = np.log(h_x_novo)
            h_x_log_list.append(h_x_novo2)
        print('h_t_log', h_x_log_list)

        # Plotagem:
        plt.plot(h_x_log_list, L2_list_malha_tt_cn, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('L2')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_x_log_list, L2_log_list,
                            grau)  # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs)  # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_x_log_list), max(h_x_log_list),
                                  100)  # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed',
                 label='Linha de Tendência Linear')
        plt.plot(h_x_log_list, L2_log_list, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('ln (L2)')
        plt.show()

        # Plotagem:
        plt.plot(h_x_log_list, E_inf_depois_list_malha_tt_cn, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('E$ \infty$')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_x_log_list, E_inf_depois_log_list,
                            grau)  # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs)  # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_x_log_list), max(h_x_log_list),
                                  100)  # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed',
                 label='Linha de Tendência Linear')
        plt.plot(h_x_log_list, E_inf_depois_log_list, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('ln (E$ \infty$)')
        plt.show()

        # Plotagem:
        plt.plot(h_x_log_list, err_rel_total_list_malha_tt_cn, linestyle='none', marker='o', color="#FF007F",label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('L1')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_x_log_list, err_rel_total_log_list,
                            grau)  # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs)  # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_x_log_list), max(h_x_log_list),
                                  100)  # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed',
                 label='Linha de Tendência Linear')
        plt.plot(h_x_log_list, err_rel_total_log_list, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('ln (L1)')
        plt.show()

        # Tabelas:
        tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'Norma Euclidiana - L2', 'Erro Absoluto Máximo - E_inf', 'Erro Relativo - Norma L1'])

        for n_x_val, L2_val, E_inf_val, err_rel_val in zip(n_x_array, L2_list_malha_tt_cn, E_inf_depois_list_malha_tt_cn, err_rel_total_list_malha_tt_cn):
            rounded_L2_val = round(L2_val, 3)
            rounded_E_inf_val = round(E_inf_val, 3)
            rounded_err_rel_val = round(err_rel_val, 3)
            rounded_n_x_val = round(n_x_val, 3)
            tabela.add_row([n_t, rounded_n_x_val, rounded_L2_val, rounded_E_inf_val, rounded_err_rel_val])

        print(tabela)

        return L2_list_malha_tt_cn, E_inf_depois_list_malha_tt_cn, err_rel_total_list_malha_tt_cn, n_x_array, n_t

#________________Análise do Solver Scipy________________

class erros_pp_solv():
    def calculate_erros_tempo():

        rho = 8.92  # g/cm³
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        qw = 25
        L = 100  # cm
        T0 = 50  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        t0 = 0
        tf = 100
        x0 = 0
        xf = L
        variancia = 'tempo'

        h_t = [3,2,1,0.5,0.05]
        h_x = 4
        j = h_x

        def calculate_n_t(tf,t0,i):

            n_t = (tf - t0) / (i)
            return n_t

        n_x = (xf - x0) / (h_x)

        def calculate_h_t_ex(variancia):
            x_ex_array = []
            t_ex_array = []
            T_ex_array = []
            n_t_array = []
            for i in h_t:

                n_t = calculate_n_t(tf, t0, i)
                x_ex, t_ex, T_ex = analitica.calculate_analitica(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_ex_array.append(x_ex)
                T_ex_array.append(T_ex)
                t_ex_array.append(t_ex)
                n_t_array.append(n_t)

            return x_ex_array, t_ex_array, T_ex_array, n_t_array


        x_ex_array, t_ex_array, T_ex_array, n_t_array = calculate_h_t_ex(variancia)


        def calculate_h_t_calc(variancia):
            x_calc_array = []
            t_calc_array = []
            T_calc_array = []
            n_t_array = []
            for i in h_t:

                n_t = calculate_n_t(tf, t0, i)
                x_calc, t_calc, T_calc = BTCS.calculate_BTCS_pp_solv(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_calc_array.append(x_calc)
                T_calc_array.append(T_calc)
                t_calc_array.append(t_calc)
                n_t_array.append(n_t)

            return x_calc_array, t_calc_array, T_calc_array, n_t_array


        x_calc_array, t_calc_array, T_calc_array, n_t_array = calculate_h_t_calc(variancia)

        # Cálculo do Erro:

        # Norma L2
        L2_list_tempo_pp_solv = []
        for i in range(len(T_calc_array)): # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            sum = 0
            y_calc = T_calc[13] # selecinar a coluna
            y_ex = T_ex[13] # selecinar a coluna
            for k in range(len(y_ex)): # acesso a cada linha
                sum = sum + ((abs((y_ex[k]-y_calc[k])/(y_ex[k])))**2)
            L2 = np.sqrt((1/(n_x**2))*sum)
            L2_list_tempo_pp_solv.append(L2)
        print('L2', L2_list_tempo_pp_solv)

        L2_log_list = np.log(L2_list_tempo_pp_solv)

        # Norma E_inf
        E_inf_depois_list_tempo_pp_solv = []

        for i in range(len(T_calc_array)): # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[13]
            y_ex = T_ex[13]
            E_inf_antes_list = []
            for k in range(len(y_ex)):
                E_inf_antes = abs((y_ex[k] - y_calc[k]))
                E_inf_antes_list.append(E_inf_antes)
            E_inf_depois = max(E_inf_antes_list)
            E_inf_depois_list_tempo_pp_solv.append(E_inf_depois)

        E_inf_depois_log_list = np.log(E_inf_depois_list_tempo_pp_solv)

        # Norma E_rel
        err_rel_total_list_tempo_pp_solv = []

        for i in range(len(T_calc_array)): # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[13]
            y_ex = T_ex[13]
            err_rel_list = []
            sum = 0
            for k in range(len(y_ex)):
                err_rel = abs((y_ex[k] - y_calc[k])/(y_ex[k]))
                err_rel_list.append(err_rel)
            for j in range(len(err_rel_list)):
                sum = sum + err_rel_list[j]
            err_rel_total = 1/n_x * sum
            err_rel_total_list_tempo_pp_solv.append(err_rel_total)

        err_rel_total_log_list = np.log(err_rel_total_list_tempo_pp_solv)

        h_t_log_list = np.log(h_t)

        # Plotagem:
        plt.plot(h_t_log_list, L2_list_tempo_pp_solv, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('L2')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_t_log_list, L2_log_list, grau) # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs) # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_t_log_list), max(h_t_log_list), 100) # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed', label='Linha de Tendência Linear')
        plt.plot(h_t_log_list, L2_log_list, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('ln (L2)')
        plt.show()

        # Plotagem:
        plt.plot(h_t_log_list, E_inf_depois_list_tempo_pp_solv, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('E$ \infty$')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_t_log_list, E_inf_depois_log_list, grau) # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs) # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_t_log_list), max(h_t_log_list), 100) # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed', label='Linha de Tendência Linear')
        plt.plot(h_t_log_list, E_inf_depois_log_list, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('ln (E$ \infty$)')
        plt.show()

        # Plotagem:
        plt.plot(h_t_log_list, err_rel_total_list_tempo_pp_solv, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('L1')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_t_log_list, err_rel_total_log_list, grau) # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs) # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_t_log_list), max(h_t_log_list), 100) # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed', label='Linha de Tendência Linear')
        plt.plot(h_t_log_list, err_rel_total_log_list, linestyle='none', marker='o', color="#FF007F", label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup t$ [s]')
        plt.ylabel('ln (L1)')
        plt.show()

        # Tabelas:
        tabela = PrettyTable(['N° de Blocos', 'Steps de Tempo', 'Norma Euclidiana - L2', 'Erro Absoluto Máximo - E_inf', 'Erro Relativo - Norma L1'])

        for n_t_val, L2_val, E_inf_val, err_rel_val in zip(n_t_array, L2_list_tempo_pp_solv, E_inf_depois_list_tempo_pp_solv, err_rel_total_list_tempo_pp_solv):
            rounded_L2_val = round(L2_val, 3)
            rounded_E_inf_val = round(E_inf_val, 3)
            rounded_err_rel_val = round(err_rel_val, 3)
            rounded_n_t_val = round(n_t_val, 3)
            tabela.add_row([n_x, rounded_n_t_val, rounded_L2_val, rounded_E_inf_val, rounded_err_rel_val])

        print(tabela)

        return L2_list_tempo_pp_solv, E_inf_depois_list_tempo_pp_solv, err_rel_total_list_tempo_pp_solv, n_t_array, n_x

    def calculate_erros_malha():

        rho = 8.92  # g/cm³
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        qw = 25
        L = 100  # cm
        T0 = 50  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        t0 = 0
        tf = 100
        x0 = 0
        xf = L
        variancia = 'malha'

        h_x = [4, 3, 2, 1, 0.5]
        h_t = 0.05
        i = h_t

        def calculate_n_x(xf, x0, j):

            n_x = (xf - x0) / (j)
            return n_x

        n_t = (tf - t0) / (h_t)

        def calculate_h_t_ex(variancia):
            x_ex_array = []
            t_ex_array = []
            T_ex_array = []
            n_x_array = []
            for j in h_x:
                n_x = calculate_n_x(xf, x0, j)
                x_ex, t_ex, T_ex = analitica.calculate_analitica(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_ex_array.append(x_ex)
                T_ex_array.append(T_ex)
                t_ex_array.append(t_ex)
                n_x_array.append(n_x)

            return x_ex_array, t_ex_array, T_ex_array, n_x_array

        x_ex_array, t_ex_array, T_ex_array, n_x_array = calculate_h_t_ex(variancia)

        def calculate_h_t_calc(variancia):
            x_calc_array = []
            t_calc_array = []
            T_calc_array = []
            n_x_array = []
            for j in h_x:
                n_x = calculate_n_x(xf, x0, j)
                x_calc, t_calc, T_calc = BTCS.calculate_BTCS_pp_solv(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_calc_array.append(x_calc)
                T_calc_array.append(T_calc)
                t_calc_array.append(t_calc)
                n_x_array.append(n_x)

            return x_calc_array, t_calc_array, T_calc_array, n_x_array

        x_calc_array, t_calc_array, T_calc_array, n_x_array = calculate_h_t_calc(variancia)

        # Cálculo do Erro:

        # Norma L2
        L2_list_malha_pp_solv = []
        for i in range(len(T_calc_array)):  # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            sum = 0
            y_calc = T_calc[:, 2]  # selecinar a coluna
            y_ex = T_ex[:, 2]  # selecinar a coluna
            n_x = n_x_array[i]
            for k in range(1, len(y_ex)):  # acesso a cada linha
                sum = sum + ((abs((y_ex[k] - y_calc[k]) / (y_ex[k]))) ** 2)
            L2 = np.sqrt((1 / (n_x ** 2)) * sum)
            L2_list_malha_pp_solv.append(L2)
        print('L2', L2_list_malha_pp_solv)

        L2_log_list = np.log(L2_list_malha_pp_solv)

        # Norma E_inf
        E_inf_depois_list_malha_pp_solv = []

        for i in range(len(T_calc_array)):  # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[:, 2]
            y_ex = T_ex[:, 2]
            E_inf_antes_list = []
            n_x = n_x_array[i]
            for k in range(len(y_ex)):
                E_inf_antes = abs((y_ex[k] - y_calc[k]))
                E_inf_antes_list.append(E_inf_antes)
            E_inf_depois = max(E_inf_antes_list)
            E_inf_depois_list_malha_pp_solv.append(E_inf_depois)

        E_inf_depois_log_list = np.log(E_inf_depois_list_malha_pp_solv)

        # Norma E_rel
        err_rel_total_list_malha_pp_solv = []

        for i in range(len(T_calc_array)):  # acesso a matriz menor
            T_calc = T_calc_array[i]
            T_ex = T_ex_array[i]
            y_calc = T_calc[:, 2]
            y_ex = T_ex[:, 2]
            err_rel_list = []
            sum = 0
            n_x = n_x_array[i]
            for k in range(1, len(y_ex)):
                err_rel = abs((y_ex[k] - y_calc[k]) / (y_ex[k]))
                err_rel_list.append(err_rel)
            for j in range(len(err_rel_list)):
                sum = sum + err_rel_list[j]
            err_rel_total = 1 / n_x * sum
            err_rel_total_list_malha_pp_solv.append(err_rel_total)

        err_rel_total_log_list = np.log(err_rel_total_list_malha_pp_solv)

        h_x_log_list = []
        # Log de h_t
        for i in range(len(h_x)):
            h_x_novo = (h_x[i]) ** 2
            h_x_novo2 = np.log(h_x_novo)
            h_x_log_list.append(h_x_novo2)
        print('h_t_log', h_x_log_list)

        # Plotagem:
        plt.plot(h_x_log_list, L2_list_malha_pp_solv, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('L2')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_x_log_list, L2_log_list,
                            grau)  # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs)  # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_x_log_list), max(h_x_log_list),
                                  100)  # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed',
                 label='Linha de Tendência Linear')
        plt.plot(h_x_log_list, L2_log_list, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Norma Euclidiana - Norma L2')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('ln (L2)')
        plt.show()

        # Plotagem:
        plt.plot(h_x_log_list, E_inf_depois_list_malha_pp_solv, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('E$ \infty$')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_x_log_list, E_inf_depois_log_list,
                            grau)  # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs)  # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_x_log_list), max(h_x_log_list),
                                  100)  # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed',
                 label='Linha de Tendência Linear')
        plt.plot(h_x_log_list, E_inf_depois_log_list, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Erro Absoluto Máximo - Norma E$ \infty$')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('ln (E$ \infty$)')
        plt.show()

        # Plotagem:
        plt.plot(h_x_log_list, err_rel_total_list_malha_pp_solv, linestyle='none', marker='o', color="#FF007F",label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('L1')
        plt.show()

        # Ajustando uma linha de tendência nos Gráficos LogxLog:
        grau = 1
        coeffs = np.polyfit(h_x_log_list, err_rel_total_log_list,
                            grau)  # ajusta a linha de tendência aos dados e retorna os coeficientes do polinômio
        tendencia = np.poly1d(coeffs)  # cria um polinômio a partir dos coeficientes retornados por polyfit
        x_tendencia = np.linspace(min(h_x_log_list), max(h_x_log_list),
                                  100)  # cria um conjunto de pontos para suavizar a linha de tendência
        plt.plot(x_tendencia, tendencia(x_tendencia), color='green', linestyle='dashed',
                 label='Linha de Tendência Linear')
        plt.plot(h_x_log_list, err_rel_total_log_list, linestyle='none', marker='o', color="#FF007F",
                 label='Erro Analítica/Explícita')
        plt.title('Erro Relativo - Norma L1')
        plt.legend()
        plt.xlabel(r'$\bigtriangleup x$ [s]')
        plt.ylabel('ln (L1)')
        plt.show()

        # Tabelas:
        tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'Norma Euclidiana - L2', 'Erro Absoluto Máximo - E_inf', 'Erro Relativo - Norma L1'])

        for n_x_val, L2_val, E_inf_val, err_rel_val in zip(n_x_array, L2_list_malha_pp_solv, E_inf_depois_list_malha_pp_solv, err_rel_total_list_malha_pp_solv):
            rounded_L2_val = round(L2_val, 3)
            rounded_E_inf_val = round(E_inf_val, 3)
            rounded_err_rel_val = round(err_rel_val, 3)
            rounded_n_x_val = round(n_x_val, 3)
            tabela.add_row([n_t, rounded_n_x_val, rounded_L2_val, rounded_E_inf_val, rounded_err_rel_val])

        print(tabela)

        return L2_list_malha_pp_solv, E_inf_depois_list_malha_pp_solv, err_rel_total_list_malha_pp_solv, n_x_array, n_t

L2_list_tempo_tt_ftcs, E_inf_depois_list_tempo_tt_ftcs, err_rel_total_list_tempo_tt_ftcs, n_t_array, n_x = erros_tt_ftcs.calculate_erros_tempo()
L2_list_malha_tt_ftcs, E_inf_depois_list_malha_tt_ftcs, err_rel_total_list_malha_tt_ftcs, n_x_array, n_t = erros_tt_ftcs.calculate_erros_malha()
L2_list_tempo_tt_btcs, E_inf_depois_list_tempo_tt_btcs, err_rel_total_list_tempo_tt_btcs, n_t_array, n_x = erros_tt_btcs.calculate_erros_tempo()
L2_list_malha_tt_btcs, E_inf_depois_list_malha_tt_btcs, err_rel_total_list_malha_tt_btcs, n_x_array, n_t = erros_tt_btcs.calculate_erros_malha()
L2_list_tempo_tt_cn, E_inf_depois_list_tempo_tt_cn, err_rel_total_list_tempo_tt_cn, n_t_array, n_x = erros_tt_cn.calculate_erros_tempo()
L2_list_malha_tt_cn, E_inf_depois_list_malha_tt_cn, err_rel_total_list_malha_tt_cn, n_x_array, n_t = erros_tt_cn.calculate_erros_malha()
print(L2_list_malha_tt_ftcs)
print(E_inf_depois_list_tempo_tt_ftcs)
print(err_rel_total_list_tempo_tt_ftcs)
print(L2_list_malha_tt_ftcs)

# Tabelas Tempo:
tabela = PrettyTable(['N° de Blocos', 'Steps de Tempo', 'L2 FTCS - Dirchlet', 'L2 BTCS - Dirchlet', 'L2 CN - Dirchlet',
                      'EAM FTCS - Dirchlet', 'EAM BTCS - Dirchlet', 'EAM CN - Dirchlet',
                      'L1 FTCS - Dirchlet', 'L1 BTCS - Dirchlet', 'L1 CN - Dirchlet'])

for n_t_val, L2_tt_ftcs_val, L2_tt_btcs_val, L2_tt_cn_val, E_inf_tt_ftcs_val, E_inf_tt_btcs_val, E_inf_tt_cn_val, err_rel_tt_ftcs_val, err_rel_tt_btcs_val, err_rel_tt_cn_val in zip(n_t_array, L2_list_tempo_tt_ftcs, L2_list_tempo_tt_btcs, L2_list_tempo_tt_cn, E_inf_depois_list_tempo_tt_ftcs, E_inf_depois_list_tempo_tt_btcs, E_inf_depois_list_tempo_tt_cn, err_rel_total_list_tempo_tt_ftcs, err_rel_total_list_tempo_tt_btcs, err_rel_total_list_tempo_tt_cn):
    rounded_L2_tt_ftcs_val = round(L2_tt_ftcs_val, 4)
    rounded_L2_tt_btcs_val = round(L2_tt_btcs_val, 4)
    rounded_L2_tt_cn_val = round(L2_tt_cn_val, 4)
    rounded_E_inf_tt_ftcs_val = round(E_inf_tt_ftcs_val, 4)
    rounded_E_inf_tt_btcs_val = round(E_inf_tt_btcs_val, 4)
    rounded_E_inf_tt_cn_val = round(E_inf_tt_cn_val, 4)
    rounded_err_rel_tt_ftcs_val = round(err_rel_tt_ftcs_val, 4)
    rounded_err_rel_tt_btcs_val = round(err_rel_tt_btcs_val, 4)
    rounded_err_rel_tt_cn_val = round(err_rel_tt_cn_val, 4)
    rounded_n_t_val = round(n_t_val, 4)
    tabela.add_row([n_x, rounded_n_t_val, rounded_L2_tt_ftcs_val, rounded_L2_tt_btcs_val, rounded_L2_tt_cn_val, rounded_E_inf_tt_ftcs_val,
                    rounded_E_inf_tt_btcs_val, rounded_E_inf_tt_cn_val, rounded_err_rel_tt_ftcs_val, rounded_err_rel_tt_btcs_val, rounded_err_rel_tt_cn_val])

print(tabela)

# Tabelas Tempo:
tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'L2 FTCS - Dirchlet', 'L2 BTCS - Dirchlet', 'L2 CN - Dirchlet',
                      'EAM FTCS - Dirchlet', 'EAM BTCS - Dirchlet', 'EAM CN - Dirchlet',
                      'L1 FTCS - Dirchlet', 'L1 BTCS - Dirchlet', 'L1 CN - Dirchlet'])

for n_x_val, L2_tt_ftcs_val, L2_tt_btcs_val, L2_tt_cn_val, E_inf_tt_ftcs_val, E_inf_tt_btcs_val, E_inf_tt_cn_val, \
        err_rel_tt_ftcs_val, err_rel_tt_btcs_val, err_rel_tt_cn_val in zip(n_x_array, L2_list_malha_tt_ftcs,
                                                   L2_list_malha_tt_btcs, L2_list_malha_tt_cn, E_inf_depois_list_malha_tt_ftcs,
                                                   E_inf_depois_list_malha_tt_btcs, E_inf_depois_list_malha_tt_cn, err_rel_total_list_malha_tt_ftcs,
                                                                           err_rel_total_list_malha_tt_btcs, err_rel_total_list_malha_tt_cn):
    rounded_L2_tt_ftcs_val = round(L2_tt_ftcs_val, 4)
    rounded_L2_tt_btcs_val = round(L2_tt_btcs_val, 4)
    rounded_L2_tt_cn_val = round(L2_tt_cn_val, 4)
    rounded_E_inf_tt_ftcs_val = round(E_inf_tt_ftcs_val, 4)
    rounded_E_inf_tt_btcs_val = round(E_inf_tt_btcs_val, 4)
    rounded_E_inf_tt_cn_val = round(E_inf_tt_cn_val, 4)
    rounded_err_rel_tt_ftcs_val = round(err_rel_tt_ftcs_val, 4)
    rounded_err_rel_tt_btcs_val = round(err_rel_tt_btcs_val, 4)
    rounded_err_rel_tt_cn_val = round(err_rel_tt_cn_val, 4)
    rounded_n_x_val = round(n_x_val, 4)
    tabela.add_row([n_t, rounded_n_x_val, rounded_L2_tt_ftcs_val, rounded_L2_tt_btcs_val, rounded_L2_tt_cn_val, rounded_E_inf_tt_ftcs_val,
                    rounded_E_inf_tt_btcs_val, rounded_E_inf_tt_cn_val, rounded_err_rel_tt_ftcs_val, rounded_err_rel_tt_btcs_val, rounded_err_rel_tt_cn_val])

print(tabela)

'''
# Tabelas Tempo:
tabela = PrettyTable(['N° de Blocos', 'Steps de Tempo', 'L2 - Dirchlet - GS', 'L2 - Dirchlet - Scipy', 'L2 - Neumann - GS', 'L2 - Neumann - Scipy'])

for n_t_val, L2_pp_gs_val, L2_pp_solv_val, L2_fp_gs_val, L2_fp_solv_val in zip(n_t_array, L2_list_tempo_pp_gs, L2_list_tempo_pp_solv,
                                                   L2_list_tempo_fp_gs, L2_list_tempo_fp_solv):
    rounded_L2_pp_gs_val = round(L2_pp_gs_val, 4)
    rounded_L2_pp_solv_val = round(L2_pp_solv_val, 4)
    rounded_L2_fp_gs_val = round(L2_fp_gs_val, 4)
    rounded_L2_fp_solv_val = round(L2_fp_solv_val, 4)
    rounded_n_t_val = round(n_t_val, 4)
    tabela.add_row([n_x, rounded_n_t_val, rounded_L2_pp_gs_val, rounded_L2_pp_solv_val, rounded_L2_fp_gs_val, rounded_L2_fp_solv_val])

print(tabela)
'''


'''
# Tabelas Malha:
tabela = PrettyTable(['Steps de Tempo', 'N° de Blocos', 'L2 - Dirchlet - GS', 'L2 - Dirchlet - Scipy', 'L2 - Neumann - GS', 'L2 - Neumann - Scipy'])

for n_x_val, L2_pp_gs_val, L2_pp_solv_val, L2_fp_gs_val, L2_fp_solv_val in zip(n_x_array, L2_list_malha_pp_gs, L2_list_malha_pp_solv,
                                                   L2_list_malha_fp_gs, L2_list_malha_fp_solv):
    rounded_L2_pp_gs_val = round(L2_pp_gs_val, 4)
    rounded_L2_pp_solv_val = round(L2_pp_solv_val, 4)
    rounded_L2_fp_gs_val = round(L2_fp_gs_val, 4)
    rounded_L2_fp_solv_val = round(L2_fp_solv_val, 4)
    rounded_n_x_val = round(n_x_val, 4)
    tabela.add_row([n_t, rounded_n_x_val, rounded_L2_pp_gs_val, rounded_L2_pp_solv_val, rounded_L2_fp_gs_val, rounded_L2_fp_solv_val])

print(tabela)
'''

