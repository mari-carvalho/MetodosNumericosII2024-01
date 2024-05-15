from analitica import analitica
from BTCS import BTCS
from FTCS import FTCS
from CN_tt import CN
import matplotlib.pyplot as plt

class comparacao_metodos():
    def calculate_comparacao_metodos():

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

        h_t = [0.05]
        h_x = 4
        j = h_x

        def calculate_n_t(tf,t0,i):

            n_t = (tf - t0) / (i)
            return n_t

        n_x = (xf - x0) / (h_x)

        def calculate_h_t_ex(variancia):
            x_ex_array_analitica = []
            t_ex_array_analitica = []
            T_ex_array_analitica = []
            n_t_array_analitica = []
            for i in h_t:

                n_t = calculate_n_t(tf, t0, i)
                x_ex_analitica, t_ex_analitica, T_ex_analitica = analitica.calculate_analitica(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_ex_array_analitica.append(x_ex_analitica)
                T_ex_array_analitica.append(T_ex_analitica)
                t_ex_array_analitica.append(t_ex_analitica)
                n_t_array_analitica.append(n_t)

            return x_ex_array_analitica, t_ex_array_analitica, T_ex_array_analitica, n_t_array_analitica


        def calculate_h_t_calc_ftcs(variancia):
            x_calc_array_ftcs = []
            t_calc_array_ftcs = []
            T_calc_array_ftcs = []
            n_t_array_ftcs = []
            for i in h_t:

                n_t = calculate_n_t(tf, t0, i)
                x_calc_ftcs, t_calc_ftcs, T_calc_ftcs = FTCS.calculate_FTCS_tt(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_calc_array_ftcs.append(x_calc_ftcs)
                T_calc_array_ftcs.append(T_calc_ftcs)
                t_calc_array_ftcs.append(t_calc_ftcs)
                n_t_array_ftcs.append(n_t)

            return x_calc_array_ftcs, t_calc_array_ftcs, T_calc_array_ftcs, n_t_array_ftcs


        def calculate_h_t_calc_btcs(variancia):
            x_calc_array_btcs = []
            t_calc_array_btcs= []
            T_calc_array_btcs = []
            n_t_array_btcs = []
            for i in h_t:

                n_t = calculate_n_t(tf, t0, i)
                x_calc_btcs, t_calc_btcs, T_calc_btcs = BTCS.calculate_BTCS_tt(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_calc_array_btcs.append(x_calc_btcs)
                T_calc_array_btcs.append(T_calc_btcs)
                t_calc_array_btcs.append(t_calc_btcs)
                n_t_array_btcs.append(n_t)

            return x_calc_array_btcs, t_calc_array_btcs, T_calc_array_btcs, n_t_array_btcs


        def calculate_h_t_calc_cn(variancia):
            x_calc_array_cn = []
            t_calc_array_cn  = []
            T_calc_array_cn  = []
            n_t_array_cn  = []
            for i in h_t:

                n_t = calculate_n_t(tf, t0, i)
                x_calc_cn , t_calc_cn , T_calc_cn  = CN.calculate_CN_tt(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia)

                x_calc_array_cn .append(x_calc_cn )
                T_calc_array_cn .append(T_calc_cn )
                t_calc_array_cn .append(t_calc_cn )
                n_t_array_cn .append(n_t)

            return x_calc_array_cn , t_calc_array_cn , T_calc_array_cn , n_t_array_cn


        x_ex_array_analitica, t_ex_array_analitica, T_ex_array_analitica, n_t_array_analitica = calculate_h_t_ex(
            variancia)
        x_calc_array_ftcs, t_calc_array_ftcs, T_calc_array_ftcs, n_t_array_ftcs = calculate_h_t_calc_ftcs(variancia)
        x_calc_array_btcs, t_calc_array_btcs, T_calc_array_btcs, n_t_array_btcs = calculate_h_t_calc_btcs(variancia)
        x_calc_array_cn, t_calc_array_cn, T_calc_array_cn, n_t_array_cn = calculate_h_t_calc_cn(variancia)

        for i in range(len(T_calc_array_cn)):
            T_ex_analitica = T_ex_array_analitica[i]
            T_calc_ftcs = T_calc_array_ftcs[i]
            T_calc_btcs = T_calc_array_btcs[i]
            T_calc_cn = T_calc_array_cn[i]
            y_ex_analitica = T_ex_analitica[1000]
            y_calc_ftcs = T_calc_ftcs[1000]
            y_calc_btcs = T_calc_btcs[1000]
            y_calc_cn = T_calc_cn[1000]
            x_calc_cn = x_calc_array_cn[i]
            plt.plot(x_calc_cn, y_ex_analitica, linestyle='dashed', color='pink', label='Analítica')
            plt.plot(x_calc_cn, y_calc_ftcs, linestyle='-', color='red', label='Explícita')
            plt.plot(x_calc_cn, y_calc_btcs, linestyle='-', color='green', label='Implícita')
            plt.plot(x_calc_cn, y_calc_cn, linestyle='-', color='blue', label='Crank Nicolson')

            plt.legend()
            plt.xlabel('Comprimento [cm]')
            plt.ylabel('Temperatura [°C]')
            plt.title('Comparação dos Métodos de Solução')
            plt.grid()
            plt.show()



calc = comparacao_metodos.calculate_comparacao_metodos()