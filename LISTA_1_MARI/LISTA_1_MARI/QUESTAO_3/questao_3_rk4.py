# Questão 3 - Solução Numérica - Método de Runge-Kutta 2ª Ordem:

import numpy as np 
import matplotlib.pyplot as plt 

# Definindo as Variáveis de Entrada:

g_rk4 = 9.81
R_rk4 = 6.37*10**6
v0_rk4 = 0
vf_rk4 = 1500
x0_rk4 = 0
t0_rk4 = 0 
h_rk4 = 0.25

# Definindo as Funções de EDO do Sistema:

def fx_rk4(t_rk4, v_rk4, x_rk4):
    return v_rk4

def fv_rk4(t_rk4, v_rk4, x_rk4):
    return (g_rk4) * ((R_rk4**2)/((R_rk4+x_rk4)**2))

# Método de Runge-Kutta de 4ª Ordem:
v_rk4_list = []
x_rk4_list = []
t_rk4_list = []
v_rk4 = v0_rk4
x_rk4 = x0_rk4
t_rk4 = t0_rk4
while v_rk4 <= vf_rk4:
    k1_x_rk4 = h_rk4*fx_rk4(t_rk4, v_rk4, x_rk4)
    k2_x_rk4 = h_rk4*fx_rk4(t_rk4 + h_rk4/2, v_rk4 + k1_x_rk4/2, x_rk4 + k1_x_rk4/2)
    k3_x_rk4 = h_rk4*fx_rk4(t_rk4 + h_rk4/2, v_rk4 + k2_x_rk4/2, x_rk4 + k2_x_rk4/2)
    k4_x_rk4 = h_rk4*fx_rk4(t_rk4 + h_rk4, v_rk4 + k3_x_rk4, x_rk4 + k3_x_rk4)
    x_rk4 = x_rk4 + (1/6)*(k1_x_rk4 + 2*k2_x_rk4 + 2*k3_x_rk4 + k4_x_rk4)
    k1_v_rk4 = h_rk4*fv_rk4(t_rk4, v_rk4, x_rk4)
    k2_v_rk4 = h_rk4*fv_rk4(t_rk4 + h_rk4/2, v_rk4 + k1_v_rk4/2, x_rk4 + k1_v_rk4/2)
    k3_v_rk4 = h_rk4*fv_rk4(t_rk4 + h_rk4/2, v_rk4 + k2_v_rk4/2, x_rk4 + k2_v_rk4/2)
    k4_x_rk4 = h_rk4*fv_rk4(t_rk4 + h_rk4, v_rk4 + k3_v_rk4, x_rk4 + k3_v_rk4)
    v_rk4 = v_rk4 + (1/6)*(k1_v_rk4 + 2*k2_v_rk4 + 2*k3_v_rk4 + k2_v_rk4)
    if v_rk4 >= vf_rk4:
        break
    x_rk4_list.append(x_rk4)
    v_rk4_list.append(v_rk4)
    t_rk4_list.append(t_rk4)
    t_rk4 = t_rk4 + h_rk4

print('t_rk4_list', t_rk4_list)
print('x_rk4__list', x_rk4_list)
print('v_rk4__list', v_rk4_list)

# Plotando as soluções:

plt.plot(t_rk4_list, v_rk4_list, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('v(t)')
plt.grid()
plt.show()

plt.plot(t_rk4_list, x_rk4_list, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('x(t)')
plt.grid()
plt.show()

plt.plot(x_rk4_list, v_rk4_list, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.grid()
plt.show()