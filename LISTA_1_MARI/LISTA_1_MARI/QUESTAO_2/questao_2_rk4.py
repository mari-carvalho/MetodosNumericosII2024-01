# Questão 2 - Solução Numérica - Método de Runge-Kutta de 4ª Ordem:

import numpy as np 
import matplotlib.pyplot as plt 

# Definindo as variáveis de Entrada:

xf_rk4 = 2
x0_rk4 = 0
y0_rk4 = 1
h_rk4 = 0.5
n_rk4 = ((xf_rk4-x0_rk4)/(h_rk4)) + 1

# Definindo os Vetores:

x_rk4 = np.zeros(int(n_rk4))
y_rk4 = np.zeros(int(n_rk4))

# Alimentando os Vetores com as Condições Iniciais:

for i in range(int(n_rk4)):
    if i == 0:
        x_rk4[i] = x0_rk4
        y_rk4[i] = y0_rk4
    else:
        x_rk4[i] = i*h_rk4
        y_rk4[i] = 0

print('x_rk4', x_rk4)
print('y_rk4', y_rk4)

# Definindo a Função de EDO:

def f_rk4(x_rk4,y_rk4):
    return (-2*y_rk4) + (x_rk4**2)

# Método de Euler:

for i in range(0,int(n_rk4) - 1):
    k1_rk4 = h_rk4*f_rk4(x_rk4[i], y_rk4[i])
    k2_rk4 = h_rk4*f_rk4(x_rk4[i] + h_rk4/2, y_rk4[i] + k1_rk4/2)
    k3_rk4 = h_rk4*f_rk4(x_rk4[i] + h_rk4/2, y_rk4[i] + k2_rk4/2)
    k4_rk4 = h_rk4*f_rk4(x_rk4[i] + h_rk4, y_rk4[i] + k3_rk4)
    y_rk4[i+1] = y_rk4[i] + (1/6)*(k1_rk4+2*k2_rk4+2*k3_rk4+k4_rk4)

print('y_rk4', y_rk4)

# Solução Analítica:

yex_rk4 = np.zeros(int(n_rk4))

for i in range(len(x_rk4)):
    yex_rk4[i] = ((1/2)*(x_rk4[i]**2)) - ((1/2)*(x_rk4[i])) + (1/4) + ((3/4)/(np.exp(2*x_rk4[i])))

print('yex_rk4', yex_rk4)

# Cálculo do Erro Percentual Verdadeiro:

erro_rk4 = np.zeros(int(n_rk4))

for i in range(len(y_rk4)):
    erro_rk4[i] = np.abs((yex_rk4[i] - y_rk4[i])/yex_rk4[i]) * 100

print('erro_rk4', erro_rk4)

# Plotagem das Soluções:

plt.plot(x_rk4, y_rk4, linestyle='-', color='blue', label='Solução Numérica')
plt.plot(x_rk4, yex_rk4, linestyle='dashed', color='green', label='Solução Analítica')

plt.legend()
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid()
plt.show()

# Plotagem do Erro Percentual Verdadeiro:

plt.plot(x_rk4, erro_rk4, linestyle='-', color='red', label='Erro')

plt.legend()
plt.xlabel('x')
plt.ylabel('Erro Percentual Verdadeiro [%]')
plt.grid()
plt.show()