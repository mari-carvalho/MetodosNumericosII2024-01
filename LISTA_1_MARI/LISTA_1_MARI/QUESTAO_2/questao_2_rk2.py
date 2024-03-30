# Questão 2 - Solução Numérica - Método de Runge-Kutta de 2ª Ordem:

import numpy as np 
import matplotlib.pyplot as plt 

# Definindo as variáveis de Entrada:

xf_rk2 = 2
x0_rk2 = 0
y0_rk2 = 1
h_rk2 = 0.5
n_rk2 = ((xf_rk2-x0_rk2)/(h_rk2)) + 1

# Definindo os Vetores:

x_rk2 = np.zeros(int(n_rk2))
y_rk2 = np.zeros(int(n_rk2))

# Alimentando os Vetores com as Condições Iniciais:

for i in range(int(n_rk2)):
    if i == 0:
        x_rk2[i] = x0_rk2
        y_rk2[i] = y0_rk2
    else:
        x_rk2[i] = i*h_rk2
        y_rk2[i] = 0

print('x_rk2', x_rk2)
print('y_rk2', y_rk2)

# Definindo a Função de EDO:

def f_rk2(x_rk2,y_rk2):
    return (-2*y_rk2) + (x_rk2**2)

# Método de Euler:

for i in range(0,int(n_rk2) - 1):
    k1_rk2 = h_rk2*f_rk2(x_rk2[i], y_rk2[i])
    k2_rk2 = h_rk2*f_rk2(x_rk2[i+1], y_rk2[i] + k1_rk2)
    y_rk2[i+1] = y_rk2[i] + (1/2)*(k1_rk2+k2_rk2)

print('y_rk2', y_rk2)

# Solução Analítica:

yex_rk2 = np.zeros(int(n_rk2))

for i in range(len(x_rk2)):
    yex_rk2[i] = ((1/2)*(x_rk2[i]**2)) - ((1/2)*(x_rk2[i])) + (1/4) + ((3/4)/(np.exp(2*x_rk2[i])))

print('yex_rk2', yex_rk2)

# Cálculo do Erro Percentual Verdadeiro:

erro_rk2 = np.zeros(int(n_rk2))

for i in range(len(y_rk2)):
    erro_rk2[i] = np.abs((yex_rk2[i] - y_rk2[i])/yex_rk2[i]) * 100

print('erro_rk2', erro_rk2)

# Plotagem das Soluções:

plt.plot(x_rk2, y_rk2, linestyle='-', color='blue', label='Solução Numérica')
plt.plot(x_rk2, yex_rk2, linestyle='dashed', color='green', label='Solução Analítica')

plt.legend()
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid()
plt.show()

# Plotagem do Erro Percentual Verdadeiro:

plt.plot(x_rk2, erro_rk2, linestyle='-', color='red', label='Erro')

plt.legend()
plt.xlabel('x')
plt.ylabel('Erro Percentual Verdadeiro [%]')
plt.grid()
plt.show()