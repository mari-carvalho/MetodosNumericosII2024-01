# Questão 1 - Letra c - Solução Numérica - Método de Runge-Kutta de 2ª Ordem:

import numpy as np 
import matplotlib.pyplot as plt 

# Definindo as Variáveis de Entrada:

x0_rk2 = 0
xf_rk2 = 1
y0_rk2 = 1
h_rk2 = 0.25
n_rk2 = ((xf_rk2-x0_rk2)/(h_rk2)) + 1

# Criando os vetores:

x_rk2 = np.zeros(int(n_rk2))
y_rk2 = np.zeros(int(n_rk2))

# Alimentando os vetores com as condições inicias:

for i in range(int(n_rk2)):
    if i == 0:
        x_rk2[i] = x0_rk2
        y_rk2[i] = y0_rk2
    else:
        x_rk2[i] = i*h_rk2
        y_rk2[i] = 0

print('x', x_rk2)
print('y', y_rk2)

# Definindo a função de EDO:

def f_rk2(x_rk2,y_rk2):
    return (1+4*x_rk2)*np.sqrt(y_rk2)

# Método de Runge-Kutta 2ª Ordem:

for i in range(0,int(n_rk2) -1):
    k1_rk2 = h_rk2*f_rk2(x_rk2[i], y_rk2[i])
    k2_rk2 = h_rk2*f_rk2(x_rk2[i+1], y_rk2[i+1]+ k1_rk2) 
    y_rk2[i+1] = y_rk2[i] + (1/2)*(k1_rk2+k2_rk2)
print('y_rk2', y_rk2)

# Solução Analítica:

yex_rk2 = np.zeros(int(n_rk2))

for i in range(len(x_rk2)):
    yex_rk2[i] = ((x_rk2[i] + 2*(x_rk2[i]**2) + 2)/2)**2

print('yex_rk2', yex_rk2)

# Cálculo do Erro Percentual Verdadeiro:

erro_rk2 = np.zeros(int(n_rk2))

for i in range(len(y_rk2)):
    erro_rk2[i] = np.abs((yex_rk2[i] - y_rk2[i])/yex_rk2[i]) * 100

print('erro_rk2', erro_rk2)

# Plotando as Soluções:

plt.plot(x_rk2, y_rk2, linestyle='-', color='blue', label='Solução Numérica')
plt.plot(x_rk2, yex_rk2, linestyle='dashed', color='green', label='Solução Analítica')

plt.legend()
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid()
plt.show()

# Plotando o Erro Percentual Verdadeiro:

plt.plot(x_rk2, erro_rk2, linestyle='-', color='red', label='Erro')

plt.legend()
plt.xlabel('x')
plt.ylabel('Erro Percentual Verdadeiro [%]')
plt.grid()
plt.show()