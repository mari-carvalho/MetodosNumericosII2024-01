# Questão 2 - Solução Numérica - Método de Euler:

import numpy as np 
import matplotlib.pyplot as plt 

# Definindo as variáveis de Entrada:

xf_euler = 2
x0_euler = 0
y0_euler = 1
h_euler = 0.5
n_euler = ((xf_euler-x0_euler)/(h_euler)) + 1

# Definindo os Vetores:

x_euler = np.zeros(int(n_euler))
y_euler = np.zeros(int(n_euler))

# Alimentando os Vetores com as Condições Iniciais:

for i in range(int(n_euler)):
    if i == 0:
        x_euler[i] = x0_euler
        y_euler[i] = y0_euler
    else:
        x_euler[i] = i*h_euler
        y_euler[i] = 0

print('x_euler', x_euler)
print('y_euler', y_euler)

# Definindo a Função de EDO:

def f_euler(x_euler,y_euler):
    return (-2*y_euler) + (x_euler**2)

# Método de Euler:

for i in range(0,int(n_euler) - 1):
    y_euler[i+1] = y_euler[i] + h_euler*f_euler(x_euler[i], y_euler[i])

print('y_euler', y_euler)

# Solução Analítica:

yex_euler = np.zeros(int(n_euler))

for i in range(len(x_euler)):
    yex_euler[i] = ((1/2)*(x_euler[i]**2)) - ((1/2)*(x_euler[i])) + (1/4) + ((3/4)/(np.exp(2*x_euler[i])))

print('yex_euler', yex_euler)

# Cálculo do Erro Percentual Verdadeiro:

erro_euler = np.zeros(int(n_euler))

for i in range(len(y_euler)):
    erro_euler[i] = np.abs((yex_euler[i] - y_euler[i])/yex_euler[i]) * 100

print('erro_euler', erro_euler)

# Plotagem das Soluções:

plt.plot(x_euler, y_euler, linestyle='-', color='blue', label='Solução Numérica')
plt.plot(x_euler, yex_euler, linestyle='dashed', color='green', label='Solução Analítica')

plt.legend()
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid()
plt.show()

# Plotagem do Erro Percentual Verdadeiro:

plt.plot(x_euler, erro_euler, linestyle='-', color='red', label='Erro')

plt.legend()
plt.xlabel('x')
plt.ylabel('Erro Percentual Verdadeiro [%]')
plt.grid()
plt.show()