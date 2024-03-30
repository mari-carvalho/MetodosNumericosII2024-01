# Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt

# Definindo os parâmetros iniciais
inicio = 0
fim = 1
passo = 0.25
num_pontos = int((fim - inicio) / passo) + 1

# Funções para resolver a equação diferencial
def equacao(x, y):
    return -2*y + x**2

def solucao_analitica(x):
    return 3/4 * np.exp(-2*x) + 0.5*x**2 - 0.5*x + 1/4

# Método de Euler
x_euler = [inicio]
y_euler = [1]
erros_euler = []

for i in range(num_pontos - 1):
    prox_x = x_euler[i] + passo
    x_euler.append(prox_x)
    prox_y = y_euler[i] + passo * equacao(x_euler[i], y_euler[i])
    y_euler.append(prox_y)
    sol_analitica = solucao_analitica(prox_x)
    erro_euler = abs((sol_analitica - prox_y) / prox_y) * 100
    erros_euler.append(erro_euler)

# Método de Runge-Kutta 2ª Ordem
x_rk2 = [inicio]
y_rk2 = [1]
erros_rk2 = []

for i in range(num_pontos - 1):
    prox_x = x_rk2[i] + passo
    x_rk2.append(prox_x)
    k1 = passo * equacao(x_rk2[i], y_rk2[i])
    k2 = passo * equacao(x_rk2[i] + passo/2, y_rk2[i] + k1/2)
    prox_y = y_rk2[i] + k2
    y_rk2.append(prox_y)
    sol_analitica = solucao_analitica(prox_x)
    erro_rk2 = abs((sol_analitica - prox_y) / prox_y) * 100
    erros_rk2.append(erro_rk2)

# Método de Runge-Kutta 4ª Ordem
x_rk4 = [inicio]
y_rk4 = [1]
erros_rk4 = []

for i in range(num_pontos - 1):
    prox_x = x_rk4[i] + passo
    x_rk4.append(prox_x)
    k1 = passo * equacao(x_rk4[i], y_rk4[i])
    k2 = passo * equacao(x_rk4[i] + passo/2, y_rk4[i] + k1/2)
    k3 = passo * equacao(x_rk4[i] + passo/2, y_rk4[i] + k2/2)
    k4 = passo * equacao(x_rk4[i] + passo, y_rk4[i] + k3)
    prox_y = y_rk4[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    y_rk4.append(prox_y)
    sol_analitica = solucao_analitica(prox_x)
    erro_rk4 = abs((sol_analitica - prox_y) / prox_y) * 100
    erros_rk4.append(erro_rk4)

# Gráfico dos métodos
plt.figure()
plt.plot(x_euler, y_euler, label='Euler', color='red')
plt.plot(x_rk2, y_rk2, label='RK23', color='green')
plt.plot(x_rk4, y_rk4, label='RK4', color='blue')
plt.plot(x_euler, [solucao_analitica(xi) for xi in x_euler], '--', label='Solução Analítica', color='purple')
plt.title('Comparação de Métodos de Resolução')
plt.grid(True)
plt.legend()
plt.show()

# Gráfico dos erros
plt.figure()
plt.title('Erros Percentuais')
plt.plot(x_euler[:-1], erros_euler, label='Euler', color='red')
plt.plot(x_rk2[:-1], erros_rk2, label='RK23', color='green')
plt.plot(x_rk4[:-1], erros_rk4, label='RK34', color='blue')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir os erros finais
print('Erros Finais:')
print('Método de Euler:', [round(erro, 4) for erro in erros_euler])
print('Método de Runge-Kutta 2ª Ordem:', [round(erro, 4) for erro in erros_rk2])
print('Método de Runge-Kutta 4ª Ordem:', [round(erro, 4) for erro in erros_rk4])
