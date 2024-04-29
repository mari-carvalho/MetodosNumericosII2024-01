# Prova 1 Métodos Numéricos II
# Discente: Bernardo Skovronski Woitas
# Docente: Prof. Dr. Diego Alex Mayer
# Data: 01/04/2001

# Questão 1)

'''
 A principal diferença entre o método de Euler, Runge-Kutta de 2ª e 4ª ordem é que em relação a Euler é por método de passo unico,
ou seja, ele faz a previsão em relação ao primeiro valor e do próximo fazendo com que seu erro seja maior e seu truncamento é de 2ª ordem. Já falando dos Runge-Kutta, eles são mais eficazes por fazer a previsão considerando
2 ou 4 correções da derivada fazendo com que a resposta se aproxime da analítica conforme o tempo passa e os erros de truncamento para os Runge-Kutta de 2ª e 4ª ordem são, respectivamente,
erros de ordem 3 e 5. Um dos principais motivos que influenciam a escolha do método é o valor computacional do RK que é menor que o de Euler, uma vez que se busque um valor muito
próximo ao analítico. Outra coisa que influencia é o número de passos que quanto menor o passo, mais preciso será a equação.

'''

# Questão 2)

# Bibliotecas
import numpy as np
import matplotlib.pyplot as plt

# Variaveis
m0 = 200 # kg (massa inicial)
t0 = 0.0 # seg (tempo inicial)
a = 2000 # newtpn (aceleração)
fr = 2 # resistência do ar
v0 = 0 # velocidade inicial
tf = 50 # seg (tempo final)
h1 = 1 # passo 1
h2 = 5 # passo 2
h3 = 10 # passo 3

n1 = int((tf-t0)/h1)+1
n2 = int((tf-t0)/h2)+1
n3 = int((tf-t0)/h3)+1

# Função

def analytic_velocity(t):
    return (10 * t) - ((t**2) / 40)

def dev_velocity(v ,t):
    return (2000 - 2*v) / (200 - t)

# Euler com h1
x1_euler = [t0]
y1_euler = [t0]
erro1_euler = []

for i in range (n1-1):
    prox_x = x1_euler[i] + h1
    x1_euler.append(prox_x)
    prox_y = y1_euler + (h1 * dev_velocity(x1_euler,y1_euler))
    y1_euler.append(prox_y)
    #print('x1_euler', x1_euler)
    solucao_analitica = analytic_velocity (x1_euler)
    erro1 = abs((solucao_analitica-prox_y)/prox_y) * 100
    erro1_euler.append(erro1)

print('x1_euler', x1_euler)
print('y1_euler', y1_euler)
print('erro1_euler', erro1_euler)

# Euler com h2
x2_euler = [t0]
y2_euler = [t0]
erro2_euler = []

for i in range (n1-1):
    for j in range (0, 50):
        x2_euler = x2_euler + 1
        y2_euler = y2_euler + 1

    prox_x = x2_euler[i] + h2
    x2_euler.append(prox_x)
    prox_y = y2_euler + (h2 * dev_velocity(x2_euler,y2_euler))
    y2_euler.append(prox_y)
    #print('x2_euler', x2_euler)
    solucao_analitica = analytic_velocity (x2_euler)
    erro2 = abs((solucao_analitica-prox_y)/prox_y) * 100
    erro2_euler.append(erro2)

# Euler com h3
x3_euler = [t0]
y3_euler = [t0]
erro3_euler = []

for i in range (n1-1):
    for j in range (0, 50):
        x3_euler = x3_euler + 1
        y3_euler = y3_euler + 1

    prox_x = x3_euler[i] + h3
    x3_euler.append(prox_x)
    prox_y = y3_euler + (h3 * dev_velocity(x3_euler,y3_euler))
    y3_euler.append(prox_y)
    #print('x3_euler', x3_euler)
    solucao_analitica = analytic_velocity (x3_euler)
    erro3 = abs((solucao_analitica-prox_y)/prox_y) * 100
    erro3_euler.append(erro3)

# Gráficos Euler
plt.figure 
plt.plot(x1_euler,y1_euler)
plt.plot(x2_euler,y2_euler)
plt.plot(x3_euler,y3_euler)
plt.xlabel('Tempo (s)')
plt.ylabel('v(t)')
plt.grid
plt.show()

# Gráficos Erros de Euler
plt.figure 
plt.plot(x1_euler,erro1_euler)
plt.plot(x2_euler,erro2_euler)
plt.plot(x3_euler,erro3_euler)
plt.xlabel('Tempo (s)')
plt.ylabel('v(t)')
plt.grid
plt.show()

# Questão 2) Letra C

'''
Caso decidirmos aplicar os métodos de Runge-Kutta de 2ª e 4ª para determinar a solução, iriamos obter
resultados cada vez mais proximos se comparados com a solução exata. O erro do RK2 iria diminuir menos 
que o de Euler e, consequentemente, o RK4 seria menor ainda, sendo extramente próximo da analítica. 
Outro fator que influenciaria positivamente seria o valor de passos (h1, h2, h3) onde o valor se aproxima
do exato quanto menor for o passo. Então para ter o resultado mais preciso seria necessário implementar o
método de Runge-Kutta de 4ª ordem com o h1 (passo=1).

Infelizmente meu código não rodou devido ao erro "TypeError: unsupported operand type(s) for -: 'int' and 'list'"
que fez com que os meus valores não entrassem nas funções "analytic_velocity" e "dev_velocity" que precisa entrar 
como valor exato e o meu código acabei fazendo por lista. Tentei fazer do jeito que o professor recomendou implementando
um for para que entrasse valores inteiros, mas como faltei as aulas da explicação acabei me perdendo no jeito de fazer.
No #Euler com h1 deixei a minha primeira tentativa e nos seguinte deixei a tentativa pela ajuda do professor.
'''