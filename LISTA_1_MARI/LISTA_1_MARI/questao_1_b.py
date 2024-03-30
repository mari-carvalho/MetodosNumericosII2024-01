# Questão 1 - Letra b - Solução Numérica - Método de Euler:

import numpy as np 
import matplotlib.pyplot as plt 

# Definindo as variáveis de entrada:

xf = 1
x0 = 0
y0 = 1
h = 0.25
n = ((xf-x0)/(h)) + 1
print(n)

# Criando os vetores:

x = np.zeros(int(n))
y = np.zeros(int(n))
print('y', y)
print('x', x)

# Alimentando os vetores com as condições inicias:

for i in range(int(n)):
    print(i) # vai percorrer de 0 até o número inteiro de elementos, i de 0 a 4 para fechar 5 elementos 
    if i == 0:
        x[i] = x0
        y[i] = y0
    else:
        x[i] = i*h
        y[i] = 0
print('x', x)
print('y', y)

# Definindo a EDO:

def f(x,y):
    return (1+4*x)*np.sqrt(y)

# Método de Euler:

for i in range(0, int(n) - 1): # O i de n-1 vai calcular o y[i+1] (último), o i de n não tem como calcular próximo y[i+1] (ultrapassa o número de elementos dos vetores
    y[i+1] = y[i] + h*f(x[i], y[i])
    
print('y', y)

# Solução Analítica:

yex = np.zeros(int(n))

for i in range(len(x)): # percorre todos os valores presentes em x 
    yex[i] = ((x[i] + 2*x[i]**2 + 2)/2)**2

print('yex', yex)

# Cálculo do Erro Percentual Verdadeiro:

erro = np.zeros(int(n))

for i in range(len(y)):
    erro[i] = np.abs((yex[i] - y[i])/yex[i])*100

print(erro)

# Plotagem das soluções:
plt.plot(x, yex, linestyle='dashed', color='blue', label='Solução Analítica')
plt.plot(x,y, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid()
plt.show()

# Plotagem dos erros:

plt.plot(x, erro, linestyle='-', color='red', label='Erro')

plt.legend()
plt.xlabel('x')
plt.ylabel('Erro Perentual Verdadeiro [%]')
plt.grid()
plt.show()
